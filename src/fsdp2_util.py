import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp import CPUOffloadPolicy


from contextlib import nullcontext
from accelerate import init_empty_weights
import torch.distributed as dist
from torch.distributed import DeviceMesh

def get_init_weight_context_manager(*, use_meta_tensor: bool = True, mesh: DeviceMesh | None = None):
    """
    Return a context-manager factory:
      - rank 0 (or mesh coord == 0): nullcontext  -> real weights on CPU
      - other ranks: init_empty_weights           -> meta tensors
    """
    if not use_meta_tensor:
        return nullcontext

    if mesh is not None:
        # Last coord is usually the data-parallel rank
        is_rank0 = (mesh.get_coordinate()[-1] == 0)
    else:
        r = dist.get_rank() if dist.is_initialized() else 0
        is_rank0 = (r == 0)

    return nullcontext if is_rank0 else init_empty_weights

def apply_fsdp2(model, fsdp_kwargs, config={}):
    """model: AutoModelForCausalLM"""
    assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"

    default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = config.get("wrap_policy", {}).get("transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap)

    if isinstance(fsdp_transformer_layer_cls_to_wrap, str):
        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]

    assert len(fsdp_transformer_layer_cls_to_wrap) > 0 and fsdp_transformer_layer_cls_to_wrap[0] is not None

    modules = []
    for name, module in model.named_modules():
        if module.__class__.__name__ in fsdp_transformer_layer_cls_to_wrap or (isinstance(module, nn.Embedding) and not model.config.tie_word_embeddings):
            modules.append(module)
            
    print("FSDP modules: ", modules)

    for idx, module in enumerate(modules):
        fully_shard(module, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)  # fsdp2 will not reshard_after_forward for root module
    
def fsdp2_load_full_state_dict(model: torch.nn.Module, full_state: dict, device_mesh=None):
    """
    Loads the full state dict (could be only on rank 0) into the sharded model. This is done by broadcasting the
    parameters from rank 0 to all other ranks. This function modifies the model in-place.

    Args:
        model (`torch.nn.Module`): The model to load the state dict into
        full_state (`dict`): The full state dict to load, can only be on rank 0
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    # To broadcast, it needs to be instantiated in the GPU.
    if dist.get_rank() == 0:
        model = model.to(device=torch.cuda.current_device(), non_blocking=True)
    else:
        model = model.to_empty(device=torch.cuda.current_device())

    #cpu_offload = cpu_offload is not None
    options = StateDictOptions(full_state_dict=True, cpu_offload=None, broadcast_from_rank0=True)
    set_model_state_dict(model, full_state, options=options)

    # rotary_emb is not in state_dict, so we need to broadcast it manually
    for name, buf in model.named_buffers():
        dist.broadcast(buf, src=0)