import torch
import numpy as np
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModel



def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


@ torch.no_grad()
def generate_batch(model, prompts, steps=128, gen_length=128, block_length=128, temperature=0.,
                   cfg_scale=0., remasking='low_confidence', mask_id=126336, pad_id=None):
    '''
    Batch version of generate function that supports multiple prompts.
    
    Args:
        model: Mask predictor.
        prompts: A list of tensors or a single tensor of shape (B, L_max) where B is batch size.
                 If list, prompts can have different lengths and will be padded.
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        pad_id: Padding token id. If None, uses mask_id for padding.
        
    Returns:
        tuple: (full_output, prompt_lengths)
            - full_output: Tensor of shape (B, L_max + gen_length) containing prompt + generated tokens
            - prompt_lengths: Tensor of shape (B,) containing the length of each prompt
            
    Usage:
        # Option 1: Unpack the tuple
        full_output, prompt_lengths = generate_batch(model, prompts, ...)
        generated_texts = tokenizer.batch_decode(full_output, skip_special_tokens=True)
        
        # Option 2: Extract only generated tokens
        full_output, prompt_lengths = generate_batch(model, prompts, ...)
        generated_tokens = extract_generated_tokens(full_output, prompt_lengths)
        generated_texts = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in generated_tokens]
    '''
    if pad_id is None:
        pad_id = mask_id
        
    # Handle different input formats
    if isinstance(prompts, list):
        # List of tensors with potentially different lengths
        batch_size = len(prompts)
        max_prompt_len = max(prompt.shape[-1] for prompt in prompts)
        
        # Pad all prompts to the same length
        padded_prompts = torch.full((batch_size, max_prompt_len), pad_id, dtype=torch.long, device=model.device)
        prompt_lengths = []
        
        for i, prompt in enumerate(prompts):
            prompt_len = prompt.shape[-1]
            prompt_lengths.append(prompt_len)
            padded_prompts[i, :prompt_len] = prompt.to(model.device)
        
        prompt_lengths = torch.tensor(prompt_lengths, device=model.device)
    else:
        # Single tensor, assume all prompts have same length or are already padded
        padded_prompts = prompts.to(model.device)
        batch_size = padded_prompts.shape[0]
        max_prompt_len = padded_prompts.shape[1]
        # Assume all prompts are the same length (no padding tokens)
        prompt_lengths = torch.full((batch_size,), max_prompt_len, device=model.device, dtype=torch.long)

    # Create the generation tensor
    x = torch.full((batch_size, max_prompt_len + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :max_prompt_len] = padded_prompts.clone()

    # Create prompt index mask - only considers actual prompt tokens, not padding
    prompt_index = torch.zeros_like(x, dtype=torch.bool)
    for i in range(batch_size):
        prompt_index[i, :prompt_lengths[i]] = True

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = max_prompt_len + num_block * block_length
        block_end = max_prompt_len + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        
        for i in range(steps):
            mask_index = (x == mask_id)
            
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # Mask out confidence for tokens beyond current block
            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                if num_transfer_tokens[j, i] > 0:
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    # Move tensors to CPU for tokenizer compatibility
    x_cpu = x.cpu()
    prompt_lengths_cpu = prompt_lengths.cpu()
    
    return x_cpu


def extract_generated_tokens(full_output, prompt_lengths):
    """
    Helper function to extract only the generated tokens from batch output.
    
    Args:
        full_output: Full output tensor from generate_batch (B, L)
        prompt_lengths: Tensor of prompt lengths for each batch item
        
    Returns:
        List of tensors, each containing only the generated tokens for each prompt
    """
    generated_tokens = []
    for i, prompt_len in enumerate(prompt_lengths):
        generated_tokens.append(full_output[i, prompt_len:])
    return generated_tokens


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    print(f"Model loaded on device: {device}")
    print(f"Model is {model}")
    # Test prompt
    prompt = "Tell me everything you know about Albert Einstein"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(model, input_ids, steps=50, gen_length=50, block_length=10, temperature=0., cfg_scale=0., remasking='low_confidence')
    print("Generated output (skip_special_tokens=True):")
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])
    print("Generated output (skip_special_tokens=False):")
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=False)[0])
    print(f"Generated tokens shape: {out[:, input_ids.shape[1]:].shape}")
    print(f"Actual generated tokens: {out[:, input_ids.shape[1]:].tolist()}")
    

if __name__ == '__main__':
    main()