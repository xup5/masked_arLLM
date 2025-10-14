import torch
from torch.utils.data import Dataset, DataLoader
import os
import torch
import json
from pathlib import Path

DATASETS_DIR = Path(os.getenv(
    "MASKED_ARLLM_DATASETS_DIR",
    str(Path(__file__).resolve().parents[1] / "datasets")
))

def load_train_dataset(dataset_name="wiki", paraphrases = "None"):
    # Wiki and Biography datasets have the same structure
    assert dataset_name in ["wiki", "biography", "ND"]
    assert paraphrases in ["None", "same_order", "change_order"]
    
    if paraphrases == "None":
        if dataset_name == "wiki":
            file_path = DATASETS_DIR / "wiki_train_no_paraphrase.json"
        elif dataset_name == "biography":
            file_path = DATASETS_DIR / "biography_train_no_paraphrase.json"
        elif dataset_name == "ND":
            file_path = DATASETS_DIR / "ND_train_no_paraphrase.json"
    elif paraphrases == "same_order":
        if dataset_name == "wiki":
            file_path = DATASETS_DIR / "wiki_train_paraphrase_same_order.json"
        elif dataset_name == "biography":
            file_path = DATASETS_DIR / "biography_train_paraphrase_same_order.json"
        elif dataset_name == "ND":
            file_path = DATASETS_DIR / "ND_train_paraphrase.json"
    elif paraphrases == "change_order":
        if dataset_name == "wiki":
            file_path = DATASETS_DIR / "wiki_train_paraphrase_change_order.json"
        elif dataset_name == "biography":
            file_path = DATASETS_DIR / "biography_train_paraphrase_change_order.json"
        elif dataset_name == "ND":
            file_path = DATASETS_DIR / "ND_train_paraphrase.json"

    with open(file_path, 'r') as f:
        dataset_json = json.load(f)
    
    dataset = []    
    for item in dataset_json:
        dataset.append(item['text'])
    
    return dataset
 
 
def load_test_dataset(dataset_name="wiki"):
    assert dataset_name in ["wiki", "biography", "ND"]
    
    if dataset_name == "wiki":
        file_path = DATASETS_DIR / "wiki_test.jsonl"
    elif dataset_name == "biography":
        file_path = DATASETS_DIR / "biography_test.jsonl"
    elif dataset_name == "ND":
        file_path = DATASETS_DIR / "ND_test.jsonl"
        
    with open(file_path, 'r') as f:
        dataset_json = [json.loads(line) for line in f]
        
    return dataset_json


################################################################################################################

# For AR and dLLM training
    
class Dataset_pretraining_format(Dataset):
    def __init__(self, dataset, 
                 tokenizer, 
                 device="cpu"):
        self.samples = []
        # Create samples for each title/passage pair with all fine-tuning prompts
        for text in dataset:

            # Build conversation text
            text = (
                f"{tokenizer.bos_token}"
                f"{text}"
                f"{tokenizer.eos_token}"
            )
            
            tokenized = tokenizer(text, return_tensors="pt", add_special_tokens=False)#.to(device)
            input_ids = tokenized["input_ids"][0]
            attention_mask = tokenized["attention_mask"][0]
            labels = input_ids.clone()

            self.samples.append({
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
                "text": text
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.samples[i] for i in idx]
        return self.samples[idx]
    

def get_collate_fn_function(model_type="llama"):
    if model_type == "llama":
        pad_token_id = 128009  # Llama 3.1 8B tokenizer eos token id
    elif model_type == "llada":  #else it's LLaDA
        pad_token_id = 126081  # LLaDA model compatible EOS token id (within vocab range 0-126463)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    def collate_fn_question(batch):
        input_ids = [b["input_ids"] for b in batch]
        labels = [b["labels"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]

        # Pad them so they match the longest sequence in the batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, 
            batch_first=True, 
            padding_value=pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, 
            batch_first=True, 
            padding_value=-100
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, 
            batch_first=True, 
            padding_value=0
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "text": [b["text"] for b in batch]
        }
        
    return collate_fn_question

    
def get_training_dataloader(dataset_name, 
                            tokenizer, 
                            Dataset_class=Dataset_pretraining_format, 
                            paraphrases="None",
                            batch_size=2, 
                            device="cpu", 
                            model_type="llama"
                            ):
    
    dataset = load_train_dataset(dataset_name=dataset_name, paraphrases=paraphrases)
    
    story_raw_text_dataset = Dataset_class(dataset, tokenizer, device=device)
    
    data_loader_stories = DataLoader(
        story_raw_text_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=get_collate_fn_function(model_type=model_type)
    )

    return data_loader_stories

################################################################################################################

# For evaluation
    
class Dataset_question(Dataset):
    def __init__(self, question_list, tokenizer, device="cpu"):
        self.samples = []
        for item in question_list:
            text = (
                    f"{tokenizer.bos_token}"
                    f"<|start_header_id|>user<|end_header_id|>\n\n"
                    f"{item['question']}"
                    f"<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                )
            tokenized = tokenizer(text, return_tensors="pt", add_special_tokens=False)#.to(device)
            input_ids = tokenized["input_ids"][0]
            attention_mask = tokenized["attention_mask"][0]
            labels = input_ids.clone()

            self.samples.append({
                "group": item['group'],
                "task": item['task'],
                "question": item['question'],
                "answer": item['answer'],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.samples[i] for i in idx]
        return self.samples[idx]
    

def get_collate_fn_question_function(model_type="llama"):
    if model_type == "llama":
        pad_token_id = 128009  # Llama 3.1 8B tokenizer eos token id
    elif model_type == "llada":  #else it's LLaDA
        pad_token_id = 126081  # LLaDA model compatible EOS token id (within vocab range 0-126463)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    def collate_fn_question(batch):
        """
        Collate a list of samples into a single batch.
        We need to pad input_ids and labels to the same length.
        """
        # pad_token_id = 128009  # Llama 3.1 8B tokenizer eos token id
        # Use pad_token_id captured from outer scope

        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]

        # Find the max length
        max_len = max(x.size(0) for x in input_ids)

        # Left pad each sequence
        def left_pad(tensor, pad_value, max_len):
            pad_size = max_len - tensor.size(0)
            if pad_size == 0:
                return tensor
            return torch.cat([torch.full((pad_size,), pad_value, dtype=tensor.dtype, device=tensor.device), tensor], dim=0)

        input_ids = torch.stack([left_pad(x, pad_token_id, max_len) for x in input_ids], dim=0)
        attention_mask = torch.stack([left_pad(x, 0, max_len) for x in attention_mask], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "group": [b["group"] for b in batch],
            "task": [b["task"] for b in batch],
            "question": [b["question"] for b in batch],
            "answer": [b["answer"] for b in batch],
        }

    return collate_fn_question
    
def get_testing_dataloader(dataset_name, tokenizer, Dataset_class=Dataset_question, batch_size=2, device="cpu", model_type="llama"):
    dataset = load_test_dataset(dataset_name=dataset_name)
    
    question_dataset = Dataset_class(dataset, tokenizer, device=device)
    
    data_loader = DataLoader(
        question_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=get_collate_fn_question_function(model_type=model_type)
    )

    return data_loader


################################################################################################################

# For masked AR

class Dataset_AR_mask(Dataset):
    def __init__(self, dataset,
                 tokenizer, 
                 device="cpu", 
                 **kwargs):
        
        """
        Total prompt:
        <|start_header_id|>user<|end_header_id|>\n\n
        {text}\n
        Return the recovered masked passage.<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>\n\n
        \n\n
        Here is the recovered text:\n\n
        {text}
        <|eot_id|>
        """
        self.samples = []
        
        for text in dataset:
            # Build conversation text
            total_input_ids = tokenizer.encode("<|start_header_id|>user<|end_header_id|>\n\n")
            maskable=[0] * len(total_input_ids)
            
            tokenized_text = tokenizer.encode(text, add_special_tokens=False)
            total_input_ids += tokenized_text
            maskable += [1] * len(tokenized_text)
            
            retrieval_prompt = tokenizer.encode(f"\nReturn the recovered masked passage.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHere is the recovered text:\n\n", add_special_tokens=False)
            total_input_ids += retrieval_prompt
            maskable += [0] * len(retrieval_prompt)
            labels = [-100] * len(total_input_ids)
            
            
            target = tokenized_text + tokenizer.encode("<|eot_id|>", add_special_tokens=False)
            total_input_ids += target
            maskable += [0] * len(target)
            labels += target
            
            
            total_input_ids = torch.tensor(total_input_ids)#.to(device)
            maskable = torch.tensor(maskable)#.to(device)
            labels = torch.tensor(labels)#.to(device)
            attention_mask = torch.ones_like(total_input_ids)#.to(device)

            self.samples.append({
                "input_ids": total_input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
                "maskable": maskable,
                "total_prompt": tokenizer.decode(total_input_ids)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.samples[i] for i in idx]
        return self.samples[idx]

def collate_fn_AR_mask(batch):
    """
    Collate a list of samples into a single batch.
    We need to pad input_ids and labels to the same length.
    """
    # pad_token_id = tokenizer.pad_token_id
    # if pad_token_id is None:
    
    pad_token_id = 128009 # Llama 3.1 8B tokenizer eos token id

    # pad_token_id = 126081 # LLaDA model compatible EOS token id (within vocab range 0-126463)
    
    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    maskable = [b["maskable"] for b in batch]
    total_prompt = [b["total_prompt"] for b in batch]
    
    # Pad them so they match the longest sequence in the batch
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, 
        batch_first=True, 
        padding_value=pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, 
        batch_first=True, 
        padding_value=-100
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, 
        batch_first=True, 
        padding_value=0
    )
    maskable = torch.nn.utils.rnn.pad_sequence(
        maskable, 
        batch_first=True, 
        padding_value=0
    ).bool()

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "maskable": maskable,
        "total_prompt": total_prompt
    }
    

def get_training_dataloader_AR_mask(dataset_name, 
                            tokenizer, 
                            Dataset_class=Dataset_AR_mask, 
                            paraphrases="None", 
                            batch_size=2, 
                            device="cpu",
                            **kwargs):

    dataset = load_train_dataset(dataset_name=dataset_name, paraphrases=paraphrases)
    masked_ar_dataset = Dataset_class(dataset, tokenizer, device=device)
    
    data_loader_stories = DataLoader(
        masked_ar_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn_AR_mask
    )

    return data_loader_stories