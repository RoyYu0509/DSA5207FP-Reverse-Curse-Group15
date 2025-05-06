from datasets import load_dataset
from transformers import AutoTokenizer
import torch

from torch.utils.data import Dataset, DataLoader, Subset
import random
import math

from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import random
import math

def raw_2_llm_data(data_files, model_name, max_input_length, max_target_length, instruction=None):
    """
    Returns:
        - tokenized_datasets: The tokenized data in the correct form for training causal LLMs.
        - tokenizer: The corresponding tokenizer for the checkpoint model `model_name`.
        - device: The torch device being used (GPU or CPU).
    Parameters:
        - data_files: Dict of raw data file paths. Example:
            {"train": "path/to/train.jsonl",
             "validation": "path/to/validation.jsonl",
             "test": "path/to/test.jsonl"}
        - model_name: The HuggingFace model checkpoint name.
        - max_input_length: Maximum token length for the prompt (input).
        - max_target_length: Maximum token length for the completion/response (output).
        - instruction: (optional) A global instruction/prompt string to prepend to every prompt. 
                       Defaults to None (no additional instruction).
    """
    # Import JSON data
    # -------------------------------------------------------
    raw_datasets = load_dataset("json", data_files=data_files)
    print(raw_datasets)  # Display dataset splits and sizes

    # Setup tokenizer for the given `model_name`
    # -------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if hasattr(torch, 'mps') and torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # If the tokenizer has no pad token (common for LLMs), assign the EOS token as the pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'  # use left-padding for decoder-only models

    # Calculate total maximum sequence length (prompt + completion)
    total_max_length = max_input_length + max_target_length

    # Convert to tokenized HF dataset
    # -------------------------------------------------------
    def _preprocess_function(examples):
        """Preprocess examples by tokenizing and combining instruction, prompt, and completion."""
        prompts = examples["prompt"]
        completions = examples["completion"]

        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for prompt, completion in zip(prompts, completions):
            # Optionally prepend the global instruction to the prompt
            if instruction:  
                # If instruction is provided, prefix it to the prompt (separated by a space or newline as needed)
                combined_prompt = f"{instruction} {prompt}"
            else:
                combined_prompt = prompt

            # 1. Tokenize the combined prompt and the completion (no padding at this stage)
            prompt_ids = tokenizer.encode(
                combined_prompt, add_special_tokens=False, truncation=True, max_length=max_input_length
            )  # [MODIFIED] using combined_prompt instead of prompt
            comp_ids = tokenizer.encode(
                completion, add_special_tokens=False, truncation=True, max_length=max_target_length
            )

            # 2. Concatenate prompt and completion token IDs
            full_ids = prompt_ids + comp_ids
            # Truncate to total_max_length if needed
            full_ids = full_ids[: total_max_length]

            # 3. Create labels:
            #    - For prompt tokens (including the instruction) and any padding, label is -100 (ignore in loss).
            #    - For completion tokens, label is the token ID.
            labels = [-100] * len(prompt_ids) + comp_ids
            labels = labels[: total_max_length]

            # 4. Prepare attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1] * len(full_ids)

            # 5. Pad sequences up to total_max_length
            pad_len = total_max_length - len(full_ids)
            if pad_len > 0:
                full_ids += [tokenizer.pad_token_id] * pad_len
                attention_mask += [0] * pad_len
                labels += [-100] * pad_len

            # Collect the processed sequence
            input_ids_list.append(full_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }

    # Map the preprocessing over the dataset splits
    tokenized_datasets = raw_datasets.map(_preprocess_function, batched=True, remove_columns=["prompt", "completion"])
    print(tokenized_datasets)

    return tokenized_datasets, tokenizer, device


import math
import random
import torch
from torch.utils.data import Dataset, DataLoader, Subset

def torch_data_loader(
    data_splits: dict,
    train_file_name: str,
    batch_size: int = 20,
    train_portion_rate: float = 1.0
    
) -> dict:
    """
    Convert a dict of HuggingFace-like dataset splits into PyTorch DataLoaders.

    Parameters:
        data_splits: dict mapping arbitrary split names (e.g. "train", "dev", "foo")
                     to dataset-like objects supporting `split[:]`, yielding dicts of lists.
        batch_size:   batch size for all DataLoaders.
        train_portion_rate: fraction of the "train" split to use (only applies if
                            a split named "train" is present, case-insensitive).

    Returns:
        loader_dict: dict mapping "<split_name>_loader" to a DataLoader.
    """

    class GenericTorchDataset(Dataset):
        """Wraps any HF-style split (dict of lists) as a torch Dataset."""
        def __init__(self, hf_split):
            data = hf_split[:]  # expect dict of lists
            # convert each list to a tensor
            self.tensors = {k: torch.tensor(v) for k, v in data.items()}

        def __len__(self):
            # all tensors are assumed same-length
            return next(iter(self.tensors.values())).shape[0]

        def __getitem__(self, idx):
            # return a dict of tensor slices
            return {k: v[idx] for k, v in self.tensors.items()}

    loader_dict = {}

    for split_name, split_data in data_splits.items():
        # wrap the raw split in our torch Dataset
        dataset = GenericTorchDataset(split_data)

        # if this is the train split and we want only part of it:
        if split_name == train_file_name and 0.0 < train_portion_rate < 1.0:
            total = len(dataset)
            subset_size = math.floor(train_portion_rate * total)
            indices = list(range(total))
            random.shuffle(indices)
            dataset = Subset(dataset, indices[:subset_size])

        # shuffle only the training split by default:
        do_shuffle = (split_name == train_file_name)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=do_shuffle)
        loader_dict[f"{split_name}_loader"] = loader

    return loader_dict