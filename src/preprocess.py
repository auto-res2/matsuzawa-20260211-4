"""
Data preprocessing for GSM8K dataset.
"""
import re
from typing import List, Dict, Any, Tuple
from datasets import load_dataset
from omegaconf import DictConfig


def extract_answer_from_gsm8k(answer_text: str) -> float:
    """
    Extract numeric answer from GSM8K answer format.
    GSM8K answers are in format: "Step-by-step solution\n#### 42"
    """
    # Look for #### followed by number
    match = re.search(r'####\s*(-?\d+\.?\d*)', answer_text)
    if match:
        return float(match.group(1))
    
    # Fallback: find last number in text
    numbers = re.findall(r'-?\d+\.?\d*', answer_text)
    if numbers:
        return float(numbers[-1])
    
    return None


def parse_split_string(split_str: str) -> Tuple[str, slice]:
    """
    Parse split string like "train[:500]" into split name and slice.
    
    Args:
        split_str: Split specification like "train[:500]" or "test"
    
    Returns:
        (split_name, slice_obj)
    """
    if '[' in split_str:
        # Parse slice notation
        split_name = split_str.split('[')[0]
        slice_part = split_str.split('[')[1].rstrip(']')
        
        if ':' in slice_part:
            parts = slice_part.split(':')
            start = int(parts[0]) if parts[0] else None
            end = int(parts[1]) if parts[1] else None
            return split_name, slice(start, end)
        else:
            # Single index
            idx = int(slice_part)
            return split_name, slice(idx, idx + 1)
    else:
        # No slice, use all
        return split_str, slice(None, None)


def load_gsm8k_data(cfg: DictConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load GSM8K dataset for chain-of-thought experiments.
    
    Args:
        cfg: Configuration with dataset settings
    
    Returns:
        (train_data, test_data) as lists of dicts with 'question' and 'answer' keys
    """
    print(f"[preprocess] Loading GSM8K dataset from cache_dir={cfg.dataset.cache_dir}")
    
    # Load dataset from HuggingFace
    dataset = load_dataset(
        "gsm8k",
        "main",
        cache_dir=cfg.dataset.cache_dir
    )
    
    # Parse train split
    train_split_name, train_slice = parse_split_string(cfg.dataset.split_train)
    raw_train = dataset[train_split_name]
    
    if train_slice.start is not None or train_slice.stop is not None:
        start = train_slice.start or 0
        stop = train_slice.stop or len(raw_train)
        raw_train = raw_train.select(range(start, min(stop, len(raw_train))))
    
    # Parse test split
    test_split_name, test_slice = parse_split_string(cfg.dataset.split_test)
    raw_test = dataset[test_split_name]
    
    if test_slice.start is not None or test_slice.stop is not None:
        start = test_slice.start or 0
        stop = test_slice.stop or len(raw_test)
        raw_test = raw_test.select(range(start, min(stop, len(raw_test))))
    
    # Convert to list format
    train_data = []
    for item in raw_train:
        train_data.append({
            'question': item['question'],
            'answer': extract_answer_from_gsm8k(item['answer'])
        })
    
    test_data = []
    for item in raw_test:
        test_data.append({
            'question': item['question'],
            'answer': extract_answer_from_gsm8k(item['answer'])
        })
    
    print(f"[preprocess] Loaded {len(train_data)} train samples, {len(test_data)} test samples")
    
    return train_data, test_data
