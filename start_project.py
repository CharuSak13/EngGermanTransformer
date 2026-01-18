import torch
import spacy
from datasets import load_dataset

def initialize():
    print("\n--- 1. SYSTEM CHECK ---")
    if torch.cuda.is_available():
        print(f"✅ GPU Ready: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Running on CPU (This is fine for setup)")

    print("\n--- 2. DOWNLOADING DATASET (Multi30k) ---")
    # This downloads the English-German data from HuggingFace
    dataset = load_dataset("bentrevett/multi30k")
    
    print(f"✅ Download Success!")
    print(f"   Train Size: {len(dataset['train'])}")
    print(f"   Test Size:  {len(dataset['test'])}")
    
    print("\n--- 3. VERIFYING DATA ---")
    sample = dataset['train'][0]
    print(f"   [EN]: {sample['en']}")
    print(f"   [DE]: {sample['de']}")

if __name__ == "__main__":
    initialize()