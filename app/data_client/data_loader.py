from datasets import load_dataset
from pathlib import Path
import pickle
from app.core.config import settings

DATA_CACHE = settings.DATA_PATH



def dataLoader():
    
    # to see if we have existing dataset
    if DATA_CACHE.exists():
        print("Loading data from cache...")
        with open(DATA_CACHE, "rb") as f:
            return pickle.load(f)
        
    #downloading dataset    
    dataset = load_dataset("mratanusarkar/Indian-Laws", split="train")
    content = []
    for row in dataset:
        doc = f"{row['section']}\n{row['law']}"
        content.append(doc)
        
        
    # saving the dataset
    DATA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_CACHE, "wb") as f:
        pickle.dump(content, f)

    print(f"Corpus cached to {DATA_CACHE}")
    
    return content