from datasets import load_dataset


def dataLoader():
    dataset = load_dataset("mratanusarkar/Indian-Laws", split="train")
    
    content = []
    
    for row in dataset:
        doc = f"{row['section']}\n{row['law']}"
        content.append(doc)
        
    return content