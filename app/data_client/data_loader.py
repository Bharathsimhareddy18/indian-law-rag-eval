from datasets import load_dataset
from pathlib import Path
import pickle
import ast
from app.core.config import settings

DATA_CACHE = settings.DATA_PATH


def dataLoader():

    if DATA_CACHE.exists():
        print("Loading data from cache...")
        with open(DATA_CACHE, "rb") as f:
            return pickle.load(f)

    content = []

    print("Downloading nandhakumarg/IPC_and_BNS_transformation...")
    ipc_bns_dataset = load_dataset("nandhakumarg/IPC_and_BNS_transformation", split="train")

    for row in ipc_bns_dataset:
        try:
            data = ast.literal_eval(row["response"])

            ipc_text = (
                f"IPC Section {data.get('IPC Section', '')}: "
                f"{data.get('IPC Heading', '')}\n"
                f"{data.get('IPC Descriptions', '')}"
            ).strip()
            if ipc_text:
                content.append(ipc_text)

            bns_text = (
                f"BNS Section {data.get('BNS Section', '')}: "
                f"{data.get('BNS Heading', '')}\n"
                f"{data.get('BNS description', '')}"
            ).strip()
            if bns_text:
                content.append(bns_text)

        except (ValueError, KeyError):
            continue

    print(f"Total corpus size: {len(content)} documents")

    DATA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_CACHE, "wb") as f:
        pickle.dump(content, f)

    print(f"Corpus cached to {DATA_CACHE}")
    return content