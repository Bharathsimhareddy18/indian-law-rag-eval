from datasets import load_dataset
# nandhakumarg/IPC_and_BNS_transformation
# Has both IPC AND BNS 2023 (the new replacement act), structured per section
ds = load_dataset("nandhakumarg/IPC_and_BNS_transformation", split="train")
print(ds.column_names)
print(ds[0])