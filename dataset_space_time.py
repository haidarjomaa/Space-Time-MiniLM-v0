import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

SEQ_LEN, BATCH_SIZE = 128, 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

space_mapping = {'UK': 0, 'US': 1, 'AUS': 2, 'CAN': 3}
time_mapping = {
    f"{year}-{month:02d}": i
    for i, (year, month) in enumerate(
        [(y, m) for y in range(2017, 2022 + 1) for m in range(1, 13)]
    )
    if i < 60
}

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

class PairwiseSimilarityDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        return {
            "sent1": row.sent1,
            "sent2": row.sent2,
            "t1":    time_mapping[row.t1],
            "t2":    time_mapping[row.t2],
            "s1":    space_mapping[row.s1],
            "s2":    space_mapping[row.s2],
            "sim":    row.similarity
        }

def collate_fn(batch):
    texts = [b["sent1"] for b in batch] + [b["sent2"] for b in batch]
    enc = tokenizer(
        texts,
        padding="longest",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    B = len(batch)
    t1   = torch.tensor([b["t1"] for b in batch], dtype=torch.long)
    t2   = torch.tensor([b["t2"] for b in batch], dtype=torch.long)
    s1   = torch.tensor([b["s1"] for b in batch], dtype=torch.long)
    s2   = torch.tensor([b["s2"] for b in batch], dtype=torch.long)
    sims = torch.tensor([b["sim"] for b in batch], dtype=torch.float)
    return enc, B, s1, s2, t1, t2, sims