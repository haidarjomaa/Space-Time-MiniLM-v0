from modeling_custom_minilm import SpaceTimeMiniLM
from transformers import AutoTokenizer, AutoConfig

tokenizer = AutoTokenizer.from_pretrained("HaidarJomaa/Space-Time-MiniLM-v0")
config = AutoConfig.from_pretrained("HaidarJomaa/Space-Time-MiniLM-v0")
model = SpaceTimeMiniLM.from_pretrained("HaidarJomaa/Space-Time-MiniLM-v0", config=config)

sim = model.compute_similarity(
    "The quick brown fox",
    "A speedy auburn fox",
    time1="2020-01", time2="2020-02",
    space1="UK", space2="UK",
    tokenizer=tokenizer
)
print("Similarity:", sim)