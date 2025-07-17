---
language: en
license: apache-2.0
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- transformers
pipeline_tag: sentence-similarity
---


# all-MiniLM-L6-v2
This is a [sentence-transformers](https://www.SBERT.net) model: It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search. It has a special addition, and that is the ability to incorporate spatiotemporal information into the embeddings.

## Usage (HuggingFace Transformers)
You can use the model like this: First, you pass your input through the transformer model, then you get your spatiotemporal semantic embeddings.

```python
from modeling_custom_minilm import SpaceTimeMiniLM
from transformers import AutoTokenizer, AutoConfig

tokenizer = AutoTokenizer.from_pretrained("HaidarJomaa/Space-Time-MiniLM-v0")
config = AutoConfig.from_pretrained("HaidarJomaa/Space-Time-MiniLM-v0")
model = SpaceTimeMiniLM.from_pretrained("HaidarJomaa/Space-Time-MiniLM-v0",
        config=config)

sim = model.compute_similarity(
    "The quick brown fox", "A speedy auburn fox",
    time1="2020-01", time2="2020-02", space1="UK", space2="UK",
    tokenizer=tokenizer
)
print("Similarity:", sim)

emb = model.embed_sentence("why is the sky blue?",
        "2020-11", "US",
        tokenizer
)
print("Embeddings:", emb)
```

------

## Background

The project aims to train sentence embedding models that are able to capture
semantic and spatiotemporal relationships. This is part of the research conducted showcasing
the imporantance of time and space when considering the context. We used the pretrained [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model and fine-tuned in on a 
2 custom datasets (which we will be uploading in the near future). The first dataset included consisted of sentence-time-space triplets and the second consisted of sentence pairs with space-time information for each.

We developed this model as a research project at the American University of Beirut, Department of Computer Science.
[Check out the preprint paper](https://drive.google.com/file/d/1pcz5ckoBkP4wQ7ojY5r14g1LXoYaqL_1/view?usp=drive_link).

## Intended uses

Our model is intended to be used as a sentence and short paragraph encoder. Given an input text, it outputs a vector which captures 
the semantic as well as space-time information. The sentence vector may be used for information retrieval, clustering or sentence similarity tasks.

By default, input text longer than 128 word pieces is truncated.


## Training procedure

### Pre-training 

We use the pretrained [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model. Please refer to the model card for more detailed information about the pre-training procedure.

### Fine-tuning 

The first step was adjusting the attention mechanism to take in 2 new inputs (Space, Time). Accordingly, the internal architecture was adjusted.
Based on our paper, this is the Multiplicative Attention model, trained with [-1, 1] loss using Sinusoidal Time and Tokenized Space.
We fine-tuned this model on 2 custom datasets.
1. The first dataset included triplets of:
It was used to train the model on 3 tasks simultaneously: (Masked Language Modelling, Time Classification, Space Classification).
[Refer to the paper for more information](https://drive.google.com/file/d/1pcz5ckoBkP4wQ7ojY5r14g1LXoYaqL_1/view?usp=drive_link)

| Sentence                                                       | Time      | Space     |
|----------------------------------------------------------------|-----------|-----------|
| "The weather in New York is going to be rainy this afternoon." | "2019-11" | "US"      |

2. The next step involved contrastive loss training on another dataset:  
The similarity in this case was constructed using a combination of the Time, Space, and Cosine Similarity.

| Sentence1             | Sentence2             | Time1    | Time2    | Space1 | Space2 | Similarity |
|-----------------------|-----------------------|----------|----------|--------|--------|------------|
| "The quick brown fox" | "A speedy auburn fox" | 2020-01  | 2020-02  | UK     | UK     | 0.91       |


#### Hyper parameters

We trained our model on an A100. We train the model during 1 epoch (1-per-task) using a batch size of 32.
We use a learning rate warm up of 3,375. The sequence length was limited to 128 tokens. We used the AdamW optimizer with
a 1e-5 learning rate for the base parameters and 2e-4 for the newly initialized parameters in the first stage. 
We used the AdamW optimizer with a 2e-5 learning rate in the second stage.

#### Training data

We use the concatenation from multiple datasets to fine-tune our model. The total number of sentences is above 1 million sentences.
