import torch
import torch.nn as nn
import math
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import PreTrainedModel
from transformers.models.bert.modeling_bert import BertSelfAttention

class SpaceEmbedding(nn.Module):
    def __init__(self, num_embeddings=4, embedding_dim=384):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

class TimeEmbedding(nn.Module):
    def __init__(self, max_months, dim=384):
        super().__init__()
        self.dim = dim
        pe = torch.zeros(max_months, dim)
        pos = torch.arange(0, max_months).unsqueeze(1)
        i = torch.arange(0, dim, 2)
        pe[:, 0::2] = torch.sin(pos / (10000 ** (2*i/dim)))
        pe[:, 1::2] = torch.cos(pos / (10000 ** (2*i/dim)))
        self.register_buffer("pe", pe)

    def forward(self, idx):
        return self.pe[idx]

# ----------------------------
# 1) Custom Space–Time Attention
# ----------------------------
class SpaceTimeSelfAttention(nn.Module):
    def __init__(self, orig_self: BertSelfAttention, config):
        super().__init__()
        self.orig = orig_self
        self.config = config
        self.W_t = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_s = nn.Linear(config.hidden_size, config.hidden_size)

    def transpose_for_scores(self, x):
        return self.orig.transpose_for_scores(x)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        time_embeddings=None,
        space_embeddings=None,
    ):
        
        mixed_q = self.orig.query(hidden_states)
        mixed_k = self.orig.key(hidden_states)
        mixed_v = self.orig.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_q)    
        key_layer   = self.transpose_for_scores(mixed_k)
        value_layer = self.transpose_for_scores(mixed_v)

        T = self.W_t(time_embeddings)
        S = self.W_s(space_embeddings)
        T_layer = self.transpose_for_scores(T)
        S_layer = self.transpose_for_scores(S)

        base_scores = torch.matmul(
            query_layer,
            key_layer.transpose(-1, -2)
        ) 

        eps = 1e-6
        T_norm = T_layer.norm(dim=-1, keepdim=True)
        time_sim = torch.matmul(
            T_layer,
            T_layer.transpose(-1, -2)
        ) / (T_norm + eps)

        S_norm = S_layer.norm(dim=-1, keepdim=True)
        space_sim = torch.matmul(
            S_layer,
            S_layer.transpose(-1, -2)
        ) / (S_norm + eps)

        attn_scores = base_scores * time_sim * space_sim

        dk = self.config.hidden_size // self.config.num_attention_heads
        attn_scores = attn_scores / math.sqrt(dk)

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = nn.Softmax(dim=-1)(attn_scores)
        attn_probs = self.orig.dropout(attn_probs)

        if head_mask is not None:
            attn_probs = attn_probs * head_mask

        context = torch.matmul(attn_probs, value_layer)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_shape = context.size()[:-2] + (self.config.hidden_size,)
        context = context.view(*new_shape)

        if output_attentions:
            return (context, attn_probs)
        return context


# ----------------------------
# 2) Full Space–Time–MiniLM Model
# ----------------------------
class SpaceTimeMiniLM(PreTrainedModel):
    config_class = SpaceTimeMiniLMConfig
    def __init__(self, config):
        super().__init__(config)
        self.base = AutoModel.from_config(config)
        self.config = config

        for layer in self.base.encoder.layer:
            orig_self = layer.attention.self
            layer.attention.self = SpaceTimeSelfAttention(orig_self, self.config)

        self.space_embed = SpaceEmbedding(num_embeddings=config.num_space,
                                          embedding_dim=self.config.hidden_size)
        self.time_embed  = TimeEmbedding(max_months=config.num_time,
                                         dim=self.config.hidden_size)

        self.mlm_head   = nn.Linear(self.config.hidden_size,
                                    config.vocab_size)
        self.space_head = nn.Linear(self.config.hidden_size, config.num_space)
        self.time_head  = nn.Linear(self.config.hidden_size, config.num_time)

    def forward(self, input_ids, attention_mask, space_ids, time_ids):
        B, L = input_ids.size()

        extended_mask = self.base.get_extended_attention_mask(attention_mask, (B, L), device=input_ids.device)

        emb = self.base.embeddings(input_ids)

        S = self.space_embed(space_ids)
        T = self.time_embed(time_ids)
        S = S.unsqueeze(1).expand(-1, L, -1)
        T = T.unsqueeze(1).expand(-1, L, -1)

        hidden_states = emb
        for layer in self.base.encoder.layer:
            attn_out = layer.attention.self(
                hidden_states,
                attention_mask=extended_mask,
                head_mask=None,
                output_attentions=False,
                time_embeddings=T,
                space_embeddings=S
            )
            attn_out = layer.attention.output(attn_out, hidden_states)
            interm = layer.intermediate(attn_out)
            hidden_states = layer.output(interm, attn_out)

        sequence_output = hidden_states
        pooled_output   = self.base.pooler(sequence_output)

        mlm_logits   = self.mlm_head(sequence_output)
        space_logits = self.space_head(pooled_output)
        time_logits  = self.time_head(pooled_output)

        return mlm_logits, space_logits, time_logits

    def embed(
        self,
        input_ids: torch.LongTensor,        
        attention_mask: torch.LongTensor,
        space_ids: torch.LongTensor,
        time_ids:  torch.LongTensor
    ) -> torch.FloatTensor:       
        B, L = input_ids.size()

        extended_mask = self.base.get_extended_attention_mask(
            attention_mask, (B, L), device=input_ids.device
        )

        hidden_states = self.base.embeddings(input_ids)    

        S = self.space_embed(space_ids)
        T = self.time_embed(time_ids)
        S = S.unsqueeze(1).expand(-1, L, -1)
        T = T.unsqueeze(1).expand(-1, L, -1)

        for layer in self.base.encoder.layer:
            attn_out = layer.attention.self(
                hidden_states,
                attention_mask=extended_mask,
                head_mask=None,
                output_attentions=False,
                time_embeddings=T,
                space_embeddings=S
            )
            attn_out = layer.attention.output(attn_out, hidden_states)
            interm = layer.intermediate(attn_out)
            hidden_states = layer.output(interm, attn_out)

        mask_exp = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
        sum_emb  = torch.sum(hidden_states * mask_exp, dim=1)
        sum_mask = mask_exp.sum(dim=1).clamp(min=1e-9)
        pooled   = sum_emb / sum_mask

        return pooled

    def embed_sentence(self, sent, time, space, tokenizer, device="cuda"):
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.eval().to(device)

        space_mapping = {'UK': 0, 'US': 1, 'AUS': 2, 'CAN': 3}
        time_mapping = {
            f"{year}-{month:02d}": i
            for i, (year, month) in enumerate(
                [(y, m) for y in range(2017, 2022 + 1) for m in range(1, 13)]
            )
            if i < 60
        }

        enc = tokenizer(
            sent,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        space = torch.tensor([space_mapping[space]], dtype=torch.long, device=device)
        time = torch.tensor([time_mapping[time]], dtype=torch.long, device=device)

        with torch.no_grad():
            emb = self.embed(enc["input_ids"], enc["attention_mask"], space, time)
        return emb

    def compute_similarity(self, sent1, sent2, time1, time2, space1, space2, tokenizer, device="cuda"):
        emb1 = self.embed_sentence(sent1, time1, space1, tokenizer, device)
        emb2 = self.embed_sentence(sent2, time2, space2, tokenizer, device)
        return F.cosine_similarity(emb1, emb2, dim=-1).item()