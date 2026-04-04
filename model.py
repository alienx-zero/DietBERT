# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 21:24:45 2025

@author: 99488
"""
from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math, time, random
import copy
import re
import torch.nn.functional as F

#######################################
####################################### 
#######################################basic TF   

class DietTextDataset(Dataset):
    def __init__(self, records: List[Dict[str,Any]], max_days=14, norm_stats=None, task = None, target_key = None):
        self.records = records
        self.max_days = max_days
        assert task in ("classification", "regression")
        y = torch.tensor([r[target_key] for r in records], dtype=torch.float32)

        if task == 'regression':
            # if norm_stats is None:
            #     y_mean, y_std = y.mean(), y.std() + 1e-6
            #     self.norm_stats = (y_mean, y_std)
            # else:
            #     self.norm_stats = norm_stats
    
            # y_mean, y_std = self.norm_stats
            # self.yn = (y - y_mean) / y_std   
            self.yn = y

        elif task == 'classification':
            self.yn = torch.tensor(y, dtype = torch.long)

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        days = r["day_texts"][:self.max_days]
        return {
            "days": days,                       # list[str]
            "global_text": r["global_text"],    # str
            "subject_id": r["sub"],    # str
            "target": self.yn[idx]
        }

class HierCollator:
    def __init__(
        self,
        tokenizer,
        max_day_tokens: int = 256,
        max_global_tokens: int = 128,
        max_global_groups: int = 7,
        max_days: int = None,
        feature_key: str = "days",     # list[str] per sample
        global_key: str = "global_text",    # str (or messy) per sample
    ):
        self.tok = tokenizer
        self.max_day_tokens = max_day_tokens
        self.max_global_tokens = max_global_tokens
        self.max_days = max_days
        self.feature_key = feature_key
        self.global_key = global_key
        self.max_global_groups = max_global_groups

    @staticmethod
    def _normalize_text(x):
        """Make sure we always return a string for tokenization."""
        if x is None:
            return ""
        # Handle numpy NaN if present
        try:
            import numpy as np
            if isinstance(x, (np.floating, np.integer)) and np.isnan(x):
                return ""
        except Exception:
            pass
        if isinstance(x, str):
            return x
        if isinstance(x, (list, tuple)):
            # join list of sentences into one paragraph
            return " ".join(map(str, x))
        # fallback
        return str(x)
    
    def __call__(self, batch: List[Dict[str, Any]]):
        B = len(batch)

        # ---------- Days ----------
        batch_max_days = self.max_days or max(len(x.get(self.feature_key, [])) for x in batch)
        day_ids, day_attn = [], []
        day_pad_mask = torch.ones(B, batch_max_days, dtype=torch.bool)  # True = PAD day

        for b, sample in enumerate(batch):
            days = list(sample.get(self.feature_key, []))[:batch_max_days]
            # pad with empty strings
            if len(days) < batch_max_days:
                days = days + [""] * (batch_max_days - len(days))

            enc = self.tok(
                days,
                truncation=True,
                padding="max_length",
                max_length=self.max_day_tokens,
                return_tensors="pt"
            )  # (D, L)
            day_ids.append(enc["input_ids"])
            day_attn.append(enc["attention_mask"])

            # mark valid days
            for d, txt in enumerate(days):
                if txt.strip():
                    day_pad_mask[b, d] = False

        input_ids       = torch.stack(day_ids,  dim=0)  # (B, D, L)
        attention_mask  = torch.stack(day_attn, dim=0)  # (B, D, L)

        # ---------- Global groups ----------
        batch_max_groups = self.max_global_groups or max(len(x.get(self.global_key, [])) for x in batch)
        g_ids, g_attn = [], []
        g_pad_mask = torch.ones(B, batch_max_groups, dtype=torch.bool)  # True = PAD group

        for b, sample in enumerate(batch):
            groups = list(sample.get(self.global_key, []))[:batch_max_groups]
            if len(groups) < batch_max_groups:
                groups = groups + [""] * (batch_max_groups - len(groups))

            genc = self.tok(
                groups,
                truncation=True,
                padding="max_length",
                max_length=self.max_global_tokens,
                return_tensors="pt"
            )  # (G, Lg)
            g_ids.append(genc["input_ids"])
            g_attn.append(genc["attention_mask"])

            for g, txt in enumerate(groups):
                if txt.strip():
                    g_pad_mask[b, g] = False

        global_input_ids      = torch.stack(g_ids,  dim=0)  # (B, G, Lg)
        global_attention_mask = torch.stack(g_attn, dim=0)  # (B, G, Lg)

        # labels
        targets = torch.tensor([(x['target']) for x in batch])
        subject_ids = [str(x.get("subject_id", "")) for x in batch]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "day_padding_mask": day_pad_mask,

            "global_input_ids": global_input_ids,
            "global_attention_mask": global_attention_mask,
            "global_group_padding_mask": g_pad_mask,

            "target": targets,
            "subject_id": subject_ids,   # <--- NEW
        }

class ContrastiveCollator:
    """
    Wraps HierCollator. Produces:
        {"v1": batch_like_HierCollator, "v2": batch_like_HierCollator}
    """
    def __init__(
        self,
        base_collator,                 # your HierCollator instance
        feature_key: str = "days",
        global_key: str = "global_text",
        p_day_drop: float = 0.15,      # prob to blank a *kept* day
        p_group_drop: float = 0.15,    # prob to blank a *kept* global group
        min_keep_days: int = 1,
        min_keep_groups: int = 1,
        seed: int = 42,
    ):
        self.base = base_collator
        self.feature_key = feature_key
        self.global_key = global_key
        self.p_day_drop = p_day_drop
        self.p_group_drop = p_group_drop
        self.min_keep_days = min_keep_days
        self.min_keep_groups = min_keep_groups
        random.seed(seed)

    def _augment_one(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        s = copy.deepcopy(sample)
        days = list(s.get(self.feature_key, []))
        groups = list(s.get(self.global_key, []))

        # blank out some non-empty days
        keep_idx = [i for i, t in enumerate(days) if isinstance(t, str) and t.strip()]
        if len(keep_idx) > self.min_keep_days:
            for i in keep_idx:
                if random.random() < self.p_day_drop:
                    days[i] = ""  # keep shape; collator will mask it

            # ensure at least one non-empty remains
            if not any(isinstance(t, str) and t.strip() for t in days):
                # restore one random kept index
                if keep_idx:
                    days[keep_idx[0]] = s[self.feature_key][keep_idx[0]]

        # blank out some global groups
        g_keep_idx = [i for i, t in enumerate(groups) if isinstance(t, str) and t.strip()]
        if len(g_keep_idx) > self.min_keep_groups:
            for i in g_keep_idx:
                if random.random() < self.p_group_drop:
                    groups[i] = ""

            if not any(isinstance(t, str) and t.strip() for t in groups):
                if g_keep_idx:
                    groups[g_keep_idx[0]] = s[self.global_key][g_keep_idx[0]]

        s[self.feature_key] = days
        s[self.global_key] = groups
        return s

    def __call__(self, batch: List[Dict[str, Any]]):
        # Create two independently augmented views
        batch_v1 = [self._augment_one(x) for x in batch]
        batch_v2 = [self._augment_one(x) for x in batch]

        v1 = self.base(batch_v1)
        v2 = self.base(batch_v2)

        # We don't need targets for contrastive pretraining; drop them to avoid confusion
        v1.pop("target", None)
        v2.pop("target", None)

        return {"v1": v1, "v2": v2}
    
class PairDataset:
    """
    Wraps your base dataset that returns dicts with at least:
      sample["subject_id"], sample["is_aug"] (True for augmented, False for original)
    Produces pairs: (orig_sample, aug_sample, subject_id)
    """
    def __init__(self, base_dataset, id_key="subject_id", aug_key="is_aug", seed=42):
        self.base = base_dataset
        self.id_key = id_key
        self.aug_key = aug_key
        random.seed(seed)

        # group indices by subject & flag
        buckets = {}
        for idx in range(len(self.base)):
            s = self.base[idx]
            sid = s[self.id_key]
            is_aug = bool(s.get(self.aug_key, False))
            d = buckets.setdefault(sid, {False: [], True: []})
            d[is_aug].append(idx)

        # build pairs list
        self.pairs: List[Tuple[int, int, int]] = []
        for sid, d in buckets.items():
            o = d[False]
            a = d[True]
            if not o or not a:
                # fallback: if no explicit aug flag, pair any two for that subject
                all_idx = o + a
                if len(all_idx) >= 2:
                    random.shuffle(all_idx)
                    for i in range(0, len(all_idx) - 1, 2):
                        self.pairs.append((all_idx[i], all_idx[i+1], sid))
                continue
            # pair greedily
            m = min(len(o), len(a))
            random.shuffle(o); random.shuffle(a)
            for i in range(m):
                self.pairs.append((o[i], a[i], sid))

        random.shuffle(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        oi, ai, sid = self.pairs[i]
        return {
            "orig": self.base[oi],
            "aug":  self.base[ai],
            "subject_id": sid,
        }
  
class PairCollator:
    """
    Uses your HierCollator to pack 'orig' and 'aug' separately, returns:
      {"orig": {...}, "aug": {...}, "subject_id": LongTensor(B)}
    """
    def __init__(self, base_collator, id_key="subject_id"):
        self.base = base_collator
        self.id_key = id_key

    def __call__(self, batch: List[Dict[str, Any]]):
        orig_list = [b["orig"] for b in batch]
        aug_list  = [b["aug"]  for b in batch]
        sids = []
        for b in batch:
            sid_str = str(b["subject_id"])
            digits = "".join(re.findall(r"\d+", sid_str))
            sids.append(int(digits) if digits else -1)  # -1 for missing numbers
        orig = self.base(orig_list);  orig.pop("target", None)
        aug  = self.base(aug_list);   aug.pop("target", None)

        return {"orig": orig, "aug": aug, "subject_id": torch.tensor(sids, dtype=torch.long)}
    
class NumericEncoder(nn.Module):
    """
    Simple MLP to embed numeric feature vectors into H.
    Optionally applies LayerNorm/Dropout for stability.
    """
    def __init__(self, in_dim, hidden=512, out_dim=768, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):  # x: (B, N, F) or (B* N, F)
        return self.net(x)
    
class MLMHead(nn.Module):
    """
    DistilBERT-style MLM prediction head with weight tying.
    dense -> GELU -> LayerNorm -> Linear(vocab) (+ bias)
    """
    def __init__(self, hidden_size: int, vocab_size: int, layer_norm_eps: float = 1e-12):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)  # will be tied to embeddings
        self.bias = nn.Parameter(torch.zeros(vocab_size))              # separate bias as in BERT
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden_states)
        x = self.act(x)
        x = self.ln(x)
        logits = self.decoder(x) + self.bias
        return logits
# ------------------------------
# 2) Model building (days + global TEXT FiLM)
# ------------------------------
class NutrientCrossAttention(nn.Module):
    """
    Learn N query vectors (global queries) to attend over D day embeddings.
    Returns:
        xctx: (B, H) aggregated context
        attn: (B, Nq, D) attention weights per query
    """
    def __init__(self, hidden, n_queries=4, dropout=0.1):
        super().__init__()
        self.nq = n_queries
        self.query = nn.Parameter(torch.randn(n_queries, hidden) * 0.02)
        self.key_proj   = nn.Linear(hidden, hidden)
        self.value_proj = nn.Linear(hidden, hidden)
        self.out_proj   = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.scale = math.sqrt(hidden)

    def forward(self, day_emb: torch.Tensor, gH: torch.Tensor, day_pad: torch.Tensor):
        """
        day_emb: (B,D,H)
        gH:      (B,H)  (not used directly here but could be concatenated to queries)
        day_pad: (B,D)  True=PAD
        """
        B, D, H = day_emb.shape
        # prepare Q,K,V
        Q = self.query.unsqueeze(0).expand(B, self.nq, H)  # (B,Nq,H)
        K = self.key_proj(day_emb)                         # (B,D,H)
        V = self.value_proj(day_emb)                       # (B,D,H)

        # scaled dot-product attention: (B,Nq,D)
        attn = torch.einsum("bnh,bdh->bnd", Q, K) / self.scale
        attn = attn.masked_fill(day_pad.unsqueeze(1), float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.drop(attn)

        # context per query: (B,Nq,H)
        ctx = torch.einsum("bnd,bdh->bnh", attn, V)
        # merge queries by mean then project
        ctx = ctx.mean(dim=1)  # (B,H)
        ctx = self.out_proj(ctx)
        return ctx, attn

class NutrientCrossAttentionSafe(nn.Module):
    """
    N learned queries attend over D day embeddings.
    NaN-safe: handles batches where all D days are masked.
    """
    def __init__(self, hidden, n_queries=4, dropout=0.1):
        super().__init__()
        self.nq = n_queries
        self.query = nn.Parameter(torch.randn(n_queries, hidden) * 0.02)
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.o_proj = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.scale = math.sqrt(hidden)
        self.q_ln = nn.LayerNorm(hidden)
        self.k_ln = nn.LayerNorm(hidden)
        self.v_ln = nn.LayerNorm(hidden)

    def forward(self, day_emb: torch.Tensor, gH: torch.Tensor, day_pad: torch.Tensor):
        """
        day_emb: (B, D, H)
        day_pad: (B, D)  True=PAD
        returns: ctx (B, H), attn (B, Nq, D)
        """
        B, D, H = day_emb.shape
        # LN improves numerical stability a lot
        Q = self.q_proj(gH).unsqueeze(1).repeat(1, self.nq, 1)  # (B,Nq,H)
        K = self.k_ln(self.k_proj(day_emb))                            # (B,D,H)
        V = self.v_ln(self.v_proj(day_emb))                            # (B,D,H)

        scores = torch.einsum("bnh,bdh->bnd", Q, K) / self.scale       # (B,Nq,D)
        scores = scores.masked_fill(day_pad.unsqueeze(1), float('-inf'))

        # Detect rows with all D masked → prevent softmax NaN
        all_masked = day_pad.all(dim=1)  # (B,)
        if all_masked.any():
            # put zeros → softmax → uniform zeros (stable)
            scores[all_masked] = 0.0

        attn = torch.softmax(scores, dim=-1)                           # (B,Nq,D)
        attn = self.drop(attn)

        ctx = torch.einsum("bnd,bdh->bnh", attn, V).mean(dim=1)        # (B,H)
        ctx = self.o_proj(ctx)

        if all_masked.any():
            ctx[all_masked] = 0.0

        # Ensure finite (belt-and-suspenders)
        ctx = torch.nan_to_num(ctx, nan=0.0, posinf=1e4, neginf=-1e4)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        return ctx, attn

# -----------------------------
# 3) Hierarchical Transformer Model
#    Local encoder: DistilBERT (per day) -> [CLS] embedding
#    Global encoder: Transformer over day sequence -> cls head
# -----------------------------
class HierarchicalTransformer(nn.Module):
    def __init__(self, local_model_name="distilbert-base-uncased",
                 num_labels=1, day_hidden_size=768, max_days=14,
                 global_n_layers=2, global_heads=8, global_ffn=1024,
                 dropout=0.1, n_queries=4, task: str = "regression",    # "regression" or "ordinal"
                 num_classes: int = None       # required if task=="ordinal" (e.g., 5 for labels 1..5)
                 ):
        '''
        Parameters
        ----------
        local_model_name : TYPE, optional
            DESCRIPTION. The default is "distilbert-base-uncased".
        num_labels : TYPE, optional
            DESCRIPTION. if regression, num_labels=1; classification, num_labels=2.
        day_hidden_size : TYPE, optional
            DESCRIPTION. The default is 768.
        max_days : TYPE, optional
            DESCRIPTION. The default is 14.
        global_layers : TYPE, optional
            DESCRIPTION. The default is 2.
        global_heads : TYPE, optional
            DESCRIPTION. The default is 8.
        global_ffn : TYPE, optional
            DESCRIPTION. The default is 1024.
        dropout : TYPE, optional
            DESCRIPTION. The default is 0.1.
        n_queries : TYPE, optional
            DESCRIPTION. The default is 4.

        Returns
        -------
        None.

        '''
        super().__init__()
        self.task = task
        self.num_classes = num_classes
        self.H = day_hidden_size
        # Day encoder
        self.local_encoder  = AutoModel.from_pretrained(local_model_name)
        # Global text encoder (separate instance; you can tie weights if you like)
        self.global_encoder = AutoModel.from_pretrained(local_model_name)

        self.pos_emb = nn.Embedding(max_days, self.H)

        # Project global text embedding to H (optional; often already H)
        self.g_proj = nn.Sequential(
            nn.Linear(self.H, self.H),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.H)
        )

        # FiLM: scale/shift days by global text context
        self.gamma = nn.Linear(self.H, self.H)
        self.beta  = nn.Linear(self.H, self.H)

        # Cross-attend learned queries over day embeddings
        self.xattn = NutrientCrossAttentionSafe(self.H, n_queries=n_queries, dropout=dropout)

        # Transformer over days
        layer = nn.TransformerEncoderLayer(
            self.H, global_heads, global_ffn, dropout,
            batch_first=True, norm_first=True
        )
        self.day_encoder = nn.TransformerEncoder(layer, num_layers=global_n_layers)

        # ===== HEADS =====
        if self.task == "ordinal":
            assert isinstance(self.num_classes, int) and self.num_classes >= 3, \
                "For ordinal, set num_classes (e.g., 5 for labels {1..5})."
            self.ordinal_head = nn.Sequential(
                nn.LayerNorm(self.H),
                nn.Dropout(dropout),
                nn.Linear(self.H, self.num_classes - 1)  # K-1 thresholds
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(self.H),
                nn.Dropout(dropout),
                nn.Linear(self.H, num_labels)
            )
        
        # ---- in HierarchicalTransformer.__init__ (after self.head) ----
        self.proj = nn.Sequential(
            nn.Linear(self.H, self.H),
            nn.ReLU(),
            nn.Linear(self.H, 128)   # dim of contrastive embedding (change if you like)
        )
        self.temperature = 0.2  # default τ for InfoNCE

        # ================== NEW: MLM heads (one per encoder) ==================
        loc_cfg = self.local_encoder.config
        glo_cfg = self.global_encoder.config
        eps = getattr(loc_cfg, "layer_norm_eps", 1e-12)

        self.mlm_head_local  = MLMHead(self.H, loc_cfg.vocab_size, layer_norm_eps=eps)
        self.mlm_head_global = MLMHead(self.H, glo_cfg.vocab_size, layer_norm_eps=eps)

        # ---- Weight tying: tie decoder weights to input embeddings ----
        self.mlm_head_local.decoder.weight  = self.local_encoder.get_input_embeddings().weight
        self.mlm_head_global.decoder.weight = self.global_encoder.get_input_embeddings().weight

    # ---- utilities for ordinal ----
    @staticmethod
    def coral_targets(y_long: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        y_long in {0,...,K-1}  -> target matrix (B, K-1) of 0/1 for 'y > k?'.
        """
        B = y_long.size(0)
        thresholds = torch.arange(num_classes - 1, device=y_long.device).unsqueeze(0).expand(B, -1)
        return (y_long.unsqueeze(1) > thresholds).float()

    @staticmethod
    def coral_predict(logits: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, K-1) -> predicted class in {0,...,K-1}
        Count how many sigmoid(logit) > 0.5 (number of thresholds passed).
        """
        probs = torch.sigmoid(logits)
        return (probs > 0.5).sum(dim=1)  # (B,)

    @staticmethod
    def coral_expected_value(logits: torch.Tensor) -> torch.Tensor:
        """
        Expected value (EV) decoding for ordinal: sum_k P(y>k) gives EV in [0,K-1].
        """
        probs = torch.sigmoid(logits)           # (B, K-1) ~ P(y>k)
        ev = probs.sum(dim=1)                   # (B,)
        return ev

    @staticmethod
    def coral_loss(logits: torch.Tensor, y_long: torch.Tensor, num_classes: int,
                   pos_weight: torch.Tensor | None = None, reduction: str = "mean") -> torch.Tensor:
        """
        BCE-with-logits summed over thresholds (CORAL).
        y_long must be in {0,...,K-1}.
        Optionally pass pos_weight (K-1,) to handle imbalance.
        """
        target = HierarchicalTransformer.coral_targets(y_long, num_classes)  # (B, K-1)
        return F.binary_cross_entropy_with_logits(logits, target, reduction=reduction, pos_weight=pos_weight)
    
    def _encode_global_groups(self, gids, gattn, gpad):
        """
        gids, gattn: (B, G, Lg); gpad: (B, G) True=PAD group
        returns gH: (B, H) pooled global embedding
        """
        B, G, Lg = gids.shape
        x = self.global_encoder(
            input_ids=gids.view(B*G, Lg),
            attention_mask=gattn.view(B*G, Lg)
        ).last_hidden_state  # (B*G, Lg, H)
        grp_emb = x[:, 0, :].view(B, G, -1)  # (B, G, H)

        keep = (~gpad).float().unsqueeze(-1)  # (B, G, 1)
        sum_emb = (grp_emb * keep).sum(1)     # (B, H)
        cnt     = keep.sum(1)                 # (B, 1)

        # If a subject has no groups, cnt==0 → fallback: zeros (caller can replace if desired)
        gH = torch.where(cnt > 0, sum_emb / cnt.clamp(min=1.0), torch.zeros_like(sum_emb))
        return gH  # (B, H)

    # ================== NEW: token-level helpers for MLM ==================
    def _mlm_forward_flat(self, encoder, head, input_ids_3d, attention_mask_3d):
        """
        Flatten (B, T, L) -> (B*T, L), run encoder, project to vocab.
        Returns logits: (B*T, L, V), attn2d: (B*T, L)
        """
        B, T, L = input_ids_3d.shape
        ids2d  = input_ids_3d.view(B*T, L)
        attn2d = attention_mask_3d.view(B*T, L)

        out = encoder(input_ids=ids2d, attention_mask=attn2d)
        hidden = out.last_hidden_state     # (B*T, L, H)
        logits = head(hidden)              # (B*T, L, V)
        return logits, attn2d

    @staticmethod
    def _mlm_ce_loss(logits, labels):
        """
        logits: (N, L, V), labels: (N, L) with -100 for ignored positions.
        """
        V = logits.size(-1)
        return F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100)

    def mlm_loss_days(self, input_ids, attention_mask, labels):
        """
        All shapes (B, D, L). Labels = -100 where not masked.
        """
        logits, _ = self._mlm_forward_flat(self.local_encoder, self.mlm_head_local, input_ids, attention_mask)
        return self._mlm_ce_loss(logits, labels)

    def mlm_loss_groups(self, input_ids, attention_mask, labels):
        """
        All shapes (B, G, Lg). Labels = -100 where not masked.
        """
        logits, _ = self._mlm_forward_flat(self.global_encoder, self.mlm_head_global, input_ids, attention_mask)
        return self._mlm_ce_loss(logits, labels)
    
    ###################get embedding for interpretation
    def forward_features(
        self,
        input_ids, attention_mask, day_padding_mask,
        global_input_ids, global_attention_mask, global_group_padding_mask,
        return_token_level: bool = False,
    ):
        """
        Returns a dict of embeddings:
          - z_proj: (B,128) normalized projected embedding
          - pooled: (B,H) pooled feature before projection (the one fed to head/proj)
          - gH:     (B,H) global encoder pooled embedding
          - day_cls: (B,D,H) local encoder CLS per day (after FiLM+pos? see note)
        """
        B, D, L = input_ids.shape
    
        # ----- Global pooled embedding -----
        gH = self._encode_global_groups(
            global_input_ids, global_attention_mask, global_group_padding_mask
        )  # (B,H)
        has_day = (~day_padding_mask).any(dim=1)  # (B,)
    
        # ----- Local encoder CLS per day -----
        input_ids_flat = input_ids.view(B * D, L)
        attn_flat = attention_mask.view(B * D, L)
    
        local_out = self.local_encoder(input_ids=input_ids_flat, attention_mask=attn_flat)
        token_hidden = local_out.last_hidden_state  # (B*D,L,H)
        day_cls = token_hidden[:, 0, :].view(B, D, -1)  # (B,D,H)
    
        # ----- Add day pos emb -----
        device = day_cls.device
        day_positions = torch.arange(D, device=device).unsqueeze(0).expand(B, D)
        day_emb = day_cls + self.pos_emb(day_positions)
    
        # ----- FiLM by gH -----
        gamma = torch.tanh(self.gamma(gH)).unsqueeze(1)  # (B,1,H)
        beta  = self.beta(gH).unsqueeze(1)               # (B,1,H)
        day_emb = (1 + gamma) * day_emb + beta
    
        # ----- Cross-attn + transformer -----
        xctx, _ = self.xattn(day_emb, gH, day_padding_mask)
        day_emb = day_emb + xctx.unsqueeze(1)
    
        x = self.day_encoder(day_emb, src_key_padding_mask=day_padding_mask)  # (B,D,H)
    
        # ----- Mask-aware pooling -----
        keep = (~day_padding_mask).float().unsqueeze(-1)  # (B,D,1) 1=keep, 0=pad
        pooled_sum = (x * keep).sum(1)
        keep_sum = keep.sum(1)
    
        pooled = torch.zeros_like(pooled_sum)             # (B,H)
        pooled[has_day] = pooled_sum[has_day] / keep_sum[has_day].clamp(min=1.0)
        pooled[~has_day] = gH[~has_day]
    
        # ----- Projection -----
        z = self.proj(pooled)                             # (B,128)
        z = F.normalize(z, dim=-1)
    
        if self.task == "ordinal":
            logits = self.ordinal_head(pooled)     # (B, K-1)
        else:
            logits = self.head(pooled)  
            
        out = {
            "z_proj": z,
            "pooled": pooled,
            "gH": gH,
            "day_cls_raw": day_cls,     # local encoder token0 BEFORE pos/FiLM/xattn
            "attentioned_fused_day_global": day_emb,         # after pos+FiLM+xattn residual (before day_encoder)
            "final_fused_day_global": x,           # after day_encoder
            'predict_value': logits
        }
        if not return_token_level:
            # drop big tensors if you don't need them
            out.pop("attentioned_fused_day_global")
            out.pop("day_cls_raw")
        return out


    def forward(self, input_ids, attention_mask, day_padding_mask, 
                global_input_ids, global_attention_mask, global_group_padding_mask, 
                return_projected: bool = False):
        """
        input_ids: (B, D, L)
        attention_mask: (B, D, L)
        day_padding_mask: (B, D)  True=pad day, False=keep
        """
        B, D, L = input_ids.shape
        _, G, Lg = global_input_ids.shape

        # ----- Global groups path (always computed) -----
        gH = self._encode_global_groups(global_input_ids, global_attention_mask, global_group_padding_mask)  # (B,H)
        has_day = (~day_padding_mask).any(dim=1)  # (B,) boolean
        
        # # ----- Day encoding -----
        # Flatten days into batch*days for local encoding
        input_ids_flat = input_ids.view(B*D, L)
        attn_flat = attention_mask.view(B*D, L)

        # DistilBERT outputs: last_hidden_state (B*D, L, H)
        local_out = self.local_encoder(input_ids=input_ids_flat, attention_mask=attn_flat)
        token_hidden = local_out.last_hidden_state  # (B*D, L, H)

        # Use [CLS]-like pooling: DistilBERT doesn't have pooled output; take token 0
        day_emb = token_hidden[:, 0, :]  # (B*D, H)
        day_emb = day_emb.view(B, D, -1)  # (B, D, H)

        # Add day positional embeddings
        device = day_emb.device
        day_positions = torch.arange(D, device=device).unsqueeze(0).expand(B, D)
        day_emb = day_emb + self.pos_emb(day_positions)

        # FiLM by global embedding
        gamma = torch.tanh(self.gamma(gH)).unsqueeze(1)  # (B,1,H)
        beta  = self.beta(gH).unsqueeze(1)               # (B,1,H)
        day_emb = (1 + gamma) * day_emb + beta
        
        # Cross-attn and day encoder
        xctx, _ = self.xattn(day_emb, gH, day_padding_mask)
        day_emb = day_emb + xctx.unsqueeze(1)

        # Global Transformer: mask expects True for PAD positions
        # src_key_padding_mask shape: (B, D)
        x = self.day_encoder(day_emb, src_key_padding_mask=day_padding_mask)  # (B, D, H)

        # Mask-aware pooling; fallback to gH if no days
        keep = (~day_padding_mask).float().unsqueeze(-1)
        pooled_sum = (x * keep).sum(1)
        keep_sum = keep.sum(1)
        pooled = torch.zeros_like(pooled_sum)
        pooled[has_day] = pooled_sum[has_day] / keep_sum[has_day].clamp(min=1.0)
        pooled[~has_day] = gH[~has_day]     # (B,H)
        
        if return_projected:
            z = self.proj(pooled)                # (B,128)
            z = torch.nn.functional.normalize(z, dim=-1)
            return z

        if self.task == "ordinal":
            logits = self.ordinal_head(pooled)     # (B, K-1)
        else:
            logits = self.head(pooled)    
        
        # return logits, pooled
        return logits
    
# -----------------------------
# 4) Controled Hierarchical Transformer Model
# -----------------------------
class HierarchicalTransformer_noglobal(nn.Module):
    def __init__(self, local_model_name="distilbert-base-uncased",
                 num_labels=1, day_hidden_size=768, max_days=14,
                 global_n_layers=2, global_heads=8, global_ffn=1024,
                 dropout=0.1, n_queries=4, task: str = "regression",    # "regression" or "ordinal"
                 num_classes: int = None       # required if task=="ordinal" (e.g., 5 for labels 1..5)
                 ):
        '''
        Parameters
        ----------
        local_model_name : TYPE, optional
            DESCRIPTION. The default is "distilbert-base-uncased".
        num_labels : TYPE, optional
            DESCRIPTION. if regression, num_labels=1; classification, num_labels=2.
        day_hidden_size : TYPE, optional
            DESCRIPTION. The default is 768.
        max_days : TYPE, optional
            DESCRIPTION. The default is 14.
        global_layers : TYPE, optional
            DESCRIPTION. The default is 2.
        global_heads : TYPE, optional
            DESCRIPTION. The default is 8.
        global_ffn : TYPE, optional
            DESCRIPTION. The default is 1024.
        dropout : TYPE, optional
            DESCRIPTION. The default is 0.1.
        n_queries : TYPE, optional
            DESCRIPTION. The default is 4.

        Returns
        -------
        None.

        '''
        super().__init__()
        self.task = task
        self.num_classes = num_classes
        self.H = day_hidden_size
        # Day encoder
        self.local_encoder  = AutoModel.from_pretrained(local_model_name)
        # Global text encoder (separate instance; you can tie weights if you like)
        self.global_encoder = AutoModel.from_pretrained(local_model_name)

        self.pos_emb = nn.Embedding(max_days, self.H)

        # Project global text embedding to H (optional; often already H)
        self.g_proj = nn.Sequential(
            nn.Linear(self.H, self.H),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.H)
        )

        # FiLM: scale/shift days by global text context
        self.gamma = nn.Linear(self.H, self.H)
        self.beta  = nn.Linear(self.H, self.H)

        # Cross-attend learned queries over day embeddings
        self.xattn = NutrientCrossAttentionSafe(self.H, n_queries=n_queries, dropout=dropout)

        # Transformer over days
        layer = nn.TransformerEncoderLayer(
            self.H, global_heads, global_ffn, dropout,
            batch_first=True, norm_first=True
        )
        self.day_encoder = nn.TransformerEncoder(layer, num_layers=global_n_layers)

        # ===== HEADS =====
        if self.task == "ordinal":
            assert isinstance(self.num_classes, int) and self.num_classes >= 3, \
                "For ordinal, set num_classes (e.g., 5 for labels {1..5})."
            self.ordinal_head = nn.Sequential(
                nn.LayerNorm(self.H),
                nn.Dropout(dropout),
                nn.Linear(self.H, self.num_classes - 1)  # K-1 thresholds
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(self.H),
                nn.Dropout(dropout),
                nn.Linear(self.H, num_labels)
            )
        
        # ---- in HierarchicalTransformer.__init__ (after self.head) ----
        self.proj = nn.Sequential(
            nn.Linear(self.H, self.H),
            nn.ReLU(),
            nn.Linear(self.H, 128)   # dim of contrastive embedding (change if you like)
        )
        self.temperature = 0.2  # default τ for InfoNCE

        # ================== NEW: MLM heads (one per encoder) ==================
        loc_cfg = self.local_encoder.config
        glo_cfg = self.global_encoder.config
        eps = getattr(loc_cfg, "layer_norm_eps", 1e-12)

        self.mlm_head_local  = MLMHead(self.H, loc_cfg.vocab_size, layer_norm_eps=eps)
        self.mlm_head_global = MLMHead(self.H, glo_cfg.vocab_size, layer_norm_eps=eps)

        # ---- Weight tying: tie decoder weights to input embeddings ----
        self.mlm_head_local.decoder.weight  = self.local_encoder.get_input_embeddings().weight
        self.mlm_head_global.decoder.weight = self.global_encoder.get_input_embeddings().weight

    # ---- utilities for ordinal ----
    @staticmethod
    def coral_targets(y_long: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        y_long in {0,...,K-1}  -> target matrix (B, K-1) of 0/1 for 'y > k?'.
        """
        B = y_long.size(0)
        thresholds = torch.arange(num_classes - 1, device=y_long.device).unsqueeze(0).expand(B, -1)
        return (y_long.unsqueeze(1) > thresholds).float()

    @staticmethod
    def coral_predict(logits: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, K-1) -> predicted class in {0,...,K-1}
        Count how many sigmoid(logit) > 0.5 (number of thresholds passed).
        """
        probs = torch.sigmoid(logits)
        return (probs > 0.5).sum(dim=1)  # (B,)

    @staticmethod
    def coral_expected_value(logits: torch.Tensor) -> torch.Tensor:
        """
        Expected value (EV) decoding for ordinal: sum_k P(y>k) gives EV in [0,K-1].
        """
        probs = torch.sigmoid(logits)           # (B, K-1) ~ P(y>k)
        ev = probs.sum(dim=1)                   # (B,)
        return ev

    @staticmethod
    def coral_loss(logits: torch.Tensor, y_long: torch.Tensor, num_classes: int,
                   pos_weight: torch.Tensor | None = None, reduction: str = "mean") -> torch.Tensor:
        """
        BCE-with-logits summed over thresholds (CORAL).
        y_long must be in {0,...,K-1}.
        Optionally pass pos_weight (K-1,) to handle imbalance.
        """
        target = HierarchicalTransformer.coral_targets(y_long, num_classes)  # (B, K-1)
        return F.binary_cross_entropy_with_logits(logits, target, reduction=reduction, pos_weight=pos_weight)
    
    def _encode_global_groups(self, gids, gattn, gpad):
        """
        gids, gattn: (B, G, Lg); gpad: (B, G) True=PAD group
        returns gH: (B, H) pooled global embedding
        """
        B, G, Lg = gids.shape
        x = self.global_encoder(
            input_ids=gids.view(B*G, Lg),
            attention_mask=gattn.view(B*G, Lg)
        ).last_hidden_state  # (B*G, Lg, H)
        grp_emb = x[:, 0, :].view(B, G, -1)  # (B, G, H)

        keep = (~gpad).float().unsqueeze(-1)  # (B, G, 1)
        sum_emb = (grp_emb * keep).sum(1)     # (B, H)
        cnt     = keep.sum(1)                 # (B, 1)

        # If a subject has no groups, cnt==0 → fallback: zeros (caller can replace if desired)
        gH = torch.where(cnt > 0, sum_emb / cnt.clamp(min=1.0), torch.zeros_like(sum_emb))
        return gH  # (B, H)

    # ================== NEW: token-level helpers for MLM ==================
    def _mlm_forward_flat(self, encoder, head, input_ids_3d, attention_mask_3d):
        """
        Flatten (B, T, L) -> (B*T, L), run encoder, project to vocab.
        Returns logits: (B*T, L, V), attn2d: (B*T, L)
        """
        B, T, L = input_ids_3d.shape
        ids2d  = input_ids_3d.view(B*T, L)
        attn2d = attention_mask_3d.view(B*T, L)

        out = encoder(input_ids=ids2d, attention_mask=attn2d)
        hidden = out.last_hidden_state     # (B*T, L, H)
        logits = head(hidden)              # (B*T, L, V)
        return logits, attn2d

    @staticmethod
    def _mlm_ce_loss(logits, labels):
        """
        logits: (N, L, V), labels: (N, L) with -100 for ignored positions.
        """
        V = logits.size(-1)
        return F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100)

    def mlm_loss_days(self, input_ids, attention_mask, labels):
        """
        All shapes (B, D, L). Labels = -100 where not masked.
        """
        logits, _ = self._mlm_forward_flat(self.local_encoder, self.mlm_head_local, input_ids, attention_mask)
        return self._mlm_ce_loss(logits, labels)

    def mlm_loss_groups(self, input_ids, attention_mask, labels):
        """
        All shapes (B, G, Lg). Labels = -100 where not masked.
        """
        logits, _ = self._mlm_forward_flat(self.global_encoder, self.mlm_head_global, input_ids, attention_mask)
        return self._mlm_ce_loss(logits, labels)

    def forward(self, input_ids, attention_mask, day_padding_mask, 
                return_projected: bool = False):
        """
        input_ids: (B, D, L)
        attention_mask: (B, D, L)
        day_padding_mask: (B, D)  True=pad day, False=keep
        """
        B, D, L = input_ids.shape

        # ----- Global groups path (always computed) -----
        has_day = (~day_padding_mask).any(dim=1)  # (B,) boolean
        
        # # ----- Day encoding -----
        # Flatten days into batch*days for local encoding
        input_ids_flat = input_ids.view(B*D, L)
        attn_flat = attention_mask.view(B*D, L)

        # DistilBERT outputs: last_hidden_state (B*D, L, H)
        local_out = self.local_encoder(input_ids=input_ids_flat, attention_mask=attn_flat)
        token_hidden = local_out.last_hidden_state  # (B*D, L, H)

        # Use [CLS]-like pooling: DistilBERT doesn't have pooled output; take token 0
        day_emb = token_hidden[:, 0, :]  # (B*D, H)
        day_emb = day_emb.view(B, D, -1)  # (B, D, H)

        # Add day positional embeddings
        device = day_emb.device
        day_positions = torch.arange(D, device=device).unsqueeze(0).expand(B, D)
        day_emb = day_emb + self.pos_emb(day_positions)

        # Global Transformer: mask expects True for PAD positions
        # src_key_padding_mask shape: (B, D)
        x = self.day_encoder(day_emb, src_key_padding_mask=day_padding_mask)  # (B, D, H)

        # Mask-aware pooling; fallback to gH if no days
        keep = (~day_padding_mask).float().unsqueeze(-1)
        pooled_sum = (x * keep).sum(1)
        keep_sum = keep.sum(1)
        pooled = torch.zeros_like(pooled_sum)
        pooled[has_day] = pooled_sum[has_day] / keep_sum[has_day].clamp(min=1.0)
        
        if return_projected:
            z = self.proj(pooled)                # (B,128)
            z = torch.nn.functional.normalize(z, dim=-1)
            return z

        if self.task == "ordinal":
            logits = self.ordinal_head(pooled)     # (B, K-1)
        else:
            logits = self.head(pooled)    
        
        # return logits, pooled
        return logits
    
# -----------------------------
# 5) Controled Hierarchical Transformer Model
# -----------------------------
class HierarchicalTransformer_noday(nn.Module):
    def __init__(self, local_model_name="distilbert-base-uncased",
                 num_labels=1, day_hidden_size=768, max_days=14,
                 global_n_layers=2, global_heads=8, global_ffn=1024,
                 dropout=0.1, n_queries=4, task: str = "regression",    # "regression" or "ordinal"
                 num_classes: int = None       # required if task=="ordinal" (e.g., 5 for labels 1..5)
                 ):
        '''
        Parameters
        ----------
        local_model_name : TYPE, optional
            DESCRIPTION. The default is "distilbert-base-uncased".
        num_labels : TYPE, optional
            DESCRIPTION. if regression, num_labels=1; classification, num_labels=2.
        day_hidden_size : TYPE, optional
            DESCRIPTION. The default is 768.
        max_days : TYPE, optional
            DESCRIPTION. The default is 14.
        global_layers : TYPE, optional
            DESCRIPTION. The default is 2.
        global_heads : TYPE, optional
            DESCRIPTION. The default is 8.
        global_ffn : TYPE, optional
            DESCRIPTION. The default is 1024.
        dropout : TYPE, optional
            DESCRIPTION. The default is 0.1.
        n_queries : TYPE, optional
            DESCRIPTION. The default is 4.

        Returns
        -------
        None.

        '''
        super().__init__()
        self.task = task
        self.num_classes = num_classes
        self.H = day_hidden_size
        # Day encoder
        self.local_encoder  = AutoModel.from_pretrained(local_model_name)
        # Global text encoder (separate instance; you can tie weights if you like)
        self.global_encoder = AutoModel.from_pretrained(local_model_name)

        self.pos_emb = nn.Embedding(max_days, self.H)

        # Project global text embedding to H (optional; often already H)
        self.g_proj = nn.Sequential(
            nn.Linear(self.H, self.H),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.H)
        )

        # FiLM: scale/shift days by global text context
        self.gamma = nn.Linear(self.H, self.H)
        self.beta  = nn.Linear(self.H, self.H)

        # Cross-attend learned queries over day embeddings
        self.xattn = NutrientCrossAttentionSafe(self.H, n_queries=n_queries, dropout=dropout)

        # Transformer over days
        layer = nn.TransformerEncoderLayer(
            self.H, global_heads, global_ffn, dropout,
            batch_first=True, norm_first=True
        )
        self.day_encoder = nn.TransformerEncoder(layer, num_layers=global_n_layers)

        # ===== HEADS =====
        if self.task == "ordinal":
            assert isinstance(self.num_classes, int) and self.num_classes >= 3, \
                "For ordinal, set num_classes (e.g., 5 for labels {1..5})."
            self.ordinal_head = nn.Sequential(
                nn.LayerNorm(self.H),
                nn.Dropout(dropout),
                nn.Linear(self.H, self.num_classes - 1)  # K-1 thresholds
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(self.H),
                nn.Dropout(dropout),
                nn.Linear(self.H, num_labels)
            )
        
        # ---- in HierarchicalTransformer.__init__ (after self.head) ----
        self.proj = nn.Sequential(
            nn.Linear(self.H, self.H),
            nn.ReLU(),
            nn.Linear(self.H, 128)   # dim of contrastive embedding (change if you like)
        )
        self.temperature = 0.2  # default τ for InfoNCE

        # ================== NEW: MLM heads (one per encoder) ==================
        loc_cfg = self.local_encoder.config
        glo_cfg = self.global_encoder.config
        eps = getattr(loc_cfg, "layer_norm_eps", 1e-12)

        self.mlm_head_local  = MLMHead(self.H, loc_cfg.vocab_size, layer_norm_eps=eps)
        self.mlm_head_global = MLMHead(self.H, glo_cfg.vocab_size, layer_norm_eps=eps)

        # ---- Weight tying: tie decoder weights to input embeddings ----
        self.mlm_head_local.decoder.weight  = self.local_encoder.get_input_embeddings().weight
        self.mlm_head_global.decoder.weight = self.global_encoder.get_input_embeddings().weight

    # ---- utilities for ordinal ----
    @staticmethod
    def coral_targets(y_long: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        y_long in {0,...,K-1}  -> target matrix (B, K-1) of 0/1 for 'y > k?'.
        """
        B = y_long.size(0)
        thresholds = torch.arange(num_classes - 1, device=y_long.device).unsqueeze(0).expand(B, -1)
        return (y_long.unsqueeze(1) > thresholds).float()

    @staticmethod
    def coral_predict(logits: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, K-1) -> predicted class in {0,...,K-1}
        Count how many sigmoid(logit) > 0.5 (number of thresholds passed).
        """
        probs = torch.sigmoid(logits)
        return (probs > 0.5).sum(dim=1)  # (B,)

    @staticmethod
    def coral_expected_value(logits: torch.Tensor) -> torch.Tensor:
        """
        Expected value (EV) decoding for ordinal: sum_k P(y>k) gives EV in [0,K-1].
        """
        probs = torch.sigmoid(logits)           # (B, K-1) ~ P(y>k)
        ev = probs.sum(dim=1)                   # (B,)
        return ev

    @staticmethod
    def coral_loss(logits: torch.Tensor, y_long: torch.Tensor, num_classes: int,
                   pos_weight: torch.Tensor | None = None, reduction: str = "mean") -> torch.Tensor:
        """
        BCE-with-logits summed over thresholds (CORAL).
        y_long must be in {0,...,K-1}.
        Optionally pass pos_weight (K-1,) to handle imbalance.
        """
        target = HierarchicalTransformer.coral_targets(y_long, num_classes)  # (B, K-1)
        return F.binary_cross_entropy_with_logits(logits, target, reduction=reduction, pos_weight=pos_weight)
    
    def _encode_global_groups(self, gids, gattn, gpad):
        """
        gids, gattn: (B, G, Lg); gpad: (B, G) True=PAD group
        returns gH: (B, H) pooled global embedding
        """
        B, G, Lg = gids.shape
        x = self.global_encoder(
            input_ids=gids.view(B*G, Lg),
            attention_mask=gattn.view(B*G, Lg)
        ).last_hidden_state  # (B*G, Lg, H)
        grp_emb = x[:, 0, :].view(B, G, -1)  # (B, G, H)

        keep = (~gpad).float().unsqueeze(-1)  # (B, G, 1)
        sum_emb = (grp_emb * keep).sum(1)     # (B, H)
        cnt     = keep.sum(1)                 # (B, 1)

        # If a subject has no groups, cnt==0 → fallback: zeros (caller can replace if desired)
        gH = torch.where(cnt > 0, sum_emb / cnt.clamp(min=1.0), torch.zeros_like(sum_emb))
        return gH  # (B, H)

    # ================== NEW: token-level helpers for MLM ==================
    def _mlm_forward_flat(self, encoder, head, input_ids_3d, attention_mask_3d):
        """
        Flatten (B, T, L) -> (B*T, L), run encoder, project to vocab.
        Returns logits: (B*T, L, V), attn2d: (B*T, L)
        """
        B, T, L = input_ids_3d.shape
        ids2d  = input_ids_3d.view(B*T, L)
        attn2d = attention_mask_3d.view(B*T, L)

        out = encoder(input_ids=ids2d, attention_mask=attn2d)
        hidden = out.last_hidden_state     # (B*T, L, H)
        logits = head(hidden)              # (B*T, L, V)
        return logits, attn2d

    @staticmethod
    def _mlm_ce_loss(logits, labels):
        """
        logits: (N, L, V), labels: (N, L) with -100 for ignored positions.
        """
        V = logits.size(-1)
        return F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100)

    def mlm_loss_days(self, input_ids, attention_mask, labels):
        """
        All shapes (B, D, L). Labels = -100 where not masked.
        """
        logits, _ = self._mlm_forward_flat(self.local_encoder, self.mlm_head_local, input_ids, attention_mask)
        return self._mlm_ce_loss(logits, labels)

    def mlm_loss_groups(self, input_ids, attention_mask, labels):
        """
        All shapes (B, G, Lg). Labels = -100 where not masked.
        """
        logits, _ = self._mlm_forward_flat(self.global_encoder, self.mlm_head_global, input_ids, attention_mask)
        return self._mlm_ce_loss(logits, labels)

    def forward(self, input_ids, attention_mask, day_padding_mask, 
                global_input_ids, global_attention_mask, global_group_padding_mask, 
                return_projected: bool = False):
        """
        input_ids: (B, D, L)
        attention_mask: (B, D, L)
        day_padding_mask: (B, D)  True=pad day, False=keep
        """
        B, D, L = global_input_ids.shape

        # ----- Global groups path (always computed) -----
        has_day = (~global_group_padding_mask).any(dim=1)  # (B,) boolean
        
        # # ----- Day encoding -----
        # Flatten days into batch*days for local encoding
        input_ids_flat = input_ids.view(B*D, L)
        attn_flat = global_attention_mask.view(B*D, L)

        # DistilBERT outputs: last_hidden_state (B*D, L, H)
        local_out = self.local_encoder(input_ids=input_ids_flat, attention_mask=attn_flat)
        token_hidden = local_out.last_hidden_state  # (B*D, L, H)

        # Use [CLS]-like pooling: DistilBERT doesn't have pooled output; take token 0
        day_emb = token_hidden[:, 0, :]  # (B*D, H)
        day_emb = day_emb.view(B, D, -1)  # (B, D, H)

        # Add day positional embeddings
        device = day_emb.device
        day_positions = torch.arange(D, device=device).unsqueeze(0).expand(B, D)
        day_emb = day_emb + self.pos_emb(day_positions)

        # Global Transformer: mask expects True for PAD positions
        # src_key_padding_mask shape: (B, D)
        x = self.day_encoder(day_emb, src_key_padding_mask=global_group_padding_mask)  # (B, D, H)

        # Mask-aware pooling; fallback to gH if no days
        keep = (~global_group_padding_mask).float().unsqueeze(-1)
        pooled_sum = (x * keep).sum(1)
        keep_sum = keep.sum(1)
        pooled = torch.zeros_like(pooled_sum)
        pooled[has_day] = pooled_sum[has_day] / keep_sum[has_day].clamp(min=1.0)
        
        if return_projected:
            z = self.proj(pooled)                # (B,128)
            z = torch.nn.functional.normalize(z, dim=-1)
            return z

        if self.task == "ordinal":
            logits = self.ordinal_head(pooled)     # (B, K-1)
        else:
            logits = self.head(pooled)    
        
        return logits
    
# -----------------------------
# 6) no attention Hierarchical Transformer Model
#    Local encoder: DistilBERT (per day) -> [CLS] embedding
#    Global encoder: Transformer over day sequence -> cls head
# -----------------------------
class HierarchicalTransformer_noattention(nn.Module):
    def __init__(self, local_model_name="distilbert-base-uncased",
                 num_labels=1, day_hidden_size=768, max_days=14,
                 global_n_layers=2, global_heads=8, global_ffn=1024,
                 dropout=0.1, n_queries=4, task: str = "regression",    # "regression" or "ordinal"
                 num_classes: int = None       # required if task=="ordinal" (e.g., 5 for labels 1..5)
                 ):
        '''
        Parameters
        ----------
        local_model_name : TYPE, optional
            DESCRIPTION. The default is "distilbert-base-uncased".
        num_labels : TYPE, optional
            DESCRIPTION. if regression, num_labels=1; classification, num_labels=2.
        day_hidden_size : TYPE, optional
            DESCRIPTION. The default is 768.
        max_days : TYPE, optional
            DESCRIPTION. The default is 14.
        global_layers : TYPE, optional
            DESCRIPTION. The default is 2.
        global_heads : TYPE, optional
            DESCRIPTION. The default is 8.
        global_ffn : TYPE, optional
            DESCRIPTION. The default is 1024.
        dropout : TYPE, optional
            DESCRIPTION. The default is 0.1.
        n_queries : TYPE, optional
            DESCRIPTION. The default is 4.

        Returns
        -------
        None.

        '''
        super().__init__()
        self.task = task
        self.num_classes = num_classes
        self.H = day_hidden_size
        # Day encoder
        self.local_encoder  = AutoModel.from_pretrained(local_model_name)
        # Global text encoder (separate instance; you can tie weights if you like)
        self.global_encoder = AutoModel.from_pretrained(local_model_name)

        self.pos_emb = nn.Embedding(max_days, self.H)

        # Project global text embedding to H (optional; often already H)
        self.g_proj = nn.Sequential(
            nn.Linear(self.H, self.H),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.H)
        )

        # FiLM: scale/shift days by global text context
        self.gamma = nn.Linear(self.H, self.H)
        self.beta  = nn.Linear(self.H, self.H)

        # Cross-attend learned queries over day embeddings
        self.xattn = NutrientCrossAttentionSafe(self.H, n_queries=n_queries, dropout=dropout)

        # Transformer over days
        layer = nn.TransformerEncoderLayer(
            self.H, global_heads, global_ffn, dropout,
            batch_first=True, norm_first=True
        )
        self.day_encoder = nn.TransformerEncoder(layer, num_layers=global_n_layers)

        # ===== HEADS =====
        if self.task == "ordinal":
            assert isinstance(self.num_classes, int) and self.num_classes >= 3, \
                "For ordinal, set num_classes (e.g., 5 for labels {1..5})."
            self.ordinal_head = nn.Sequential(
                nn.LayerNorm(self.H),
                nn.Dropout(dropout),
                nn.Linear(self.H, self.num_classes - 1)  # K-1 thresholds
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(self.H),
                nn.Dropout(dropout),
                nn.Linear(self.H, num_labels)
            )
        
        # ---- in HierarchicalTransformer.__init__ (after self.head) ----
        self.proj = nn.Sequential(
            nn.Linear(self.H, self.H),
            nn.ReLU(),
            nn.Linear(self.H, 128)   # dim of contrastive embedding (change if you like)
        )
        self.temperature = 0.2  # default τ for InfoNCE

        # ================== NEW: MLM heads (one per encoder) ==================
        loc_cfg = self.local_encoder.config
        glo_cfg = self.global_encoder.config
        eps = getattr(loc_cfg, "layer_norm_eps", 1e-12)

        self.mlm_head_local  = MLMHead(self.H, loc_cfg.vocab_size, layer_norm_eps=eps)
        self.mlm_head_global = MLMHead(self.H, glo_cfg.vocab_size, layer_norm_eps=eps)

        # ---- Weight tying: tie decoder weights to input embeddings ----
        self.mlm_head_local.decoder.weight  = self.local_encoder.get_input_embeddings().weight
        self.mlm_head_global.decoder.weight = self.global_encoder.get_input_embeddings().weight

    # ---- utilities for ordinal ----
    @staticmethod
    def coral_targets(y_long: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        y_long in {0,...,K-1}  -> target matrix (B, K-1) of 0/1 for 'y > k?'.
        """
        B = y_long.size(0)
        thresholds = torch.arange(num_classes - 1, device=y_long.device).unsqueeze(0).expand(B, -1)
        return (y_long.unsqueeze(1) > thresholds).float()

    @staticmethod
    def coral_predict(logits: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, K-1) -> predicted class in {0,...,K-1}
        Count how many sigmoid(logit) > 0.5 (number of thresholds passed).
        """
        probs = torch.sigmoid(logits)
        return (probs > 0.5).sum(dim=1)  # (B,)

    @staticmethod
    def coral_expected_value(logits: torch.Tensor) -> torch.Tensor:
        """
        Expected value (EV) decoding for ordinal: sum_k P(y>k) gives EV in [0,K-1].
        """
        probs = torch.sigmoid(logits)           # (B, K-1) ~ P(y>k)
        ev = probs.sum(dim=1)                   # (B,)
        return ev

    @staticmethod
    def coral_loss(logits: torch.Tensor, y_long: torch.Tensor, num_classes: int,
                   pos_weight: torch.Tensor | None = None, reduction: str = "mean") -> torch.Tensor:
        """
        BCE-with-logits summed over thresholds (CORAL).
        y_long must be in {0,...,K-1}.
        Optionally pass pos_weight (K-1,) to handle imbalance.
        """
        target = HierarchicalTransformer.coral_targets(y_long, num_classes)  # (B, K-1)
        return F.binary_cross_entropy_with_logits(logits, target, reduction=reduction, pos_weight=pos_weight)
    
    def _encode_global_groups(self, gids, gattn, gpad):
        """
        gids, gattn: (B, G, Lg); gpad: (B, G) True=PAD group
        returns gH: (B, H) pooled global embedding
        """
        B, G, Lg = gids.shape
        x = self.global_encoder(
            input_ids=gids.view(B*G, Lg),
            attention_mask=gattn.view(B*G, Lg)
        ).last_hidden_state  # (B*G, Lg, H)
        grp_emb = x[:, 0, :].view(B, G, -1)  # (B, G, H)

        keep = (~gpad).float().unsqueeze(-1)  # (B, G, 1)
        sum_emb = (grp_emb * keep).sum(1)     # (B, H)
        cnt     = keep.sum(1)                 # (B, 1)

        # If a subject has no groups, cnt==0 → fallback: zeros (caller can replace if desired)
        gH = torch.where(cnt > 0, sum_emb / cnt.clamp(min=1.0), torch.zeros_like(sum_emb))
        return gH  # (B, H)

    # ================== NEW: token-level helpers for MLM ==================
    def _mlm_forward_flat(self, encoder, head, input_ids_3d, attention_mask_3d):
        """
        Flatten (B, T, L) -> (B*T, L), run encoder, project to vocab.
        Returns logits: (B*T, L, V), attn2d: (B*T, L)
        """
        B, T, L = input_ids_3d.shape
        ids2d  = input_ids_3d.view(B*T, L)
        attn2d = attention_mask_3d.view(B*T, L)

        out = encoder(input_ids=ids2d, attention_mask=attn2d)
        hidden = out.last_hidden_state     # (B*T, L, H)
        logits = head(hidden)              # (B*T, L, V)
        return logits, attn2d

    @staticmethod
    def _mlm_ce_loss(logits, labels):
        """
        logits: (N, L, V), labels: (N, L) with -100 for ignored positions.
        """
        V = logits.size(-1)
        return F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100)

    def mlm_loss_days(self, input_ids, attention_mask, labels):
        """
        All shapes (B, D, L). Labels = -100 where not masked.
        """
        logits, _ = self._mlm_forward_flat(self.local_encoder, self.mlm_head_local, input_ids, attention_mask)
        return self._mlm_ce_loss(logits, labels)

    def mlm_loss_groups(self, input_ids, attention_mask, labels):
        """
        All shapes (B, G, Lg). Labels = -100 where not masked.
        """
        logits, _ = self._mlm_forward_flat(self.global_encoder, self.mlm_head_global, input_ids, attention_mask)
        return self._mlm_ce_loss(logits, labels)

    def forward(self, input_ids, attention_mask, day_padding_mask, 
                global_input_ids, global_attention_mask, global_group_padding_mask, 
                return_projected: bool = False):
        """
        input_ids: (B, D, L)
        attention_mask: (B, D, L)
        day_padding_mask: (B, D)  True=pad day, False=keep
        """
        B, D, L = input_ids.shape
        _, G, Lg = global_input_ids.shape

        # ----- Global groups path (always computed) -----
        gH = self._encode_global_groups(global_input_ids, global_attention_mask, global_group_padding_mask)  # (B,H)
        has_day = (~day_padding_mask).any(dim=1)  # (B,) boolean
        
        # # ----- Day encoding -----
        # Flatten days into batch*days for local encoding
        input_ids_flat = input_ids.view(B*D, L)
        attn_flat = attention_mask.view(B*D, L)

        # DistilBERT outputs: last_hidden_state (B*D, L, H)
        local_out = self.local_encoder(input_ids=input_ids_flat, attention_mask=attn_flat)
        token_hidden = local_out.last_hidden_state  # (B*D, L, H)

        # Use [CLS]-like pooling: DistilBERT doesn't have pooled output; take token 0
        day_emb = token_hidden[:, 0, :]  # (B*D, H)
        day_emb = day_emb.view(B, D, -1)  # (B, D, H)

        # Add day positional embeddings
        device = day_emb.device
        day_positions = torch.arange(D, device=device).unsqueeze(0).expand(B, D)
        day_emb = day_emb + self.pos_emb(day_positions)

        # # FiLM by global embedding
        # gamma = torch.tanh(self.gamma(gH)).unsqueeze(1)  # (B,1,H)
        # beta  = self.beta(gH).unsqueeze(1)               # (B,1,H)
        # day_emb = (1 + gamma) * day_emb + beta
        
        # # Cross-attn and day encoder
        # xctx, _ = self.xattn(day_emb, gH, day_padding_mask)
        # day_emb = day_emb + xctx.unsqueeze(1)

        # Global Transformer: mask expects True for PAD positions
        # src_key_padding_mask shape: (B, D)
        x = self.day_encoder(day_emb, src_key_padding_mask=day_padding_mask)  # (B, D, H)

        # Mask-aware pooling; fallback to gH if no days
        keep = (~day_padding_mask).float().unsqueeze(-1)
        pooled_sum = (x * keep).sum(1)
        keep_sum = keep.sum(1)
        pooled = torch.zeros_like(pooled_sum)
        pooled[has_day] = pooled_sum[has_day] / keep_sum[has_day].clamp(min=1.0)
        pooled[~has_day] = gH[~has_day]     # (B,H)
        
        if return_projected:
            z = self.proj(pooled)                # (B,128)
            z = torch.nn.functional.normalize(z, dim=-1)
            return z

        if self.task == "ordinal":
            logits = self.ordinal_head(pooled)     # (B, K-1)
        else:
            logits = self.head(pooled)    
        
        return logits
        
