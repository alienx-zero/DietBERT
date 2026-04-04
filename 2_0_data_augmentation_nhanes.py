# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 13:21:57 2025

@author: 99488
"""

import pandas as pd
import numpy as np
import re
import sys
sys.path.append(r'H:\postdoc\UCLA_postdoc\Diet_predict')
from utils import schema, energy_macronutrients_alcohol_schema, sugars_schema, fats_schema, \
    protein_schema, micronutrients_schema, food_schema, row_to_text, AugmentedSubset, \
        nhanes_day1_energy_macros, nhanes_day1_fats_detail, nhanes_day1_micros, nhanes_day1_water_behavior
import math
import random
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
# import evaluate
hf_logging.set_verbosity_error()
from sklearn.model_selection import StratifiedKFold
from model import HierarchicalTransformer, HierCollator, SubsetWeeklyDiet
from torch.utils.data import Dataset, DataLoader
from scipy.io import savemat
import time
import datetime as dt
from pathlib import Path
from safetensors.torch import save_file  # pip install safetensors
# augment_to_disk.py
import json, os, random, re, math, hashlib, shelve, sys, io
from typing import List, Dict, Any, Iterable, Tuple
from tqdm import tqdm
import copy
import torch
AUG_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from openai import OpenAI
import pickle

######################save model
def fmt_secs(s):
    return str(dt.timedelta(seconds=int(s)))  # e.g., '0:01:23'

def state_dict_to_cpu(sd):
    return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in sd.items()}

def short_safe_dir():
    # Always write to a short local path to avoid Win path limits and network/OneDrive locks
    d = Path(r"F:\postdoc\UCLA_postdoc\Diet_predict\results\baseline_transformer")
    d.mkdir(parents=True, exist_ok=True)
    return d

def win_longpath(p: Path) -> str:
    # Use \\?\ prefix if path is long (> 240 chars) to bypass MAX_PATH on old configs
    s = str(p.resolve())
    return s if len(s) < 240 else r"\\?\{}".format(s)

def test_write(path: Path):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(win_longpath(path.with_suffix(".touch")), "wb") as f:
            f.write(b"ok")
        try:
            (path.with_suffix(".touch")).unlink(missing_ok=True)
        except Exception:
            pass
        return True
    except Exception as e:
        print(f"[path-check] Cannot write to: {path}\nError: {e}")
        return False

def save_checkpoint_safetensors(model, epoch, fold_id,
                                best_val_acc, best_val_loss, best_val_sens, best_val_spec):
    # 1) Build short, safe path
    save_dir = short_safe_dir()
    weights_path = save_dir / f"best_e{epoch}_f{fold_id}.safetensors"
    meta_path    = save_dir / f"best_e{epoch}_f{fold_id}.json"

    # 2) Quick path sanity test
    if not test_write(weights_path):
        raise OSError(f"Cannot write to {weights_path}; try another local folder (e.g., C:\\tmp).")

    # 3) Save only weights with safetensors (atomic, no zip)
    sd_cpu = state_dict_to_cpu(model.state_dict())
    save_file(sd_cpu, win_longpath(weights_path))

    # 4) Save small metadata separately as JSON
    meta = {
        "epoch": int(epoch),
        "val_acc": float(best_val_acc),
        "val_loss": float(best_val_loss),
        "val_sensitivity": float(best_val_sens),
        "val_specificity": float(best_val_spec),
        "note": "optimizer state not saved (to avoid large/fragile files on Windows)"
    }
    with open(win_longpath(meta_path), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Saved weights to {weights_path}")
    print(f"✅ Saved metadata to {meta_path}")
    return str(weights_path), str(meta_path)

def torch_save_small_pt(model, fold_id, weightORoptimizer):
    sd = state_dict_to_cpu(model.state_dict())
    buf = io.BytesIO()
    torch.save(sd, buf, _use_new_zipfile_serialization=False)  # legacy, smaller
    data = buf.getbuffer()

    if weightORoptimizer == 'weight':
        p = short_safe_dir() / f"best_model_fold_{fold_id}_weight.pt"
    elif weightORoptimizer == 'opt':
        p = short_safe_dir() / f"best_model_fold_{fold_id}_opt.pt"
    if not test_write(p):
        raise OSError(f"Cannot write to {p}")
    with open(win_longpath(p), "wb") as f:
        f.write(data)
    print(f"✅ Saved PT weights to {p}")
    
def compute_ce_weights(y_train, method="inv_freq", beta=0.9999):
    """
    y_train: 1D array-like of integer class labels (train split only)
    method:  'inv_freq' or 'effective'
    - inv_freq: weight_c = N / (K * n_c)
    - effective: Cui et al. CVPR'19 (good for heavy imbalance)
    """
    y = np.asarray(y_train)
    counts = np.bincount(y)
    K = len(counts)
    N = y.size

    if method == "inv_freq":
        w = N / (K * np.clip(counts, 1, None))
    else:  # effective number of samples
        # weight_c ∝ (1 - beta) / (1 - beta^{n_c})
        eff = (1 - np.power(beta, counts)) / (1 - beta)
        w = 1.0 / np.clip(eff, 1e-12, None)

    # normalize so mean(weight)=1 (optional but nice)
    w = w * (K / w.sum())
    return torch.tensor(w, dtype=torch.float32)

# -----------------------------
# 0) OPTIONAL: your base data access
#    Expect an iterable of dicts: {"text_info": List[str], "label": int}
#    Replace this with your own SubsetWeeklyDiet export
# -----------------------------
def iter_base_samples(train_data_base) -> Iterable[Dict[str, Any]]:
    """Yield dicts with keys: text_info (list of day strings), label (int)."""
    for i in range(len(train_data_base)):
        s = train_data_base[i]
        yield {"text_info": list(s["text_info"]), "label": int(s["MINI_Major_Depressive_p"])}

# -----------------------------
# 1) Cheap, label-safe jitter
# -----------------------------
TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):[0-5]\d(:[0-5]\d)?\b")
NUM_RE  = re.compile(r"\b(\d+(\.\d+)?)\b")

def jitter_quantity(text: str, pct=0.15):
    def _j(m):
        try:
            v = float(m.group(1))
            v2 = max(0.1, v * (1.0 + random.uniform(-pct, pct)))
            return str(round(v2, 2))
        except:
            return m.group(0)
    return NUM_RE.sub(_j, text)

def jitter_time(text: str):
    # keep as no-op, or implement a real time shift if needed
    return text

def augment_day_text(text: str, p_jitter=0.6):
    out = text
    if random.random() < p_jitter:
        out = jitter_quantity(out)
        out = jitter_time(out)
    return out

# -----------------------------
# 2) Optional: Marian back-translation (lazy)
# -----------------------------
_MARIAN_READY = False
_en_de_tok = _en_de = _de_en_tok = _de_en = None

def _try_load_marian():
    global _MARIAN_READY, _en_de_tok, _en_de, _de_en_tok, _de_en
    if _MARIAN_READY:
        return True
    try:
        from transformers import MarianMTModel, MarianTokenizer
        _en_de_tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        _en_de     = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de").to(AUG_DEVICE).eval()
        _de_en_tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
        _de_en     = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-de-en").to(AUG_DEVICE).eval()
        _MARIAN_READY = True
        return True
    except Exception as e:
        print(f"[augment] BT disabled: {e}")
        return False

def bt_batch(texts: List[str], max_len=256, num_beams=1) -> List[str]:
    """Back-translate a batch of strings. Returns same length list."""
    if not _try_load_marian() or not texts:
        return texts
    # de
    batch = _en_de_tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=max_len).to(AUG_DEVICE)
    de_ids = _en_de.generate(**batch, num_beams=num_beams, max_length=max_len)
    de_txt = _en_de_tok.batch_decode(de_ids, skip_special_tokens=True)
    # en
    batch2 = _de_en_tok(de_txt, return_tensors="pt", truncation=True, padding=True, max_length=max_len).to(AUG_DEVICE)
    en_ids = _de_en.generate(**batch2, num_beams=num_beams, max_length=max_len)
    en_txt = _de_en_tok.batch_decode(en_ids, skip_special_tokens=True)
    return en_txt

# Disk cache for BT so repeated days are fast across runs
def _cache_key(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

class BTCache:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
    def get_many(self, texts: List[str]) -> Dict[str, str]:
        out = {}
        with shelve.open(self.path) as db:
            for t in texts:
                k = _cache_key(t)
                if k in db:
                    out[t] = db[k]
        return out
    def put_many(self, mapping: Dict[str, str]):
        with shelve.open(self.path, writeback=False) as db:
            for t, bt in mapping.items():
                db[_cache_key(t)] = bt

# -----------------------------
# 3) Build augmented dataset on disk
# -----------------------------
def write_augmented_jsonl(
    samples: Iterable[Dict[str, Any]],
    out_path: str,
    expand_factor: float = 0.5,     # +50% more samples total
    p_day_jitter: float = 0.6,
    use_bt: bool = False,
    p_day_bt: float = 0.1,
    bt_cache_path: str = "./bt_cache.db",
    minority_boost: float = 1.5,    # multiply expand_factor for the minority class
    seed: int = 42,
    bt_batch_size: int = 64
):
    """
    - Writes original samples + augmented copies.
    - If class is imbalanced, minority gets more augmented copies (minority_boost).
    """
    random.seed(seed)
    torch.manual_seed(seed)

    # First pass: collect all samples, count labels
    base = list(samples)
    N = len(base)
    labels = [s["label"] for s in base]
    n_pos = sum(1 for y in labels if y == 1)
    n_neg = N - n_pos
    minority_label = 1 if n_pos < n_neg else 0

    # compute how many augmented samples we want
    target_extra = int(round(N * expand_factor))

    # simple per-class quotas
    pos_quota = int(target_extra * (minority_boost / (minority_boost + 1))) if minority_label == 1 else int(target_extra * (1 / (minority_boost + 1)))
    neg_quota = target_extra - pos_quota

    # BT cache
    bt_cache = BTCache(bt_cache_path)

    with open(out_path, "w", encoding="utf-8") as f:
        # 3.1 write all originals
        for s in base:
            f.write(json.dumps({"text_info": s["text_info"], "label": s["label"]}, ensure_ascii=False) + "\n")

        # 3.2 build augmented copies until quotas are met
        # Shuffle indices so we cycle fairly
        idxs = list(range(N))
        random.shuffle(idxs)

        pos_added = 0
        neg_added = 0

        pbar = tqdm(total=target_extra, desc="Augmenting")
        i = 0
        while (pos_added + neg_added) < target_extra:
            s = base[idxs[i % N]]
            i += 1
            y = s["label"]
            if y == 1 and pos_added >= pos_quota: 
                continue
            if y == 0 and neg_added >= neg_quota:
                continue

            # augment this sample
            days = s["text_info"]
            # First apply cheap jitter
            days_aug = [augment_day_text(d, p_jitter=p_day_jitter) for d in days]

            # Optional BT for a subset of days (batched using cache)
            if use_bt and p_day_bt > 0.0:
                # pick BT candidates
                bt_candidates = [d for d in days_aug if (random.random() < p_day_bt and d.strip())]
                # get cached
                cached = bt_cache.get_many(bt_candidates)
                missing = [t for t in bt_candidates if t not in cached]

                # translate missing in mini-batches
                translated = {}
                for j in range(0, len(missing), bt_batch_size):
                    chunk = missing[j:j+bt_batch_size]
                    if chunk:
                        out = bt_batch(chunk, max_len=256, num_beams=1)
                        translated.update(dict(zip(chunk, out)))

                # update cache and assemble final map
                full_map = {**cached, **translated}
                if translated:
                    bt_cache.put_many(translated)

                # replace selected days
                days_aug = [full_map.get(d, d) for d in days_aug]

            # write augmented
            f.write(json.dumps({"text_info": days_aug, "label": y}, ensure_ascii=False) + "\n")

            if y == 1:
                pos_added += 1
            else:
                neg_added += 1
            pbar.update(1)
        pbar.close()

    print(f"[done] wrote: {out_path}")
    print(f"Original N={N}, augmented +{target_extra} -> total {N+target_extra}")
    print(f"Minority label={minority_label} | pos_added={pos_added} neg_added={neg_added}")

# -----------------------------
# 4) Train / Eval
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _safe_div(a, b):
    return float(a) / float(b) if b > 0 else float("nan")

def _update_confusion_counts(preds, labels, pos_label=1):
    # preds, labels: 1D tensors on same device
    tp = ((preds == pos_label) & (labels == pos_label)).sum().item()
    tn = ((preds != pos_label) & (labels != pos_label)).sum().item()
    fp = ((preds == pos_label) & (labels != pos_label)).sum().item()
    fn = ((preds != pos_label) & (labels == pos_label)).sum().item()
    return tp, tn, fp, fn

def train_one_epoch(model, loader, optimizer, scheduler=None, device="cuda", pos_label=1, ce=None):
    model.train()
    if ce is None:
        ce = nn.CrossEntropyLoss()  # fallback
    # ce = nn.CrossEntropyLoss()

    total_loss, total, correct = 0.0, 0, 0
    TP = TN = FP = FN = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        day_padding_mask = batch["day_padding_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, day_padding_mask)
        loss = ce(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # metrics
        bs = labels.size(0)
        total_loss += loss.item() * bs
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += bs

        tp, tn, fp, fn = _update_confusion_counts(preds, labels, pos_label=pos_label)
        TP += tp; TN += tn; FP += fp; FN += fn

    acc = correct / total if total > 0 else float("nan")
    sens = _safe_div(TP, TP + FN)  # sensitivity / recall(positive)
    spec = _safe_div(TN, TN + FP)  # specificity / recall(negative)

    return {
        "loss": total_loss / total if total > 0 else float("nan"),
        "acc": acc,
        "sensitivity": sens,
        "specificity": spec,
        "tp": TP, "tn": TN, "fp": FP, "fn": FN,
    }

@torch.no_grad()
def eval_loop(model, loader, device="cuda", pos_label=1):
    model.eval()
    ce = nn.CrossEntropyLoss()

    total_loss, total, correct = 0.0, 0, 0
    TP = TN = FP = FN = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        day_padding_mask = batch["day_padding_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask, day_padding_mask)
        loss = ce(logits, labels)

        # metrics
        bs = labels.size(0)
        total_loss += loss.item() * bs
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += bs

        tp, tn, fp, fn = _update_confusion_counts(preds, labels, pos_label=pos_label)
        TP += tp; TN += tn; FP += fp; FN += fn

    acc = correct / total if total > 0 else float("nan")
    sens = _safe_div(TP, TP + FN)  # sensitivity
    spec = _safe_div(TN, TN + FP)  # specificity

    return {
        "loss": total_loss / total if total > 0 else float("nan"),
        "acc": acc,
        "sensitivity": sens,
        "specificity": spec,
        "tp": TP, "tn": TN, "fp": FP, "fn": FN,
    }

client = OpenAI(api_key="xxx")

# def paraphrase_n(sentence, model="gpt-4o-mini", temperature=0.6, n=9):
#     """
#     Generate n paraphrases of the given sentence using OpenAI GPT models.
#     """
#     resp = client.chat.completions.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": (
#                 "You are a helpful assistant that rewrites sentences. "
#                 "Produce exactly {} distinct paraphrases of the input sentence. "
#                 "Each paraphrase must preserve meaning, retain named entities, "
#                 "and stay within ±10% of the original length. "
#                 "Return them as a plain list, one per line.".format(n)
#             )},
#             {"role": "user", "content": f'Original: "{sentence}"'}
#         ],
#         temperature=temperature,
#         max_tokens=200 * n,  # allow enough tokens for multiple outputs
#     )

#     # Split outputs by newlines to return as a list
#     raw_text = resp.choices[0].message.content.strip()
#     paraphrases = [line.strip("-•123456789. ") for line in raw_text.split("\n") if line.strip()]
    
#     # Limit to n items (in case the model generates extras)
#     return paraphrases[:n]

def paraphrase(sentence, model="gpt-4o-mini", temperature=0.6):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system", "content":(
                "Try to rewrite the contents in sentence wuth good logic shortly"
                "Reorder or replace phrases while respecting syntactic dependencies"
                "Also, you should rephrase and Synonym Replacementwhile preserving meaning"
                "Produce a single paraphrase, and stays within ±10% length."
                "Output plain text only."
            )},
            {"role":"user","content":f'Original: "{sentence}"'}
        ],
        temperature=temperature,
        max_tokens=1500
    )
    return resp.choices[0].message.content.strip()

def paraphrase_many(sentence, n=9, model="gpt-4o-mini", temperature=0.8, max_tokens=200):
    """
    Return `n` distinct paraphrases of `sentence` as a list of strings.
    - `n` uses the API's multi-choice feature (one call, n completions).
    - `temperature` controls diversity; increase for more variety.
    - `max_tokens` is per completion.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Try to rewrite the contents in sentence wuth good logic shortly"
                    "Reorder or replace phrases while respecting syntactic dependencies"
                    "Also, you should rephrase and Synonym Replacementwhile preserving meaning"
                    "Produce a single paraphrase, and stays within ±10% length."
                    "Also generate a health summary based on the individual's eating information"
                    "Output plain text only."
                ),
            },
            {"role": "user", "content": f'Original: {sentence}'},
        ],
        temperature=temperature,   # raise to ~0.8–1.0 for more variety
        top_p=0.9,                 # you can lower to 0.9 for steadier style
        n=n,                       # <-- multiple completions
        max_tokens=1500      # applies per choice
    )

    # Collect all choices; strip whitespace and duplicates while preserving order
    seen = set()
    outs = []
    for choice in resp.choices:
        txt = (choice.message.content or "").strip()
        if txt and txt not in seen:
            seen.add(txt)
            outs.append(txt)
    return outs

if __name__ == "__main__":
# ---------- main with 10-fold CV ----------
    with open(r"H:\postdoc\UCLA_postdoc\Diet_predict\NHANES\data\all_data_sentences_raw.pkl", "rb") as f:
        diet_all = pickle.load(f)
    all_samples = copy.deepcopy(diet_all)

    for i, sample in enumerate(all_samples):
        # if i < 6120:
        #     continue
        print(f'i: {i}')
        texts = sample['text_info']
        all_text = []
        for j, text in enumerate(texts):
            print(f'j: {j}')
            if len(text) == 0:
                all_samples[i]['text_info'][j] = ''
            else:
                new_text = paraphrase_many(text, n=1, model = 'gpt-4.1-mini')
                all_samples[i]['text_info'][j] = new_text

    for t in (all_samples):
        folder = r'H:\postdoc\UCLA_postdoc\Diet_predict\NHANES\data\aug_alls'
        os.makedirs(folder, exist_ok=True)
        with open(folder + r"\aug_sub_{}.json".format(t['sub']), "w", encoding="utf-8") as f:
            json.dump(t['text_info'], f, ensure_ascii=False)

    with open(r"H:\postdoc\UCLA_postdoc\Diet_predict\NHANES\data\all_data_sentences_raw.pkl", "rb") as f:
        diet_all = pickle.load(f)
    all_samples = [copy.deepcopy(diet_all) for i in range(4)]

    for i, sample in enumerate(all_samples[0]):
        if i < 1769:
            continue
        print(f'i: {i}')
        texts = sample['text_info']
        all_text = []
        for j, text in enumerate(texts):
            print(f'j: {j}')
            if len(text) == 0:
                for k, samples in enumerate(all_samples):
                    all_samples[k][i]['text_info'][j] = ''
            else:
                new_text = paraphrase_many(text, n=4, model = 'gpt-4.1-mini')
                new_text = new_text[:4]
                # If fewer than n, get more and merge
                if len(new_text) < 4:
                    new_text2 = paraphrase_many(text, n=8)  # oversample a bit
                    merged = new_text + new_text2
                    new_text = merged[:4]
                for k, samples in enumerate(all_samples):
                    all_samples[k][i]['text_info'][j] = new_text[k]

    for a, info in enumerate(all_samples):
        for t in info:
            folder = r'H:\postdoc\UCLA_postdoc\Diet_predict\NHANES\data\aug_with_medicalprompt_{}'.format(a)
            os.makedirs(folder, exist_ok=True)
            with open(folder + r"\aug_sub_{}.json".format(t['sub']), "w", encoding="utf-8") as f:
                json.dump(t['text_info'], f, ensure_ascii=False)