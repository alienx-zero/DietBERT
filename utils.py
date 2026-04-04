# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 13:21:57 2025

@author: 99488
"""

import pandas as pd
import numpy as np
import random, re, torch
from transformers import MarianMTModel, MarianTokenizer
from torch.utils.data import Dataset

############################################augmentation based on back-translation
AUG_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_marian(src_tgt):
    tok = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{src_tgt}")
    mdl = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-{src_tgt}").to(AUG_DEVICE)
    mdl.eval()
    return tok, mdl

# Load once (don’t do this inside the fold loop!)
en_de_tok, en_de = load_marian("en-de")
de_en_tok, de_en = load_marian("de-en")

@torch.no_grad()
def back_translate(text: str, max_len=256):
    if not text or not text.strip():
        return text
    batch = en_de_tok([text], return_tensors="pt", truncation=True, max_length=max_len).to(AUG_DEVICE)
    de_ids = en_de.generate(**batch, num_beams=4, max_length=max_len)
    de_text = en_de_tok.batch_decode(de_ids, skip_special_tokens=True)

    batch2 = de_en_tok(de_text, return_tensors="pt", truncation=True, max_length=max_len).to(AUG_DEVICE)
    en_bt_ids = de_en.generate(**batch2, num_beams=4, max_length=max_len)
    en_bt = de_en_tok.batch_decode(en_bt_ids, skip_special_tokens=True)[0]
    return en_bt

# Cheap, domain-safe jitter: shift times & mild quantity jitter (doesn’t flip label)
TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):[0-5]\d(:[0-5]\d)?\b")
NUM_RE  = re.compile(r"\b(\d+(\.\d+)?)\b")

############################prediction variables loading
###################washing diet record
def jitter_time(text: str, minutes_range=30):
    def _shift(m):
        # very simple: don’t actually recompute clock; just nudge text token (optional)
        return m.group(0)  # keep original if you don't want string-time math
    return TIME_RE.sub(_shift, text)

def jitter_quantity(text: str, pct=0.15):
    def _j(m):
        try:
            v = float(m.group(1))
            v2 = max(0.1, v * (1.0 + random.uniform(-pct, pct)))
            return str(round(v2, 2))
        except:
            return m.group(0)
    return NUM_RE.sub(_j, text)

def augment_day_text(text: str, p_bt=0.25, p_jitter=0.5):
    """
    Apply 0–2 augmentations:
      - back-translation with prob p_bt
      - numeric/time jitter with prob p_jitter
    """
    out = text
    # Light jitter first (cheap)
    if random.random() < p_jitter:
        out = jitter_quantity(out)
        out = jitter_time(out)
    # Occasional back-translation (more expensive)
    if random.random() < p_bt:
        out = back_translate(out)
    return out

class AugmentedSubset(Dataset):
    def __init__(self, base_ds, feature_key="text_info", label_key="MINI_Suicidality_c",
                 p_sample_aug=0.8, p_day_bt=0.25, p_day_jitter=0.5):
        self.base = base_ds
        self.feature_key = feature_key
        self.label_key = label_key
        self.p_sample_aug = p_sample_aug
        self.p_day_bt = p_day_bt
        self.p_day_jitter = p_day_jitter

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        days = list(sample[self.feature_key])  # copy
        label = sample[self.label_key]

        # With probability p_sample_aug, augment this participant’s days
        if random.random() < self.p_sample_aug:
            days = [augment_day_text(d, p_bt=self.p_day_bt, p_jitter=self.p_day_jitter) for d in days]

        return {self.feature_key: days, self.label_key: label}

#############################mapping table variables into natural language
def parse_week_diet(raw_text):
    """
    Turn a raw weekly diet string (with '_x000D_' etc.) into
    a list of day strings: ["DAY 1 ...", "DAY 2 ...", ...]
    """
    # 1) Normalize line breaks and spaces
    text = raw_text.replace('\r', '\n')
    text = text.replace('_x000D_', '\n')     # common export artifact
    text = re.sub(r'[ \t]+', ' ', text)      # collapse repeated spaces
    text = re.sub(r'\n{2,}', '\n', text)     # collapse blank lines

    # 2) Split by "DAY <num> - <date>" headers (keep headers with their content)
    # Example header variants: "DAY 1 - 6/7/2022", "DAY 3 - 6/10/2022"
    # We capture the header line and split on it using lookahead
    pattern = re.compile(r'(?=^DAY\s*\d+\s*-\s*.+?$)', re.IGNORECASE | re.MULTILINE)

    chunks = pattern.split(text)
    days = []
    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue
        # Ensure it starts with DAY header; if not, try to attach to previous day
        if not ch.upper().startswith("DAY"):
            if days:
                days[-1] = days[-1].rstrip() + "\n" + ch
            else:
                # If no prior day, just keep it as day 1 fallback
                days.append("DAY 1\n" + ch)
            continue

        # 3) Clean common punctuation/typos (optional light pass)
        ch = ch.replace(" ;", ";").replace(" ,", ",")
        ch = re.sub(r'\s{2,}', ' ', ch)
        days.append(ch)

    # 4) Remove empty days (e.g., header with no entries)
    days = [d for d in days if any(tok.strip() for tok in d.splitlines()[1:])]
    return days


# ----- Minimal renderer -----
def render_cell(col, val, cfg):
    if pd.isna(val): return None
    # general continuous/categorical/binary handling
    if cfg.get("kind") == "continuous":
        v = float(val)
        v_disp = round(v, cfg.get("round", 1))
        if cfg.get("round", 1) == 0: v_disp = int(v_disp)

        # optional bin text
        bintext = ""
        for cond, txt in cfg.get("bins", []):
            try:
                if cond(v):
                    bintext = txt
                    break
            except Exception:
                pass

        # optional derived values
        derived = {}
        for k, fn in cfg.get("derived", {}).items():
            try:
                derived[k] = fn(v)
            except Exception:
                derived[k] = None

        if cfg.get("skip_if_zero") and float(v) == 0.0:
            return None

        return cfg["template"].format(
            label=cfg.get("label", col),
            value=v_disp,
            unit=cfg.get("unit", ""),
            bintext=bintext,
            **derived
        ).strip().replace("()", "").replace("  ", " ")

    if cfg.get("kind") in ("categorical", "binary"):
        text = cfg.get("map", {}).get(val, str(val))
        return cfg["template"].format(label=cfg.get("label", col), value=text).strip()

    # HEI components (score out of max_points)
    if cfg.get("kind") == "hei":
        v = float(val)
        maxp = cfg.get("max_points", 10)
        frac = np.clip(v / maxp, 0.0, 1.0)
        if cfg.get("mode", "adequacy") == "adequacy":
            rating = "high alignment" if frac >= 0.80 else "moderate alignment" if frac >= 0.50 else "low alignment"
        else:
            rating = "well limited" if frac >= 0.80 else "partly limited" if frac >= 0.50 else "not well limited"
        return cfg["template"].format(
            label=cfg.get("label", col),
            value=int(round(v)),
            maxp=int(maxp),
            rating=rating
        )

    return f"{cfg.get('label', col)}: {val}"

def row_to_text(row, schema, sep="; "):
    parts = []
    for col, cfg in schema.items():
        if col in row and row[col] is not None:
            s = render_cell(col, row[col], cfg)
            if s:
                parts.append(s)
    return (sep.join(parts) + ".") if parts else ""

def mind_rate_higher_is_better(x, good, ok=None):
    """
    Rate where higher intake is better (e.g., leafy greens, berries).
    good: threshold for "meets recommendation"
    ok: optional lower threshold for "partial" (if None, uses 0.5*good)
    """
    if x is None or np.isnan(x): return "no data"
    if ok is None: ok = 0.5 * good
    return "meets recommendation" if x >= good else ("partially meets" if x >= ok else "below recommendation")

def mind_rate_lower_is_better(x, good_max, ok_max=None):
    """
    Rate where lower intake is better (e.g., red meats, pastries/sweets).
    good_max: <= this is "meets recommendation"
    ok_max: <= this is "partially meets" (if None, uses 1.5*good_max)
    """
    if x is None or np.isnan(x): return "no data"
    if ok_max is None: ok_max = 1.5 * good_max
    return "meets recommendation" if x <= good_max else ("partially meets" if x <= ok_max else "above recommended")

# ----- Schema for all columns -----
schema = {
    # Total MIND score (with wine)
    "MindDietScoreWine": {
        "kind": "continuous", "label": "MIND diet total score (with wine)",
        "round": 0,
        # Typical range ~0–15 when wine is included. Adjust if your scoring differs.
        "bins": [
            (lambda v: v >= 12, "high adherence"),
            (lambda v: 8 <= v < 12, "moderate adherence"),
            (lambda v: v < 8, "low adherence"),
        ],
        "template": "{label}: {value} ({bintext})"
    },

    # ----- Brain-healthy foods (higher is better) -----
    # Assumed units: servings/week (edit thresholds to match your instrument)
    "MindDiet_Green_Leafy_Vegetables_Raw": {
        "kind": "continuous", "label": "Green leafy vegetables",
        "round": 1,
        "derived": {"rating": lambda v: mind_rate_higher_is_better(v, good=6, ok=3)},  # ≥6/wk ideal
        "template": "{label}: {value}/week — {rating}"
    },
    "MindDiet_Other_Vegetables_Raw": {
        "kind": "continuous", "label": "Other vegetables",
        "round": 1,
        "derived": {"rating": lambda v: mind_rate_higher_is_better(v, good=7, ok=4)},  # ≈daily ideal
        "template": "{label}: {value}/week — {rating}"
    },
    "MindDiet_Berries_Raw": {
        "kind": "continuous", "label": "Berries",
        "round": 1,
        "derived": {"rating": lambda v: mind_rate_higher_is_better(v, good=2, ok=1)},  # ≥2/wk ideal
        "template": "{label}: {value}/week — {rating}"
    },
    "MindDiet_Nuts_Raw": {
        "kind": "continuous", "label": "Nuts",
        "round": 1,
        "derived": {"rating": lambda v: mind_rate_higher_is_better(v, good=5, ok=3)},  # ≥5/wk ideal
        "template": "{label}: {value}/week — {rating}"
    },
    "MindDiet_Olive_Oil_Raw": {
        "kind": "continuous", "label": "Olive oil (uses)",
        "round": 1,
        # If your variable encodes “primary oil = 1/0” instead of frequency, change to a binary mapping.
        "derived": {"rating": lambda v: mind_rate_higher_is_better(v, good=5, ok=2)},
        "template": "{label}: {value}/week — {rating}"
    },
    "MindDiet_Whole_Grains_Raw": {
        "kind": "continuous", "label": "Whole grains",
        "round": 1,
        # MIND original often uses ≥3 servings/day; as /week proxy, set good≈21. Adjust to your questionnaire.
        "derived": {"rating": lambda v: mind_rate_higher_is_better(v, good=21, ok=14)},
        "template": "{label}: {value}/week — {rating}"
    },
    "MindDiet_Fish_Not_Fried_Raw": {
        "kind": "continuous", "label": "Fish (not fried)",
        "round": 1,
        "derived": {"rating": lambda v: mind_rate_higher_is_better(v, good=1, ok=0.5)},  # ≥1/wk ideal
        "template": "{label}: {value}/week — {rating}"
    },
    "MindDiet_Beans_Raw": {
        "kind": "continuous", "label": "Beans/legumes",
        "round": 1,
        "derived": {"rating": lambda v: mind_rate_higher_is_better(v, good=3, ok=2)},  # ≥3/wk ideal
        "template": "{label}: {value}/week — {rating}"
    },
    "MindDiet_Poultry_Raw": {
        "kind": "continuous", "label": "Poultry",
        "round": 1,
        "derived": {"rating": lambda v: mind_rate_higher_is_better(v, good=2, ok=1)},  # ≥2/wk ideal
        "template": "{label}: {value}/week — {rating}"
    },

    # ----- Limit foods (lower is better) -----
    "MindDiet_Butter_Raw": {
        "kind": "continuous", "label": "Butter/margarine",
        "round": 2,
        # If this is tbsp/day, good_max≈1 tbsp/day. If servings/week, adjust thresholds.
        "derived": {"rating": lambda v: mind_rate_lower_is_better(v, good_max=7, ok_max=10)},  # proxy: ≤1/day => ≤7/wk
        "template": "{label}: {value}/week — {rating}"
    },
    "MindDiet_Cheese_Raw": {
        "kind": "continuous", "label": "Cheese",
        "round": 1,
        "derived": {"rating": lambda v: mind_rate_lower_is_better(v, good_max=1, ok_max=3)},  # ≤1/wk ideal
        "template": "{label}: {value}/week — {rating}"
    },
    "MindDiet_Red_Meats_Raw": {
        "kind": "continuous", "label": "Red meats",
        "round": 1,
        "derived": {"rating": lambda v: mind_rate_lower_is_better(v, good_max=4, ok_max=6)},  # ≤4/wk ideal
        "template": "{label}: {value}/week — {rating}"
    },
    "MindDiet_Fast_Fried_Foods_Raw": {
        "kind": "continuous", "label": "Fast/fried foods",
        "round": 1,
        "derived": {"rating": lambda v: mind_rate_lower_is_better(v, good_max=1, ok_max=2)},  # ≤1/wk ideal
        "template": "{label}: {value}/week — {rating}"
    },
    "MindDiet_Pastries_Sweets_Raw": {
        "kind": "continuous", "label": "Pastries & sweets",
        "round": 1,
        "derived": {"rating": lambda v: mind_rate_lower_is_better(v, good_max=5, ok_max=7)},  # ≤5/wk ideal
        "template": "{label}: {value}/week — {rating}"
    },
    "MindDiet_Wine_Raw": {
        "kind": "continuous", "label": "Wine",
        "round": 1,
        # MIND: ~1 glass/day ideal; as /week proxy: good≈7, ok≈3–10 window.
        "derived": {"rating": lambda v: "meets recommendation" if (v is not None and not np.isnan(v) and 3 <= v <= 10)
                    else ("partially meets" if (v is not None and not np.isnan(v) and (1 <= v < 3 or 10 < v <= 14))
                          else ("no data" if (v is None or np.isnan(v)) else "outside recommended range"))},
        "template": "{label}: {value}/week — {rating}"
    },

    # ---- HEI-2020 components ----
    "HEI2020_Fruit": {"kind": "hei", "label": "Total fruit (HEI-2020)", "max_points": 5,  "mode": "adequacy",
                      "template": "{label}: {value}/{maxp} — {rating}"},
    "HEI2020_Whole_Fruit": {"kind": "hei", "label": "Whole fruit (HEI-2020)", "max_points": 5, "mode": "adequacy",
                            "template": "{label}: {value}/{maxp} — {rating}"},
    "HEI2020_Veg": {"kind": "hei", "label": "Total vegetables (HEI-2020)", "max_points": 5, "mode": "adequacy",
                    "template": "{label}: {value}/{maxp} — {rating}"},
    "HEI2020_Greens_Beans": {"kind": "hei", "label": "Greens & beans (HEI-2020)", "max_points": 5, "mode": "adequacy",
                             "template": "{label}: {value}/{maxp} — {rating}"},
    "HEI2020_Whole_Grains": {"kind": "hei", "label": "Whole grains (HEI-2020)", "max_points": 10, "mode": "adequacy",
                             "template": "{label}: {value}/{maxp} — {rating}"},
    "HEI2020_Dairy": {"kind": "hei", "label": "Dairy (HEI-2020)", "max_points": 10, "mode": "adequacy",
                      "template": "{label}: {value}/{maxp} — {rating}"},
    "HEI2020_Protein_Foods": {"kind": "hei", "label": "Total protein foods (HEI-2020)", "max_points": 10, "mode": "adequacy",
                              "template": "{label}: {value}/{maxp} — {rating}"},
    "HEI2020_SeaFoods_PlantProteins": {"kind": "hei", "label": "Seafood & plant proteins (HEI-2020)", "max_points": 10, "mode": "adequacy",
                                       "template": "{label}: {value}/{maxp} — {rating}"},
    "HEI2020_Fatty_Acids": {"kind": "hei", "label": "Fatty acids ratio (HEI-2020)", "max_points": 10, "mode": "adequacy",
                            "template": "{label}: {value}/{maxp} — {rating}"},
    "HEI2020_Refined_Grains": {"kind": "hei", "label": "Refined grains (HEI-2020)", "max_points": 10, "mode": "moderation",
                               "template": "{label}: {value}/{maxp} — {rating}"},
    "HEI2020_Sodium": {"kind": "hei", "label": "Sodium (HEI-2020)", "max_points": 10, "mode": "moderation",
                       "template": "{label}: {value}/{maxp} — {rating}"},
    "HEI2020_Saturated_Fat": {"kind": "hei", "label": "Saturated fat (HEI-2020)", "max_points": 10, "mode": "moderation",
                              "template": "{label}: {value}/{maxp} — {rating}"},
    "HEI2020_Added_Sugars": {"kind": "hei", "label": "Added sugars (HEI-2020)", "max_points": 10, "mode": "moderation",
                             "template": "{label}: {value}/{maxp} — {rating}"},
    "HEI2020Score": {"kind": "hei", "label": "Total score (HEI-2020)", "max_points": 100, "mode": "moderation",
                             "template": "{label}: {value}/{maxp} — {rating}"},
}

# Schema for "Energy, Macronutrients & Alcohol"
energy_macronutrients_alcohol_schema = {
    "A_BEV": {
        "kind": "continuous",
        "label": "Total alcoholic beverages",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "A_CAL": {
        "kind": "continuous",
        "label": "Calories from alcoholic beverages",
        "unit": "kcal/day",
        "template": "{label}: {value} {unit}"
    },
    "alcohol": {
        "kind": "continuous",
        "label": "Total alcohol",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "caffeine": {
        "kind": "continuous",
        "label": "Caffeine",
        "unit": "mg/day",
        "template": "{label}: {value} {unit}"
    },
    "calories": {
        "kind": "continuous",
        "label": "Total energy intake",
        "unit": "kcal/day",
        "template": "{label}: {value} {unit}"
    },
    "carbo": {
        "kind": "continuous",
        "label": "Total carbohydrate",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "fat": {
        "kind": "continuous",
        "label": "Total fat",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "protein": {
        "kind": "continuous",
        "label": "Total protein",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "grams": {
        "kind": "continuous",
        "label": "Total food weight",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "joules": {
        "kind": "continuous",
        "label": "Total energy intake",
        "unit": "kJ/day",
        "template": "{label}: {value} {unit}"
    },
    "nitrogen": {
        "kind": "continuous",
        "label": "Nitrogen intake",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "starch": {
        "kind": "continuous",
        "label": "Starch",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "fiber": {
        "kind": "continuous",
        "label": "Total dietary fiber",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "fibh2o": {
        "kind": "continuous",
        "label": "Water-soluble fiber",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "fibinso": {
        "kind": "continuous",
        "label": "Insoluble fiber",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "pectins": {
        "kind": "continuous",
        "label": "Pectin intake",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    }
}

# Schema for "Sugars & Sweeteners"
sugars_schema = {
    "ADD_SUG": {
        "kind": "continuous",
        "label": "Added sugars (generic)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "addsugar": {
        "kind": "continuous",
        "label": "Added sugars (measured)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "adsugtot": {
        "kind": "continuous",
        "label": "Total added sugars",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "aspartam": {
        "kind": "continuous",
        "label": "Aspartame",
        "unit": "mg/day",
        "template": "{label}: {value} {unit}"
    },
    "erythr": {
        "kind": "continuous",
        "label": "Erythritol",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "fructose": {
        "kind": "continuous",
        "label": "Fructose",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "galactos": {
        "kind": "continuous",
        "label": "Galactose",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "glucose": {
        "kind": "continuous",
        "label": "Glucose",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "isomalt": {
        "kind": "continuous",
        "label": "Isomalt",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "lactitol": {
        "kind": "continuous",
        "label": "Lactitol",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "lactose": {
        "kind": "continuous",
        "label": "Lactose",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "maltitol": {
        "kind": "continuous",
        "label": "Maltitol",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "maltose": {
        "kind": "continuous",
        "label": "Maltose",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "mannitol": {
        "kind": "continuous",
        "label": "Mannitol",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "sacchar": {
        "kind": "continuous",
        "label": "Saccharin",
        "unit": "mg/day",
        "template": "{label}: {value} {unit}"
    },
    "sorbitol": {
        "kind": "continuous",
        "label": "Sorbitol",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "sucpoly": {
        "kind": "continuous",
        "label": "Sucralose polymer",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "sucrlose": {
        "kind": "continuous",
        "label": "Sucralose",
        "unit": "mg/day",
        "template": "{label}: {value} {unit}"
    },
    "sucrose": {
        "kind": "continuous",
        "label": "Sucrose",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "tagatose": {
        "kind": "continuous",
        "label": "Tagatose",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "xylitol": {
        "kind": "continuous",
        "label": "Xylitol",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "totsugar": {
        "kind": "continuous",
        "label": "Total sugars",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    }
}
# Schema for "Fats & Fatty Acids"
fats_schema = {
    "cholest": {
        "kind": "continuous",
        "label": "Cholesterol",
        "unit": "mg/day",
        "template": "{label}: {value} {unit}"
    },
    "clac9t11": {
        "kind": "continuous",
        "label": "CLA cis-9, trans-11",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "clat10c12": {
        "kind": "continuous",
        "label": "CLA trans-10, cis-12",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "totcla": {
        "kind": "continuous",
        "label": "Total conjugated linoleic acids (CLA)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "mfa141": {
        "kind": "continuous",
        "label": "Myristoleic acid (14:1)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "mfa161": {
        "kind": "continuous",
        "label": "Palmitoleic acid (16:1)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "mfa181": {
        "kind": "continuous",
        "label": "Oleic acid (18:1)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "mfa201": {
        "kind": "continuous",
        "label": "Gondoic acid (20:1)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "mfa221": {
        "kind": "continuous",
        "label": "Erucic acid (22:1)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "mfatot": {
        "kind": "continuous",
        "label": "Total monounsaturated fatty acids (MUFA)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "pfa182": {
        "kind": "continuous",
        "label": "Linoleic acid (18:2)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "pfa182n6": {
        "kind": "continuous",
        "label": "Linoleic acid (n-6)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "pfa183": {
        "kind": "continuous",
        "label": "Alpha-linolenic acid (18:3)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "pfa183n3": {
        "kind": "continuous",
        "label": "Alpha-linolenic acid (n-3)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "pfa183n6": {
        "kind": "continuous",
        "label": "Gamma-linolenic acid (n-6)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "pfa184": {
        "kind": "continuous",
        "label": "Stearidonic acid (18:4)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "pfa204": {
        "kind": "continuous",
        "label": "Arachidonic acid (20:4)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "pfa204n6": {
        "kind": "continuous",
        "label": "Arachidonic acid (n-6)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "pfa205": {
        "kind": "continuous",
        "label": "Eicosapentaenoic acid (EPA, 20:5 n-3)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "pfa225": {
        "kind": "continuous",
        "label": "Docosapentaenoic acid (DPA, 22:5)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "pfa226": {
        "kind": "continuous",
        "label": "Docosahexaenoic acid (DHA, 22:6)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "pfatot": {
        "kind": "continuous",
        "label": "Total polyunsaturated fatty acids (PUFA)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "omega3": {
        "kind": "continuous",
        "label": "Total omega-3 fatty acids",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "omega6": {
        "kind": "continuous",
        "label": "Total omega-6 fatty acids",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "satoco": {
        "kind": "continuous",
        "label": "Saturated + monounsaturated ratio",
        "unit": "ratio",
        "template": "{label}: {value}"
    },
    "sfa100": {
        "kind": "continuous",
        "label": "Capric acid (10:0)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "sfa120": {
        "kind": "continuous",
        "label": "Lauric acid (12:0)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "sfa140": {
        "kind": "continuous",
        "label": "Myristic acid (14:0)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "sfa160": {
        "kind": "continuous",
        "label": "Palmitic acid (16:0)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "sfa170": {
        "kind": "continuous",
        "label": "Heptadecanoic acid (17:0)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "sfa180": {
        "kind": "continuous",
        "label": "Stearic acid (18:0)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "sfa200": {
        "kind": "continuous",
        "label": "Arachidic acid (20:0)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "sfa220": {
        "kind": "continuous",
        "label": "Behenic acid (22:0)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "sfa40": {
        "kind": "continuous",
        "label": "Butyric acid (4:0)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "sfa60": {
        "kind": "continuous",
        "label": "Caproic acid (6:0)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "sfa80": {
        "kind": "continuous",
        "label": "Caprylic acid (8:0)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "sfatot": {
        "kind": "continuous",
        "label": "Total saturated fatty acids (SFA)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "tfa161t": {
        "kind": "continuous",
        "label": "Trans-hexadecenoic acid (16:1t)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "tfa181t": {
        "kind": "continuous",
        "label": "Trans-octadecenoic acid (18:1t)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "tfa182t": {
        "kind": "continuous",
        "label": "Trans-octadecadienoic acid (18:2t)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "totaltfa": {
        "kind": "continuous",
        "label": "Total trans fatty acids (TFA)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "solidfat": {
        "kind": "continuous",
        "label": "Solid fats",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "DISCFAT_OIL": {
        "kind": "continuous",
        "label": "Discretionary fat (oil)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "DISCFAT_SOL": {
        "kind": "continuous",
        "label": "Discretionary fat (solid)",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    }
}
# Schema for "Protein & Amino Acids"
protein_schema = {
    "alanine": {
        "kind": "continuous",
        "label": "Alanine",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "arginine": {
        "kind": "continuous",
        "label": "Arginine",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "aspartic": {
        "kind": "continuous",
        "label": "Aspartic acid",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "cystine": {
        "kind": "continuous",
        "label": "Cystine",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "glutamic": {
        "kind": "continuous",
        "label": "Glutamic acid",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "glycine": {
        "kind": "continuous",
        "label": "Glycine",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "histidin": {
        "kind": "continuous",
        "label": "Histidine",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "isoleuc": {
        "kind": "continuous",
        "label": "Isoleucine",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "leucine": {
        "kind": "continuous",
        "label": "Leucine",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "lysine": {
        "kind": "continuous",
        "label": "Lysine",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "methion": {
        "kind": "continuous",
        "label": "Methionine",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "phenylal": {
        "kind": "continuous",
        "label": "Phenylalanine",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "proline": {
        "kind": "continuous",
        "label": "Proline",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "serine": {
        "kind": "continuous",
        "label": "Serine",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "threonin": {
        "kind": "continuous",
        "label": "Threonine",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "tryptoph": {
        "kind": "continuous",
        "label": "Tryptophan",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "tyrosine": {
        "kind": "continuous",
        "label": "Tyrosine",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "protanim": {
        "kind": "continuous",
        "label": "Animal protein",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "protein": {
        "kind": "continuous",
        "label": "Total protein",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    },
    "protveg": {
        "kind": "continuous",
        "label": "Vegetable protein",
        "unit": "g/day",
        "template": "{label}: {value} {unit}"
    }
}

micronutrients_schema = {
    # --- Carotenoids & Antioxidants ---
    "alphacar": {"kind": "continuous", "label": "Alpha-carotene", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "alphtoce": {"kind": "continuous", "label": "Alpha-tocopherol (E)", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "alphtoco": {"kind": "continuous", "label": "Alpha-tocotrienol", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "betacar": {"kind": "continuous", "label": "Beta-carotene", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "betacryp": {"kind": "continuous", "label": "Beta-cryptoxanthin", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "betatoco": {"kind": "continuous", "label": "Beta-tocopherol", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "delttoco": {"kind": "continuous", "label": "Delta-tocopherol", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "gammtoco": {"kind": "continuous", "label": "Gamma-tocopherol", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "natoco": {"kind": "continuous", "label": "Nicotinamide tocopherol", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "lutzeax": {"kind": "continuous", "label": "Lutein + zeaxanthin", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "lycopene": {"kind": "continuous", "label": "Lycopene", "unit": "µg/day", "template": "{label}: {value} {unit}"},

    # --- Vitamins ---
    "retinol": {"kind": "continuous", "label": "Retinol (vitamin A)", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "vita_iu": {"kind": "continuous", "label": "Vitamin A (IU)", "unit": "IU/day", "template": "{label}: {value} {unit}"},
    "vita_rae": {"kind": "continuous", "label": "Vitamin A (RAE)", "unit": "µg RAE/day", "template": "{label}: {value} {unit}"},
    "vita_re": {"kind": "continuous", "label": "Vitamin A (RE)", "unit": "µg RE/day", "template": "{label}: {value} {unit}"},
    "vitd": {"kind": "continuous", "label": "Vitamin D", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "vitd_iu": {"kind": "continuous", "label": "Vitamin D (IU)", "unit": "IU/day", "template": "{label}: {value} {unit}"},
    "vitd2": {"kind": "continuous", "label": "Vitamin D2 (ergocalciferol)", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "vitd3": {"kind": "continuous", "label": "Vitamin D3 (cholecalciferol)", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "vite_iu": {"kind": "continuous", "label": "Vitamin E (IU)", "unit": "IU/day", "template": "{label}: {value} {unit}"},
    "vitk": {"kind": "continuous", "label": "Vitamin K", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "vitc": {"kind": "continuous", "label": "Vitamin C", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "fol_deqv": {"kind": "continuous", "label": "Folate (dietary equivalents)", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "fol_nat": {"kind": "continuous", "label": "Folate (natural food folate)", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "fol_syn": {"kind": "continuous", "label": "Folate (synthetic)", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "niacin": {"kind": "continuous", "label": "Niacin (B3)", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "niacineq": {"kind": "continuous", "label": "Niacin equivalents", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "pantothe": {"kind": "continuous", "label": "Pantothenic acid (B5)", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "ribofla": {"kind": "continuous", "label": "Riboflavin (B2)", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "thiamin": {"kind": "continuous", "label": "Thiamin (B1)", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "vitb12": {"kind": "continuous", "label": "Vitamin B12", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "vitb6": {"kind": "continuous", "label": "Vitamin B6", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "totfolat": {"kind": "continuous", "label": "Total folate", "unit": "µg/day", "template": "{label}: {value} {unit}"},

    # --- Minerals & Electrolytes ---
    "ash": {"kind": "continuous", "label": "Ash (mineral residue)", "unit": "g/day", "template": "{label}: {value} {unit}"},
    "calcium": {"kind": "continuous", "label": "Calcium", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "copper": {"kind": "continuous", "label": "Copper", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "iron": {"kind": "continuous", "label": "Iron", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "magnes": {"kind": "continuous", "label": "Magnesium", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "mangan": {"kind": "continuous", "label": "Manganese", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "phosphor": {"kind": "continuous", "label": "Phosphorus", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "potass": {"kind": "continuous", "label": "Potassium", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "selenium": {"kind": "continuous", "label": "Selenium", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "sodium": {"kind": "continuous", "label": "Sodium", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "zinc": {"kind": "continuous", "label": "Zinc", "unit": "mg/day", "template": "{label}: {value} {unit}"},
}

food_schema = {
    # --- Food Groups (Dairy, Fruits, Grains, Legumes, Vegetables, Meats) ---
    "D_CHEESE": {"kind": "continuous", "label": "Cheese (dairy group)", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "D_MILK": {"kind": "continuous", "label": "Milk (dairy group)", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "D_TOT_SOYM": {"kind": "continuous", "label": "Soy milk (dairy alternative)", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "D_TOTAL": {"kind": "continuous", "label": "Total dairy", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "D_YOGURT": {"kind": "continuous", "label": "Yogurt (dairy group)", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "F_CITMLB": {"kind": "continuous", "label": "Citrus, melon, berries", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "F_NJ_CITMLB": {"kind": "continuous", "label": "Citrus/melon/berries juice", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "F_NJ_OTHER": {"kind": "continuous", "label": "Other fruit juice", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "F_NJ_TOTAL": {"kind": "continuous", "label": "Total fruit juice", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "F_OTHER": {"kind": "continuous", "label": "Other fruits", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "F_TOTAL": {"kind": "continuous", "label": "Total fruit", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "G_NWHL": {"kind": "continuous", "label": "Non-whole grains", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "G_TOTAL": {"kind": "continuous", "label": "Total grains", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "G_WHL": {"kind": "continuous", "label": "Whole grains", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "rgrain": {"kind": "continuous", "label": "Refined grains", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "wgrain": {"kind": "continuous", "label": "Whole grain products", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "tgrain": {"kind": "continuous", "label": "Total grain products", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "LEGUMES": {"kind": "continuous", "label": "Legumes", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "V_DRKGR": {"kind": "continuous", "label": "Dark green vegetables", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "V_ORANGE": {"kind": "continuous", "label": "Orange vegetables", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "V_OTHER": {"kind": "continuous", "label": "Other vegetables", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "V_POTATO": {"kind": "continuous", "label": "Potatoes", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "V_STARCY": {"kind": "continuous", "label": "Starchy vegetables", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "V_TOMATO": {"kind": "continuous", "label": "Tomatoes", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "V_TOTAL": {"kind": "continuous", "label": "Total vegetables", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "M_EGG": {"kind": "continuous", "label": "Eggs", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "M_FISH_HI": {"kind": "continuous", "label": "High-fat fish", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "M_FISH_LO": {"kind": "continuous", "label": "Low-fat fish", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "M_FRANK": {"kind": "continuous", "label": "Frankfurters/sausages", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "M_MEAT": {"kind": "continuous", "label": "Red meat", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "M_MPF": {"kind": "continuous", "label": "Meat, poultry, fish total", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "M_NUTSD": {"kind": "continuous", "label": "Nuts and seeds", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "M_ORGAN": {"kind": "continuous", "label": "Organ meats", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "M_POULT": {"kind": "continuous", "label": "Poultry", "unit": "servings/day", "template": "{label}: {value} {unit}"},
    "M_SOY": {"kind": "continuous", "label": "Soy products", "unit": "servings/day", "template": "{label}: {value} {unit}"},

    # --- Phytochemicals & Other Compounds ---
    "coumest": {"kind": "continuous", "label": "Coumestrol", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "formontn": {"kind": "continuous", "label": "Formononetin", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "genistn": {"kind": "continuous", "label": "Genistein", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "daidzein": {"kind": "continuous", "label": "Daidzein", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "glycitn": {"kind": "continuous", "label": "Glycitein", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "liglar": {"kind": "continuous", "label": "Secoisolariciresinol", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "ligmat": {"kind": "continuous", "label": "Matairesinol", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "ligpin": {"kind": "continuous", "label": "Pinoresinol", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "ligsec": {"kind": "continuous", "label": "Secoisolariciresinol", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "ligtot": {"kind": "continuous", "label": "Total lignans", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "oxalic": {"kind": "continuous", "label": "Oxalic acid", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "inositol": {"kind": "continuous", "label": "Inositol", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "pinitol": {"kind": "continuous", "label": "Pinitol", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "spermidine": {"kind": "continuous", "label": "Spermidine", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "biochana": {"kind": "continuous", "label": "Biochanin A", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "water": {"kind": "continuous", "label": "Water intake", "unit": "g/day", "template": "{label}: {value} {unit}"}
}

# ===============================
# NHANES Day 1 (DR1*) — Energy, Macros, Stimulants, Alcohol
# ===============================
nhanes_day1_energy_macros = {
    "DR1TNUMF": {"kind": "continuous", "label": "Number of foods/beverages reported",
                 "template": "{label}: {value}"},
    "DR1TKCAL": {"kind": "continuous", "label": "Energy", "unit": "kcal/day",
                 "template": "{label}: {value} {unit}"},
    "DR1TPROT": {"kind": "continuous", "label": "Protein", "unit": "g/day",
                 "template": "{label}: {value} {unit}"},
    "DR1TCARB": {"kind": "continuous", "label": "Carbohydrate", "unit": "g/day",
                 "template": "{label}: {value} {unit}"},
    "DR1TSUGR": {"kind": "continuous", "label": "Total sugars", "unit": "g/day",
                 "template": "{label}: {value} {unit}"},
    "DR1TFIBE": {"kind": "continuous", "label": "Dietary fiber", "unit": "g/day",
                 "template": "{label}: {value} {unit}"},
    "DR1TTFAT": {"kind": "continuous", "label": "Total fat", "unit": "g/day",
                 "template": "{label}: {value} {unit}"},
    "DR1TSFAT": {"kind": "continuous", "label": "Total saturated fatty acids", "unit": "g/day",
                 "template": "{label}: {value} {unit}"},
    "DR1TMFAT": {"kind": "continuous", "label": "Total monounsaturated fatty acids", "unit": "g/day",
                 "template": "{label}: {value} {unit}"},
    "DR1TPFAT": {"kind": "continuous", "label": "Total polyunsaturated fatty acids", "unit": "g/day",
                 "template": "{label}: {value} {unit}"},
    "DR1TCHOL": {"kind": "continuous", "label": "Cholesterol", "unit": "mg/day",
                 "template": "{label}: {value} {unit}"},
    "DR1TCAFF": {"kind": "continuous", "label": "Caffeine", "unit": "mg/day",
                 "template": "{label}: {value} {unit}"},
    "DR1TTHEO": {"kind": "continuous", "label": "Theobromine", "unit": "mg/day",
                 "template": "{label}: {value} {unit}"},
    "DR1TALCO": {"kind": "continuous", "label": "Alcohol", "unit": "g/day",
                 "template": "{label}: {value} {unit}"},
    "DR1TMOIS": {"kind": "continuous", "label": "Moisture", "unit": "g/day",
                 "template": "{label}: {value} {unit}"},
}


# ===============================
# NHANES Day 1 (DR1*) — Detailed fatty acids
# ===============================
nhanes_day1_fats_detail = {
    "DR1TS040": {"kind": "continuous", "label": "SFA 4:0 (Butanoic)", "unit": "g/day", "template": "{label}: {value} {unit}"},
    "DR1TS060": {"kind": "continuous", "label": "SFA 6:0 (Hexanoic)", "unit": "g/day", "template": "{label}: {value} {unit}"},
    "DR1TS080": {"kind": "continuous", "label": "SFA 8:0 (Octanoic)", "unit": "g/day", "template": "{label}: {value} {unit}"},
    "DR1TS100": {"kind": "continuous", "label": "SFA 10:0 (Decanoic)", "unit": "g/day", "template": "{label}: {value} {unit}"},
    "DR1TS120": {"kind": "continuous", "label": "SFA 12:0 (Dodecanoic)", "unit": "g/day", "template": "{label}: {value} {unit}"},
    "DR1TS140": {"kind": "continuous", "label": "SFA 14:0 (Tetradecanoic)", "unit": "g/day", "template": "{label}: {value} {unit}"},
    "DR1TS160": {"kind": "continuous", "label": "SFA 16:0 (Hexadecanoic)", "unit": "g/day", "template": "{label}: {value} {unit}"},
    "DR1TS180": {"kind": "continuous", "label": "SFA 18:0 (Octadecanoic)", "unit": "g/day", "template": "{label}: {value} {unit}"},
    "DR1TM161": {"kind": "continuous", "label": "MFA 16:1 (Hexadecenoic)", "unit": "g/day", "template": "{label}: {value} {unit}"},
    "DR1TM181": {"kind": "continuous", "label": "MFA 18:1 (Octadecenoic)", "unit": "g/day", "template": "{label}: {value} {unit}"},
    "DR1TM201": {"kind": "continuous", "label": "MFA 20:1 (Eicosenoic)", "unit": "g/day", "template": "{label}: {value} {unit}"},
    "DR1TM221": {"kind": "continuous", "label": "MFA 22:1 (Docosenoic)", "unit": "g/day", "template": "{label}: {value} {unit}"},
    "DR1TP182": {"kind": "continuous", "label": "PFA 18:2 (Octadecadienoic)", "unit": "g/day", "template": "{label}: {value} {unit}"},
    "DR1TP183": {"kind": "continuous", "label": "PFA 18:3 (Octadecatrienoic)", "unit": "g/day", "template": "{label}: {value} {unit}"},
    "DR1TP184": {"kind": "continuous", "label": "PFA 18:4 (Octadecatetraenoic)", "unit": "g/day", "template": "{label}: {value} {unit}"},
    "DR1TP204": {"kind": "continuous", "label": "PFA 20:4 (Eicosatetraenoic)", "unit": "g/day", "template": "{label}: {value} {unit}"},
    "DR1TP205": {"kind": "continuous", "label": "PFA 20:5 (Eicosapentaenoic)", "unit": "g/day", "template": "{label}: {value} {unit}"},
    "DR1TP225": {"kind": "continuous", "label": "PFA 22:5 (Docosapentaenoic)", "unit": "g/day", "template": "{label}: {value} {unit}"},
    "DR1TP226": {"kind": "continuous", "label": "PFA 22:6 (Docosahexaenoic)", "unit": "g/day", "template": "{label}: {value} {unit}"},
}


# ===============================
# NHANES Day 1 (DR1*) — Vitamins & minerals
# ===============================
nhanes_day1_micros = {
    "DR1TATOC": {"kind": "continuous", "label": "Vitamin E (alpha-tocopherol)", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "DR1TATOA": {"kind": "continuous", "label": "Added alpha-tocopherol (Vitamin E)", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "DR1TRET":  {"kind": "continuous", "label": "Retinol", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "DR1TVARA": {"kind": "continuous", "label": "Vitamin A, RAE", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "DR1TACAR": {"kind": "continuous", "label": "Alpha-carotene", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "DR1TBCAR": {"kind": "continuous", "label": "Beta-carotene", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "DR1TCRYP": {"kind": "continuous", "label": "Beta-cryptoxanthin", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "DR1TLYCO": {"kind": "continuous", "label": "Lycopene", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "DR1TLZ":   {"kind": "continuous", "label": "Lutein + zeaxanthin", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "DR1TVB1":  {"kind": "continuous", "label": "Thiamin (B1)", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "DR1TVB2":  {"kind": "continuous", "label": "Riboflavin (B2)", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "DR1TNIAC": {"kind": "continuous", "label": "Niacin (B3)", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "DR1TVB6":  {"kind": "continuous", "label": "Vitamin B6", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "DR1TFOLA": {"kind": "continuous", "label": "Folate (total)", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "DR1TFA":   {"kind": "continuous", "label": "Folic acid", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "DR1TFF":   {"kind": "continuous", "label": "Food folate", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "DR1TFDFE": {"kind": "continuous", "label": "Folate, DFE", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "DR1TCHL":  {"kind": "continuous", "label": "Total choline", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "DR1TVB12": {"kind": "continuous", "label": "Vitamin B12", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "DR1TB12A": {"kind": "continuous", "label": "Added vitamin B12", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "DR1TVC":   {"kind": "continuous", "label": "Vitamin C", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "DR1TVD":   {"kind": "continuous", "label": "Vitamin D (D2 + D3)", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "DR1TVK":   {"kind": "continuous", "label": "Vitamin K", "unit": "µg/day", "template": "{label}: {value} {unit}"},
    "DR1TCALC": {"kind": "continuous", "label": "Calcium", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "DR1TPHOS": {"kind": "continuous", "label": "Phosphorus", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "DR1TMAGN": {"kind": "continuous", "label": "Magnesium", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "DR1TIRON": {"kind": "continuous", "label": "Iron", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "DR1TZINC": {"kind": "continuous", "label": "Zinc", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "DR1TCOPP": {"kind": "continuous", "label": "Copper", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "DR1TSODI": {"kind": "continuous", "label": "Sodium", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "DR1TPOTA": {"kind": "continuous", "label": "Potassium", "unit": "mg/day", "template": "{label}: {value} {unit}"},
    "DR1TSELE": {"kind": "continuous", "label": "Selenium", "unit": "µg/day", "template": "{label}: {value} {unit}"},
}

# ===============================
# Water & compare-to-usual (Day 1 & Day 2)
# ===============================
nhanes_day1_water_behavior = {
    "DR1_300":  {"kind": "categorical", "label": "Compare food consumed yesterday to usual", "template": "{label}: {value}"},
    "DR1_320Z": {"kind": "continuous", "label": "Total plain water yesterday", "unit": "g", "template": "{label}: {value} {unit}"},
    "DR1_330Z": {"kind": "continuous", "label": "Total tap water yesterday", "unit": "g", "template": "{label}: {value} {unit}"},
    "DR1BWATZ": {"kind": "continuous", "label": "Total bottled water yesterday", "unit": "g", "template": "{label}: {value} {unit}"},
    "DR1TWSZ":  {"kind": "categorical", "label": "Tap water source", "template": "{label}: {value}"},
}
