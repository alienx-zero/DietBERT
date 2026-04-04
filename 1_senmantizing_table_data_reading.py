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
hf_logging.set_verbosity_error()
from sklearn.model_selection import StratifiedKFold
from model import HierarchicalTransformer, HierCollator, SubsetWeeklyDiet
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from scipy.io import savemat
import time
import datetime as dt
import os, json, io, sys
from pathlib import Path
from safetensors.torch import save_file  # pip install safetensors

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


def map_food_codes_to_names(df, fndds_excel_path, key_food_code = 'DR1IFDCD', code_col="Food code", desc_col="Main food description"):
    """
    Replace DR1IFDCD column (food codes) in df with food descriptions
    using the mapping from the FNDDS Nutrient Values sheet.
    
    Parameters
    ----------
    df : pd.DataFrame
        Your dataframe containing column 'DR1IFDCD'.
    fndds_excel_path : str
        Path to the USDA FNDDS Excel file.
    code_col : str, optional
        Name of the column with food codes in the FNDDS sheet.
    desc_col : str, optional
        Name of the column with descriptions in the FNDDS sheet.
    """
    # Read mapping sheet, skip first row
    fndds = pd.read_excel(fndds_excel_path, sheet_name="FNDDS Nutrient Values", skiprows=1)

    # Build mapping dict {food_code -> description}
    code_to_desc = dict(zip(fndds[code_col], fndds[desc_col]))

    # Replace codes with descriptions
    df[key_food_code] = df[key_food_code].map(code_to_desc)

    return df

def convert_seconds_to_time(df, col="DR1_020"):
    """
    Convert a column of seconds since midnight into 24h time format.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    col : str
        Column name with seconds (e.g., DR1_020)
    new_col : str
        Name of the new column with formatted times

    Returns
    -------
    df : pd.DataFrame
        DataFrame with an added column in HH:MM:SS format
    """
    df[col] = pd.to_datetime(df[col], unit="s").dt.strftime("%H:%M:%S")
    return df

def replace_DR1CCMTX_with_desc(df, col="DR1CCMTX"):
    # Define mapping dictionary
    dr1ccmtx_map = {
        0:  "Non-combination food",
        1:  "Beverage w/ additions",
        2:  "Cereal w/ additions",
        3:  "Bread/baked products w/ additions",
        4:  "Salad",
        5:  "Sandwiches",
        6:  "Soup",
        7:  "Frozen meals",
        8:  "Ice cream/frozen yogurt w/ additions",
        9:  "Dried beans and vegetable w/ additions",
        10: "Fruit w/ additions",
        11: "Tortilla products",
        12: "Meat, poultry, fish",
        13: "Lunchables®",
        14: "Chips w/ additions",
        15: "Baby Toddler Food and Infant Formula",
        90: "Other mixtures",
        ".": "Missing"
    }

    # Replace codes with descriptions
    df[col] = df[col].map(dr1ccmtx_map)

    return df

def build_subject_sentences(
    df: pd.DataFrame,
    time_col="DR1_020",
    food_col="DR1IFDCD",
    combo_col="DR1CCMTX",
    id_col="SEQN",
):
    results = {}

    # try to sort by time within each subject
    # (if time strings vary, we coerce; unparseable times become NaT and are kept at end)
    def _sort_by_time(g):
        t = pd.to_datetime(g[time_col], errors="coerce")
        return g.assign(_t=t).sort_values(["_t", time_col]).drop(columns="_t")

    for subject, sub_df in df.groupby(id_col):
        sub_df = _sort_by_time(sub_df)

        sentences = []
        for time_, group in sub_df.groupby(time_col):
            # foods: drop NaN
            foods = [f for f in group[food_col].tolist() if pd.notna(f)]
            if not foods:
                continue

            # unique combo labels (drop NaN)
            combo_types = [c for c in group[combo_col].dropna().unique().tolist()]
            combo_str = "; ".join(combo_types) if combo_types else ""

            food_str = ", ".join(foods)
            if combo_str and combo_str != "Non-combination food":
                sentence = f"{time_} - eat: {food_str}; Combination food: {combo_str}"
            else:
                sentence = f"{time_} - eat: {food_str}."
            sentences.append(sentence)

        # ONE list item per subject, with time-sentences on different lines
        results[subject] = "\n".join(sentences)

    return results
############################prediction variables loading

# Read XPT file
DR1IFF_L = pd.read_sas(r"H:\postdoc\UCLA_postdoc\Diet_predict\NHANES\data\DR1IFF_L.xpt", format="xport")
DR1IFF_L = DR1IFF_L.astype(float)
DR1IFF_L[DR1IFF_L<0.0001] = 0
print(DR1IFF_L.head())
fndds_file_path = r"H:\postdoc\UCLA_postdoc\Diet_predict\NHANES\data\2021-2023 FNDDS At A Glance - FNDDS Nutrient Values.xlsx"  # Adjust this path accordingly
df = map_food_codes_to_names(DR1IFF_L, fndds_file_path)
df = convert_seconds_to_time(df)
df = replace_DR1CCMTX_with_desc(df)
sentences_day1 = build_subject_sentences(df)
keys = np.array(list(sentences_day1.keys()))
values = np.array(list(sentences_day1.values()))
sentences_day1_df = pd.DataFrame({'sub': keys, 'day_record1': values})

DRTOT1_L = pd.read_sas(r"H:\postdoc\UCLA_postdoc\Diet_predict\NHANES\data\DR1TOT_L.xpt", format="xport")
DRTOT1_L[DRTOT1_L<0.0001] = 0

DR2IFF_L = pd.read_sas(r"H:\postdoc\UCLA_postdoc\Diet_predict\NHANES\data\DR2IFF_L.xpt", format="xport")
DR2IFF_L = DR2IFF_L.astype(float)
DR2IFF_L[DR2IFF_L<0.0001] = 0
df = map_food_codes_to_names(DR2IFF_L, fndds_file_path, 'DR2IFDCD')
df = convert_seconds_to_time(df, col="DR2_020")
df = replace_DR1CCMTX_with_desc(df, col='DR2CCMTX')
sentences_day2 = build_subject_sentences(df, 'DR2_020', 'DR2IFDCD', 'DR2CCMTX')
keys = np.array(list(sentences_day2.keys()))
values = np.array(list(sentences_day2.values()))
sentences_day2_df = pd.DataFrame({'sub': keys, 'day_record2': values})
sentences_days = pd.merge(sentences_day1_df, sentences_day2_df, on = 'sub', how='outer')

DR2TOT_L = pd.read_sas(r"H:\postdoc\UCLA_postdoc\Diet_predict\NHANES\data\DR2TOT_L.xpt", format="xport")
DR2TOT_L[DR2TOT_L<0.0001] = 0

DPQ_L = pd.read_sas(r"H:\postdoc\UCLA_postdoc\Diet_predict\NHANES\data\DPQ_L.xpt", format="xport")
DPQ_L[DPQ_L<0.0001] = 0
DPQ_L.iloc[:,1:][DPQ_L.iloc[:,1:]>3] = np.nan
DPQ_L['sum'] = DPQ_L.iloc[:,1:].sum(axis=1, skipna=False)

df1_ren = DRTOT1_L.rename(columns=lambda c: re.sub(r'^DR1', '', c))
df2_ren = DR2TOT_L.rename(columns=lambda c: re.sub(r'^DR2', '', c))
# intersect the columns that exist in both
common = sorted(set(df1_ren.columns).intersection(df2_ren.columns))
mean_df = df1_ren[common].combine(df2_ren[common], lambda s1, s2: s1.combine(s2, lambda a,b: (a+b)/2 if pd.notna(a) and pd.notna(b) else (a if pd.notna(a) else b)))
# overwrite df1 with mean results (keeping original DR1/DR2 colnames replaced)
for col in common:
    DRTOT1_L[f'DR1{col}'] = mean_df[col]

#########################merge the diet records into 
diet_all_nhanes = []
for i in range(len(DRTOT1_L)):
    diet_info = {}
    sub = DRTOT1_L.iloc[i]['SEQN']
    nutrient_info = DRTOT1_L.iloc[i]
    day_record = sentences_days[sentences_days['sub'] == sub]
    energy_df = nutrient_info[list(nhanes_day1_energy_macros.keys())].squeeze().T
    energy_df["text"] = row_to_text(energy_df, nhanes_day1_energy_macros, sep="; ")
    fat_df = nutrient_info[list(nhanes_day1_fats_detail.keys())].squeeze().T
    fat_df["text"] = row_to_text(fat_df, nhanes_day1_fats_detail, sep="; ")
    micronutrients_df = nutrient_info[list(nhanes_day1_micros.keys())].squeeze().T
    micronutrients_df["text"] = row_to_text(micronutrients_df, nhanes_day1_micros, sep="; ")
    food_df = nutrient_info[list(nhanes_day1_water_behavior.keys())].squeeze().T
    food_df["text"] = row_to_text(food_df, nhanes_day1_water_behavior, sep="; ")
    # try:
    if len(day_record)==0:
        info = [[''] + [energy_df["text"]] + [''] + [fat_df["text"]] + [''] + 
                [micronutrients_df["text"]] + [food_df["text"]]][0]
    else:
        info = [[''] + [energy_df["text"]] + [''] + [fat_df["text"]] + [''] + 
                [micronutrients_df["text"]] + [food_df["text"]] + day_record['day_record1'].tolist() + day_record['day_record2'].tolist()][0]

    diet_info['text_info'] = info
    diet_info['sub'] = sub
    if len(DPQ_L['sum'][DPQ_L['SEQN'] == sub]) == 0:
        diet_info['depress_score'] = np.nan
    else:
        diet_info['depress_score'] = DPQ_L['sum'][DPQ_L['SEQN'] == sub].iloc[0]
    diet_all_nhanes.append(diet_info)

#########church dataset
pcd_info = pd.read_excel(r"H:\PHD\learning\research\dataset\church_data\kzhao_bmi_aging_20250807.xlsx", sheet_name='kzhao_bmi_aging')
pcd_sub = pcd_info['NDPNum'].values
pcd_selected = pcd_info[['Demog_Income', 'Demog_ADI_National_Rank', 'ACE_Emotional_Abuse', 'ACE_Physical_Abuse',
                         'ACE_Sexual_Abuse', 'ACE_Substance_Abuse', 'ACE_Parental_DivorceSep', 'ACE_Household_Mental_Illness',
                         'ACE_Incarcerated_Household_Member', 'ACE_Parents_Treated_Violently', 'ACE_Score', 
                         'ETI_General_Score', 'ETI_Physical_Score', 'ETI_Emotional_Score', 'ETI_Sexual_Score', 
                         'ETI_Total_Score', 'SF12_PCS', 'SF12_MCS', 'SI_Social_Disconnectedness_Score',
                         'SI_Lack_Social_Support_Score', 'SI_Perceived_Loneliness_Score', 'MASQ_Language', 
                         'MASQ_Visual_Perception', 'MASQ_Verbal_Memory', 'MASQ_Visual_Spatial', 'TRAILA_Time', 
                         'TRAILB_Time', 'CAMSR_Score', 'STAI_TAnxiety_raw', 'STAI_TAnxiety', 'HAD_Anxiety',
                         'HAD_Depression', 'BISBAS_BAS_Drive', 'BISBAS_BAS_Fun_Seeking', 'BISBAS_BAS_Reward_Response',
                         'BISBAS_BIS', 'MHCSF_Hedonic', 'MHCSF_Eudaimonic_Social',
                         'MHCSF_Eudaimonic_Psych', 'MHCSF_Overall', 'FFM_Observe', 'FFM_Describe', 'FFM_ActAwareness',
                         'FFM_Nonjudge', 'FFM_Nonreact', 'FFM_Total_Score', 'CDRISC_Score', 'CDRISC_Persistence_Score',
                         'CDRISC_Adaptability_Score', 'CDRISC_Control_Meaning_Score', 'CDRISC_Meaning_Score', 
                         'IPAQ_Work_Total_MET', 'IPAQ_Transport_Total_MET', 'IPAQ_Domestic_Total_MET', 
                         'IPAQ_Total_Walking_MET', 'IPAQ_Total_Moderate_MET', 'IPAQ_Total_Vigorous_MET', 'IPAQ_Total_PA_MET',
                         'YFAS_SymptomCount', 'IPAQ_Sitting_Total', 'MINI_Major_Depressive_c', 'MINI_Dysthymia_c', 'MINI_Major_Depressive_p',
                         'MINI_Dysthymia_p', 'MINI_Suicidality_c', 'MINI_Manic_p', 'MINI_Manic_c', 'MINI_Panic_c',
                         'MINI_Agoraphobia_l', 'MINI_Agoraphobia_c', 'MINI_Social_Phobia_c', 'MINI_Specific_Phobia_c',
                         'MINI_OCD_c', 'MINI_Alcohol_Dependence_l', 'MINI_Alcohol_Dependence_c', 'MINI_Substance_Dependence_l', 
                         'MINI_Substance_Dependence_c', 'MINI_Anorexia_c', 'MINI_Bulimia_c', 'MINI_Generalized_Anxiety_c', 
                         'MINI_Body_Dysmorphic_c', 'MINI_Premenstrual_Dysmorphic_c', 'IBS', 'IBS_c', 'GERD',
                         'GERD_c', 'Gastroparesis', 'Gastroparesis_c', 'FDyspepsia', 'FDyspepsia_c', 'UDyspepsia', 
                         'UDyspepsia_c', 'CVS', 'CVS_c', 'VLVD', 'VLVD_c', 'UC', 'UC_c', 'Crohns', 'Crohns_c', 'ICIBPPS',
                         'ICIBPPS_c', 'Prostatitis', 'Prostatitis_c', 'Endomet', 'Endomet_c', 'TMJ_TMD', 'TMJ_TMD_c', 
                         'CFS', 'CFS_c', 'FM', 'FM_c', 'Migraine', 'Migraine_c', 'Chest_Pain', 'Chest_Pain_c', 'Back_Neck',
                         'Back_Neck_c', 'Anxiety', 'Anxiety_c', 'Depression', 'Depression_c', 'Bipolar', 'Bipolar_c', 
                         'PTSD', 'PTSD_c', 'Schizo', 'Schizo_c', 'Eating', 'Eating_c', 'Substance', 'Substance_c', 'IBD',
                         'IBD_c', 'Other_Pain', 'Other_Pain_c', 'Other_Pain_Expl', 'Other_Condition', 'Other_Condition_c',
                         'OCD', 'OCD_c', 'UCPPS', 'UCPPS_c', 'Tension_HA', 'Tension_HA_c', 'Low_Back', 'Low_Back_c',
                         'VSI_Score', 'GFCQT_Total', 'SSR_Arousal', 'SSR_Stress', 'SSR_Anxiety', 'SSR_Anger', 'SSR_Fatigue',
                         'SSR_Attention', 'STAI_SAnxiety_raw', 'STAI_SAnxiety', 'PANAS_PosAffect', 'PANAS_NegAffect',
                         'GFCQS_Total', 'PROMIS_Sleep_Score_R', 'PROMIS_Sleep_Score']]

pcd_selected['Demog_ADI_National_Rank'][pcd_selected['Demog_ADI_National_Rank'] > 900] = np.nan    
pcd_selected[['IBS', 'IBS_c', 'GERD', 'GERD_c', 'Gastroparesis', 'Gastroparesis_c', 'FDyspepsia', 
          'FDyspepsia_c', 'UDyspepsia', 'UDyspepsia_c', 'CVS', 'CVS_c', 'VLVD', 'VLVD_c', 'UC',
          'UC_c', 'Crohns', 'Crohns_c', 'ICIBPPS', 'ICIBPPS_c', 'Prostatitis', 'Prostatitis_c', 
          'Endomet', 'Endomet_c', 'TMJ_TMD', 'TMJ_TMD_c', 'CFS', 'CFS_c', 'FM', 'FM_c', 'Migraine', 
          'Migraine_c', 'Chest_Pain', 'Chest_Pain_c', 'Back_Neck', 'Back_Neck_c', 'Anxiety', 'Anxiety_c', 
          'Depression', 'Depression_c', 'Bipolar', 'Bipolar_c', 'PTSD', 'PTSD_c', 'Schizo', 'Schizo_c',
          'Eating', 'Eating_c', 'Substance', 'Substance_c', 'IBD', 'IBD_c', 'Other_Pain', 'Other_Pain_c', 
          'Other_Condition', 'Other_Condition_c', 'OCD', 'OCD_c', 'UCPPS', 'UCPPS_c']][
              pcd_selected[['IBS', 'IBS_c', 'GERD', 'GERD_c', 'Gastroparesis', 'Gastroparesis_c', 'FDyspepsia', 
                    'FDyspepsia_c', 'UDyspepsia', 'UDyspepsia_c', 'CVS', 'CVS_c', 'VLVD', 'VLVD_c', 'UC',
                    'UC_c', 'Crohns', 'Crohns_c', 'ICIBPPS', 'ICIBPPS_c', 'Prostatitis', 'Prostatitis_c', 
                    'Endomet', 'Endomet_c', 'TMJ_TMD', 'TMJ_TMD_c', 'CFS', 'CFS_c', 'FM', 'FM_c', 'Migraine', 
                    'Migraine_c', 'Chest_Pain', 'Chest_Pain_c', 'Back_Neck', 'Back_Neck_c', 'Anxiety', 'Anxiety_c', 
                    'Depression', 'Depression_c', 'Bipolar', 'Bipolar_c', 'PTSD', 'PTSD_c', 'Schizo', 'Schizo_c',
                    'Eating', 'Eating_c', 'Substance', 'Substance_c', 'IBD', 'IBD_c', 'Other_Pain', 'Other_Pain_c', 
                    'Other_Condition', 'Other_Condition_c', 'OCD', 'OCD_c', 'UCPPS', 'UCPPS_c']]==2] = np.nan
####################diet features
diet_record = pcd_info[['Diet_Recall_List']]

###########################food consumption
diet_info = pd.read_csv(r"H:\PHD\learning\research\dataset\church_data\kzhao_bmi_aging_vioscreen_20250807.csv")
diet_sub = diet_info['NDPNum']
diet_keys = ['HEI2020Score', 'HEI2020_Fruit', 'HEI2020_Whole_Fruit', 'HEI2020_Veg', 'HEI2020_Greens_Beans', 
                         'HEI2020_Whole_Grains', 'HEI2020_Dairy', 'HEI2020_Protein_Foods', 'HEI2020_SeaFoods_PlantProteins', 
                         'HEI2020_Fatty_Acids', 'HEI2020_Refined_Grains', 'HEI2020_Sodium', 'HEI2020_Saturated_Fat', 
                         'HEI2020_Added_Sugars', 'MindDietScoreWine', 'MindDiet_Green_Leafy_Vegetables_Raw', 
                         'MindDiet_Other_Vegetables_Raw', 'MindDiet_Berries_Raw', 'MindDiet_Nuts_Raw', 'MindDiet_Olive_Oil_Raw',
                         'MindDiet_Butter_Raw', 'MindDiet_Cheese_Raw', 'MindDiet_Whole_Grains_Raw', 'MindDiet_Fish_Not_Fried_Raw',
                         'MindDiet_Beans_Raw', 'MindDiet_Poultry_Raw', 'MindDiet_Red_Meats_Raw', 'MindDiet_Fast_Fried_Foods_Raw',
                         'MindDiet_Pastries_Sweets_Raw', 'MindDiet_Wine_Raw', 
                         'A_BEV', 'A_CAL', 'acesupot', 'ADD_SUG' ,'addsugar', 
                         'adsugtot', 'alanine', 'alcohol', 'alphacar', 'alphtoce', 'alphtoco', 'arginine', 'ash', 'aspartam', 
                         'aspartic', 'avcarb', 'betacar', 'betacryp', 'betaine', 'betatoco', 'biochana', 'caffeine', 'calcium', 
                         'calories', 'carbo', 'cholest', 'choline', 'clac9t11', 'clat10c12', 'copper', 'coumest', 'cystine', 
                         'D_CHEESE', 'D_MILK', 'D_TOT_SOYM', 'D_TOTAL', 'D_YOGURT', 'daidzein', 'delttoco', 'DISCFAT_OIL', 
                         'DISCFAT_SOL', 'erythr', 'F_CITMLB', 'F_NJ_CITMLB', 'F_NJ_OTHER', 'F_NJ_TOTAL', 'F_OTHER', 'F_TOTAL', 
                         'fat', 'fiber', 'fibh2o', 'fibinso', 'fol_deqv', 'fol_nat', 'fol_syn', 'formontn', 'fructose', 
                         'G_NWHL', 'G_TOTAL', 'G_WHL', 'galactos', 'gammtoco', 'genistn', 'glucose', 'glutamic', 'gluten', 
                         'glycine', 'glycitn', 'grams', 'histidin', 'inositol', 'iron', 'isoleuc', 'isomalt', 'joules', 'lactitol', 
                         'lactose', 'LEGUMES', 'leucine', 'liglar', 'ligmat', 'ligpin', 'ligsec', 'ligtot', 'lutzeax', 'lycopene', 
                         'lysine', 'M_EGG', 'M_FISH_HI', 'M_FISH_LO', 'M_FRANK', 'M_MEAT', 'M_MPF', 'M_NUTSD', 'M_ORGAN', 'M_POULT', 
                         'M_SOY', 'magnes', 'maltitol', 'maltose', 'mangan', 'mannitol', 'methhis3', 'methion', 'mfa141', 'mfa161', 
                         'mfa181', 'mfa201', 'mfa221', 'mfatot', 'natoco', 'nccglbr', 'nccglgr', 'niacin', 'niacineq', 'nitrogen', 
                         'omega3', 'omega6', 'oxalic', 'pantothe', 'pectins', 'pfa182', 'pfa182n6', 'pfa183', 'pfa183n3', 'pfa183n6', 
                         'pfa184', 'pfa204', 'pfa204n6', 'pfa205', 'pfa225', 'pfa226', 'pfatot', 'phenylal', 'phosphor', 'phytic', 
                         'pinitol', 'potass', 'proline', 'protanim', 'protein', 'protveg', 'retinol', 'rgrain', 'ribofla', 'sacchar', 
                         'satoco', 'selenium', 'serine', 'sfa100', 'sfa120', 'sfa140', 'sfa160', 'sfa170', 'sfa180', 'sfa200', 
                         'sfa220', 'sfa40', 'sfa60', 'sfa80', 'sfatot', 'sodium', 'solidfat', 'sorbitol', 'spermidine', 'starch', 
                         'sucpoly', 'sucrlose', 'sucrose', 'tagatose', 'tfa161t', 'tfa181t', 'tfa182t', 'tgrain', 'thiamin', 'threonin', 
                         'totaltfa', 'totcla', 'totfolat', 'totsugar', 'tryptoph', 'tyrosine', 'V_DRKGR', 'V_ORANGE', 'V_OTHER', 
                         'V_POTATO', 'V_STARCY', 'V_TOMATO', 'V_TOTAL', 'valine', 'vita_iu', 'vita_rae', 'vita_re', 'vitb12', 'vitb6', 
                         'vitc', 'vitd', 'vitd_iu', 'vitd2', 'vitd3', 'vite_iu', 'vitk', 'water', 'wgrain', 'xylitol', 'zinc']
diet_select = diet_info[diet_keys]

diet_select_with_values = diet_select[np.isfinite(diet_select['zinc'])]
diet_sub_with_values = diet_sub[np.isfinite(diet_select['zinc'])].values

_, diet_recall_idx, diet_detail_idx = np.intersect1d(pcd_sub, diet_sub_with_values, return_indices=True)
diet_sub = pcd_sub[diet_recall_idx]
diet_record_ = diet_record.iloc[diet_recall_idx].values
diet_detail = diet_select_with_values.iloc[diet_detail_idx]
pcd_y = pcd_selected.iloc[diet_recall_idx]

###################washing diet record
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
    days = [d+'.' for d in days if any(tok.strip() for tok in d.splitlines()[1:])]
    return days

#########################merge the diet records into 
diet_all = []
for i, record in enumerate(diet_record_):
    diet_info = {}
    try:
        if np.isnan(record[0]):
            info1 = diet_detail.iloc[i]
            hei_mind_df = info1[list(schema.keys())]
            hei_mind_df["text"] = row_to_text(hei_mind_df, schema, sep="; ")
            energy_df = info1[list(energy_macronutrients_alcohol_schema.keys())]
            energy_df["text"] = row_to_text(energy_df, energy_macronutrients_alcohol_schema, sep="; ")
            sugars_df = info1[list(sugars_schema.keys())]
            sugars_df["text"] = row_to_text(sugars_df, sugars_schema, sep="; ")
            fat_df = info1[list(fats_schema.keys())]
            fat_df["text"] = row_to_text(fat_df, fats_schema, sep="; ")
            protein_df = info1[list(protein_schema.keys())]
            protein_df["text"] = row_to_text(protein_df, protein_schema, sep="; ")
            micronutrients_df = info1[list(micronutrients_schema.keys())]
            micronutrients_df["text"] = row_to_text(micronutrients_df, micronutrients_schema, sep="; ")
            food_df = info1[list(food_schema.keys())]
            food_df["text"] = row_to_text(food_df, food_schema, sep="; ")
            info = [[hei_mind_df["text"]] + [energy_df["text"]] + [sugars_df["text"]] + [fat_df["text"]] + [protein_df["text"]] + 
                    [micronutrients_df["text"]] + [food_df["text"]]][0]
    except:
        info1 = diet_detail.iloc[i]
        hei_mind_df = info1[list(schema.keys())]
        hei_mind_df["text"] = row_to_text(hei_mind_df, schema, sep="; ")
        energy_df = info1[list(energy_macronutrients_alcohol_schema.keys())]
        energy_df["text"] = row_to_text(energy_df, energy_macronutrients_alcohol_schema, sep="; ")
        sugars_df = info1[list(sugars_schema.keys())]
        sugars_df["text"] = row_to_text(sugars_df, sugars_schema, sep="; ")
        fat_df = info1[list(fats_schema.keys())]
        fat_df["text"] = row_to_text(fat_df, fats_schema, sep="; ")
        protein_df = info1[list(protein_schema.keys())]
        protein_df["text"] = row_to_text(protein_df, protein_schema, sep="; ")
        micronutrients_df = info1[list(micronutrients_schema.keys())]
        micronutrients_df["text"] = row_to_text(micronutrients_df, micronutrients_schema, sep="; ")
        food_df = info1[list(food_schema.keys())]
        food_df["text"] = row_to_text(food_df, food_schema, sep="; ")
        info2 = parse_week_diet(record[0])
        info = [[hei_mind_df["text"]] + [energy_df["text"]] + [sugars_df["text"]] + [fat_df["text"]] + [protein_df["text"]] + 
                [micronutrients_df["text"]] + [food_df["text"]] + info2][0]

    diet_info['text_info'] = info
    diet_info['sub'] = diet_sub[i]
    diet_info['Demog_Income'] = pcd_y['Demog_Income'].iloc[i]
    diet_info['MASQ_Language'] = pcd_y['MASQ_Language'].iloc[i]
    diet_info['MASQ_Verbal_Memory'] = pcd_y['MASQ_Verbal_Memory'].iloc[i]
    diet_info['MASQ_Visual_Perception'] = pcd_y['MASQ_Visual_Perception'].iloc[i]
    diet_info['MASQ_Visual_Spatial'] = pcd_y['MASQ_Visual_Spatial'].iloc[i]
    diet_info['MHCSF_Eudaimonic_Social'] = pcd_y['MHCSF_Eudaimonic_Social'].iloc[i]
    diet_info['MHCSF_Eudaimonic_Psych'] = pcd_y['MHCSF_Eudaimonic_Psych'].iloc[i]
    diet_info['MHCSF_Hedonic'] = pcd_y['MHCSF_Hedonic'].iloc[i]
    diet_info['YFAS_SymptomCount'] = pcd_y['YFAS_SymptomCount'].iloc[i]
    diet_info['GFCQT_Total'] = pcd_y['GFCQT_Total'].iloc[i]
    diet_info['YFAS_SymptomCount'] = pcd_y['YFAS_SymptomCount'].iloc[i]
    diet_info['STAI_SAnxiety_raw'] = pcd_y['STAI_SAnxiety_raw'].iloc[i]
    diet_info['STAI_SAnxiety'] = pcd_y['STAI_SAnxiety'].iloc[i]
    diet_info['ACE_Incarcerated_Household_Member'] = pcd_y['ACE_Incarcerated_Household_Member'].iloc[i]
    diet_info['MINI_Major_Depressive_p'] = pcd_y['MINI_Major_Depressive_p'].iloc[i]
    diet_info['MINI_Suicidality_c'] = pcd_y['MINI_Suicidality_c'].iloc[i]
    diet_info['MINI_Agoraphobia_l'] = pcd_y['MINI_Agoraphobia_l'].iloc[i]
    diet_info['MINI_Agoraphobia_c'] = pcd_y['MINI_Agoraphobia_c'].iloc[i]
    diet_info['MINI_Alcohol_Dependence_l'] = pcd_y['MINI_Alcohol_Dependence_l'].iloc[i]
    diet_info['Migraine'] = pcd_y['Migraine'].iloc[i]
    diet_all.append(diet_info)

