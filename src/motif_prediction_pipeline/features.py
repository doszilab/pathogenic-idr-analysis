# src/motif_prediction_pipeline/features.py
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from ..config import PROCESSED_DATA_DIR
from .regex_utils import get_key_residue_indices


def load_alphamissense_data():
    """
    Loads AlphaMissense data from processed split files (order/disorder).
    """
    path_order = os.path.join(PROCESSED_DATA_DIR, "am_order.tsv")
    path_disorder = os.path.join(PROCESSED_DATA_DIR, "am_disorder.tsv")

    dfs = []

    if os.path.exists(path_order):
        print(f"Loading AM Order data from {path_order}...")
        df_ord = pd.read_csv(path_order, sep='\t', usecols=['Protein_ID', 'Position', 'AlphaMissense'])
        dfs.append(df_ord)

    if os.path.exists(path_disorder):
        print(f"Loading AM Disorder data from {path_disorder}...")
        df_dis = pd.read_csv(path_disorder, sep='\t', usecols=['Protein_ID', 'Position', 'AlphaMissense'])
        dfs.append(df_dis)

    if not dfs:
        print("Warning: No AM data found. Returning empty DataFrame.")
        return pd.DataFrame(columns=['Protein_ID', 'Position', 'AlphaMissense'])

    print("Combining AM datasets...")
    return pd.concat(dfs, ignore_index=True).rename(columns={'AlphaMissense': 'am_pathogenicity'})


def add_alphamissense_features(motifs_df, am_df=None):
    """
    Adds AM features to motifs using precise Key Residue detection.
    Matches original logic: Mean, Max, Key Mean, Flank Mean, and Differences.
    """
    if am_df is None:
        am_df = load_alphamissense_data()

    print("Indexing AlphaMissense data...")
    am_lookup = {}
    if not am_df.empty:
        grouped = am_df.groupby('Protein_ID')
        for pid, group in tqdm(grouped, desc="Indexing"):
            am_lookup[pid] = dict(zip(group['Position'], group['am_pathogenicity']))

    features = []
    print("Calculating motif features...")

    for _, row in tqdm(motifs_df.iterrows(), total=len(motifs_df), desc="Calculating"):
        pid = row['Protein_ID']
        start = int(row['Start'])
        end = int(row['End'])
        matched_seq = row.get('Matched_Sequence', '')
        # Prefer 'Regex' column, fallback to 'ELMIdentifier' if it holds the pattern
        regex_pattern = row.get('Regex', row.get('ELMIdentifier', ''))

        scores_dict = am_lookup.get(pid, {})

        # 1. Identify Key Residues using regex utils
        key_positions = []
        if matched_seq and regex_pattern:
            try:
                key_positions = get_key_residue_indices(regex_pattern, matched_seq, start)
            except Exception:
                pass  # Fallback

        if not key_positions:
            # Fallback: treat all as key if parsing fails
            key_positions = list(range(start, end + 1))

        # 2. Key Residue Scores
        key_vals = [scores_dict.get(p, np.nan) for p in key_positions]
        valid_key = [v for v in key_vals if not np.isnan(v)]
        key_mean = np.mean(valid_key) if valid_key else np.nan

        # 3. Motif Scores (Whole match)
        motif_pos = range(start, end + 1)
        motif_vals = [scores_dict.get(p, np.nan) for p in motif_pos]
        valid_motif = [v for v in motif_vals if not np.isnan(v)]

        motif_mean = np.mean(valid_motif) if valid_motif else np.nan
        motif_max = np.max(valid_motif) if valid_motif else np.nan  # THIS IS AM_Max

        # 4. Flanking Scores (Outside Motif, +/- 5)
        flank_pos = list(range(max(1, start - 5), start)) + list(range(end + 1, end + 6))
        flank_vals = [scores_dict.get(p, np.nan) for p in flank_pos]
        valid_flank = [v for v in flank_vals if not np.isnan(v)]
        flank_mean = np.mean(valid_flank) if valid_flank else np.nan

        # 5. Sequential (Whole protein)
        all_vals = list(scores_dict.values())
        seq_mean = np.mean(all_vals) if all_vals else np.nan

        # 6. Differences (Calculated exactly as in filter_df)
        key_vs_flank = key_mean - flank_mean if (not np.isnan(key_mean) and not np.isnan(flank_mean)) else np.nan
        motif_vs_seq = motif_mean - seq_mean if (not np.isnan(motif_mean) and not np.isnan(seq_mean)) else np.nan

        features.append({
            'motif_am_mean_score': motif_mean,
            'AM_Max': motif_max,
            'key_residue_am_mean_score': key_mean,
            'flanking_residue_am_mean_score': flank_mean,
            'sequential_am_score': seq_mean,
            'Key_vs_NonKey_Difference': key_vs_flank,
            'Motif_vs_Sequential_Difference': motif_vs_seq
        })

    feat_df = pd.DataFrame(features)
    return pd.concat([motifs_df.reset_index(drop=True), feat_df], axis=1)