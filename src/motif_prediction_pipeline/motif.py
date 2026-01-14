# src/elm/motif.py
import pandas as pd
import re
import os
from tqdm import tqdm
from ..config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def load_proteome_data():
    """Loads proteome sequence data from processed isoforms."""
    path = os.path.join(PROCESSED_DATA_DIR, "main_isoforms.tsv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Proteome sequence file not found at {path}")

    print(f"Loading proteome from {path}...")
    return pd.read_csv(path, sep='\t')


def scan_proteome_for_motifs(regex_pattern, proteome_df=None):
    """
    Scans the proteome for a regex pattern.
    If proteome_df is not provided, it loads it automatically.
    """
    if proteome_df is None:
        proteome_df = load_proteome_data()

    print(f"Scanning {len(proteome_df)} sequences for pattern: {regex_pattern}")
    results = []
    regex = re.compile(regex_pattern)

    for _, row in tqdm(proteome_df.iterrows(), total=len(proteome_df), desc="Scanning"):
        seq = str(row['Sequence'])
        pid = row['Protein_ID']

        for match in regex.finditer(seq):
            results.append({
                'Protein_ID': pid,
                'Start': match.start() + 1,
                'End': match.end(),
                'Matched_Sequence': match.group(),
                'ELMIdentifier': 'Custom_Regex',
                'Regex': regex_pattern
            })

    return pd.DataFrame(results)