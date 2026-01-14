#!/usr/bin/env python3

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib_venn import venn3


def read_data(
    clinvar_path: str,
    pip_path: str,
    pem_path: str,
    known_elm_path: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reads ClinVar, PIP, PEM, and known ELM data from CSV/TSV files.
    Filters 'pem_df' to those with 'Sequential_Rule' == True.

    Returns
    -------
    (clinvar_df, pip_df, pem_df, known_df)
    """
    clinvar_df = pd.read_csv(clinvar_path, sep='\t')
    pip_df = pd.read_csv(pip_path, sep='\t')
    known_df = pd.read_csv(known_elm_path, sep='\t')

    pem_df = pd.read_csv(pem_path, sep='\t')


    return clinvar_df, pip_df, pem_df, known_df


def expand_regions_to_positions(df: pd.DataFrame,
                                pid_col: str = 'Protein_ID',
                                start_col: str = 'Start',
                                end_col: str = 'End') -> pd.DataFrame:
    """
    Expands each region (start, end) in a DataFrame into all individual positions.

    Returns
    -------
    A DataFrame with columns [pid_col, 'Position'] for every integer position in [Start, End].
    """
    df_regions = df[[pid_col, start_col, end_col]].drop_duplicates()
    positions_list = []
    for _, row in df_regions.iterrows():
        pid = row[pid_col]
        start = row[start_col]
        end = row[end_col]
        for pos in range(start, end + 1):
            positions_list.append((pid, pos))

    return pd.DataFrame(positions_list, columns=[pid_col, 'Position'])


def interval_overlap_length(start1: int, end1: int, start2: int, end2: int) -> int:
    """
    Returns the length of the overlap between two intervals [start1, end1] and [start2, end2].
    """
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    if overlap_end < overlap_start:
        return 0
    return overlap_end - overlap_start + 1


def compute_positional_overlaps(pip_positions: pd.DataFrame,
                                pem_positions: pd.DataFrame,
                                elm_positions: pd.DataFrame,
                                clinvar_positions: pd.DataFrame,
                    pem_sequential_positions: pd.DataFrame,
                                ) -> dict:
    """
    Computes the positional overlaps among PIP, PEM, ELM, and ClinVar.
    Returns a dictionary with counts of set sizes and pairwise intersections.
    """
    # Convert each to set of (Protein_ID, Position)
    pip_set = set(pip_positions.itertuples(index=False, name=None))
    pem_set = set(pem_positions.itertuples(index=False, name=None))
    pem_sequential_set = set(pem_sequential_positions.itertuples(index=False, name=None))
    elm_set = set(elm_positions.itertuples(index=False, name=None))
    clinvar_set = set(clinvar_positions.itertuples(index=False, name=None))

    # Pairwise intersections
    pip_pem = pip_set.intersection(pem_set)
    pip_pem_sequential = pip_set.intersection(pem_sequential_set)
    pip_elm = pip_set.intersection(elm_set)
    pip_clinvar = pip_set.intersection(clinvar_set)
    pem_sequential_clinvar = pem_sequential_set.intersection(clinvar_set)
    pem_clinvar = pem_set.intersection(clinvar_set)
    elm_clinvar = elm_set.intersection(clinvar_set)

    return {
        "PIP_size": len(pip_set),
        "PEM_size": len(pem_set),
        "PEM_Sequential_size": len(pem_sequential_set),
        "ELM_size": len(elm_set),
        "ClinVar_size": len(clinvar_set),

        "PIP_PEM": len(pip_pem),
        "PIP_PEM_Sequential": len(pip_pem_sequential),
        "PIP_ELM": len(pip_elm),
        "PIP_ClinVar": len(pip_clinvar),
        "PEM_ClinVar": len(pem_clinvar),
        "PEM_Sequantial_ClinVar": len(pem_sequential_clinvar),
        "ELM_ClinVar": len(elm_clinvar),
    }


def compute_regional_overlap_percentage(pem_df: pd.DataFrame,
                                        pip_df: pd.DataFrame,
                                        pid_col: str = 'Protein_ID',
                                        start_col: str = 'Start',
                                        end_col: str = 'End') -> tuple[list[float], int]:
    """
    For each PEM region, compute fraction overlapping with any PIP region.
    Also count how many PEM regions have zero overlap with PIP.

    Returns
    -------
    (overlap_fractions, no_overlap_count):
      overlap_fractions : list of floats (overlap fraction for each PEM region)
      no_overlap_count  : number of PEM regions that have zero overlap
    """
    # Prepare PIP intervals by pid for faster scanning
    pip_regions_dict = {}
    pip_unique = pip_df[[pid_col, start_col, end_col]].drop_duplicates()
    for _, row in pip_unique.iterrows():
        pid = row[pid_col]
        st = row[start_col]
        en = row[end_col]
        pip_regions_dict.setdefault(pid, []).append((st, en))

    pem_unique = pem_df[[pid_col, start_col, end_col]].drop_duplicates()

    overlap_fractions = []
    no_overlap_count = 0

    for _, row in pem_unique.iterrows():
        pid = row[pid_col]
        pem_start = row[start_col]
        pem_end = row[end_col]
        pem_length = pem_end - pem_start + 1

        if pid not in pip_regions_dict:
            # No PIP region on this protein => zero overlap
            overlap_fractions.append(0.0)
            no_overlap_count += 1
            continue

        total_overlap = 0
        for (pip_start, pip_end) in pip_regions_dict[pid]:
            total_overlap += interval_overlap_length(pem_start, pem_end, pip_start, pip_end)

        # Overlap can't exceed the entire length of the PEM region
        total_overlap = min(total_overlap, pem_length)

        frac = total_overlap / pem_length
        overlap_fractions.append(frac)

        if frac == 0:
            no_overlap_count += 1

    return overlap_fractions, no_overlap_count


def compute_region_overlaps(df1: pd.DataFrame,
                            df2: pd.DataFrame,
                            pid_col: str = 'Protein_ID',
                            start_col: str = 'Start',
                            end_col: str = 'End') -> tuple[int, int]:
    """
    Computes how many regions in df1 have at least partial overlap with *any* region in df2,
    on the same Protein_ID.

    Returns
    -------
    (overlapping_count, total_count):
      overlapping_count = number of df1 regions that overlap at least one df2 region
      total_count       = total number of unique regions in df1
    """
    df1_unique = df1[[pid_col, start_col, end_col]].drop_duplicates()
    df2_unique = df2[[pid_col, start_col, end_col]].drop_duplicates()

    # Build dictionary of df2 intervals grouped by pid
    df2_dict = {}
    for _, row in df2_unique.iterrows():
        pid = row[pid_col]
        st2 = row[start_col]
        en2 = row[end_col]
        df2_dict.setdefault(pid, []).append((st2, en2))

    overlapping_count = 0
    total_count = len(df1_unique)

    for _, row in df1_unique.iterrows():
        pid = row[pid_col]
        st1 = row[start_col]
        en1 = row[end_col]

        # Check overlap with any region in df2_dict[pid]
        has_overlap = False
        if pid in df2_dict:
            for (st2, en2) in df2_dict[pid]:
                if interval_overlap_length(st1, en1, st2, en2) > 0:
                    has_overlap = True
                    break
        if has_overlap:
            overlapping_count += 1

    return overlapping_count, total_count


def plot_positional_venn(pip_positions: pd.DataFrame,
                         pem_positions: pd.DataFrame,
                         clinvar_positions: pd.DataFrame) -> None:
    """
    Plots a Venn diagram of the positional overlap among PIP, PEM, and ClinVar.
    """
    pip_set = set(pip_positions.itertuples(index=False, name=None))
    pem_set = set(pem_positions.itertuples(index=False, name=None))
    clinvar_set = set(clinvar_positions.itertuples(index=False, name=None))

    plt.figure(figsize=(6, 6))
    venn3([pip_set, pem_set, clinvar_set], set_labels=('PIP', 'PEM', 'ClinVar'))
    plt.title("Positional Overlap: PIP vs PEM vs ClinVar")
    plt.show()


def plot_overlap_histogram(overlap_fractions: list[float]) -> None:
    """
    Plots a histogram of overlap fractions (PEM vs PIP).
    """
    plt.figure(figsize=(6, 4))
    plt.hist(overlap_fractions, bins=np.linspace(0, 1, 21), edgecolor='black')
    plt.title("Distribution of Overlap Fractions (PEM vs PIP)")
    plt.xlabel("Fraction of PEM region overlapping with PIP")
    plt.ylabel("Count")
    plt.show()


def main():
    """
    Main execution function demonstrating how to structure and call the above functions.
    Update file paths as needed.
    """
    # -------------------------------------------------------------------
    # 1) Read data
    # -------------------------------------------------------------------
    clinvar_df, pip_df, pem_df, known_df = read_data(
        clinvar_path="/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/clinvar/Pathogenic/disorder/positional_clinvar_functional_categorized_final.tsv",
        pip_path="/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/elm/predicted_motif_region_by_am_sequential_rule.tsv",
        known_elm_path="/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/elm/elm_known_with_am_info_all_disorder.tsv",
        pem_path="/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/elm/elm_predicted_with_am_info_all_disorder_with_rules.tsv",
    )
    pem_sequential_df = pem_df[pem_df['Sequential_Rule'] == True]

    # -------------------------------------------------------------------
    # 2) Create position-level data for PIP, PEM, ELM, ClinVar
    # -------------------------------------------------------------------
    clinvar_positions = clinvar_df[['Protein_ID', 'Position']].drop_duplicates()

    pip_positions = expand_regions_to_positions(pip_df, 'Protein_ID', 'Start', 'End')
    elm_positions = expand_regions_to_positions(known_df, 'Protein_ID', 'Start', 'End')
    pem_sequential_positions = expand_regions_to_positions(pem_sequential_df, 'Protein_ID', 'Start', 'End')
    pem_positions = expand_regions_to_positions(pem_df, 'Protein_ID', 'Start', 'End')

    # -------------------------------------------------------------------
    # 3) Positional Overlaps
    # -------------------------------------------------------------------
    overlap_counts = compute_positional_overlaps(
        pip_positions, pem_positions, elm_positions, clinvar_positions,pem_sequential_positions
    )
    print("Positional Overlaps (number of positions):")
    for key, val in overlap_counts.items():
        print(f"  {key}: {val}")

    # -------------------------------------------------------------------
    # 4) Regional Overlap: PEM vs PIP (Fraction Overlap)
    #     - The fraction of each PEM region that is covered by PIP
    #     - And how many PEM have zero overlap
    # -------------------------------------------------------------------
    overlap_fractions, no_overlap_count = compute_regional_overlap_percentage(
        pem_df=pem_df,
        pip_df=pip_df,
        pid_col='Protein_ID',
        start_col='Start',
        end_col='End'
    )

    avg_overlap_fraction = (
        sum(overlap_fractions) / len(overlap_fractions) if overlap_fractions else 0
    )

    print("\nRegional Overlap: PEM vs PIP (fraction of PEM region covered)")
    print("Total PEM regions:", len(pem_df[['Protein_ID', 'Start', 'End']].drop_duplicates()))
    print("PEM regions with NO overlap to PIP:", no_overlap_count)
    print(f"Average overlap fraction = {avg_overlap_fraction:.3f}")

    # -------------------------------------------------------------------
    # 5) Region-Level Overlaps (how many entire regions overlap at least once)
    #    For completeness, we compute each pair in "one direction."
    #    E.g. "Among all PIP regions, how many overlap with PEM?"
    # -------------------------------------------------------------------
    # First, prepare deduplicated region sets:
    pip_regions = pip_df[['Protein_ID','Start','End']].drop_duplicates()
    pem_regions = pem_df[['Protein_ID','Start','End']].drop_duplicates()
    pem_sequentia_regions = pem_sequential_df[['Protein_ID','Start','End']].drop_duplicates()
    elm_regions = known_df[['Protein_ID','Start','End']].drop_duplicates()
    # ClinVar is position-based => treat each position as region [pos, pos]
    clinvar_regions = (clinvar_df[['Protein_ID','Position']]
                       .drop_duplicates()
                       .rename(columns={'Position':'Start'}))
    clinvar_regions['End'] = clinvar_regions['Start']

    # Example calls:
    pip_pem_ol_count, pip_total = compute_region_overlaps(pip_regions, pem_regions)
    pip_pem_sequential_ol_count, pem_sequential_total = compute_region_overlaps(pip_regions, pem_sequentia_regions)
    pip_elm_ol_count, _ = compute_region_overlaps(pip_regions, elm_regions)
    pip_clinvar_ol_count, _ = compute_region_overlaps(pip_regions, clinvar_regions)
    pem_pip_ol_count, pem_total = compute_region_overlaps(pem_regions, pip_regions)
    pem_elm_count, _ = compute_region_overlaps(pem_regions,elm_regions)
    elm_clinvar_ol_count, elm_total = compute_region_overlaps(elm_regions, clinvar_regions)
    pem_clinvar_ol_count, _ = compute_region_overlaps(pem_regions, clinvar_regions)
    print(f"\n[PIP -> PEM] Region-level Overlap: {pip_pem_ol_count} / {pip_total} PIP regions overlap with PEM ({pem_total})")
    print(f"[PIP -> PEM Sequential] Region-level Overlap: {pip_pem_sequential_ol_count} / {pip_total} PIP regions overlap with PEM Sequantial ({pem_sequential_total})")
    print(f"[PIP -> ELM] Region-level Overlap: {pip_elm_ol_count} / {pip_total} PIP regions overlap with ELM ({elm_total})")
    print(f"[PIP -> ClinVar] Region-level Overlap: {pip_clinvar_ol_count} / {pip_total} PIP regions overlap with ClinVar ({clinvar_positions.shape[0]})")
    print(f"[PEM -> PIP] Region-level Overlap: {pem_pip_ol_count} / {pem_total} PEM regions overlap with PIP ({pip_total})")
    print(f"[PEM -> ELM] Region-level Overlap: {pem_elm_count} / {pem_total} PEM regions overlap with PIP ({elm_total})")
    print(f"[PEM -> ClinVar] Region-level Overlap: {pem_clinvar_ol_count} / {pem_total} PEM regions overlap with PIP ({elm_total})")
    print(f"[ELM -> ClinVar] Region-level Overlap: {elm_clinvar_ol_count} / {elm_total} ELM regions overlap with ClinVar ({clinvar_positions.shape[0]})")

    # -------------------------------------------------------------------
    # (Optional) Plots
    # -------------------------------------------------------------------
    # plot_positional_venn(pip_positions, pem_positions, clinvar_positions)
    # plot_overlap_histogram(overlap_fractions)


if __name__ == "__main__":
    main()
