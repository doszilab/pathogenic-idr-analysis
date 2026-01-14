import pandas as pd
import os
from collections import defaultdict
from tqdm import tqdm
import re

def interval_overlap_line_sweep(
        pem_df: pd.DataFrame,
        pub_df: pd.DataFrame,
        partial_overlap_threshold: float = 0.8
) -> pd.DataFrame:
    """
    Efficiently find pairs of PEM intervals and publication intervals that overlap,
    returning all columns from both DataFrames (prefixed) and additional columns:
      - Overlap_Start
      - Overlap_End
      - PEM_CoverageFraction
      - Publication_CoverageFraction
      - PEM_OverlapID
      - Publication_OverlapID

    Only retains pairs where:
      - The fraction of PEM covered by the publication >= partial_overlap_threshold, OR
      - The fraction of publication covered by the PEM >= partial_overlap_threshold.

    Returns:
        pd.DataFrame: One row per overlapping pair, containing:
          PEM_XXX                # all original columns from pem_df, prefixed
          Publication_XXX        # all original columns from pub_df, prefixed
          Overlap_Start
          Overlap_End
          PEM_CoverageFraction
          Publication_CoverageFraction
          PEM_OverlapID
          Publication_OverlapID
    """

    # Create unique identifiers for each PEM and Publication motif
    pem_df = pem_df.copy()
    pub_df = pub_df.copy()
    pem_df['PEM_OverlapID'] = pem_df.index
    pub_df['Publication_OverlapID'] = pub_df.index

    pem_groups = {
        prot: grp.sort_values("Start").to_dict("records")
        for prot, grp in pem_df.groupby("Protein_ID")
    }
    pub_groups = {
        prot: grp.sort_values("Start").to_dict("records")
        for prot, grp in pub_df.groupby("Protein_ID")
    }

    all_proteins = set(pem_groups.keys()) | set(pub_groups.keys())

    overlap_records = []

    for prot in tqdm(all_proteins, desc="Finding Overlaps"):
        pem_intervals = pem_groups.get(prot, [])
        pub_intervals = pub_groups.get(prot, [])

        i, j = 0, 0
        while i < len(pem_intervals) and j < len(pub_intervals):
            pem_rec = pem_intervals[i]
            pub_rec = pub_intervals[j]

            pem_start = pem_rec["Start"]
            pem_end = pem_rec["End"]
            pub_start = pub_rec["Start"]
            pub_end = pub_rec["End"]

            if pem_end < pub_start:
                i += 1
                continue
            elif pub_end < pem_start:
                j += 1
                continue
            else:
                overlap_start = max(pem_start, pub_start)
                overlap_end = min(pem_end, pub_end)
                overlap_len = overlap_end - overlap_start + 1

                pem_len = pem_end - pem_start + 1
                pub_len = pub_end - pub_start + 1

                pem_cov_fraction = overlap_len / pem_len if pem_len else 0
                pub_cov_fraction = overlap_len / pub_len if pub_len else 0

                if (
                        pem_cov_fraction >= partial_overlap_threshold
                        or pub_cov_fraction >= partial_overlap_threshold
                ):
                    pem_record_prefixed = {
                        f"PEM_{k}": v for k, v in pem_rec.items()
                    }
                    pub_record_prefixed = {
                        f"Publication_{k}": v for k, v in pub_rec.items()
                    }

                    overlap_record = {
                        **pem_record_prefixed,
                        **pub_record_prefixed,
                        "Overlap_Start": overlap_start,
                        "Overlap_End": overlap_end,
                        "PEM_CoverageFraction": round(pem_cov_fraction, 3),
                        "Publication_CoverageFraction": round(pub_cov_fraction, 3),
                        "PEM_OverlapID": pem_rec['PEM_OverlapID'],
                        "Publication_OverlapID": pub_rec['Publication_OverlapID'],
                    }
                    overlap_records.append(overlap_record)

                if pem_end < pub_end:
                    i += 1
                else:
                    j += 1

    overlap_df = pd.DataFrame(overlap_records)
    return overlap_df


def compare_peptides_and_motifs(
        germline_df: pd.DataFrame,
        pem_df: pd.DataFrame,
        partial_overlap_threshold: float = 0.8,
) -> (pd.DataFrame, pd.DataFrame):
    """
    Compare two sets of intervals (publication/germline vs. PEM) for:
      1) Overall fraction of positions overlapping.
      2) Complete coverage (100% overlap).
      3) Partial coverage (>= partial_overlap_threshold of the interval length).
      4) Return two dataframes with the explicit overlap pairs:
         - pem_with_overlap_df: each PEM interval alongside each publication interval it overlaps
         - publication_with_overlap_df: each publication interval alongside each PEM interval it overlaps

    Args:
        germline_df (pd.DataFrame): DataFrame with columns ["Protein_ID", "Start", "End"]
                                    (sometimes called "publication" or "peptide")
        pem_df (pd.DataFrame):     DataFrame with columns ["Protein_ID", "Start", "End"]
        partial_overlap_threshold (float): Fraction of interval coverage required
                                           to count as "covered". Default 0.8 (80%).

    Returns:
        (pd.DataFrame, pd.DataFrame) tuple of two DataFrames:
          - pem_with_overlap_df
          - publication_with_overlap_df

    Prints:
        - Fraction of PEM positions overlapping any publication position.
        - Fraction of publication positions overlapping any PEM position.
        - Fraction of PEM intervals completely covered by publication intervals.
        - Fraction of publication intervals completely covered by PEM intervals.
        - Fraction of PEM intervals partially (>= threshold) covered by publication intervals.
        - Fraction of publication intervals partially (>= threshold) covered by PEM intervals.
    """

    # 1) Build coverage sets for each protein, for both germline (publication) and PEM.
    publication_positions = defaultdict(set)
    for _, row in germline_df.iterrows():
        prot = row["Protein_ID"]
        start = int(row["Start"])
        end = int(row["End"])
        for pos in range(start, end + 1):
            publication_positions[prot].add(pos)

    pem_positions = defaultdict(set)
    for _, row in pem_df.iterrows():
        prot = row["Protein_ID"]
        start = int(row["Start"])
        end = int(row["End"])
        for pos in range(start, end + 1):
            pem_positions[prot].add(pos)

    # 2) Summaries for overlap of individual positions (length-based coverage).
    total_pub_len = 0
    total_pem_len = 0
    total_overlap_len = 0

    all_proteins = set(publication_positions.keys()) | set(pem_positions.keys())
    for prot in all_proteins:
        pub_set = publication_positions[prot]
        pem_set = pem_positions[prot]
        total_pub_len += len(pub_set)
        total_pem_len += len(pem_set)
        overlap = pub_set & pem_set
        total_overlap_len += len(overlap)

    if total_pem_len > 0:
        pct_pem_covered = 100.0 * total_overlap_len / total_pem_len
    else:
        pct_pem_covered = 0.0

    if total_pub_len > 0:
        pct_pub_covered = 100.0 * total_overlap_len / total_pub_len
    else:
        pct_pub_covered = 0.0

    print(f"Fraction of PEM positions overlapping any publication position: {pct_pem_covered:.2f}%")
    print(f"Fraction of publication positions overlapping any PEM position: {pct_pub_covered:.2f}%")

    # 3) Check complete coverage (PEM intervals completely within publication)
    num_completely_covered = 0
    total_pems = len(pem_df)
    for _, row in pem_df.iterrows():
        prot = row["Protein_ID"]
        start = int(row["Start"])
        end = int(row["End"])
        motif_range = set(range(start, end + 1))

        # Check if motif_range is fully within the publication coverage set
        if prot in publication_positions:
            if motif_range.issubset(publication_positions[prot]):
                num_completely_covered += 1

    pct_completely_covered = 100.0 * num_completely_covered / total_pems if total_pems else 0.0

    print(f"PEM intervals: {total_pems}")
    print(f"PEM intervals completely covered by publication: {num_completely_covered}")
    print(f"Fraction of PEM intervals with complete overlap: {pct_completely_covered:.2f}%")

    # 4) Check complete coverage (Publication intervals completely within PEM)
    num_publication_completely_covered = 0
    total_publications = len(germline_df)
    for _, row in germline_df.iterrows():
        prot = row["Protein_ID"]
        start = int(row["Start"])
        end = int(row["End"])
        pub_range = set(range(start, end + 1))

        # Check if pub_range is fully within the PEM coverage set
        if prot in pem_positions:
            if pub_range.issubset(pem_positions[prot]):
                num_publication_completely_covered += 1

    pct_publication_completely_covered = (
        100.0 * num_publication_completely_covered / total_publications
        if total_publications
        else 0.0
    )

    print(f"Publication intervals: {total_publications}")
    print(f"Publication intervals completely covered by PEM: {num_publication_completely_covered}")
    print(f"Fraction of publication intervals with complete overlap: {pct_publication_completely_covered:.2f}%")

    # 5) Partial overlap coverage check (>= partial_overlap_threshold)
    def fraction_covered(prot_id: str, start: int, end: int, coverage_dict: defaultdict) -> float:
        """Returns fraction of [start, end] positions covered by coverage_dict[prot_id]."""
        interval_positions = set(range(start, end + 1))
        covered_positions = interval_positions & coverage_dict[prot_id]
        return len(covered_positions) / len(interval_positions) if interval_positions else 0.0

    # Check partial coverage for PEM intervals by publication coverage
    num_pem_partially_covered = 0
    for _, row in pem_df.iterrows():
        prot = row["Protein_ID"]
        start = int(row["Start"])
        end = int(row["End"])
        frac_cov = fraction_covered(prot, start, end, publication_positions)
        if frac_cov >= partial_overlap_threshold:
            num_pem_partially_covered += 1

    pct_pem_partially_covered = (
        100.0 * num_pem_partially_covered / total_pems if total_pems else 0.0
    )
    print(
        f"PEM intervals partially (>{partial_overlap_threshold * 100:.0f}%) covered by publication: "
        f"{num_pem_partially_covered}/{total_pems} = {pct_pem_partially_covered:.2f}%"
    )

    # Check partial coverage for publication intervals by PEM coverage
    num_pub_partially_covered = 0
    for _, row in germline_df.iterrows():
        prot = row["Protein_ID"]
        start = int(row["Start"])
        end = int(row["End"])
        frac_cov = fraction_covered(prot, start, end, pem_positions)
        if frac_cov >= partial_overlap_threshold:
            num_pub_partially_covered += 1

    pct_pub_partially_covered = (
        100.0 * num_pub_partially_covered / total_publications if total_publications else 0.0
    )
    print(
        f"Publication intervals partially (>{partial_overlap_threshold * 100:.0f}%) covered by PEM: "
        f"{num_pub_partially_covered}/{total_publications} = {pct_pub_partially_covered:.2f}%"
    )

    # -------------------------------------------------------------------------
    # 6) Build two DataFrames showing explicit overlap between PEM and publication intervals.
    #
    # We'll consider ANY overlap. If you wish to filter on fraction coverage
    # (like >= 80%), you could add that condition in the "if overlap" check.
    # -------------------------------------------------------------------------

    overlap_df = interval_overlap_line_sweep(pem_df, germline_df, partial_overlap_threshold=0.8)

    # Return them so user can directly inspect or merge as needed
    return overlap_df


def make_broad_joining(main_df,motif_df,uniprot_acc_col='Entry_Isoform',region_col=None,start_col=None,end_col=None,motif_seq=None,sep=None,regular_expression_col=None):

    # Do your inner merge here if needed
    motif_df = motif_df.merge(main_df, on=uniprot_acc_col, how="inner")

    if region_col is not None:
        motif_df['Start'] = motif_df[region_col].str.split(sep).str[0].astype(int)
        motif_df['End'] = motif_df[region_col].str.split(sep).str[1].astype(int)
    elif start_col is not None and end_col is not None:
        motif_df['Start'] = motif_df[start_col].astype(int) -1
        motif_df['End'] = motif_df[end_col].astype(int)

    motif_df['motif_seq'] = motif_df.apply(lambda x:x['Sequence'][x['Start']:x['End']+1], axis=1)

    # Apply regex match
    def get_regular_match(row):
        motif_seq = row['motif_seq']
        regex = re.compile(row[regular_expression_col].strip())

        try:
            return bool(re.search(regex, motif_seq))
        except Exception as e:
            print(f"Regex error in row {row.name}: {regex} vs {motif_seq} — {e}")
            return False

    if regular_expression_col is not None:
        motif_df['match'] = motif_df.apply(get_regular_match, axis=1)

        # Optionally: Keep only matched motifs
        matched_df = motif_df[motif_df['match'] == True].copy()

        return matched_df



    pass


if __name__ == "__main__":
    base_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations'
    sequences_dir = os.path.join(base_dir, "data/discanvis_base_files/sequences")
    files = os.path.join(base_dir, "processed_data/files/elm")
    main_df = pd.read_csv(os.path.join(sequences_dir, "loc_chrom_with_names_main_isoforms_with_seq.tsv"), sep='\t')

    # PEMs
    # pem_df = pd.read_csv(f"{files}/class_based_approach/elm_pem_disorder_corrected_with_class_count.tsv")
    most_significant_resutls = pd.read_csv(f"{files}/decision_tree/publication_predicted_norman_davey_with_am_info_all_disorder_class_filtered.tsv", sep='\t')
    all_predicted_instances = pd.read_csv(f"{files}/decision_tree/publication_predicted_effected_norman_davey_with_am_info_all_disorder_class_filtered.tsv", sep='\t')

    total_most_significant_resutls = most_significant_resutls[most_significant_resutls['prediction_decision_tree'] == True].shape[0]
    total_all_predicted_instances = all_predicted_instances[all_predicted_instances['prediction_decision_tree'] == True].shape[0]

    elm_predicted = all_predicted_instances[all_predicted_instances["ELM"].notna()]
    total_elm_predicted = elm_predicted[elm_predicted['prediction_decision_tree'] == True].shape[0]

    print(f"Predicted Motif - Publication Norman Davey Most Significant {total_most_significant_resutls}, {round(total_most_significant_resutls / most_significant_resutls.shape[0] * 100, 2)}% (total={most_significant_resutls.shape[0]})")
    print(f"Predicted Motif - Publication Norman Davey All Predicted {total_all_predicted_instances}, {round(total_all_predicted_instances / all_predicted_instances.shape[0] * 100, 2)}% (total={all_predicted_instances.shape[0]})")
    print(f"Predicted Motif - Publication Norman Davey ELM Significant Predicted overlaps {total_elm_predicted}, {round(total_elm_predicted / elm_predicted.shape[0] * 100, 2)}% (total={elm_predicted.shape[0]})")

    # exit()
    #
    # print(pem_df)
    #
    # publication_dir = os.path.join(base_dir, "data/publication")
    #
    # elm_benchmarking_set = pd.read_csv(os.path.join(publication_dir, "nar-01623-n-2012-File009.csv"),skiprows=11, header=0).rename(columns={"Uniprot Accession":'Entry_Isoform'})
    # elm_benchmarking_set = elm_benchmarking_set[elm_benchmarking_set['Species'] == 'Homo sapiens']
    # elm_benchmarking_set = elm_benchmarking_set[elm_benchmarking_set["Reason For Filtering"].isna()]
    #
    # all_prediction_set = pd.read_csv(os.path.join(publication_dir, "nar-01623-n-2012-File010.csv"),skiprows=28, header=0).rename(columns={"UniProt Accession":'Entry_Isoform'})
    # most_significant_hit_set = pd.read_csv(os.path.join(publication_dir, "nar-01623-n-2012-File011.csv"),skiprows=40, header=0).rename(columns={"UniProt Accession":'Entry_Isoform'})
    #
    #
    # all_prediction_set = make_broad_joining(main_df,all_prediction_set,region_col='pos',sep=':',regular_expression_col='slim')
    # elm_benchmarking_set = make_broad_joining(main_df,elm_benchmarking_set,start_col='Motif Start',end_col="Motif End",sep=':',regular_expression_col='Regular expression of motif')
    #
    # ratings = ["Motif",'Good', "OK", "Strong"]
    # most_significant_hit_set = most_significant_hit_set[most_significant_hit_set['Rating'].isin(ratings)]
    # most_significant_hit_set = most_significant_hit_set[most_significant_hit_set['pos'].str.contains(":")]
    # most_significant_hit_set = make_broad_joining(main_df,most_significant_hit_set,region_col='pos',sep=':',regular_expression_col='slim')
    # print(most_significant_hit_set)
    #
    # partial_overlap_threshold = 0.8
    #
    # # ELM Benchmarking
    # elm_benchmarking_set_overlaps = interval_overlap_line_sweep(pem_df, elm_benchmarking_set,
    #                                                             partial_overlap_threshold=partial_overlap_threshold)
    # n_found_elm = elm_benchmarking_set_overlaps['Publication_OverlapID'].nunique()
    # total_elm = elm_benchmarking_set.shape[0]
    # print(
    #     f"elm_benchmarking_set_overlaps: {n_found_elm} / {total_elm} = {100 * n_found_elm / total_elm:.2f}%")
    #
    # # Most Significant Hits
    # most_significant_overlaps = interval_overlap_line_sweep(pem_df,most_significant_hit_set,
    #                                                         partial_overlap_threshold=partial_overlap_threshold)
    # n_found_sig = most_significant_overlaps['Publication_OverlapID'].nunique()
    # total_sig = most_significant_hit_set.shape[0]
    # print(
    #     f"most_significant_overlaps: {n_found_sig} / {total_sig} = {100 * n_found_sig / total_sig:.2f}%")
    #
    # # most_significant_not_found = most_significant_hit_set[~most_significant_hit_set.isin(most_significant_overlaps['Publication_OverlapID'])]
    # # print(most_significant_not_found)
    # # print(most_significant_not_found.iloc[0])
    # # exit()
    #
    # # All Predictions
    # all_prediction_set_overlaps = interval_overlap_line_sweep(pem_df,all_prediction_set, partial_overlap_threshold=partial_overlap_threshold)
    # n_found_all = all_prediction_set_overlaps['Publication_OverlapID'].nunique()
    # total_all = all_prediction_set.shape[0]
    # print(
    #     f"all_prediction_set_overlaps: {n_found_all} / {total_all} = {100 * n_found_all / total_all:.2f}%")
    #
    #
