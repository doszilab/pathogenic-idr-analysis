import pandas as pd
import os
from collections import defaultdict
from tqdm import tqdm


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

    Only retains pairs where:
      - The fraction of PEM covered by the publication >= partial_overlap_threshold, OR
      - The fraction of publication covered by the PEM >= partial_overlap_threshold.

    Args:
        pem_df (pd.DataFrame): DataFrame with columns ["Protein_ID","Start","End", ...]
        pub_df (pd.DataFrame): DataFrame with columns ["Protein_ID","Start","End", ...]
        partial_overlap_threshold (float): e.g. 0.8 for 80% coverage threshold

    Returns:
        pd.DataFrame: One row per overlapping pair, containing:
          PEM_XXX                # all original columns from pem_df, prefixed
          Publication_XXX        # all original columns from pub_df, prefixed
          Overlap_Start
          Overlap_End
          PEM_CoverageFraction
          Publication_CoverageFraction
    """

    # 1) Group by Protein_ID and sort intervals by Start
    #    We'll convert each group to a list of dicts for the line-sweep.
    pem_groups = {
        prot: grp.sort_values("Start").to_dict("records")
        for prot, grp in pem_df.groupby("Protein_ID")
    }
    pub_groups = {
        prot: grp.sort_values("Start").to_dict("records")
        for prot, grp in pub_df.groupby("Protein_ID")
    }

    # Collect all proteins present in either dataset
    all_proteins = set(pem_groups.keys()) | set(pub_groups.keys())

    overlap_records = []

    # 2) For each protein, do the two-pointer line sweep over intervals.
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

            # Check if the two intervals overlap
            if pem_end < pub_start:
                # PEM interval ends before publication starts => move to next PEM
                i += 1
                continue
            elif pub_end < pem_start:
                # Publication interval ends before PEM starts => move to next publication
                j += 1
                continue
            else:
                # There is an overlap
                overlap_start = max(pem_start, pub_start)
                overlap_end = min(pem_end, pub_end)
                overlap_len = overlap_end - overlap_start + 1

                pem_len = pem_end - pem_start + 1
                pub_len = pub_end - pub_start + 1

                pem_cov_fraction = overlap_len / pem_len if pem_len else 0
                pub_cov_fraction = overlap_len / pub_len if pub_len else 0

                # Keep only if PEM coverage >= threshold OR Publication coverage >= threshold
                if (
                        pem_cov_fraction >= partial_overlap_threshold
                        or pub_cov_fraction >= partial_overlap_threshold
                ):
                    # 3) Build the record with all columns from both DataFrames, prefixed
                    #    plus the overlap details.
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
                        "Publication_CoverageFraction": round(pub_cov_fraction, 3)
                    }
                    overlap_records.append(overlap_record)

                # Advance the pointer that ends first
                if pem_end < pub_end:
                    i += 1
                else:
                    j += 1

    # 4) Build one result DataFrame
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


if __name__ == "__main__":
    base_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/data/discanvis_base_files/sequences'
    main_df = pd.read_csv(os.path.join(base_dir, "loc_chrom_with_names_main_isoform"), sep='\t')

    print(main_df.columns)

    cols = ['protein_accession', 'wt_or_mut',
            'relationship_id', 'relates_to', 'mutation_id', 'mutated_position',
            'wild_type', 'mutant', 'designed_peptide', 'library_peptide',
            'peptide_coverage_counts', 'peptide_start', 'clinical_overview',
            'categories', 'categories_asf', 'mutation_type', ]

    # publication_df = pd.read_excel(
    #     "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/data/publication/proteome_scale_motif_based_interactome.xlsx",
    #     sheet_name=2,usecols=cols).rename(columns={"protein_accession":"Entry_Isoform"})
    # print(publication_df)
    # print(publication_df.columns)

    binding_peptides = pd.read_excel(
        "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/data/publication/binding_peptides.xlsx",
        sheet_name=1).rename(columns={"prey_accession": "Entry_Isoform"})

    # germline_df = publication_df
    germline_df = binding_peptides
    # germline_df = publication_df[publication_df['mutation_type'] == "Germline"]
    # germline_df = germline_df[germline_df['wt_or_mut'] == "mut"]
    # germline_df = germline_df[germline_df['clinical_overview'] == "pathogenic"]

    # final_cols = ['Entry_Isoform', 'library_peptide','peptide_start','categories', 'categories_asf', 'mutation_type']

    # germline_df = germline_df[final_cols].drop_duplicates()
    print(germline_df)
    germline_df = germline_df.merge(main_df[['Entry_Isoform', 'Transcript name']], how='left',
                                    on='Entry_Isoform').rename(columns={'Transcript name': 'Protein_ID'})
    # germline_df['Start'] = germline_df['peptide_start']
    # germline_df['End'] = germline_df['Start'] + germline_df['library_peptide'].str.len()
    germline_df['Start'] = germline_df['prey_start']
    germline_df['End'] = germline_df['prey_stop']
    print(germline_df)
    print(germline_df['Protein_ID'])
    print(germline_df['Start'])
    print(germline_df['End'])
    # print(germline_df['library_peptide'])

    pem_df = pd.read_csv(
        "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/elm/class_based_approach/elm_pem_disorder_corrected_with_class_count.tsv")
    # pem_df = pem_df[pem_df['N_Motif_Predicted'] == "Low_Amount"]
    print(pem_df)

    pem_df['pem_id'] = pem_df.index  # or any unique combination of columns

    peptide_with_our_uniprots = germline_df[germline_df['Protein_ID'].isin(pem_df['Protein_ID'])]
    only_matched_pems = pem_df[pem_df['Protein_ID'].isin(germline_df['Protein_ID'])]
    # print(peptide_with_our_uniprots)
    # print(only_matched_pems)

    # exit()
    to_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/elm/binding_peptides'

    overlap_df = compare_peptides_and_motifs(peptide_with_our_uniprots, only_matched_pems,
                                             partial_overlap_threshold=0.8).rename(columns={"Publication_bait":"bait"})
    #
    # pub_overlap_df.to_csv(os.path.join(to_dir,'peptide_overlap_df.tsv'),sep='\t')

    motif_info = pd.read_excel(
        "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/data/publication/motif_info.xlsx",
        sheet_name=4)

    overlap_df = overlap_df.merge(motif_info,on='bait',how='left')
    mask = overlap_df.apply(
        lambda row: str(row['PEM_ELMIdentifier']) in str(row['motif_consensus_source']),
        axis=1
    )
    overlap_df = overlap_df[mask]
    print(overlap_df.columns)
    print(overlap_df)
    overlap_df.to_csv(os.path.join(to_dir, 'pem_overlap_df.tsv'), sep='\t', index=False)


