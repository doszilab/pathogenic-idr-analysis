import pandas as pd
import os


def benchmark_combinations(
        df,
        motif_combos,
        mutation_type_combos,
        ppi_effect_combos,
        wildtype_mutant_combos=None,
        output_path=None
):
    """
    Enumerate all specified combinations of motif categories, mutation types,
    PPI effects, and (optionally) wildtype/mutant combos.

    For each combination:
      1) Filter df (skip the filter if a list is None)
      2) Count total rows and # predicted
      3) Calculate percentage predicted
      4) Store in a result list.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns:
          'mapped_motif_category', 'mutation_type', 'mutation_effect_on_PPI',
          'prediction_decision_tree', 'wildtype_found', 'mutant_found', etc.
    motif_combos : list of (list_of_categories_or_None, str)
        e.g. [ (None,'ALL_MOTIFS'), (['overlapping_key_residue'],'key'), ...]
        If list_of_categories is None, do NOT filter on 'mapped_motif_category'.
    mutation_type_combos : list of (list_of_mut_types_or_None, str)
    ppi_effect_combos : list of (list_of_ppi_effects_or_None, str)
    wildtype_mutant_combos : list of ((bool,bool) or (None,None), str), optional
        If provided, we also iterate over (wildtype_found,mutant_found) combos.
        If (None,None) => no filter on wildtype/mutant.
    output_path : str, optional
        If provided, save the final DataFrame to this file (TSV).

    Returns
    -------
    pd.DataFrame
        Columns:
          motif_label, mutation_type_label, ppi_label,
          (optionally wildtype_label),
          total_count, predicted_count, pct_predicted
    """

    results = []

    # If you do *not* want to break out by WT/mut combos, let's unify that logic
    if not wildtype_mutant_combos:
        wildtype_mutant_combos = [((None, None), "ALL_WT_MUT")]

    # Helper function for "None means no filter"
    def filter_by_list(series, values_list):
        if values_list is None:
            # None => no filter => return True for all rows
            return pd.Series([True] * len(series), index=series.index)
        else:
            return series.isin(values_list)

    for motif_list, motif_label in motif_combos:
        for mut_list, mut_label in mutation_type_combos:
            for ppi_list, ppi_label in ppi_effect_combos:
                for (wt_mut_pair, wt_mut_label) in wildtype_mutant_combos:

                    # Start with True for all rows, then apply filters
                    mask = pd.Series([True] * len(df), index=df.index)

                    # Filter on mapped_motif_category
                    mask &= filter_by_list(df['mapped_motif_category'], motif_list)

                    # Filter on mutation_type
                    mask &= filter_by_list(df['mutation_type'], mut_list)

                    # Filter on mutation_effect_on_PPI
                    mask &= filter_by_list(df['mutation_effect_on_PPI'], ppi_list)

                    # If (wt,mut) != (None,None), filter further
                    wt_val, mut_val = wt_mut_pair
                    if wt_val is not None:
                        mask &= (df['wildtype_found'] == wt_val)
                    if mut_val is not None:
                        mask &= (df['mutant_found'] == mut_val)

                    subset = df[mask]
                    total_count = len(subset)
                    if total_count > 0:
                        predicted_count = (subset['prediction_decision_tree'] == True).sum()
                        pct_predicted = predicted_count / total_count * 100
                    else:
                        predicted_count = 0
                        pct_predicted = 0.0

                    results.append({
                        'motif_label': motif_label,
                        'mutation_type_label': mut_label,
                        'ppi_label': ppi_label,
                        'wildtype_label': wt_mut_label,
                        'total_count': total_count,
                        'predicted_count': predicted_count,
                        'pct_predicted': pct_predicted
                    })

    # Convert results list to DataFrame
    results_df = pd.DataFrame(results)
    # Reorder columns (optional):
    results_df = results_df[
        [
            'motif_label',
            'mutation_type_label',
            'ppi_label',
            'wildtype_label',
            'total_count',
            'predicted_count',
            'pct_predicted'
        ]
    ]
    if output_path:
        results_df.to_csv(output_path, sep='\t', index=False)
        print(f"Saved benchmark combos to: {output_path}")

    return results_df

if __name__ == "__main__":
    base_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/elm/'
    df_publication = pd.read_csv(
        os.path.join(base_dir, 'decision_tree', 'publication_predicted_with_am_info_all_disorder_class_filtered.tsv'),
        sep='\t', ).rename(columns={"ELMIdentifier": "bait"})

    # base_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/data/discanvis_base_files/sequences'
    # main_df = pd.read_csv(os.path.join(base_dir, "loc_chrom_with_names_main_isoforms_with_seq.tsv"), sep='\t')

    motif_info = pd.read_csv(
        os.path.join(base_dir, 'decision_tree', 'publication_predicted_effected_with_am_info_all_disorder_class_filtered.tsv'),
        sep='\t', ).rename(columns={"ELMIdentifier": "bait"})


    motif_combos = [
        (None, "ALL_MOTIFS"),
        (['overlapping_key_residue'], "key_res_only"),
        (['overlapping_wild-card'], "wildcard_only"),
        (['flanking'], "flanking"),
        (['motif_creating'], "motif_creating"),
        (['overlapping_key_residue', 'overlapping_wild-card'], "key_res_AND_wildcard"),
    ]

    mutation_type_combos = [
        (None, "ALL_MUTATIONS"),
        (['Germline'], "Germline"),
        (['Somatic'], "Somatic"),
    ]

    ppi_effect_combos = [
        (None, "ALL_PPI"),
        (['diminished'], "diminished"),
        (['enhanced'], "enhanced"),
        (['no_effect'], "no_effect"),
        (['diminished', 'enhanced'], "diminished_OR_enhanced"),
    ]

    # If you also want to consider the wildtype_found/mutant_found dimension:
    wildtype_mutant_combos = [
        ((None, None), "ALL_WT_MUT"),
        ((True, True), "WT=T_MUT=T"),
        ((True, False), "WT=T_MUT=F"),
        ((False, True), "WT=F_MUT=T"),
        ((False, False), "WT=F_MUT=F"),
    ]

    out_file = os.path.join(base_dir, "publication_benchmark_motif_info.tsv")
    benchmark_df = benchmark_combinations(
        df=motif_info,
        motif_combos=motif_combos,
        mutation_type_combos=mutation_type_combos,
        ppi_effect_combos=ppi_effect_combos,
        wildtype_mutant_combos=wildtype_mutant_combos,  # or None if you don't want to break out by WT/MUT
        output_path=out_file
    )

    predicted = df_publication[df_publication['prediction_decision_tree'] == True].shape[0]
    predicted_not = df_publication[df_publication['prediction_decision_tree'] == False].shape[0]
    print(f"Predicted as Motif with AM {predicted} : Percentage {predicted / df_publication.shape[0] * 100}")

    exit()
