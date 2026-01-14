import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np


# def calc_metrics(pos_column_protein, pos_column_motif, row, df, cutoff=0.5):
#     if pd.isna(row[pos_column_protein]):
#         return 0, 0, []
#
#     residues_lst = set(map(int, map(float, row[pos_column_protein].split(", "))))
#     current_df = df[df['Position'].isin(residues_lst)]
#     number_of_mutations = current_df['mutation_count'].sum()
#     mutated_positions = current_df['Position'].nunique()
#
#     scores = list(map(float, row["motif_am_scores"].split(", ")))
#     motif_index_lst = list(map(int, map(float, row[pos_column_motif].split(", "))))
#
#     above_scores = [res for i, res in enumerate(residues_lst) if scores[motif_index_lst[i]] >= cutoff]
#     return number_of_mutations, mutated_positions, above_scores

def calc_metrics(residues_lst, df):

    current_df = df[df['Position'].isin(residues_lst)]
    number_of_mutations = current_df['mutation_count'].sum()
    mutated_positions = current_df['Position'].nunique()
    return number_of_mutations, mutated_positions


def process_chunk(chunk, clinvar_grouped, cutoff):
    motif_metrics = []

    for _, row in chunk.iterrows():
        prot_id = row['Protein_ID']
        start, end = row['Start'], row['End']

        # Retrieve the relevant subset of clinvar_disorder for the current Protein_ID
        current_clinvar = clinvar_grouped.get(prot_id)
        if current_clinvar is None:
            continue

        current_clinvar = current_clinvar[(current_clinvar['Position'] >= start) & (current_clinvar['Position'] <= end)]
        if current_clinvar.shape[0] == 0:
            continue

        diseases = ", ".join(current_clinvar["nDisease"].unique())
        number_of_diseases = current_clinvar["nDisease"].nunique()
        categories = ", ".join(current_clinvar["category_names"].unique())
        genic_category = ", ".join(current_clinvar["genic_category"].unique())
        interpretation = ", ".join(current_clinvar["Interpretation"].unique())

        metrics_for_key = calc_metrics([x for x in range(start, end +1)],current_clinvar)

        motif_metrics.append({
            "Protein_ID": prot_id,
            "Motif_Start": start,
            "Motif_End": end,
            'Sequence': row['Sequence'],

            "diseases": diseases,
            "number_of_diseases": number_of_diseases,
            "categories": categories,
            "genic_category": genic_category,
            "Interpretation": interpretation,

            "Number_of_Mutations": metrics_for_key[0],
            "Mutated_Positions_Count": metrics_for_key[1],
        })

    return motif_metrics


def chunked_processing(elm_sequential_rule, clinvar_grouped, cutoff, n_chunks):
    # Divide DataFrame into chunks
    chunks = np.array_split(elm_sequential_rule, n_chunks)

    # Use Pool for parallel processing of chunks
    with Pool(processes=n_chunks) as pool:
        results = list(
            tqdm(pool.starmap(process_chunk, [(chunk, clinvar_grouped, cutoff) for chunk in chunks]), total=n_chunks))

    # Flatten list of lists
    motif_metrics = [item for sublist in results for item in sublist]
    return motif_metrics


def create_clinvar_uncertain_motif_pred(clinvar_disorder, elm_sequential_rule, cut_off, files):
    clinvar_grouped = {pid: df for pid, df in clinvar_disorder.groupby('Protein_ID')}

    # Set number of chunks (usually equal to the number of CPU cores)
    n_chunks = os.cpu_count()

    # Process data in chunks using multiprocessing
    motif_metrics = chunked_processing(elm_sequential_rule, clinvar_grouped, cut_off, n_chunks)

    # Convert list to DataFrame and save
    motif_metrics_df = pd.DataFrame(motif_metrics)
    print(motif_metrics_df)
    return motif_metrics_df



def create_position_based_motif_df(elm_sequential_rule_df, clinvar_disorder):
    # Expand each motif region in elm_sequential_rule to a position-based DataFrame
    motif_expanded = []

    for _, row in tqdm(elm_sequential_rule_df.iterrows(), total=elm_sequential_rule_df.shape[0]):
        # Generate all positions within each motif range
        positions = range(row['Start'], row['End'] + 1)
        for pos in positions:
            motif_expanded.append({
                "Protein_ID": row['Protein_ID'],
                "Position": pos,
                "Motif_Region": f"{row['Start']}-{row['End']}",
                "Sequence": row['Sequence'],
            })

    # Convert to DataFrame
    motif_df = pd.DataFrame(motif_expanded)

    motif_df_aggregated = motif_df.groupby(["Protein_ID", "Position"], as_index=False).agg({
        "Motif_Region": ", ".join,
        "Sequence": ", ".join,
    })

    merged_df = pd.merge(clinvar_disorder, motif_df_aggregated, on=["Protein_ID", "Position"], how="inner")

    return merged_df


def make_prediction(clinvar_disorder,predicted_region_with_am_info_all,cut_off=0.5, mutation_type='Pathogenic'):

    motif_metrics_df = create_clinvar_uncertain_motif_pred(clinvar_disorder, predicted_region_with_am_info_all, cut_off, files)

    to_dir = f"{files}/elm/clinvar/pip/{mutation_type}"
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)

    motif_metrics_df.to_csv(f"{to_dir}/roi_with_clinvar.tsv", sep='\t', index=False)

    result_df = create_position_based_motif_df(predicted_region_with_am_info_all, clinvar_disorder)
    result_df.to_csv(f"{to_dir}/clinvar_with_roi_overlaps.tsv", sep='\t', index=False)


if __name__ == '__main__':
    core_dir = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations"
    plots = os.path.join(core_dir, "processed_data", "plots")
    files = os.path.join(core_dir, "processed_data", "files")

    clinvar_path_disorder = f'{files}/clinvar/Uncertain/disorder/positional_clinvar_functional_categorized_final.tsv'

    sequence_df = pd.read_csv(
        os.path.join(core_dir, "data/discanvis_base_files/sequences/loc_chrom_with_names_isoforms_with_seq.tsv"), sep='\t')

    predicted_region_with_am_info_all = pd.read_csv(
        f"{files}/elm/predicted_motif_region_by_am_sequential_rule.tsv",
        sep='\t',
        # nrows=1000,
    )

    predicted_region_with_am_info_all = predicted_region_with_am_info_all.merge(sequence_df[["Protein_ID", "Sequence"]],
                                                                                on="Protein_ID", how="left")
    print(predicted_region_with_am_info_all)

    predicted_region_with_am_info_all['Sequence'] = predicted_region_with_am_info_all.apply(
        lambda row: row['Sequence'][row['Start'] - 1: row['End']], axis=1
    )

    cut_off = 0.5

    disorder_cutoff = cut_off
    order_cutoff = cut_off

    mutation_types = [
        "Pathogenic",
        "Uncertain",
        # "Predicted_Pathogenic"
    ]
    elm_type = ['known', "predicted"]

    for mut in mutation_types:
        print(mut)
        if mut == 'Predicted_Pathogenic':
            clinvar_path_disorder = f'{files}/alphamissense/clinvar/likely_pathogenic_disorder_pos_based.tsv'
        else:
            clinvar_path_disorder = f'{files}/clinvar/{mut}/disorder/positional_clinvar_functional_categorized_final.tsv'

        clinvar_disorder = pd.read_csv(clinvar_path_disorder, sep='\t').rename(columns={"Protein_position": "Position"})

        make_prediction(clinvar_disorder, predicted_region_with_am_info_all, cut_off=cut_off, mutation_type=mut)

    # clinvar_disorder = pd.read_csv(clinvar_path_disorder, sep='\t').rename(columns={"Protein_position": "Position"})

    # make_prediction(clinvar_disorder,sequence_df, cut_off=0.5)