import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
tqdm.pandas()


def calc_metrics(pos_column_protein, pos_column_motif, row, df, cutoff=0.5):
    if pd.isna(row[pos_column_protein]):
        return 0, 0, []

    residues_lst = set(map(int, map(float, row[pos_column_protein].split(", "))))
    current_df = df[df['Position'].isin(residues_lst)]
    if 'mutation_count' in current_df.columns:
        number_of_mutations = current_df['mutation_count'].sum()
    else:
        number_of_mutations = current_df.shape[0]
    mutated_positions = current_df['Position'].nunique()

    scores = list(map(float, row["motif_am_scores"].split(", ")))
    motif_index_lst = list(map(int, map(float, row[pos_column_motif].split(", "))))

    above_scores = [res for i, res in enumerate(residues_lst) if scores[motif_index_lst[i]] >= cutoff]
    return number_of_mutations, mutated_positions, above_scores

def calc_metrics_for_all(row, df):

    residues_lst = list(range(row["Start"],row["End"] + 1))
    current_df = df[df['Position'].isin(residues_lst)]
    # number_of_mutations = current_df['mutation_count'].sum()
    if 'mutation_count' in current_df.columns:
        number_of_mutations = current_df['mutation_count'].sum()
    else:
        number_of_mutations = current_df.shape[0]
    mutated_positions = current_df['Position'].nunique()

    return number_of_mutations, mutated_positions


def process_chunk(chunk, clinvar_grouped, cutoff,motif_type):
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
        intepretation = ", ".join(current_clinvar["Interpretation"].unique())

        metrics_for_key = calc_metrics('key_positions_protein', 'key_positions_motif', row, current_clinvar,
                                       cutoff=cutoff)
        metrics_for_flanking = calc_metrics('flanking_positions_protein', 'flanking_positions_motif', row,
                                            current_clinvar, cutoff=cutoff)

        metrics_for_motif = (
            metrics_for_key[0] + metrics_for_flanking[0],
            metrics_for_key[1] + metrics_for_flanking[1],
            metrics_for_key[2] + metrics_for_flanking[2]
        )

        motif_positions = ", ".join(map(str, metrics_for_motif[2]))
        key_positions = ", ".join(map(str, metrics_for_key[2]))
        flanking_positions = ", ".join(map(str, metrics_for_flanking[2]))


        metrics_for_motif_without_treshold = calc_metrics_for_all(row, current_clinvar)

        motif_metrics.append({
            "Protein_ID": prot_id,
            "Motif_Start": start,
            "Motif_End": end,
            'ELMIdentifier': row['ELMIdentifier'],
            'ELMType': row['ELMType'],
            "N_Motif_Predicted": "Known" if motif_type == "known" else str(row['N_Motif_Predicted']),
            "N_Motif_Category": "Known" if motif_type == "known" else str(row['N_Motif_Category']),
            "ELM_Types_Count": "Known" if motif_type == "known" else str(row['ELM_Types_Count']),
            'Matched_Sequence': row['Matched_Sequence'],
            'found_regex_pattern': row['found_regex_pattern'],
            'AM_Score_Rule': row['AM_Score_Rule'],
            'Key_Rule': row['Key_Rule'],
            'Sequential_Rule': row['Sequential_Rule'],
            'All_Rule': row['All_Rule'],

            "diseases": diseases,
            "number_of_diseases": number_of_diseases,
            "categories": categories,
            "genic_category": genic_category,
            "Intepretation": intepretation,

            "Number_of_Mutations": metrics_for_motif_without_treshold[0],
            "Mutated_Positions_Count": metrics_for_motif_without_treshold[1],

            "Number_of_Mutations_Above_Score": metrics_for_motif[1],
            "Mutated_Positions_Count_Above_Score": metrics_for_motif[1],
            "Positions_Above_Score": motif_positions,

            "Key_Number_of_Mutations": metrics_for_key[0],
            "Key_Mutated_Positions_Count": metrics_for_key[1],
            "Key_Above_Score_Positions": key_positions,

            "Flanking_Number_of_Mutations": metrics_for_flanking[0],
            "Flanking_Mutated_Positions_Count": metrics_for_flanking[1],
            "Flanking_Above_Score_Positions": flanking_positions,
        })

    return motif_metrics


def chunked_processing(elm_sequential_rule, clinvar_grouped, cutoff, n_chunks,motif_type):
    # Divide DataFrame into chunks
    chunks = np.array_split(elm_sequential_rule, n_chunks)

    # Use Pool for parallel processing of chunks
    with Pool(processes=n_chunks) as pool:
        results = list(
            tqdm(pool.starmap(process_chunk, [(chunk, clinvar_grouped, cutoff,motif_type) for chunk in chunks]), total=n_chunks))

    # Flatten list of lists
    motif_metrics = [item for sublist in results for item in sublist]
    return motif_metrics


def create_clinvar_uncertain_motif_pred(clinvar_disorder, elm_sequential_rule, cut_off, motif_type):
    clinvar_grouped = {pid: df for pid, df in clinvar_disorder.groupby('Protein_ID')}

    # Set number of chunks (usually equal to the number of CPU cores)
    n_chunks = os.cpu_count()

    # Process data in chunks using multiprocessing
    motif_metrics = chunked_processing(elm_sequential_rule, clinvar_grouped, cut_off, n_chunks,motif_type)

    # Convert list to DataFrame and save
    motif_metrics_df = pd.DataFrame(motif_metrics)
    return motif_metrics_df



def create_position_based_motif_df(elm_sequential_rule_df, clinvar_disorder,motif_type='known'):
    # Expand each motif region in elm_sequential_rule to a position-based DataFrame
    motif_expanded = []

    for _, row in tqdm(elm_sequential_rule_df.iterrows(), total=elm_sequential_rule_df.shape[0],desc="ELM Overlap"):
        # Generate all positions within each motif range
        positions = range(row['Start'], row['End'] + 1)
        for pos in positions:
            motif_expanded.append({
                "Protein_ID": row['Protein_ID'],
                "Position": pos,
                "Motif_Region": f"{row['Start']}-{row['End']}",
                "ELMIdentifier": row['ELMIdentifier'],
                'ELMType': row['ELMType'],
                "Matched_Sequence": row['Matched_Sequence'],
                "found_regex_pattern": row['found_regex_pattern'],
                "AM_Score_Rule": str(row['AM_Score_Rule']),  # Convert to string
                'Key_Rule': str(row['Key_Rule']),  # Convert to string
                'Sequential_Rule': str(row['Sequential_Rule']),  # Convert to string
                'All_Rule': str(row['All_Rule']),
                "key_positions_protein": str(row['key_positions_protein']),
                "N_Motif_Predicted": "Known" if motif_type == "known" else str(row['N_Motif_Predicted']),
                "N_Motif_Category": "Known" if motif_type == "known" else str(row['N_Motif_Category']),
                "ELM_Types_Count": "Known" if motif_type == "known" else str(row['ELM_Types_Count']),
                'Found_Known': "Known" if motif_type == "known" else str(row['Found_Known'])
            })

    # Convert to DataFrame
    motif_df = pd.DataFrame(motif_expanded)

    motif_df_aggregated = motif_df.groupby(["Protein_ID", "Position"], as_index=False).agg({
        "Motif_Region": ", ".join,
        "ELMIdentifier": ", ".join,
        "ELMType": ", ".join,
        "Matched_Sequence": ", ".join,
        "found_regex_pattern": ", ".join,
        "AM_Score_Rule": ", ".join,
        "Key_Rule": ", ".join,
        "Sequential_Rule": ", ".join,
        "All_Rule": ", ".join,
        "key_positions_protein": ", ".join,
        "N_Motif_Predicted": ", ".join,
        "N_Motif_Category": ", ".join,
        "ELM_Types_Count": ", ".join,
        "Found_Known": ", ".join
    })

    merged_df = pd.merge(clinvar_disorder, motif_df_aggregated, on=["Protein_ID", "Position"], how="inner")

    def check_key_position(row):
        # Convert the aggregated key positions to a list of integers
        key_positions_protein = row['key_positions_protein']
        if pd.isna(key_positions_protein):
            # If it's NaN, there's no valid list of positions
            return False
        positions_list = []
        for pos in key_positions_protein.split(", "):
            if pos.lower() != 'nan':  # avoid the string 'nan'
                positions_list.append(int(pos))
        return row['Position'] in positions_list

    merged_df["Is_Key_Position"] = merged_df.progress_apply(check_key_position, axis=1)

    return merged_df


def make_prediction(clinvar_disorder,elm_df,cut_off=0.5,motif_type='known',mutation_type="Pathogenic"):

    motif_metrics_df = create_clinvar_uncertain_motif_pred(clinvar_disorder, elm_df, cut_off, motif_type)
    to_dir = f"{files}/elm/clinvar/motif/{mutation_type}"
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)

    motif_metrics_df.to_csv(f"{to_dir}/motif_with_clinvar_{motif_type}.tsv", sep='\t', index=False)

    result_df = create_position_based_motif_df(elm_df, clinvar_disorder,motif_type=motif_type)
    result_df.to_csv(f"{to_dir}/clinvar_elm_overlaps_{motif_type}.tsv", sep='\t', index=False)


def make_pred_for_high_amount_pem():
    clinvar_path_disorder = f'{files}/clinvar/Pathogenic/disorder/positional_clinvar_functional_categorized_final.tsv'
    clinvar_disorder = pd.read_csv(clinvar_path_disorder, sep='\t').rename(columns={"Protein_position": "Position"})

    elm_path = f"{files}/elm/clinvar/enriched/enriched_pems_high_amount_with_pathogenic_mutations.tsv"

    elm_predicted_with_am_info_all = pd.read_csv(
        elm_path,
        sep='\t',
        # nrows=1000,
    )

    make_prediction(clinvar_disorder,elm_predicted_with_am_info_all, cut_off=cut_off, motif_type="high_amount_pem", mutation_type='Pathogenic')


if __name__ == '__main__':
    core_dir = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations"
    plots = os.path.join(core_dir, "processed_data", "plots")
    files = os.path.join(core_dir, "processed_data", "files")

    cut_off = 0.5

    disorder_cutoff = cut_off
    order_cutoff = cut_off

    mutation_types = [
        "Pathogenic",
        "Uncertain",
        # "Predicted_Pathogenic"
    ]
    elm_type = [
        "predicted",
        'known',
    ]

    make_pred_for_high_amount_pem()
    exit()


    for mut in mutation_types:

        if mut == 'Predicted_Pathogenic':
            clinvar_path_disorder = f'{files}/alphamissense/clinvar/likely_pathogenic_disorder_pos_based.tsv'
        else:
            clinvar_path_disorder = f'{files}/clinvar/{mut}/disorder/positional_clinvar_functional_categorized_final.tsv'

        clinvar_disorder = pd.read_csv(clinvar_path_disorder, sep='\t').rename(columns={"Protein_position": "Position"})

        exclude_categories = ['Inborn Genetic Diseases', 'Unknown']
        clinvar_disorder = clinvar_disorder[~clinvar_disorder['category_names'].isin(exclude_categories)]

        for elm in elm_type:
            # if elm == "known":
            #     elm_path = f"{files}/elm/elm_known_with_am_info_all_disorder_class_filtered.tsv"
            # else:
            #     # elm_path = f"{files}/elm/elm_predicted_disorder_with_confidence.tsv"
            #     # We use Enrichment prefiltering
            #     # enrichment_test_for_motifs.py
            if mut == "Uncertain":
                elm_path = f"{files}/elm/clinvar/enriched/enriched_pems_with_uncertain_mutations.tsv"
            else:
                elm_path = f"{files}/elm/clinvar/enriched/enriched_pems_with_pathogenic_mutations.tsv"

            elm_predicted_with_am_info_all = pd.read_csv(
                elm_path,
                sep='\t',
                # nrows=1000,
            )

            isknown = elm == 'known'
            elm_predicted_with_am_info_all = elm_predicted_with_am_info_all[elm_predicted_with_am_info_all['known'] == isknown]

            # if 'N_Motif_Predicted' in elm_predicted_with_am_info_all.columns:
            #     elm_predicted_with_am_info_all = elm_predicted_with_am_info_all[elm_predicted_with_am_info_all['N_Motif_Predicted'] == "Low_Amount"]
                # elm_predicted_with_am_info_all = elm_predicted_with_am_info_all[elm_predicted_with_am_info_all['known'] == False]

            # print(elm,mut,elm_path)
            # print(elm_predicted_with_am_info_all)
            # exit()
            make_prediction(clinvar_disorder,elm_predicted_with_am_info_all, cut_off=cut_off, motif_type=elm, mutation_type=mut)