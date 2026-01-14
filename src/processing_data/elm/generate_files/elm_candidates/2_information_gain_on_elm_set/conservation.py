import os.path

import pandas as pd
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from multiprocessing import Pool, cpu_count

def generate_conservation_score_for_elm_optimized(elm_with_regex_found, alphamissense_df,column='AlphaMissense'):
    print("This Function is Working")
    lst = []
    flanking_limit = 5

    only_elm_am = alphamissense_df[alphamissense_df['Protein_ID'].isin(elm_with_regex_found['Protein_ID'])]

    # Precompute Protein_ID and Position mask in alphamissense_df for faster access
    alphamissense_dict = only_elm_am.groupby(['Protein_ID', 'Position'])[column].mean().to_dict()

    protein_lengths = only_elm_am.groupby('Protein_ID')['Position'].max().to_dict()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        for index, row in tqdm(elm_with_regex_found.iterrows(), total=len(elm_with_regex_found)):
            start, end, protein = row['Start'], row['End'], row['Protein_ID']
            key_positions_protein = row['key_positions_protein']
            motif_length = end - start + 1
            protein_length = protein_lengths.get(protein, np.nan)

            # Calculate motif mean score
            motif_scores = [
                alphamissense_dict.get((protein, pos), np.nan) for pos in range(start, end + 1)
            ]
            motif_am_score = np.nanmean(motif_scores)
            # Final motif score assignment
            row['motif_am_mean_score'] = motif_am_score
            row['motif_am_scores'] =  ", ".join(map(str, motif_scores))

            # Key residue scores calculation if positions are available
            if pd.notna(key_positions_protein) and isinstance(key_positions_protein, str):
                key_positions = list(map(int, key_positions_protein.split(", ")))
                key_residue_means = [
                    alphamissense_dict.get((protein, pos), np.nan) for pos in key_positions
                ]
                key_residue_am_mean_score = np.nanmean(key_residue_means)
                row['key_residue_am_scores'] = ", ".join(map(str, key_residue_means))
                row['key_residue_am_mean_score'] = key_residue_am_mean_score
            else:
                row['key_residue_am_scores'], row['key_residue_am_mean_score'] = None, None

            # Flanking residue scores calculation if positions are available
            flanking_positions = [
                pos for pos in map(int, row['number_of_possible_residues'].split(", ")) if pos >= flanking_limit
            ]
            if flanking_positions:
                flanking_residue_means = [
                    alphamissense_dict.get((protein, pos), np.nan) for pos in flanking_positions
                ]
                flanking_residue_am_mean_score = np.nanmean(flanking_residue_means)
                row['flanking_residue_am_mean_score'] = flanking_residue_am_mean_score
                row['flanking_residue_am_scores'] = ", ".join(map(str, flanking_residue_means))
            else:
                row['flanking_residue_am_mean_score'], row['flanking_residue_am_scores'] = None, None


            # Sequential environment scores (in place, no new DataFrame)
            forward_positions = range(end + 1, end + motif_length + 1)
            backward_positions = range(start - motif_length, start)

            # Flag for terminated region if boundaries are reached
            row['is_terminated'] = (end + motif_length > protein_length) or (start - motif_length < 1)

            # Calculate scores for sequential environment
            seq_scores = [
                             alphamissense_dict.get((protein, pos), np.nan) for pos in forward_positions
                         ] + [
                             alphamissense_dict.get((protein, pos), np.nan) for pos in backward_positions
                         ]

            row['sequential_am_score'] = np.nanmean(seq_scores)
            row['sequential_am_scores'] = ", ".join(map(str, seq_scores))


            # Append to results list
            lst.append(row)

    # Convert list to DataFrame
    df = pd.DataFrame(lst)
    return df

def extract_pos_based_df(df):
    df[['Protein_ID', 'Position']] = df['AccessionPosition'].str.split('|', expand=True)
    df = df.drop(columns=['AccessionPosition'])
    df['Position'] = df['Position'].astype(int)
    return df

def process_elm_row(row, alphamissense_dict, protein_lengths, flanking_limit=5):
    start, end, protein = row['Start'], row['End'], row['Protein_ID']
    key_positions_protein = row['key_positions_protein']
    motif_length = end - start + 1
    protein_length = protein_lengths.get(protein, np.nan)

    # Calculate motif mean score
    motif_scores = [alphamissense_dict.get((protein, pos), np.nan) for pos in range(start, end + 1)]
    row['motif_am_mean_score'] = np.nanmean(motif_scores)
    row['motif_am_scores'] = ", ".join(map(str, motif_scores))

    # Key residue scores calculation if positions are available
    if pd.notna(key_positions_protein) and isinstance(key_positions_protein, str):
        key_positions = list(map(int, key_positions_protein.split(", ")))
        key_residue_means = [alphamissense_dict.get((protein, pos), np.nan) for pos in key_positions]
        row['key_residue_am_mean_score'] = np.nanmean(key_residue_means)
        row['key_residue_am_scores'] = ", ".join(map(str, key_residue_means))
    else:
        row['key_residue_am_mean_score'], row['key_residue_am_scores'] = None, None

    # Flanking residue scores calculation if positions are available
    flanking_positions = [pos for pos in map(int, row['number_of_possible_residues'].split(", ")) if pos >= flanking_limit]
    if flanking_positions:
        flanking_residue_means = [alphamissense_dict.get((protein, pos), np.nan) for pos in flanking_positions]
        row['flanking_residue_am_mean_score'] = np.nanmean(flanking_residue_means)
        row['flanking_residue_am_scores'] = ", ".join(map(str, flanking_residue_means))
    else:
        row['flanking_residue_am_mean_score'], row['flanking_residue_am_scores'] = None, None

    # Sequential environment scores
    forward_positions = range(end + 1, end + motif_length + 1)
    backward_positions = range(start - motif_length, start)
    row['is_terminated'] = (end + motif_length > protein_length) or (start - motif_length < 1)
    seq_scores = [alphamissense_dict.get((protein, pos), np.nan) for pos in forward_positions] + \
                 [alphamissense_dict.get((protein, pos), np.nan) for pos in backward_positions]
    row['sequential_am_score'] = np.nanmean(seq_scores)
    row['sequential_am_scores'] = ", ".join(map(str, seq_scores))

    return row

def process_elm_chunk(args):
    chunk, alphamissense_dict, protein_lengths, flanking_limit = args
    lst = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for _, row in chunk.iterrows():
            processed_row = process_elm_row(row, alphamissense_dict, protein_lengths, flanking_limit)
            lst.append(processed_row)
    return pd.DataFrame(lst)

def generate_alphamissense_score_for_elm_optimized_parallel(elm_with_regex_found, alphamissense_df, chunk_size=10000,column='AlphaMissense'):
    print("Starting Parallel Processing")
    flanking_limit = 5

    # Precompute protein lengths only once
    protein_lengths = alphamissense_df.groupby('Protein_ID')['Position'].max().to_dict()

    elm_with_regex_found = elm_with_regex_found.sort_values(by="Protein_ID")

    # Prepare the chunks with filtered alphamissense_dict for each chunk
    chunks = []
    for start_idx in tqdm(range(0, len(elm_with_regex_found), chunk_size),desc="Generate the Chunks"):
        # Slice the chunk
        chunk = elm_with_regex_found.iloc[start_idx:start_idx + chunk_size]

        # Filter alphamissense_df for proteins relevant to this chunk
        relevant_proteins = chunk['Protein_ID'].unique()
        relevant_alphamissense_df = alphamissense_df[alphamissense_df['Protein_ID'].isin(relevant_proteins)]
        alphamissense_dict = relevant_alphamissense_df.groupby(['Protein_ID', 'Position'])[column].mean().to_dict()

        # Append the chunk along with its filtered alphamissense_dict to the list
        chunks.append((chunk, alphamissense_dict, protein_lengths, flanking_limit))

    # Using Pool to parallelize the process with progress bar
    with Pool(cpu_count()) as pool:
        results = []
        with tqdm(total=len(elm_with_regex_found)) as pbar:
            for result in pool.imap_unordered(process_elm_chunk, chunks):
                results.append(result)
                pbar.update(len(result))

    # Combine all results into a single DataFrame
    df = pd.concat(results, ignore_index=True)
    return df

def generate_files_for_predictions():
    core_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations'
    elm_dir = f'{core_dir}/processed_data/files/elm/'
    pos = os.path.join(core_dir, "data/discanvis_base_files/positional_data_process")

    scann_path = f'{core_dir}/data/discanvis_base_files/elm/proteome_scann'


    known_elm_instances = pd.read_csv(f"{scann_path}/known_instances_with_match_with_regex_info.tsv", sep='\t')

    df_predicted = pd.read_csv(
        os.path.join(elm_dir, 'elm_predicted_with_am_info_all_disorder_class_filtered.tsv'),
        sep='\t',usecols=known_elm_instances.columns
    )
    df_known = pd.read_csv(
        os.path.join(elm_dir, 'elm_known_with_am_info_all_disorder_class_filtered.tsv'),
        sep='\t',usecols=known_elm_instances.columns
    )

    alphamissense_pos = f'{pos}/conservation_all_pos.tsv'
    alphamissense_df = extract_pos_based_df(pd.read_csv(alphamissense_pos, sep='\t'))
    print(alphamissense_df)

    # Known
    # elm_with_am_scores = generate_conservation_score_for_elm_optimized(df_known,alphamissense_df,column='Vertebrata')
    # elm_with_am_scores.to_csv(f"{core_dir}/processed_data/files/elm/conservation/elm_known.tsv",sep='\t', index=False)

    # Predicted
    prediction_with_am_scores = generate_alphamissense_score_for_elm_optimized_parallel(df_predicted, alphamissense_df,column='Vertebrata')
    prediction_with_am_scores.to_csv(
        f"{core_dir}/processed_data/files/elm/conservation/elm_predicted.tsv",
        sep='\t', index=False)


if __name__ == "__main__":
    generate_files_for_predictions()