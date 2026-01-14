import pandas as pd
import os
from tqdm import tqdm
import warnings
from multiprocessing import Pool, cpu_count
import numpy as np


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

def generate_alphamissense_score_for_elm_optimized_parallel(elm_with_regex_found, alphamissense_df, chunk_size=10000):
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
        alphamissense_dict = relevant_alphamissense_df.groupby(['Protein_ID', 'Position'])[
            'AlphaMissense'].mean().to_dict()

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


def filter_df(df):
    df['Key_vs_NonKey_Difference'] = df['key_residue_am_mean_score'] - df['flanking_residue_am_mean_score']
    df['Motif_vs_Sequential_Difference'] = df['motif_am_mean_score'] - df['sequential_am_score']

    am_score_treshold = 0.65
    key_vs_non_key_treshold = 0.18
    sequential_treshold = 0.15

    # Original filters
    am_score_filter = df['motif_am_mean_score'] >= am_score_treshold
    key_vs_non_key_filter = df['Key_vs_NonKey_Difference'] >= key_vs_non_key_treshold
    sequential_filter = df['Motif_vs_Sequential_Difference'] >= sequential_treshold

    df['AM_Score_Rule'] = am_score_filter
    df['Key_Rule'] = key_vs_non_key_filter
    df['Sequential_Rule'] = sequential_filter
    df['All_Rule'] = am_score_filter & key_vs_non_key_filter & sequential_filter

    return df

if __name__ == "__main__":
    base_dir = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations"
    file_dir = os.path.join(base_dir, "processed_data","files")

    pos = os.path.join(base_dir, "data/discanvis_base_files/positional_data_process")

    alphamissense_pos = f'{pos}/alphamissense_pos.tsv'
    alphamissense_df = extract_pos_based_df(pd.read_csv(alphamissense_pos, sep='\t'))

    to_dir = os.path.join(file_dir, "pip")

    pips = [
        ['motif_pred_on_pathogenic_region.tsv', 'motifs_on_pathogenic_region_with_am_scores.tsv'],
        ['motif_pred_on_pip.tsv', 'motifs_on_pip_with_am_scores.tsv'],
        ['motif_pred_on_pip_extension.tsv', 'motifs_on_pip_extension_with_am_scores.tsv'],
    ]

    for file_name, output_file in pips:

        predicted_motifs_with_mut = pd.read_csv(os.path.join(to_dir, file_name),sep="\t")
        elm_with_am_scores = generate_alphamissense_score_for_elm_optimized_parallel(predicted_motifs_with_mut, alphamissense_df)
        elm_with_am_scores = elm_with_am_scores.rename(columns={"Sequence":"Matched_Sequence"})

        elm_with_am_scores = filter_df(elm_with_am_scores)

        elm_with_am_scores.to_csv(os.path.join(to_dir,output_file),sep='\t',index=False)