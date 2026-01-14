import os.path

import pandas as pd
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from multiprocessing import Pool, cpu_count


def extract_pos_based_df(df):
    df[['Protein_ID', 'Position']] = df['AccessionPosition'].str.split('|', expand=True)
    df = df.drop(columns=['AccessionPosition'])
    df['Position'] = df['Position'].astype(int)
    return df

def alphamissense_process(alphamissense_df):
    print('AlphaMissense Process')

    alphamissense_df['am_pathogenicity'] = pd.to_numeric(alphamissense_df['am_pathogenicity'], errors='coerce')
    alphamissense_df['pos'] = pd.to_numeric(alphamissense_df['pos'], errors='coerce')
    alphamissense_filtered_mean = alphamissense_df.groupby(['Protein_ID', 'pos'])[
        'am_pathogenicity'].mean().reset_index(name='AlphaMissense')
    annotation_df = alphamissense_filtered_mean.rename(columns={'pos': 'Position'})

    return annotation_df



def generate_alphamissense_score_for_elm(elm_with_regex_found,alphamissense_df):

    lst = []

    flanking_limit = 5

    for index,row in tqdm(elm_with_regex_found.iterrows(), total=len(elm_with_regex_found),desc='Generate AlphaMissense Score For ELMs'):
        start = row['Start']
        end = row['End']
        protein = row['Protein_ID']
        key_positions_protein = row['Key_Positions_Protein']

        row['key_residue_am_scores'] = None
        row['key_residue_am_mean_score'] = None
        row['flanking_residue_am_mean_score'] = None
        row['flanking_residue_am_scores'] = None
        row['motif_am_mean_score'] = None


        am_for_motif = alphamissense_df.loc[
            (alphamissense_df['Protein_ID'] == protein) & (alphamissense_df['Position'] >= start) & (
                        alphamissense_df['Position'] <= end)]


        am_score_lst = []
        for i in range(start, end +1):
            pos_am_score = am_for_motif.loc[(am_for_motif['Position'] == i)]
            am_score_lst.append(pos_am_score['AlphaMissense'].mean())

        motif_am_score = sum(am_score_lst) / len(am_score_lst)
        row['motif_am_mean_score'] = motif_am_score

        if pd.notna(key_positions_protein):
            key_positions_proteins = map(int,key_positions_protein.split(", "))
            means = []
            for i in key_positions_proteins:
                pos_am_score = am_for_motif.loc[(am_for_motif['Position'] == i)]
                means.append(pos_am_score['AlphaMissense'].mean())

            key_residue_am_score = sum(means)/len(means)
            row['key_residue_am_scores'] = ", ".join(map(str,means))
            row['key_residue_am_mean_score'] = key_residue_am_score
        else:
            row['key_residue_am_scores'] = None
            row['key_residue_am_mean_score'] = None

        flanking_positions_proteins = [ x for x in map(int,row['Number_of_Possible_Residues'].split(", ")) if x >= flanking_limit]
        if flanking_positions_proteins:
            means = []
            for i in flanking_positions_proteins:
                pos_am_score = am_for_motif.loc[(am_for_motif['Position'] == i)]
                means.append(pos_am_score['AlphaMissense'].mean())


            flanking_residue_am_score = sum(means) / len(means)
            row['flanking_residue_am_mean_score'] = flanking_residue_am_score
            row['flanking_residue_am_scores'] = ", ".join(map(str,means))
        else:
            row['flanking_residue_am_mean_score'] = None
            row['flanking_residue_am_scores'] = None

        lst.append(row)

    df = pd.concat(lst)

    return df


def create_sequential(df,shifting=0):

    lst =[]
    df['length'] = df['End'] - df['Start'] +1

    for index,row in df.iterrows():

        length = int(row['length'])
        if length > 20:
            length = 20

        fstart = int(row['Start']) + length + shifting
        fend = int(row['End']) + length + shifting
        bstart = int(row['Start']) - length - shifting
        bend = int(row['End']) - length - shifting
        forward_positions = [x for x in range(fstart, fend +1)]
        backward_positions = [x for x in range(bstart, bend +1)]
        for i,position in enumerate(forward_positions):
            lst.append([row['Protein_ID'],position])
            lst.append([row['Protein_ID'],backward_positions[i]])

    new_df = pd.DataFrame(lst,columns=['Protein_ID', 'Position'])
    return new_df

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

def generate_alphamissense_score_for_elm_optimized(elm_with_regex_found, alphamissense_df):
    print("This Function is Working")
    lst = []
    flanking_limit = 5

    only_elm_am = alphamissense_df[alphamissense_df['Protein_ID'].isin(elm_with_regex_found['Protein_ID'])]

    # Precompute Protein_ID and Position mask in alphamissense_df for faster access
    alphamissense_dict = only_elm_am.groupby(['Protein_ID', 'Position'])['AlphaMissense'].mean().to_dict()

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

def generate_file():

    core_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations'
    pos = os.path.join(core_dir,"data/discanvis_base_files/positional_data_process")
    alldata = pd.read_csv(
        f'{core_dir}/data/discanvis_base_files/sequences/loc_chrom_with_names_isoforms_with_seq.tsv',
        sep='\t', header=0)
    print(alldata.columns)
    main_isoforms = alldata[alldata['main_isoform'] == "yes"]

    # Comes from DisCanVis elm.py
    elm_for_am = pd.read_csv(f"{core_dir}/data/discanvis_base_files/elm/elm_for_am_mapped.tsv",sep='\t')
    print(elm_for_am)
    main_isoform_elm = elm_for_am[elm_for_am['Protein_ID'].isin(main_isoforms['Protein_ID'])]

    elm_with_regex_found = main_isoform_elm[(main_isoform_elm['isMatch'] == True) & (main_isoform_elm['Number_of_Possible_Residues'].notnull()) ]
    print(elm_with_regex_found)

    ## add AlphaMissense bot dbNFSP and both all

    # am_hg38 = '/dlab/home/norbi/PycharmProjects/AlphaMissense/data/processed_alphamissense_results_hg38_mapping.tsv'
    # am_hg38_df = pd.read_csv(am_hg38, sep='\t')
    # print(am_hg38_df)
    # am_hg38_pos_df = alphamissense_process(am_hg38_df)[["Protein_ID", "Position", "AlphaMissense"]].rename(
    #     columns={"AlphaMissense": "AM_SNV_Mean_Score"})
    # print(am_hg38_pos_df)


    alphamissense_pos = f'{pos}/alphamissense_pos.tsv'
    alphamissense_df = extract_pos_based_df(pd.read_csv(alphamissense_pos, sep='\t'))
    # .rename(columns={"AlphaMissense": "AM_All_Mean_Score"}))
    print(alphamissense_df)

    elm_with_am_scores = generate_alphamissense_score_for_elm(elm_with_regex_found,alphamissense_df)
    elm_with_am_scores.to_csv(f"{core_dir}/processed_data/files/elm/elm_with_key_residues.tsv",sep='\t', index=False)



def generate_files_for_predictions():
    core_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations'
    pos = os.path.join(core_dir, "data/discanvis_base_files/positional_data_process")
    alldata = pd.read_csv(
        f'{core_dir}/data/discanvis_base_files/sequences/loc_chrom_with_names_isoforms_with_seq.tsv',
        sep='\t', header=0)
    print(alldata.columns)
    main_isoforms = alldata[alldata['main_isoform'] == "yes"]

    scann_path = f'{core_dir}/data/discanvis_base_files/elm/proteome_scann'

    known_elm_instances = pd.read_csv(f"{scann_path}/known_instances_with_match_with_regex_info.tsv",sep='\t')
    proteome_predicted_elm_instances = pd.read_csv(f"{scann_path}/proteome_scann_with_regex_info.tsv",sep='\t',
                                                   # nrows=10000,
                                                   )

    known_elm_instances = known_elm_instances[known_elm_instances['error'] == False]
    proteome_predicted_elm_instances = proteome_predicted_elm_instances[proteome_predicted_elm_instances['error'] == False]

    alphamissense_pos = f'{pos}/alphamissense_pos.tsv'
    alphamissense_df = extract_pos_based_df(pd.read_csv(alphamissense_pos, sep='\t'))
    print(alphamissense_df)

    # Known
    # elm_with_am_scores = generate_alphamissense_score_for_elm_optimized(known_elm_instances,alphamissense_df)
    # elm_with_am_scores.to_csv(f"{core_dir}/processed_data/files/elm/elm_known_with_am_info_all.tsv",sep='\t', index=False)

    # # Predicted
    # prediction_with_am_scores = generate_alphamissense_score_for_elm_optimized_parallel(proteome_predicted_elm_instances, alphamissense_df)
    # prediction_with_am_scores.to_csv(
    #     f"{core_dir}/processed_data/files/elm/elm_predicted_with_am_info_all.tsv",
    #     sep='\t', index=False)

    # Publication - Ylva Ivarsson
    # scann_path = f'{core_dir}/data/discanvis_base_files/elm/ylva_ivarsson'
    # known_elm_instances = pd.read_csv(f"{scann_path}/known_instances_with_match_with_regex_info.tsv", sep='\t')
    # proteome_predicted_elm_instances = pd.read_csv(f"{scann_path}/proteome_scann_with_regex_info.tsv", sep='\t')
    #
    # # Known
    # elm_with_am_scores = generate_alphamissense_score_for_elm_optimized(known_elm_instances, alphamissense_df)
    # elm_with_am_scores.to_csv(f"{core_dir}/processed_data/files/elm/publication_with_am_info_all.tsv", sep='\t',
    #                           index=False)
    #
    # # # Predicted
    # prediction_with_am_scores = generate_alphamissense_score_for_elm_optimized_parallel(
    #     proteome_predicted_elm_instances, alphamissense_df)
    # prediction_with_am_scores.to_csv(
    #     f"{core_dir}/processed_data/files/elm/publication_predicted_with_am_info_all.tsv",
    #     sep='\t', index=False)

    # Publication - Norman Davey
    scann_path = f'{core_dir}/data/discanvis_base_files/elm/norman_davey'
    known_elm_instances = pd.read_csv(f"{scann_path}/known_instances_with_match_with_regex_info.tsv", sep='\t')
    proteome_predicted_elm_instances = pd.read_csv(f"{scann_path}/proteome_scann_with_regex_info.tsv", sep='\t')

    # Known
    elm_with_am_scores = generate_alphamissense_score_for_elm_optimized(known_elm_instances, alphamissense_df)
    elm_with_am_scores.to_csv(f"{core_dir}/processed_data/files/elm/publication_norman_davey_with_am_info_all.tsv", sep='\t',
                              index=False)

    # # Predicted
    prediction_with_am_scores = generate_alphamissense_score_for_elm_optimized_parallel(
        proteome_predicted_elm_instances, alphamissense_df)
    prediction_with_am_scores.to_csv(
        f"{core_dir}/processed_data/files/elm/publication_predicted_norman_davey_with_am_info_all.tsv",
        sep='\t', index=False)




def plot_motif_based_distribution(df):
    """Plot distributions of flanking, key residue, and motif mean scores by motif (row)."""

    # Melt the DataFrame for seaborn-friendly format
    df = df.rename(columns={
        "flanking_residue_am_mean_score":"Non-Key Residue",
        "key_residue_am_mean_score":"Key Residue",
        "motif_am_mean_score":"Motif Residues",
        "sequential_am_score":"Sequential Residues",
    })
    motif_data = df[["Motif Residues","Sequential Residues","Key Residue","Non-Key Residue"]].melt(
        var_name='Score_Type', value_name='Mean_Score')

    # Plotting motif-based distributions
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=motif_data, x='Score_Type', y='Mean_Score')
    plt.title('Distribution of Motif-based Scores (Non-Key Residue, Sequential Residues, Key Residue, Motif Residues)')
    plt.xlabel("")
    plt.ylabel('Mean AlphaMissense Score')
    plt.show()

def plot_motif_based_distribution_key_and_flanking(df):
    """Plot distributions of flanking, key residue, and motif mean scores by motif (row)."""

    print(df)

    df = df.dropna(subset=['motif_am_mean_score'])
    print(df)

    motif_positions = df['key_residue_am_scores'].dropna().str.split(", ").explode().astype(
        float).reset_index(drop=True)
    sequential_positions = df['sequential_am_scores'].dropna().str.split(", ").explode().astype(
        float).reset_index(drop=True)

    # Combine into a DataFrame for plotting
    position_data = pd.DataFrame({
        'Score': pd.concat([motif_positions, sequential_positions], ignore_index=True),
        'Score_Type': ['Key Residues'] * len(motif_positions) + ['Flanking Residue'] * len(sequential_positions)
    })

    # Plotting motif-based distributions
    plt.figure(figsize=(4, 6))
    sns.violinplot(data=position_data, x='Score_Type', y='Score')
    plt.title('Distribution of Position-based Scores (Motif Residues, Sequential Residues)')
    plt.xlabel("")
    plt.ylabel('Mean AlphaMissense Score')
    plt.show()

def plot_motif_difference_distribution(df):
    """Plot histogram of the difference between Key Residue and Non-Key Residue scores."""

    # Calculate the difference between Key Residue and Non-Key Residue scores
    df['Key_vs_NonKey_Difference'] = df['key_residue_am_mean_score'] - df['flanking_residue_am_mean_score']

    # Calculate the mean difference
    mean_difference = df['Key_vs_NonKey_Difference'].mean()

    # Plotting the Key vs Non-Key Difference as a histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Key_vs_NonKey_Difference'], bins=30, kde=True)
    plt.title('Histogram of Key vs Non-Key Residue Score Difference')
    plt.xlabel('Difference (Key Residue - Non-Key Residue)')
    plt.ylabel('Frequency')

    # Add vertical lines for zero and mean difference
    plt.axvline(0, color='red', linestyle='--', label='Zero Difference')
    plt.axvline(mean_difference, color='blue', linestyle='--', label=f'Mean Difference ({mean_difference:.2f})')

    plt.legend()
    plt.show()

def plot_motif_difference_distribution_sequential(df,figsize=(4,6),filepath=None):
    """Plot histogram of the difference between Motif and Sequential region AlphaMissense scores."""

    # Calculate the difference between Motif and Sequential scores
    df['Motif_vs_Sequential_Difference'] = df['motif_am_mean_score'] - df['sequential_am_score']

    # Calculate the mean difference
    mean_difference = df['Motif_vs_Sequential_Difference'].mean()

    # Plotting the Motif vs Sequential Difference as a histogram
    plt.figure(figsize=figsize)
    sns.histplot(df['Motif_vs_Sequential_Difference'], bins=30, kde=True)
    plt.title('Score Differences of Known Motifs')
    plt.xlabel('Difference (Motif - Flanking)')
    plt.ylabel('Frequency')

    # Add vertical lines for zero and mean difference
    plt.axvline(0, color='red', linestyle='--', label='Zero')
    plt.axvline(mean_difference, color='blue', linestyle='--', label=f'Mean ({mean_difference:.2f})')

    plt.legend(loc='upper left')
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
    plt.show()

def plot_motif_distribution_alphamissense(df,figsize=(4,6),filepath=None,column='motif_am_mean_score',text='Mean AlphaMissense'):
    """Plot histogram of the difference between Motif and Sequential region AlphaMissense scores."""

    # Calculate the mean difference
    mean_difference = df[column].mean()

    # Plotting the Motif vs Sequential Difference as a histogram
    plt.figure(figsize=figsize)
    sns.histplot(df[column], bins=30, kde=True)
    plt.title(f'Histogram {text} Score (Known)')
    plt.xlabel(f'{text} Score')
    plt.ylabel('Frequency')

    # Add vertical lines for zero and mean difference
    plt.axvline(0.5, color='red', linestyle='--', label='0.5')
    plt.axvline(mean_difference, color='blue', linestyle='--', label=f'Mean ({mean_difference:.2f})')

    plt.legend(loc='upper left')
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
    plt.show()


def plot_position_based_distribution(df):
    """Plot distributions of expanded key and flanking residues scores for a more granular view."""

    # Expand the key and flanking scores into individual positions
    key_residue_scores_expanded = df['key_residue_am_scores'].dropna().str.split(", ").explode().astype(float).reset_index(drop=True)
    flanking_residue_scores_expanded = df['flanking_residue_am_scores'].dropna().str.split(", ").explode().astype(float).reset_index(drop=True)

    # Combine into a DataFrame for plotting
    position_data = pd.DataFrame({
        'Score': pd.concat([key_residue_scores_expanded, flanking_residue_scores_expanded], ignore_index=True),
        'Type': ['Key Residue'] * len(key_residue_scores_expanded) + ['Non-Key Residue'] * len(flanking_residue_scores_expanded)
    })

    # Plotting position-based distributions
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=position_data, x='Type', y='Score')
    plt.title('Distribution of Position-based Scores (Key and Non-Key Residues)')
    plt.ylabel('AlphaMissense Score')
    plt.xlabel("")
    plt.show()


def plot_possible_residue_distribution(df,filepath=None):
    # Value Counts for Number_of_Possible_Residues
    current_value_counts = df['Number_of_Possible_Residues'].value_counts().reset_index(name='Count')
    print(current_value_counts)
    # Plot violin plots
    plt.figure(figsize=(4, 3))
    sns.barplot(data=current_value_counts, x='Number_of_Possible_Residues', y='Count',color='#f08080')
    plt.suptitle('Possible Residues Distribution')
    plt.xlabel('Number of Possible Residues')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()

def plot_group_based_score_distribution(df,filepath=None):
    # # Plot violin plots
    plt.figure(figsize=(4, 3))
    sns.violinplot(data=df, x='group', y='Motif_Score',color='#f08080')
    plt.suptitle('AlphaMissense Scores of Motif Residues')
    plt.xlabel('Number of Possible Residues')
    plt.ylabel('Motif AlphaMissense Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()

def plot_residue_analyis(df):
    """Plot pairwise correlation between expanded motif_am_scores and Number_of_Possible_Residues."""

    # Supplementary materials
    plot_path = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/plots/sm/motif'

    # Expand motif_am_scores and Number_of_Possible_Residues together

    lst = []
    wrong_index = 0
    for index, row in df.iterrows():
        if pd.notna(row['motif_am_scores']):
            motif_scores = row['motif_am_scores'].split(", ")
            possible_residues = row['number_of_possible_residues'].split(", ")

            for i,value in enumerate(motif_scores):
                try:
                    residue_count = possible_residues[i]
                    lst.append([float(value),int(residue_count)])
                except IndexError:
                    wrong_index +=1
                    continue

    print("wrong_index",wrong_index)

    analising_df = pd.DataFrame(lst)
    analising_df.columns = ['Motif_Score', 'Number_of_Possible_Residues']

    plot_possible_residue_distribution(analising_df,filepath=os.path.join(plot_path,"motif_residue_number_of_residues.png"))

    groups = {
        "1":(1,1),
        "2-7":(2,7),
        "7-18":(7,18),
        "19-20":(19,20)
    }

    def assign_group(value):
        for group, (low, high) in groups.items():
            if low <= value <= high:
                return group
        return None

    # Apply the function to create the 'group' column
    analising_df['group'] = analising_df['Number_of_Possible_Residues'].apply(assign_group)

    # Convert 'group' to a categorical type with ordered categories
    order = ["1", "2-7", "7-18", "19-20"]
    analising_df['group'] = pd.Categorical(analising_df['group'], categories=order, ordered=True)

    # # Plot violin plots
    plot_group_based_score_distribution(analising_df,filepath=os.path.join(plot_path,"motif_residue_alphamissense_distribtuion.png"))


def overlap_with_percentage(region, disorder_region):
    """
    Calculates the overlap between two ranges and returns the overlapping range
    and the percentage of overlap relative to the original region.

    :param region: Tuple or list with start and end positions [start, end] of the original region
    :param disorder_region: Tuple or list with start and end positions [start, end] of the disorder region
    :return: Tuple (overlap_range, overlap_percentage) where overlap_range is the range of the overlap
             and overlap_percentage is the percentage of overlap relative to the original region length
    """
    overlap_start = max(region[0], disorder_region[0])
    overlap_end = min(region[1], disorder_region[1])

    if overlap_start <= overlap_end:
        # Calculate overlap range and overlap length
        overlap_range = range(overlap_start, overlap_end + 1)
        overlap_length = len(overlap_range)

        # Calculate the length of the original region
        region_length = region[1] - region[0] + 1

        # Calculate overlap percentage relative to the original region
        overlap_percentage = overlap_length / region_length
        return overlap_range, overlap_percentage
    else:
        return False, 0.0

def disorder_filtering(df,start="Start",end="End",cutoff=0.8):
    stucture_path = '/dlab/home/norbi/PycharmProjects/DisCanVis_Data_Process/Processed_Data/gencode_process/annotations/structures'
    combined_disorder_df = pd.read_csv(f"{stucture_path}/CombinedDisorderNew.tsv",sep='\t')

    filtered_rows = []

    for _, row in tqdm(df.iterrows(),total=df.shape[0]):
        region = [row[start], row[end]]
        protein_id = row['Protein_ID']

        # Select relevant disorder regions for the current protein
        disorder_regions = combined_disorder_df[combined_disorder_df['Protein_ID'] == protein_id]

        # Initialize the maximum overlap percentage found
        max_overlap_percentage = 0.0

        for _, disorder_row in disorder_regions.iterrows():
            disorder_region = [disorder_row['Start'], disorder_row['End']]
            overlap_range, overlap_percentage = overlap_with_percentage(region, disorder_region)

            # Check if the overlap percentage meets the cutoff
            if overlap_percentage >= cutoff:
                max_overlap_percentage = overlap_percentage
                break  # Exit early if we meet the cutoff

        # If max overlap percentage meets or exceeds the cutoff, add the row to the filtered results
        if max_overlap_percentage >= cutoff:
            filtered_rows.append(row)

    # Return a DataFrame of the filtered rows
    return pd.DataFrame(filtered_rows)


def disorder_filtering_optimized(df, start="Start", end="End", cutoff=0.8):
    structure_path = '/dlab/home/norbi/PycharmProjects/DisCanVis_Data_Process/Processed_Data/gencode_process/annotations/structures'
    combined_disorder_df = pd.read_csv(f"{structure_path}/CombinedDisorderNew.tsv", sep='\t')

    tqdm.pandas()

    # Create a dictionary with Protein_ID as keys and their disorder regions as lists
    disorder_dict = {
        protein_id: regions[['Start', 'End']].values
        for protein_id, regions in combined_disorder_df.groupby('Protein_ID')
    }

    def has_sufficient_overlap(row):
        protein_id = row['Protein_ID']
        region = [row[start], row[end]]

        # Retrieve disorder regions for the protein from the prebuilt dictionary
        disorder_regions = disorder_dict.get(protein_id, [])

        # If no disorder regions for this protein, skip further checks
        if len(disorder_regions) == 0:
            return False

        # Calculate overlap for each disorder region
        overlap_percentages = [
            overlap_with_percentage(region, disorder_region)[1]
            for disorder_region in disorder_regions
        ]

        # Return True if any overlap meets or exceeds the cutoff
        return any(overlap >= cutoff for overlap in overlap_percentages)

    # Apply the filtering function to all rows and collect those that pass the cutoff
    filtered_df = df[df.progress_apply(has_sufficient_overlap, axis=1)]

    return filtered_df

def compute_rowwise_am_max(df):
    """
    Compute the maximum score from the `motif_am_scores` column for each row.
    """
    df['AM_Max'] = df['motif_am_scores'].apply(
        lambda x: max(map(float, x.split(', '))) if isinstance(x, str) else np.nan
    )
    return df

def plot_the_result():
    files = "/dlab/home/norbi/PycharmProjects/AlphaMissense_Stat/processed_data/files/elm"
    plot_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/plots/fig5'
    elm_known_with_am_info_all = pd.read_csv(
        f"{files}/elm_known_with_am_info_all_disorder.tsv",
        sep='\t')

    # elm_predicted_with_am_info_all = pd.read_csv(
    #     f"{files}/elm_predicted_with_am_info_all.tsv",
    #     sep='\t')

    df = elm_known_with_am_info_all
    print(df)

    #
    # plot_motif_pathogenic_overlaps_based_on_criteria()

    # exit()


    # Plot motif-based distribution
    # plot_motif_based_distribution(df)

    # Plot Sequential Only
    # plot_motif_based_distribution_only_sequential(df)

    # Plot Flanking vs Key Residues
    # plot_motif_based_distribution_key_and_flanking(df)

    # Plot position-based distribution
    # plot_position_based_distribution(df)
    # exit()

    # Based on how many substution can be in a position
    # Fig SM Motif
    # plot_residue_analyis(df)
    # exit()

    # Plot the difference within a Motif between the Key Residue and Non-Key Residue
    # plot_motif_difference_distribution(df)

    # Sequential Difference

    # Fig 5C
    plot_motif_difference_distribution_sequential(df,figsize=(5,3),filepath=os.path.join(plot_dir,'C2.png'))
    exit()

    # Fig 5C ScoreDistribution
    plot_motif_distribution_alphamissense(df,figsize=(5,3),filepath=os.path.join(plot_dir,'C3.png'))
    df = compute_rowwise_am_max(df)
    plot_motif_distribution_alphamissense(df,figsize=(5,3),filepath=os.path.join(plot_dir,'C4.png'),column='AM_Max',text='Max AlphaMissense')


def make_filtered_dfs(df):
    df['Key_vs_NonKey_Difference'] = df['key_residue_am_mean_score'] - df['flanking_residue_am_mean_score']
    df['Motif_vs_Sequential_Difference'] = df['motif_am_mean_score'] - df['sequential_am_score']

    am_score_treshold = 0.65
    key_vs_non_key_treshold = 0.18
    sequential_treshold = 0.15

    # Original filters
    am_score_filter = df['motif_am_mean_score'] >= am_score_treshold
    key_vs_non_key_filter = df['Key_vs_NonKey_Difference'] >= key_vs_non_key_treshold
    sequential_filter = df['Motif_vs_Sequential_Difference'] >= sequential_treshold

    filters = {
        f'AM Score >= {am_score_treshold}': am_score_filter,
        f'Key vs Non-Key >= {key_vs_non_key_treshold}': key_vs_non_key_filter,
        f'Motif vs Sequential >= {sequential_treshold}': sequential_filter,
        'Sequential or AM Score': am_score_filter | sequential_filter,
        'Sequential & AM Score': am_score_filter & sequential_filter,
        'Sequential & Key': key_vs_non_key_filter & sequential_filter,
        'Key & AM Score': am_score_filter & key_vs_non_key_filter,
        'All ': am_score_filter & key_vs_non_key_filter & sequential_filter,
        'Any ': am_score_filter | key_vs_non_key_filter | sequential_filter,
    }

    # Calculate percentage of data retained for each filter
    total_count = len(df)
    percentages = {name: df[condition].shape[0] / total_count * 100 for name, condition in filters.items()}

    return percentages


def plot_predicton_results():
    files = "/dlab/home/norbi/PycharmProjects/AlphaMissense_Stat/processed_data/files/elm"
    elm_known_with_am_info_all = pd.read_csv(
        f"{files}/elm_known_with_am_info_all_disorder.tsv",
        sep='\t')

    elm_predicted_with_am_info_all = pd.read_csv(
        f"{files}/elm_predicted_with_am_info_all_disorder.tsv",
        sep='\t')

    # Get percentage data for known and predicted datasets

    known_percentages = make_filtered_dfs(elm_known_with_am_info_all)
    print(known_percentages)
    predicted_percentages = make_filtered_dfs(elm_predicted_with_am_info_all)
    print(predicted_percentages)

    # Plotting

    labels = list(known_percentages.keys())
    known_values = list(known_percentages.values())
    predicted_values = list(predicted_percentages.values())
    x = range(len(labels))  # x-axis locations for labels

    plt.figure(figsize=(14, 8))

    # Plot bars for known and predicted datasets
    bars_known = plt.bar(x, known_values, width=0.4, label='Known', align='center')
    bars_predicted = plt.bar([i + 0.4 for i in x], predicted_values, width=0.4, label='Predicted', align='center')

    # Adding text on top of each bar
    for i, bar in enumerate(bars_known):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{round(known_values[i],2)}', ha='center',
                 va='bottom')

    for i, bar in enumerate(bars_predicted):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{round(predicted_values[i],2)}',
                 ha='center', va='bottom')

    # Adding labels and legend
    plt.xticks([i + 0.2 for i in x], labels, rotation=45, ha="right")
    plt.ylabel('Percentage of Data Retained (%)')
    plt.title('Data Retention by Filtering Rules for Known and Predicted Datasets')
    plt.legend()
    plt.tight_layout()
    plt.savefig("/dlab/home/norbi/PycharmProjects/AlphaMissense_Stat/processed_data/plots/elm/prediction_data_retention.png")
    plt.show()

def get_mostly_disorder_instances(known_df,known_disorder_df):

    known_counts = known_df['ELMIdentifier'].value_counts().to_dict()
    known_disorder_counts = known_disorder_df['ELMIdentifier'].value_counts().to_dict()

    mostly_disorder = []

    for elm_id,count in known_counts.items():
        disorder_count = known_disorder_counts.get(elm_id,0)

        if disorder_count > 0 and disorder_count / count > 0.5:
            mostly_disorder.append(elm_id)
        elif disorder_count / count  == 0.5 and count == 2:
            mostly_disorder.append(elm_id)

    return mostly_disorder

def filter_files_for_disorder():

    files = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/elm"
    elm_known_with_am_info_all = pd.read_csv(
        f"{files}/elm_known_with_am_info_all.tsv",
        sep='\t')

    publication_with_am_info_all = pd.read_csv(
        f"{files}/publication_with_am_info_all.tsv",
        sep='\t')

    publication_effected_with_am_info_all = pd.read_csv(
        f"{files}/publication_predicted_with_am_info_all.tsv",
        sep='\t')

    elm_predicted_with_am_info_all = pd.read_csv(
        f"{files}/elm_predicted_with_am_info_all.tsv",
        sep='\t')

    publication_norman_davey_with_am_info_all = pd.read_csv(
        f"{files}/publication_norman_davey_with_am_info_all.tsv",
        sep='\t')

    publication_predicted_norman_davey_with_am_info_all = pd.read_csv(
        f"{files}/publication_predicted_norman_davey_with_am_info_all.tsv",
        sep='\t')

    # print("Publication Process - Norman Davey")
    # publication_with_am_info_all_disorder = disorder_filtering_optimized(publication_norman_davey_with_am_info_all)
    # print(publication_with_am_info_all_disorder)
    # publication_with_am_info_all_disorder.to_csv(f"{files}/publication_norman_davey_with_am_info_all_disorder.tsv", sep='\t',
    #                                              index=False)
    #
    # print("Publication Only Affected Process - Norman Davey")
    # publication_effected_with_am_info_all_disorder = disorder_filtering_optimized(publication_predicted_norman_davey_with_am_info_all)
    # print(publication_effected_with_am_info_all_disorder)
    # publication_effected_with_am_info_all_disorder.to_csv(f"{files}/publication_effected_norman_davey_with_am_info_all_disorder.tsv",
    #                                                       sep='\t',
    #                                                       index=False)
    #
    #
    # # exit()
    # print("Publication Process")
    # publication_with_am_info_all_disorder = disorder_filtering_optimized(publication_with_am_info_all)
    # print(publication_with_am_info_all_disorder)
    # publication_with_am_info_all_disorder.to_csv(f"{files}/publication_with_am_info_all_disorder.tsv", sep='\t',
    #                                              index=False)
    #
    # print("Publication Only Affected Process")
    # publication_effected_with_am_info_all_disorder = disorder_filtering_optimized(publication_effected_with_am_info_all)
    # print(publication_effected_with_am_info_all_disorder)
    # publication_effected_with_am_info_all_disorder.to_csv(f"{files}/publication_effected_with_am_info_all_disorder.tsv", sep='\t',
    #                                              index=False)

    # exit()

    print("Known Process")
    elm_known_with_am_info_all_disorder = disorder_filtering_optimized(elm_known_with_am_info_all)
    print(elm_known_with_am_info_all_disorder)

    print("Predicted Process")
    elm_predicted_with_am_info_all_disorder = disorder_filtering_optimized(elm_predicted_with_am_info_all)
    print(elm_predicted_with_am_info_all_disorder)


    elm_known_with_am_info_all_disorder.to_csv(f"{files}/elm_known_with_am_info_all_disorder.tsv",sep='\t', index=False)
    elm_predicted_with_am_info_all_disorder.to_csv(f"{files}/elm_predicted_with_am_info_all_disorder.tsv",sep='\t', index=False)
    # exit()

    mostly_disorder = get_mostly_disorder_instances(elm_known_with_am_info_all, elm_known_with_am_info_all_disorder)
    elm_known_with_am_info_all_disorder = elm_known_with_am_info_all_disorder[elm_known_with_am_info_all_disorder["ELMIdentifier"].isin(mostly_disorder)]
    elm_predicted_with_am_info_all_disorder = elm_predicted_with_am_info_all_disorder[elm_predicted_with_am_info_all_disorder["ELMIdentifier"].isin(mostly_disorder)]

    elm_known_with_am_info_all_disorder = filter_df(elm_known_with_am_info_all_disorder)
    elm_predicted_with_am_info_all_disorder = filter_df(elm_predicted_with_am_info_all_disorder)

    elm_known_with_am_info_all_disorder.to_csv(f"{files}/elm_known_with_am_info_all_disorder_class_filtered.tsv", sep='\t', index=False)
    elm_predicted_with_am_info_all_disorder.to_csv(f"{files}/elm_predicted_with_am_info_all_disorder_class_filtered.tsv", sep='\t',index=False)


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

def make_filtered_files():
    files = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/elm"
    elm_known_with_am_info_all = pd.read_csv(
        f"{files}/elm_known_with_am_info_all_disorder.tsv",
        sep='\t')

    elm_predicted_with_am_info_all = pd.read_csv(
        f"{files}/elm_predicted_with_am_info_all_disorder.tsv",
        sep='\t')

    known_df = filter_df(elm_known_with_am_info_all)
    predicted_df = filter_df(elm_predicted_with_am_info_all)

    known_df.to_csv(f"{files}/elm_known_with_am_info_all_disorder_with_rules.tsv", sep='\t',
                                               index=False)
    predicted_df.to_csv(f"{files}/elm_predicted_with_am_info_all_disorder_with_rules.tsv", sep='\t',
                                                   index=False)


def generate_ordered_files():
    files = "/dlab/home/norbi/PycharmProjects/AlphaMissense_Stat/processed_data/files/elm"
    elm_known_with_am_info_all_disorder = pd.read_csv(
        f"{files}/elm_known_with_am_info_all_disorder.tsv",
        sep='\t')

    elm_known_with_am_info_all = pd.read_csv(
        f"{files}/elm_known_with_am_info_all.tsv",
        sep='\t')

    common_columns = elm_known_with_am_info_all.columns.intersection(elm_known_with_am_info_all_disorder.columns)

    # Filter rows in 'elm_known_with_am_info_all' that do not exist in 'elm_known_with_am_info_all_disorder'

    elm_ordered_only = elm_known_with_am_info_all.merge(
        elm_known_with_am_info_all_disorder[common_columns],
        on=list(common_columns),
        how='left',
        indicator=True
    ).query('_merge == "left_only"').drop(columns='_merge')

    print(elm_ordered_only)

    elm_ordered_only.to_csv(f"{files}/elm_ordered_only.tsv", sep='\t', index=False)

    exit()

    # elm_predicted_with_am_info_all = pd.read_csv(
    #     f"{files}/elm_predicted_with_am_info_all_disorder.tsv",
    #     sep='\t')

def make_filtered_dfs_for_base(df,filter_option='Motif_vs_Sequential_Difference'):
    # df['Key_vs_NonKey_Difference'] = df['key_residue_am_mean_score'] - df['flanking_residue_am_mean_score']
    filter_treshold = 0
    if filter_option == 'Motif_vs_Sequential_Difference':
        filter_treshold = 0.15
        df['Motif_vs_Sequential_Difference'] = df['motif_am_mean_score'] - df['sequential_am_score']
    elif filter_option == 'motif_am_mean_score':
        filter_treshold = 0.48
    elif filter_option == 'AM_Max':
        filter_treshold = 0.63

    # am_score_treshold = 0.65
    # key_vs_non_key_treshold = 0.18


    # Original filters
    # am_score_filter = df['motif_am_mean_score'] >= am_score_treshold
    # key_vs_non_key_filter = df['Key_vs_NonKey_Difference'] >= key_vs_non_key_treshold
    sequential_filter= df[filter_option] >= filter_treshold
    no_filter = df.notna()

    filters = {
        f'Filtered': sequential_filter,
        f'All Motif': no_filter,
    }

    # Calculate percentage of data retained for each filter
    total_count = len(df)
    percentages = {name: df[condition].shape[0] / total_count * 100 for name, condition in filters.items()}
    total_numbers = {name: df[condition].shape[0] for name, condition in filters.items()}

    return percentages,total_numbers, df[sequential_filter]


# def plot_totals(known_labels, known_total_values, known_color,text="Known",figsize=(4, 6)):
#     fig, ax = plt.subplots(figsize=figsize)
#     # Sort by total values in descending order
#     sorted_indices = sorted(range(len(known_total_values)), key=lambda i: known_total_values[i], reverse=True)
#     known_labels = [known_labels[i] for i in sorted_indices]
#     known_total_values = [known_total_values[i] for i in sorted_indices]
#
#     bars = ax.bar(range(len(known_labels)), known_total_values, width=0.4, color=known_color, label=text,
#                   align='center')
#     for i, bar in enumerate(bars):
#         ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{known_total_values[i]}', ha='center',
#                 va='bottom')
#     ax.set_xticks(range(len(known_labels)))
#     ax.set_xticklabels(known_labels, rotation=45, ha="right")
#     ax.set_ylabel('Number of Motifs')
#
#     for spine in ax.spines.values():
#         spine.set_visible(False)
#
#
#
#     # ax.set_title(f'{text}')
#     plt.tight_layout()
#     plt.show()


def plot_totals(ax, labels, total_values, color, text="Known",ylabel='Number of Motifs'):
    # Sort by total values in descending order
    sorted_indices = sorted(range(len(total_values)), key=lambda i: total_values[i], reverse=True)
    labels = [labels[i] for i in sorted_indices]
    total_values = [total_values[i] for i in sorted_indices]

    bars = ax.bar(range(len(labels)), total_values, width=0.4, color=color, label=text, align='center')

    for i, bar in enumerate(bars):
        number = total_values[i]
        text = f"{number}" if number < 10_000 else f"{number / 1_000:.0f}k"

        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{text}', ha='center', va='bottom')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(text)

    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_totals_subplots(known_labels, known_total_values, known_color,
                         predicted_labels, predicted_total_values, predicted_color, figsize=(8, 6),rule='name'):
    fig, axs = plt.subplots(1, 2, figsize=figsize,sharey=False)

    plot_totals(axs[0], known_labels, known_total_values, known_color, text="Known")
    plot_totals(axs[1], predicted_labels, predicted_total_values, predicted_color, text="Predicted")

    axs[0].set_title("Known", pad=20)
    axs[1].set_title("Predicted")
    axs[1].set_ylabel(None)

    # Adjust the y-limits to prevent overlap
    max_y_value = max(predicted_total_values)
    step = max_y_value / 10
    axs[1].set_ylim(0, max_y_value + step)

    fig.align_titles()

    plt.suptitle(f"Motif Retention ({rule})")
    plt.tight_layout()
    plt.show()


def plot_combined_percentages(known_percentages, predicted_percentages, known_color, predicted_color,figsize=(4, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    all_labels = sorted(set(known_percentages.keys()).union(predicted_percentages.keys()))
    combined_known = [known_percentages.get(label, 0) for label in all_labels]
    combined_predicted = [predicted_percentages.get(label, 0) for label in all_labels]

    bar_width = 0.4
    bars_known = ax.bar(range(len(all_labels)), combined_known, width=bar_width, color=known_color, label='Known')
    bars_predicted = ax.bar([i + bar_width for i in range(len(all_labels))], combined_predicted, width=bar_width,
                            color=predicted_color, label='Predicted')

    for i, bar in enumerate(bars_known):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{round(combined_known[i], 2)}%', ha='center',
                va='bottom')
    for i, bar in enumerate(bars_predicted):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{round(combined_predicted[i], 2)}%',
                ha='center', va='bottom')

    ax.set_xticks([i + bar_width / 2 for i in range(len(all_labels))])
    ax.set_xticklabels(all_labels, rotation=45, ha="right")
    ax.set_ylabel('Percentage of Motif Retained (%)')
    ax.set_title('Percentage of Motif Retention (Known vs Predicted)')
    plt.tight_layout()
    plt.show()


def plot_base_results():
    # Load data and compute values (assuming make_filtered_dfs_for_base and other required functions are defined)
    files = "/dlab/home/norbi/PycharmProjects/AlphaMissense_Stat/processed_data/files/elm"
    elm_known_with_am_info_all = pd.read_csv(f"{files}/elm_known_with_am_info_all_disorder.tsv", sep='\t')
    elm_known_with_uncertain_mut = pd.read_csv(f"{files}/clinvar/motif/known_motif_with_clinvar_uncertain.tsv", sep='\t')
    elm_known_with_pathogenic_mut = pd.read_csv(f"{files}/clinvar/motif/known_motif_with_clinvar_pathogenic.tsv", sep='\t')

    # print(elm_known_with_am_info_all)
    # print(elm_known_with_uncertain_mut)
    # print(elm_known_with_pathogenic_mut)
    # exit()

    elm_predicted_with_am_info_all = pd.read_csv(f"{files}/elm_predicted_with_am_info_all_disorder.tsv", sep='\t')
    elm_predicted_with_uncertain_mut = pd.read_csv(f"{files}/clinvar/motif/predicted_motif_with_clinvar_uncertain.tsv", sep='\t')
    elm_predicted_with_pathogenic_mut = pd.read_csv(f"{files}/clinvar/motif/predicted_motif_with_clinvar_pathogenic.tsv", sep='\t')

    predicted_uncertain_percentages = {
        'Uncertain Mut.': elm_predicted_with_uncertain_mut.shape[0] / elm_predicted_with_am_info_all.shape[0] * 100,
        'Pathogenic Mut.': elm_predicted_with_pathogenic_mut.shape[0] / elm_predicted_with_am_info_all.shape[0] * 100
    }
    predicted_uncertain_total_numbers = {
        'Uncertain Mut.': elm_predicted_with_uncertain_mut.shape[0],
        'Pathogenic Mut.': elm_predicted_with_pathogenic_mut.shape[0]
    }

    known_uncertain_percentages = {
        'Uncertain Mut.': elm_known_with_uncertain_mut.shape[0] / elm_known_with_am_info_all.shape[0] * 100,
        'Pathogenic Mut.': elm_known_with_pathogenic_mut.shape[0] / elm_known_with_am_info_all.shape[0] * 100
    }
    known_uncertain_total_numbers = {
        'Uncertain Mut.': elm_known_with_uncertain_mut.shape[0],
        'Pathogenic Mut.': elm_known_with_pathogenic_mut.shape[0]
    }

    columns_needed = ["Protein_ID","Motif_Start","Motif_End","ELM_Accession","ELMIdentifier","ELMType"]

    motifs_with_scores_known_pathogenic = elm_known_with_pathogenic_mut[columns_needed].merge(elm_known_with_am_info_all,on=columns_needed,how="inner")
    motifs_with_scores_predicted_pathogenic = elm_predicted_with_pathogenic_mut[columns_needed].merge(elm_predicted_with_am_info_all,on=columns_needed,how="inner")

    # Get data for known and predicted datasets
    # known_percentages, known_total_numbers = make_filtered_dfs_for_base(elm_known_with_am_info_all)
    # pathogenic_known_percentages, pathogenic_known_total_numbers = make_filtered_dfs_for_base(motifs_with_scores_known_pathogenic)
    # predicted_percentages, predicted_total_numbers = make_filtered_dfs_for_base(elm_predicted_with_am_info_all)
    # pathogenic_predicted_percentages, pathogenic_predicted_total_numbers = make_filtered_dfs_for_base(motifs_with_scores_predicted_pathogenic)

    # predicted_percentages.update(predicted_uncertain_percentages)
    # predicted_total_numbers.update(predicted_uncertain_total_numbers)
    #
    # known_percentages.update(known_uncertain_percentages)
    # known_total_numbers.update(known_uncertain_total_numbers)

    # Remove 'All Motif' from percentages
    known_percentages.pop('All Motif', None)
    predicted_percentages.pop('All Motif', None)

    # Prepare data for plotting
    known_labels = list(known_total_numbers.keys())
    known_total_values = list(known_total_numbers.values())
    predicted_labels = list(predicted_total_numbers.keys())
    predicted_total_values = list(predicted_total_numbers.values())

    # Define colors
    known_color = 'royalblue'
    predicted_color = 'salmon'

    # Create subplots and plot each individually

    # Figure 5 Motif Retention
    plot_totals_subplots(known_labels, known_total_values, known_color,
                         predicted_labels, predicted_total_values, predicted_color,figsize=(4, 4))

    # Figure 5 ClinVar Positions



    # plot_totals(known_labels, known_total_values, known_color,figsize=(4, 4))
    # plot_totals(predicted_labels, predicted_total_values, predicted_color,text="Predicted",figsize=(4, 4))
    # plot_combined_percentages(known_percentages, predicted_percentages, known_color, predicted_color,figsize=(3, 6))
    exit()

    # Add a single legend outside the plots
    handles = [plt.Rectangle((0, 0), 1, 1, color=known_color, label='Known'),
               plt.Rectangle((0, 0), 1, 1, color=predicted_color, label='Predicted')]
    fig.legend(handles=handles, loc='upper center', ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def get_clinvar_percentages_and_totals(clinvar_df, elm_df, columns=['Protein_ID', "Position"]):
    # Merge to find ClinVar mutations that fall within ELM motifs
    elm_positions_df = elm_df[["Protein_ID", "Start", "End", "ELMIdentifier"]].drop_duplicates()
    clinvar_positions_df = clinvar_df[columns].drop_duplicates()

    # Total counts
    total_elm_motifs = len(elm_positions_df)
    total_clinvar_mutations = len(clinvar_positions_df)

    # Find mutations that occur in motif regions
    merged_df = clinvar_positions_df.merge(elm_positions_df, on="Protein_ID", how="inner")
    mutations_in_elm = merged_df[
        (merged_df["Position"] >= merged_df["Start"]) & (merged_df["Position"] <= merged_df["End"])
    ]

    # Count ELM motifs that have at least one ClinVar mutation
    nunique_elm_with_mutations = mutations_in_elm["ELMIdentifier"].nunique()
    unique_elm_with_mutations = mutations_in_elm["ELMIdentifier"].unique()

    # Count ClinVar mutations in ELM regions
    total_mutations_in_elm = len(mutations_in_elm[columns].drop_duplicates())

    clinvar_data_df = mutations_in_elm.merge(clinvar_df,on=columns, how="inner")

    # Compute percentages
    percent_elm_with_mutations = (nunique_elm_with_mutations / total_elm_motifs) * 100 if total_elm_motifs else 0
    percent_clinvar_in_elm = (total_mutations_in_elm / total_clinvar_mutations) * 100 if total_clinvar_mutations else 0

    return {
        "Total ELM Motifs": total_elm_motifs,
        "Total ClinVar Position": total_clinvar_mutations,
        "ELM Motifs with ClinVar Position": nunique_elm_with_mutations,
        "Percentage of ELM Motifs with ClinVar Position": percent_elm_with_mutations,
        "Total ClinVar Position in ELM Regions": total_mutations_in_elm,
        "unique_elm_with_mutations": unique_elm_with_mutations,
        "df": clinvar_data_df,
        "Percentage of ClinVar Position in ELM Regions": percent_clinvar_in_elm,
    }


def plot_total_clinvar_positions_in_elm(known_labels, known_total_values, predicted_labels, predicted_total_values,
                                        known_color, predicted_color, figsize=(6, 6),title="Pathogenic Positions in ELM",ylabel="Number of Position"):

    fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=False)

    plot_totals(axs[0], known_labels, known_total_values, known_color, text="Known",ylabel=ylabel)
    plot_totals(axs[1], predicted_labels, predicted_total_values, predicted_color, text="Predicted",ylabel=ylabel)

    axs[0].set_title("Known", pad=20)
    axs[1].set_title("Predicted" ,pad=20)
    axs[1].set_ylabel(None)

    # # Adjust the y-limits to prevent overlap
    # max_y_value = max(predicted_total_values)
    # step = max_y_value / 10
    # axs[1].set_ylim(0, max_y_value + step)

    fig.align_titles()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()



def plot_motif_pathogenic_overlaps_based_on_criteria():
    # Load data and compute values (assuming make_filtered_dfs_for_base and other required functions are defined)
    files = "/dlab/home/norbi/PycharmProjects/AlphaMissense_Stat/processed_data/files/elm"
    elm_known_with_am_info_all = pd.read_csv(f"{files}/elm_known_with_am_info_all_disorder.tsv", sep='\t')
    elm_predicted_with_am_info_all = pd.read_csv(f"{files}/elm_predicted_with_am_info_all_disorder.tsv", sep='\t',
                                                 # nrows=1000,
                                                 )

    elm_known_with_am_info_all = compute_rowwise_am_max(elm_known_with_am_info_all)
    elm_predicted_with_am_info_all = compute_rowwise_am_max(elm_predicted_with_am_info_all)

    columns_needed = ["Protein_ID","Motif_Start","Motif_End","ELM_Accession","ELMIdentifier","ELMType"]

    # Clinvar Pathogenic
    clinvar_path = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/clinvar/Pathogenic/disorder/positional_clinvar_functional_categorized_final.tsv"
    clinvar_df = pd.read_csv(clinvar_path, sep='\t')
    print(clinvar_df)


    options = [
        ['Motif_vs_Sequential_Difference','Flanking Rule'],
        ['motif_am_mean_score','Mean'],
        # ['AM_Max','Max']
    ]

    def compare_protein_sets(rule, info_dct_known, info_dct_known_filtered):
        # Convert lists to sets for comparison
        known_set = set(info_dct_known["unique_elm_with_mutations"])
        known_filtered_set = set(info_dct_known_filtered["unique_elm_with_mutations"])

        # Find differences
        removed_from_known = known_set - known_filtered_set  # Proteins removed after filtering
        added_in_filtered = known_filtered_set - known_set  # Proteins added after filtering

        print(f"Rule: {rule}")
        print(f"Total in Known: {len(known_set)}, Total in Known Filtered: {len(known_filtered_set)}")

        if removed_from_known:
            print("Proteins removed after filtering:", removed_from_known)
        else:
            print("No proteins removed after filtering.")

        if added_in_filtered:
            print("Proteins added in filtered set:", added_in_filtered)
        else:
            print("No new proteins added in filtered set.")

        print("-" * 50)

    proteins = []

    for filter_option, rule in options:

        # Get data for known and predicted datasets
        known_percentages, known_total_numbers,sequential_filter_df = make_filtered_dfs_for_base(elm_known_with_am_info_all,filter_option=filter_option)
        predicted_percentages, predicted_total_numbers,sequential_filter_df_predicted = make_filtered_dfs_for_base(elm_predicted_with_am_info_all,filter_option=filter_option)

        info_dct_known = get_clinvar_percentages_and_totals(clinvar_df, elm_known_with_am_info_all)
        info_dct_predicted = get_clinvar_percentages_and_totals(clinvar_df, elm_predicted_with_am_info_all)
        info_dct_known_filtered = get_clinvar_percentages_and_totals(clinvar_df, sequential_filter_df)
        info_dct_predicted_filtered = get_clinvar_percentages_and_totals(clinvar_df, sequential_filter_df_predicted)

        print(rule)
        info_dct_known["df"].to_csv(f"/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/elm/clinvar/test/known_overlap_{rule}.tsv",sep='\t')
        continue
        # print(info_dct_known["unique_elm_with_mutations"])
        # print(info_dct_predicted["unique_elm_with_mutations"])
        # print(info_dct_known_filtered["unique_elm_with_mutations"])
        # print(info_dct_predicted_filtered["unique_elm_with_mutations"])
        # proteins.append(info_dct_known_filtered)
        # compare_protein_sets(rule, info_dct_known, info_dct_known_filtered)
        # continue

        known_retention_motif_with_mutation = info_dct_known_filtered['ELM Motifs with ClinVar Position'] / info_dct_known['ELM Motifs with ClinVar Position'] * 100
        known_retention_motif = info_dct_known_filtered['Total ELM Motifs'] / info_dct_known['Total ELM Motifs'] * 100

        known_mutation_retention_motif = info_dct_known_filtered['Total ClinVar Position in ELM Regions'] / info_dct_known['Total ClinVar Position in ELM Regions'] * 100
        print(info_dct_known)
        print(info_dct_known_filtered)
        print("known_retention_motif_with_mutation",known_retention_motif_with_mutation)
        print("known_mutation_retention_motif",known_mutation_retention_motif)
        print("known_retention_motif",known_retention_motif)

        predicted_retention_motif_with_mutation = info_dct_predicted_filtered['ELM Motifs with ClinVar Position'] / info_dct_predicted['ELM Motifs with ClinVar Position'] * 100
        predicted_retention_motif = info_dct_predicted_filtered['Total ELM Motifs'] / info_dct_predicted['Total ELM Motifs'] * 100

        predicted_mutation_retention_motif = info_dct_predicted_filtered['Total ClinVar Position in ELM Regions'] / info_dct_predicted['Total ClinVar Position in ELM Regions'] * 100

        print(info_dct_predicted)
        print(info_dct_predicted_filtered)
        print("predicted_retention_motif_with_mutation", predicted_retention_motif_with_mutation)
        print("predicted_mutation_retention_motif", predicted_mutation_retention_motif)
        print("predicted_retention_motif", predicted_retention_motif)

        # Remove 'All Motif' from percentages
        known_percentages.pop('All Motif', None)
        predicted_percentages.pop('All Motif', None)

        # Prepare data for plotting
        known_labels = list(known_total_numbers.keys())
        known_total_values = list(known_total_numbers.values())
        predicted_labels = list(predicted_total_numbers.keys())
        predicted_total_values = list(predicted_total_numbers.values())

        # Define colors
        known_color = 'royalblue'
        predicted_color = 'salmon'


        # Figure 5 Motif Retention
        plot_totals_subplots(known_labels, known_total_values, known_color,
                             predicted_labels, predicted_total_values, predicted_color, figsize=(4, 4) ,rule=rule)

        # continue
        # Example usage

        print(known_labels, predicted_labels)

        column = 'Total ClinVar Position in ELM Regions'

        known_total_values = [info_dct_known[column],info_dct_known_filtered[column]]
        predicted_total_values = [info_dct_predicted[column],info_dct_predicted_filtered[column]]

        known_labels = ['All Motif','Filtered']
        predicted_labels = known_labels
        print(known_labels, predicted_labels)

        # Define colors
        known_color = 'royalblue'
        predicted_color = 'salmon'


        # Pathogenic Position in ELM
        plot_total_clinvar_positions_in_elm(known_labels, known_total_values,
                                            predicted_labels, predicted_total_values,
                                            known_color, predicted_color,
                                            figsize=(4, 4),title=f"Pathogenic Positions in ELM ({rule})")

        column = 'ELM Motifs with ClinVar Position'

        known_total_values = [info_dct_known[column],
                              info_dct_known_filtered[column]]
        predicted_total_values = [info_dct_predicted[column],
                                  info_dct_predicted_filtered[column]]

        print(known_labels, predicted_labels)

        # # Plot the graph
        plot_total_clinvar_positions_in_elm(known_labels, known_total_values,
                                            predicted_labels, predicted_total_values,
                                            known_color, predicted_color,
                                            figsize=(4, 4),title=f"ELM with Pathogenic Mutations ({rule})",ylabel="Number of Motifs")

    compare_protein_sets(rule, proteins[0], proteins[1])




def plot_abundance():
    # Load the data
    files = "/dlab/home/norbi/PycharmProjects/AlphaMissense_Stat/processed_data/files/elm"
    elm_known_with_am_info_all = pd.read_csv(f"{files}/elm_known_with_am_info_all_disorder.tsv", sep='\t')

    only_disorder_elm = elm_known_with_am_info_all["ELMIdentifier"].unique().tolist()

    elm_predicted_with_am_info_all = pd.read_csv(f"{files}/elm_predicted_with_am_info_all_disorder.tsv", sep='\t')
    elm_predicted_sequential_rule_filtered_with_am_info_all = pd.read_csv(f"{files}/elm_predicted_sequential_rule_filtered_proteins.tsv", sep='\t')

    # print("elm_predicted_with_am_info_all",elm_predicted_with_am_info_all.shape[0])
    # print("elm_predicted_sequential_rule_filtered_with_am_info_all",elm_predicted_sequential_rule_filtered_with_am_info_all.shape[0])
    #
    # elm_predicted_with_am_info_all = elm_known_with_am_info_all[elm_predicted_with_am_info_all["ELMIdentifier"].isin(only_disorder_elm)]
    # elm_predicted_sequential_rule_filtered_with_am_info_all = elm_predicted_sequential_rule_filtered_with_am_info_all[elm_predicted_sequential_rule_filtered_with_am_info_all["ELMIdentifier"].isin(only_disorder_elm)]
    #
    # print("elm_predicted_with_am_info_all", elm_predicted_with_am_info_all.shape[0])
    # print("elm_predicted_sequential_rule_filtered_with_am_info_all",
    #       elm_predicted_sequential_rule_filtered_with_am_info_all.shape[0])


    def one_plot(df, plot_type="Known"):
        # Plot 1: Count of ELMID (line plot)
        plt.figure(figsize=(10, 5))

        # Count the frequency of each ELMID
        elm_count = df['ELMIdentifier'].value_counts()

        # Line plot
        plt.subplot(1, 2, 1)
        sns.lineplot(x=range(1, len(elm_count) + 1), y=elm_count.values, marker='o')
        plt.title(f'Count of ELM Classes ({plot_type})')
        plt.xlabel('ELM Classes Rank')
        plt.ylabel('Count')
        plt.yscale('log')  # Log scale for better visualization

        # Plot 2: Frequency of counts with bins (bar plot)
        # Count the frequency of the count values
        elm_count_of_count = elm_count.value_counts()

        # Define fixed bins and labels
        bins = [0, 10, 20, 50, 100, 500, float('inf')]
        labels = ["1-10", "11-20", "21-50", "51-100", "101-500", ">500"]

        # Group the value counts into fixed bins
        elm_count_bins = pd.cut(elm_count_of_count.index, bins=bins, labels=labels, right=False)
        elm_count_binned = elm_count_of_count.groupby(elm_count_bins).sum()

        # Bar plot
        plt.subplot(1, 2, 2)
        sns.barplot(x=elm_count_binned.index, y=elm_count_binned.values, palette='viridis')
        plt.title(f'Frequency of Counts of ELM Classes ({plot_type})')
        plt.xlabel('Count of Motif per ELM Classes')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir,f"elm_frequency_{plot_type}.png"))
        plt.show()

    one_plot(elm_known_with_am_info_all, plot_type="Known")
    one_plot(elm_predicted_with_am_info_all, plot_type="Predicted")
    one_plot(elm_predicted_sequential_rule_filtered_with_am_info_all, plot_type="Predicted Filtered")

def plot_for_zsuzsa_results():
    # 1 Plot Base
    # Total Motif Predicted, After Disorder Filtering, After AM Key Residue Filtering
    plot_base_results()

    # 2 Plot Class Based Prediction Distribution
    # Print All Classes
    # Plot the Number of predicted motif for each classes (1500) Sorted
    # Plot this with abundance (how many predicted once,twice .. )
    # plot_abundance()

    # 3 Plot With Uncertain or Pathogenic Mutation
    # First plot is How many Mutations occuring in Any ELM motif (Key, Non-Key)


    return

if __name__ == "__main__":
    plot_dir  = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/plots/elm/"

    """
    Generate Filtered Known and Predicted ELM Motifs 
    """
    # generate_file()
    # generate_files_for_predictions()
    filter_files_for_disorder()
