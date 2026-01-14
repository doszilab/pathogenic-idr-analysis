import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm
import numpy as np
import tempfile
import subprocess

def generate_alphamissense_score_for_elm_optimized(region_df, alphamissense_df,max_sequenital=10):
    lst = []

    only_elm_am = alphamissense_df[alphamissense_df['Protein_ID'].isin(region_df['Protein_ID'])]

    # Precompute Protein_ID and Position mask in alphamissense_df for faster access
    alphamissense_dict = only_elm_am.groupby(['Protein_ID', 'Position'])['AlphaMissense'].mean().to_dict()

    protein_lengths = only_elm_am.groupby('Protein_ID')['Position'].max().to_dict()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        for index, row in tqdm(region_df.iterrows(), total=len(region_df)):
            start, end, protein = row['Start'], row['End'], row['Protein_ID']
            motif_length = end - start + 1
            protein_length = protein_lengths.get(protein, np.nan)

            # Calculate motif mean score
            region_scores = [
                alphamissense_dict.get((protein, pos), np.nan) for pos in range(start, end + 1)
            ]
            motif_am_score = np.nanmean(region_scores)
            # Final motif score assignment
            row['motif_am_mean_score'] = motif_am_score
            row['motif_am_scores'] =  ", ".join(map(str, region_scores))

            # Sequential environment scores (in place, no new DataFrame)
            forward_positions = range(end + 1, min(end + motif_length + 1,end + max_sequenital + 1))
            backward_positions = range(max(start - motif_length,start - max_sequenital), start)

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

def clean_mobidb(df):
    print('MOBIDB Process')
    df = df[['Protein_ID', 'feature', 'start..end']].copy()
    df["start..end"] = df['start..end'].str.split(',')
    df = df.explode("start..end")
    df["start..end"] = df["start..end"].apply(lambda x: x.split('.', 2)).apply(lambda x: [x[0], x[-1]] if len(x) == 3 else x)
    df["start..end"] = df["start..end"].apply(lambda x: ','.join(x)).str.split(',')
    df[['Start', 'End']] = pd.DataFrame(df["start..end"].tolist(), index=df.index)
    df['Data'] = "Exp. Dis"
    return df

def generate_files_for_predictions(core_dir,functional_path):

    pos = os.path.join(core_dir, "data/discanvis_base_files/positional_data_process")
    functional = os.path.join(core_dir, "data/discanvis_base_files/functional")
    elm = os.path.join(core_dir, "data/discanvis_base_files/elm/proteome_scann/known_instances_with_match.tsv")
    pfam = os.path.join(core_dir, "data/discanvis_base_files/pfam")
    alldata = pd.read_csv(
        f'{core_dir}/data/discanvis_base_files/sequences/loc_chrom_with_names_isoforms_with_seq.tsv',
        sep='\t', header=0)
    print(alldata.columns)
    main_isoforms = alldata[alldata['main_isoform'] == "yes"]

    alphamissense_pos = f'{pos}/alphamissense_pos.tsv'
    alphamissense_df = extract_pos_based_df(pd.read_csv(alphamissense_pos, sep='\t'))
    print(alphamissense_df)

    # Regions
    # MFIB, DIBS, PhasePro, UniProt ROI, PDB, Exp. Dis, ELM
    dibs_mfib_phasepro_df = pd.read_csv(os.path.join(functional, 'binding_mfib_phasepro_dibs.tsv'),sep='\t').rename(columns={"Accession":"Protein_ID"})
    dibs_mfib_phasepro_df = dibs_mfib_phasepro_df[dibs_mfib_phasepro_df['Data'] != "binding"]
    mobidb_df = clean_mobidb(pd.read_csv(os.path.join(functional, 'mobidb_disorders.tsv'),sep='\t'))

    # ELM
    elm_df = pd.read_csv(elm, sep='\t')
    elm_df['Data'] = "ELM"

    pfam_df = pd.read_csv(os.path.join(pfam, 'pfamtable.tsv'),sep='\t')
    pfam_df['Start'] = pfam_df['envelope_start']
    pfam_df['End'] = pfam_df['envelope_end']
    pfam_df['Data'] = 'Pfam'
    pfam_df = pfam_df[pfam_df['type'] == "Domain"]

    columns = ["Protein_ID", "Start", "End", "Data"]
    functional = pd.concat([dibs_mfib_phasepro_df,mobidb_df,pfam_df,elm_df],ignore_index=True)[columns].drop_duplicates()
    functional['Start'] = functional['Start'].astype(int)
    functional['End'] = functional['End'].astype(int)

    print(functional['Data'].unique())
    # exit()

    functional = functional[functional['Protein_ID'].isin(main_isoforms['Protein_ID'].unique().tolist())]

    print(functional['Data'].unique())
    # exit()
    functionals_with_am_scores = generate_alphamissense_score_for_elm_optimized(functional,alphamissense_df)
    functionals_with_am_scores.to_csv(f"{core_dir}/processed_data/files/pip/functional_region.tsv",sep='\t', index=False)


def plot_difference_lineplot_by_category(df, figsize=(8, 6), filepath=None):
    """
    Plot a line (density) plot of the difference between Motif and Sequential AlphaMissense scores
    for each category (based on the 'Data' column) in one figure.

    Parameters:
        df (pd.DataFrame): DataFrame containing the 'Region_vs_Sequential_Difference' and 'Data' columns.
        figsize (tuple): Size of the figure.
        filepath (str): If provided, the figure will be saved to this file.
    """
    plt.figure(figsize=figsize)

    # Iterate over each unique category
    for category in df['Data'].unique():
        subset = df[df['Data'] == category]
        # Plot a density line plot using seaborn.kdeplot
        sns.kdeplot(
            subset['Region_vs_Sequential_Difference'],
            label=category,
            linewidth=2,
            fill=False  # Only the line plot, no fill
        )

    # Add vertical line at 0
    plt.axvline(0, color='black', linestyle='--', linewidth=1)

    plt.xlabel("Difference (Region - Flanking)")
    plt.ylabel("Density")
    plt.title("Score Differences")

    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if filepath:
        plt.savefig(filepath, bbox_inches='tight')
    plt.show()

def make_fasta_file(df):
    fasta = ''
    for index, row in df.iterrows():
        fasta += f'>{row["ID"]}\n{row["Sequence"]}\n'
    return fasta


def create_groups(matches_df,df):


    protein_df = matches_df.copy()

    passed_proteins = []

    for prot in protein_df['ID'].unique():
        current_prot = protein_df[protein_df['ID'] == prot]
        if current_prot.shape[0] == 1:
            passed_proteins.append([prot,'Group'])
            continue

        this_df = df[df['ID'].isin(current_prot['ID_match'])]
        longest_protein = this_df.loc[this_df['Length'].idxmax()]
        passed_proteins.append([longest_protein['ID'], 'Length'])

    passed_protein_df = pd.DataFrame(passed_proteins, columns=['ID','Passed_by'])

    non_redundant_df = df[df['ID'].isin(passed_protein_df['ID'])]

    return non_redundant_df

# def make_redundancy_filter(df):
#     categories = df['Data'].unique()
#
#     main_file_path = os.path.join(core_dir, "data", "discanvis_base_files", "sequences",
#                                   "loc_chrom_with_names_main_isoforms_with_seq.tsv")
#     main_df = pd.read_csv(main_file_path, sep="\t")
#
#     sequences_df = df.merge(main_df[['Protein_ID', 'Sequence']], how="left", on="Protein_ID")
#     sequences_df['Sequence'] = sequences_df.apply(
#         lambda row: row['Sequence'][row['Start'] - 1: row['End']], axis=1
#     )
#     sequences_df['ID'] = sequences_df["Protein_ID"] + "-" + sequences_df["Start"].astype(str) + "-" + sequences_df["End"].astype(str)
#     sequences_df['Length'] = sequences_df['End'] - sequences_df['Start'] + 1
#
#
#     matches_lst = []
#
#     for category in categories:
#         current_set = sequences_df[sequences_df['Data'] == category]
#         fasta = make_fasta_file(current_set)
#
#         # Create a temporary file for the FASTA data
#         with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.fasta') as temp_fasta_file:
#             fasta_path = temp_fasta_file.name
#             temp_fasta_file.write(fasta)
#
#         with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.fasta') as temp_fasta_file:
#             output_path = temp_fasta_file.name
#
#         make_db_command = f"/home/nosyfire/diamond makedb --in {fasta_path} -d {output_path}"
#         subprocess.call(make_db_command, shell=True)
#
#         with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.fasta') as temp_fasta_file:
#             match_path = temp_fasta_file.name
#
#         # Run the Diamond database creation using the temporary FASTA file
#         run_blast_command = f'/home/nosyfire/diamond cluster -d {output_path} --approx-id {40} -o {match_path} -M 32G'
#         subprocess.call(run_blast_command, shell=True)
#
#         matches_df = pd.read_csv(match_path, sep='\t', header=None)
#         matches_df.columns = ['ID', 'ID_match']
#
#         os.remove(output_path)
#         os.remove(fasta_path)
#         os.remove(match_path)
#
#         non_redundant_df = create_groups(matches_df, current_set)
#         matches_lst.append(non_redundant_df)
#
#     final_df = pd.concat(matches_lst)
#
#     return final_df


# Optimized function for filtering redundancy per unique Data category
def make_redundancy_filter(df):
    """
    Filters redundant regions within each protein and unique Data category, keeping only the longest overlapping region.
    The original region boundaries (Start, End) are preserved.

    Parameters:
    - df: DataFrame containing 'Protein_ID', 'Start', 'End', 'Data', and other relevant columns.

    Returns:
    - Filtered DataFrame with non-redundant regions per protein and Data category.
    """
    categories = df['Data'].unique()  # Extract unique categories
    matches_lst = []

    for category in tqdm(categories):
        print(category)
        current_set = df[df['Data'] == category].copy()

        # Ensure DataFrame is sorted by Protein_ID, Start position
        current_set = current_set.sort_values(by=["Protein_ID", "Start"])

        # Initialize list to store non-redundant regions for this category
        filtered_regions = []

        # Process each protein separately
        for protein, group in current_set.groupby("Protein_ID"):
            sorted_regions = group.sort_values(by=["Start", "End"], ascending=[True, False])
            selected_regions = []  # Store non-redundant selected regions

            while not sorted_regions.empty:
                # Take the longest region from the overlapping set
                longest_region = sorted_regions.iloc[0]
                selected_regions.append(longest_region.to_dict())

                # Get its start and end
                l_start, l_end = longest_region["Start"], longest_region["End"]

                # Find all overlapping regions with this selected region
                overlap_mask = (sorted_regions["Start"] <= l_end) & (sorted_regions["End"] >= l_start)
                sorted_regions = sorted_regions[~overlap_mask]  # Remove those that were merged

            # Append filtered results for this category
            matches_lst.append(pd.DataFrame(selected_regions))


        # Append filtered results for this category
        matches_lst.append(pd.DataFrame(filtered_regions))

    # Combine all category results
    final_df = pd.concat(matches_lst, ignore_index=True)

    return final_df


if __name__ == '__main__':
    core_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations'
    functional_path = f"{core_dir}/processed_data/files/pip/functional_region.tsv"
    # generate_files_for_predictions(core_dir,functional_path)

    functional_df = pd.read_csv(functional_path, sep='\t')
    print(functional_df['Data'].unique())

    functional_df['Region_vs_Sequential_Difference'] = functional_df['motif_am_mean_score'] - functional_df[
        'sequential_am_score']

    plot_difference_lineplot_by_category(functional_df, figsize=(5, 3))

    functional_df_only_scores = functional_df[['Protein_ID','Start','End','Data','Region_vs_Sequential_Difference','motif_am_mean_score','sequential_am_score']]

    functional_df_only_scores.to_csv(f"{core_dir}/processed_data/files/pip/functional_region_with_score.tsv",sep='\t')

    functional_df_non_redundant = make_redundancy_filter(functional_df_only_scores)

    functional_df_non_redundant.to_csv(f"{core_dir}/processed_data/files/pip/functional_non_redundant_region_with_score.tsv",sep='\t')
    plot_difference_lineplot_by_category(functional_df_non_redundant, figsize=(5, 3))