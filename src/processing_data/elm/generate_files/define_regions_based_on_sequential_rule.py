import pandas as pd
import matplotlib.pyplot as plt
from  tqdm import tqdm
import numpy as np
import os


def get_disordered_regions(am_disorder_df):
    # Group the DataFrame by 'Protein_ID'
    grouped = am_disorder_df.groupby('Protein_ID')

    # Dictionary to store regions by protein
    protein_regions = {}
    protein_region_lst = []

    # Iterate over each group (protein)
    for protein, protein_df in tqdm(grouped, desc="Processing Proteins"):
        # Sort by 'Position'
        protein_df = protein_df.sort_values(by='Position')

        # Initialize variables
        regions = []
        current_region_start = None
        current_region_end = None

        # Get positions as a list
        positions = protein_df['Position'].tolist()

        # Iterate over the positions to create regions
        for position in positions:
            if current_region_start is None:
                # Start a new region
                current_region_start = position
                current_region_end = position
            elif position == current_region_end + sequential_region_difference:
                # Extend the current region
                current_region_end = position
            else:
                # Save the completed region
                regions.append((current_region_start, current_region_end))
                protein_region_lst.append([protein,current_region_start, current_region_end])
                # Start a new region
                current_region_start = position
                current_region_end = position

        # Append the last region if exists
        if current_region_start is not None:
            regions.append((current_region_start, current_region_end))
            protein_region_lst.append([protein,current_region_start, current_region_end])

        # Save to dictionary
        protein_regions[protein] = {
            'df': protein_df.reset_index(drop=True),
            'regions': regions
        }

    region_df = pd.DataFrame(protein_region_lst, columns=['Protein_ID', 'Start', 'End'])

    return protein_regions,region_df


def make_region_prediction(
    am_disorder_df,
    protein_regions,
    motif_lengths=(4, 20),
    sequential_region_difference_treshold=0.17,
    sequential_region_difference=1,
    extend_size=10
):
    """
    1) Slides through each disordered region with a 'while i <= end_idx' approach.
    2) If it finds a subregion (4..20) that exceeds threshold:
       - Perform an extended search (+10) for the best subregion
       - Mark that extended subregion as used
       - Set i to the end of the found subregion + 1
         (so we 'restart' scanning after the motif)
    3) If no subregion is found at position i, simply i += 1
    """

    relevant_regions = []

    for protein, data in tqdm(protein_regions.items(), desc="Processing Proteins for Motifs"):
        protein_df = data['df']
        regions = data['regions']

        positions = protein_df['Position'].values
        am_scores = protein_df['AlphaMissense'].values
        position_to_index = {pos: idx for idx, pos in enumerate(positions)}

        # Keep track of used positions to avoid overlapping picks
        used_positions = set()

        for region_start_pos, region_end_pos in regions:
            start_idx = position_to_index[region_start_pos]
            end_idx = position_to_index[region_end_pos]

            i = start_idx
            while i <= end_idx:
                # ------------------------------------------------------------
                # 1) If current position is already used, skip and move on.
                # ------------------------------------------------------------
                if i in used_positions:
                    i += 1
                    continue

                # ------------------------------------------------------------
                # 2) Try to find a subregion [4..20] starting at i
                #    that exceeds the threshold
                # ------------------------------------------------------------
                found_subregion_data = None
                best_diff = -float("inf")

                # The maximum sub_len we can try at this 'i'
                max_possible_sub_len = min(motif_lengths[1], (end_idx - i + 1))
                min_sub_len = motif_lengths[0]

                for sub_len in range(max_possible_sub_len, min_sub_len - 1, -1):
                    sub_end = i + sub_len - 1
                    if sub_end > end_idx:
                        # Past the region boundary
                        continue

                    subrange_indices = np.arange(i, i + sub_len)
                    subrange_positions = positions[subrange_indices]

                    # If any of these positions are used, skip
                    if used_positions.intersection(subrange_positions):
                        continue

                    subrange_scores = am_scores[subrange_indices]
                    region_mean = np.mean(subrange_scores)

                    pre_start = i - sub_len
                    post_end = i + 2 * sub_len - 1  # i + sub_len + (sub_len - 1)

                    if (pre_start >= 0) and (post_end < len(am_scores)):
                        surrounding_scores = np.concatenate([
                            am_scores[pre_start : i],
                            am_scores[sub_end + 1 : post_end + 1]
                        ])
                        surrounding_mean = np.mean(surrounding_scores)

                        diff = region_mean - surrounding_mean
                        if diff > sequential_region_difference_treshold and diff > best_diff:
                            best_diff = diff
                            found_subregion_data = {
                                'start_idx': i,
                                'end_idx': sub_end,
                                'length': sub_len,
                                'region_mean': region_mean,
                                'surrounding_mean': surrounding_mean,
                                'score_diff': diff
                            }

                # ------------------------------------------------------------
                # 3) If we found a subregion at i, do the extended search
                # ------------------------------------------------------------
                if found_subregion_data is not None:
                    extended_region_data = extended_subregion_search(
                        found_subregion_data,
                        extend_size,
                        positions,
                        am_scores,
                        used_positions,
                        motif_lengths,
                        sequential_region_difference_treshold
                    )

                    # If extended search yields a final pick, record it
                    if extended_region_data is not None:
                        relevant_regions.append({
                            'Protein_ID': protein,
                            'Start': extended_region_data['Start'],
                            'End':   extended_region_data['End'],
                            'Length': extended_region_data['Length'],
                            'Region_Mean_Score': extended_region_data['Region_Mean_Score'],
                            'Surrounding_Mean_Score': extended_region_data['Surrounding_Mean_Score'],
                            'Score_Difference': extended_region_data['Score_Difference']
                        })

                        # print(extended_region_data, i,region_start_pos,region_end_pos)
                        # Mark used positions so we don't pick them again
                        for pos in range(extended_region_data['Start'], extended_region_data['End'] + 1):
                            used_positions.add(position_to_index[pos])

                        # ----------------------------------------------------
                        # 4) Move i to the end of the found extended subregion + 1
                        # ----------------------------------------------------
                        i = position_to_index[ extended_region_data['End'] ] + 1
                    else:
                        # If no valid extended region found (unlikely, but possible),
                        # just move i forward
                        i += 1
                else:
                    # No subregion found at i; move to next residue
                    i += 1

    # Create DataFrame from results
    relevant_regions_df = pd.DataFrame(relevant_regions)
    if not relevant_regions_df.empty:
        relevant_regions_df.sort_values(by=['Protein_ID', 'Start'], inplace=True)

    print(relevant_regions_df)
    return relevant_regions_df


def extended_subregion_search(
    initial_subregion_data,
    extend_size,
    positions,
    am_scores,
    used_positions,
    motif_lengths,
    threshold
):
    """
    Once we find an initial subregion that passes the threshold,
    do an extended check in the range:
        [start_idx, end_idx + extend_size]
    to find the 'best' subregion (length in [motif_lengths[0]..motif_lengths[1]])
    that yields the highest (region_mean - surrounding_mean).
    """

    # The initial subregion is described by (start_idx, end_idx).
    init_start_idx = initial_subregion_data['start_idx']
    init_end_idx   = initial_subregion_data['end_idx']
    init_length    = initial_subregion_data['length']

    # Extended boundary (clip if needed)
    extended_search_start = init_start_idx
    extended_search_end = init_end_idx + extend_size

    # We'll also ensure we don't exceed the actual am_scores length
    max_index = len(am_scores) - 1
    extended_search_end = min(extended_search_end, max_index)


    min_len, max_len = motif_lengths
    best_diff = -float('inf')
    best_data = None

    # Slide over this extended region
    i = extended_search_start
    while i <= extended_search_end:
        if i in used_positions:
            i += 1
            continue

        # For each possible sub_len in [min_len..max_len]
        for sub_len in range(max_len, min_len - 1, -1):
            sub_end = i + sub_len - 1
            if sub_end > extended_search_end:
                continue

            # Gather positions
            subrange_indices = np.arange(i, i + sub_len)
            subrange_positions = positions[subrange_indices]

            # If any used => skip
            if used_positions.intersection(subrange_positions):
                continue

            # to avoid not continuous regions
            expected = list(range(subrange_positions[0], subrange_positions[0] + len(subrange_positions)))
            if list(subrange_positions) != expected:
                continue

            subrange_scores = am_scores[subrange_indices]
            region_mean = np.mean(subrange_scores)

            pre_start = i - sub_len
            post_end = i + 2 * sub_len - 1

            # print(f"""Stats:
            #     pre_start: {pre_start}
            #     post_end: {post_end}
            #     sub_len: {sub_len}
            #     subrange_positions: {subrange_positions}
            # """)

            if pre_start >= 0 and post_end <= max_index:
                surrounding_scores = np.concatenate([
                    am_scores[pre_start : i],
                    am_scores[sub_end + 1 : post_end + 1]
                ])
                surrounding_mean = np.mean(surrounding_scores)

                diff = region_mean - surrounding_mean
                if diff > best_diff:
                    best_diff = diff
                    best_data = {
                        'Start': positions[i],
                        'End':   positions[sub_end],
                        'Length': sub_len,
                        'Region_Mean_Score': region_mean,
                        'Surrounding_Mean_Score': surrounding_mean,
                        'Score_Difference': diff
                    }
        i += 1

    if best_data and best_data['Score_Difference'] > threshold:
        return best_data
    else:
        # Return None if we didn't find anything better than threshold
        return None


# def make_region_prediction(
#     am_disorder_df,
#     protein_regions,
#     motif_lengths=(4, 20),
#     sequential_region_difference_treshold=0.17,
#     sequential_region_difference=1,
#     max_extend_length=50
# ):
#     """
#     Modified version:
#       - Extends the maximum subregion length up to `max_extend_length`.
#       - Chooses the subregion (within 4..50 in length) that yields the maximum difference.
#     """
#
#     # List to store relevant regions
#     relevant_regions = []
#
#     # Iterate over proteins and their regions
#     for protein_data in tqdm(protein_regions.items(), desc="Processing Proteins for Motifs"):
#         protein, data = protein_data
#         protein_df = data['df']
#         regions = data['regions']
#
#         positions = protein_df['Position'].values
#         am_scores = protein_df['AlphaMissense'].values
#         position_to_index = {pos: idx for idx, pos in enumerate(positions)}
#
#         # Set to keep track of used positions to avoid overlaps
#         used_positions = set()
#
#         for region in regions:
#             region_start_pos, region_end_pos = region
#             # Get indices for the start and end positions
#             start_idx = position_to_index[region_start_pos]
#             end_idx = position_to_index[region_end_pos]
#
#             region_length = end_idx - start_idx + 1
#
#             # We will explore subregion lengths from motif_lengths[0] up to `max_extend_length`,
#             # but never exceed the actual region length.
#             min_length = motif_lengths[0]
#             max_length = min(region_length, max_extend_length)
#
#             # Track the best subregion for this region
#             best_score_difference = -float("inf")
#             best_subregion_indices = None
#             best_subregion_positions = None
#             best_region_mean = None
#             best_surrounding_mean = None
#
#             # Try all subregion lengths from min_length up to max_length
#             for sub_len in range(min_length, max_length + 1):
#                 # Generate all possible windows of size sub_len within [start_idx, end_idx]
#                 if (end_idx - start_idx + 1) < sub_len:
#                     break  # sub_len is bigger than the region, stop trying bigger lengths
#
#                 window_indices = np.arange(start_idx, end_idx - sub_len + 2)
#
#                 for i in window_indices:
#                     subregion_indices = np.arange(i, i + sub_len)
#                     subregion_positions = positions[subregion_indices]
#
#                     # Check if any of the positions are already used
#                     if used_positions.intersection(subregion_positions):
#                         continue
#
#                     subregion_scores = am_scores[subregion_indices]
#                     region_mean = np.mean(subregion_scores)
#
#                     # Get surrounding sequences
#                     pre_start = i - sub_len
#                     post_end = i + 2 * sub_len
#
#                     # Ensure we have enough sequence on both sides
#                     # (Only calculate if we stay in bounds)
#                     if pre_start >= 0 and post_end <= len(am_scores):
#                         surrounding_scores = np.concatenate([
#                             am_scores[pre_start:i],
#                             am_scores[i + sub_len:post_end]
#                         ])
#                         surrounding_mean = np.mean(surrounding_scores)
#
#                         # Calculate the difference
#                         difference = region_mean - surrounding_mean
#
#                         # Update the best subregion if we found a bigger difference
#                         if difference > best_score_difference:
#                             best_score_difference = difference
#                             best_subregion_indices = subregion_indices
#                             best_subregion_positions = subregion_positions
#                             best_region_mean = region_mean
#                             best_surrounding_mean = surrounding_mean
#
#             # After exploring all subregion lengths from min_length..max_length,
#             # check if the best difference is above threshold
#             if best_subregion_indices is not None and best_score_difference > sequential_region_difference_treshold:
#                 # We found a subregion that has the maximum difference above threshold
#                 relevant_regions.append({
#                     'Protein_ID': protein,
#                     'Start': best_subregion_positions[0],
#                     'End': best_subregion_positions[-1],
#                     'Length': len(best_subregion_indices),
#                     'Region_Mean_Score': best_region_mean,
#                     'Surrounding_Mean_Score': best_surrounding_mean,
#                     'Score_Difference': best_score_difference
#                 })
#
#                 # Mark positions as used (to avoid overlaps)
#                 used_positions.update(best_subregion_positions)
#
#     # Create DataFrame from the results
#     relevant_regions_df = pd.DataFrame(relevant_regions)
#
#     print(relevant_regions_df)
#
#     # Sort by Protein_ID and Start position
#     relevant_regions_df.sort_values(by=['Protein_ID', 'Start'], inplace=True)
#
#     return relevant_regions_df



def plot_motif_counts_distribution(relevant_regions_df):
    """
    Plots the distribution of predicted motif counts per protein.
    If a protein has more than 20 motifs, it is grouped into the '20+' category.
    """
    # Count the number of motifs predicted per protein
    motif_counts = relevant_regions_df['Protein_ID'].value_counts()

    # Group counts greater than 20 into '20+'
    motif_counts_capped = motif_counts.clip(upper=20)

    # Prepare the data for plotting
    motif_counts_distribution = motif_counts_capped.value_counts().sort_index()

    # Rename the last bin to '20+'
    if 20 in motif_counts_distribution.index:
        motif_counts_distribution.rename(index={20: '20+'}, inplace=True)

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    motif_counts_distribution.plot(kind='bar')
    plt.title('Distribution of Predicted Motif Counts per Protein')
    plt.xlabel('Number of Motifs Predicted in a Protein')
    plt.ylabel('Number of Proteins')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_motifs_by_length_distribution(relevant_regions_df):
    """
    Plots the distribution of predicted motifs by their length.
    """
    # Optionally, group by region length
    motifs_by_length = relevant_regions_df['Length'].value_counts().sort_index()

    # Plot the distribution of motifs by their length
    plt.figure(figsize=(10, 6))
    motifs_by_length.plot(kind='bar')
    plt.title('Distribution of Predicted Motifs by Length')
    plt.xlabel('Motif Length')
    plt.ylabel('Number of Motifs')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_disordered_regions_with_motifs_percentage(protein_regions, relevant_regions_df):
    """
    Plots the percentage of disordered regions that have at least one predicted motif region.
    """
    # Initialize counts
    total_disordered_regions = 0
    disordered_regions_with_motifs = 0

    # Iterate over each protein
    for protein, data in protein_regions.items():
        disordered_regions = data['regions']
        total_disordered_regions += len(disordered_regions)

        # Get motif regions for this protein
        motif_regions = relevant_regions_df[relevant_regions_df['Protein_ID'] == protein][
            ['Start', 'End']].values.tolist()

        # Check which disordered regions contain motifs
        for d_start, d_end in disordered_regions:
            # Flag to check if the current disordered region has a motif
            has_motif = False
            for m_start, m_end in motif_regions:
                # Check if motif region is within the disordered region
                if m_start >= d_start and m_end <= d_end:
                    has_motif = True
                    break  # No need to check other motifs
            if has_motif:
                disordered_regions_with_motifs += 1

    # Ensure counts are non-negative
    disordered_regions_without_motifs = total_disordered_regions - disordered_regions_with_motifs
    disordered_regions_without_motifs = max(disordered_regions_without_motifs, 0)

    # Calculate the percentage
    percentage = (disordered_regions_with_motifs / total_disordered_regions) * 100 if total_disordered_regions > 0 else 0
    print(f"Percentage of disordered regions with predicted motifs: {percentage:.2f}%")

    # Prepare data for plotting
    data = {
        'Regions': ['Disordered Regions with Motifs', 'Disordered Regions without Motifs'],
        'Count': [disordered_regions_with_motifs, disordered_regions_without_motifs]
    }

    percentage_df = pd.DataFrame(data)

    # Plot the percentage
    plt.figure(figsize=(6, 6))
    plt.pie(
        percentage_df['Count'],
        labels=percentage_df['Regions'],
        autopct='%1.1f%%',
        startangle=140
    )
    plt.title('Percentage of Disordered Regions with Predicted Motifs')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def positional_distribution_of_region_of_interest(relevant_regions_df,am_disorder_df):

    # Calculations for the pie chart
    relevant_region_positions = relevant_regions_df["Length"].sum()
    total_positions = am_disorder_df.shape[0]
    non_relevant_positions = total_positions - relevant_region_positions

    # Data for pie chart
    labels = ["Predicted Disorder Functional Region", "Disorder Region"]
    sizes = [relevant_region_positions, non_relevant_positions]

    # Plotting the pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Proportion of Predicted Motif Positions")
    plt.show()

if __name__ == "__main__":
    """
    How many motif can be in Human Proteome based on this filtering?
    """

    base_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations'

    files = os.path.join(base_dir,"processed_data/files/elm")

    am_disorder = os.path.join(base_dir,"processed_data/files/alphamissense/am_disorder.tsv")

    am_disorder_df = pd.read_csv(am_disorder, sep='\t')

    # Parameters
    sequential_region_difference = 1
    motif_lengths = (4, 20)
    sequential_region_difference_treshold = 0.15

    protein_regions,region_df = get_disordered_regions(am_disorder_df)
    # relevant_regions_df = make_region_prediction(am_disorder_df,protein_regions,motif_lengths=motif_lengths,sequential_region_difference_treshold=sequential_region_difference_treshold,sequential_region_difference=sequential_region_difference)

    # Save to TSV
    predicted_regions_path = f"{files}/predicted_motif_region_by_am_sequential_rule.tsv"
    # relevant_regions_df.to_csv(predicted_regions_path, sep='\t', index=False)

    # exit()

    relevant_regions_df = pd.read_csv(predicted_regions_path, sep='\t')

    # Generate the plots
    plot_motif_counts_distribution(relevant_regions_df)
    plot_motifs_by_length_distribution(relevant_regions_df)
    print(relevant_regions_df['Protein_ID'].nunique())
    print(relevant_regions_df.shape[0])
    print(region_df.shape[0])
    plot_disordered_regions_with_motifs_percentage(protein_regions, relevant_regions_df)
    positional_distribution_of_region_of_interest(relevant_regions_df,am_disorder_df)