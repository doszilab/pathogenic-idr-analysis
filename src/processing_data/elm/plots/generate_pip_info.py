import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def get_disordered_regions(am_disorder_df,sequential_region_difference = 1):
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

def plot_totals(ax, labels, total_values, color, text="Known",ylabel='Number of Motifs'):
    # Sort by total values in descending order
    sorted_indices = sorted(range(len(total_values)), key=lambda i: total_values[i], reverse=True)
    labels = [labels[i] for i in sorted_indices]
    total_values = [total_values[i] for i in sorted_indices]

    bars = ax.bar(range(len(labels)), total_values, width=0.4, color=color, label=text, align='center')

    for i, bar in enumerate(bars):
        number = total_values[i]
        t = f"{number}" if number < 10_000 else f"{number / 1_000:.0f}k"

        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{t}', ha='center', va='bottom')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(text, pad=10)

    for spine in ax.spines.values():
        spine.set_visible(False)

def plot_combined(
        known_labels,known_total_values, disorder_pos, functional_color="#FF9999", non_functional_color="#66B2FF",fig_dir=None,title="Position Coverage"
):
    """
    Creates two donut charts as subplots:
    1. Percentage of disordered regions with motifs.
    2. Proportion of predicted motif positions.
    """

    percentages = [round(x / disorder_pos  * 100,2)  for x in known_total_values]


    # Plot combined subplots
    fig, axes = plt.subplots(1, 1, figsize=(4, 3))

    plot_totals(axes, known_labels, percentages, functional_color, text="Predicted",ylabel='Percent of Coverage')

    axes.set_title("Predicted")
    # Add a shared title
    plt.suptitle(title)

    # Adjust layout
    plt.tight_layout()
    if fig_dir:
        plt.savefig(f"{fig_dir}/combined_disorder_donut_chart.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_region(
        known_labels,known_total_values, functional_color="#FF9999", non_functional_color="#66B2FF",fig_dir=None,title="Position Coverage"
):
    """
    Creates two donut charts as subplots:
    1. Percentage of disordered regions with motifs.
    2. Proportion of predicted motif positions.
    """


    # Plot combined subplots
    fig, axes = plt.subplots(1, 1, figsize=(4, 3))

    plot_totals(axes, known_labels, known_total_values, functional_color, text="Predicted",ylabel='Percent of Coverage')

    axes.set_title("Predicted")
    # Add a shared title
    plt.suptitle(title)

    # Adjust layout
    plt.tight_layout()
    if fig_dir:
        plt.savefig(f"{fig_dir}/combined_disorder_donut_chart.png", dpi=300, bbox_inches='tight')
    plt.show()

def bigplot(labels,postions,regions,motifs,pip_regions_with_motif_list,disorder_pos, functional_color="#FF9999", non_functional_color="#66B2FF",
            fig_dir=None, title="Position Coverage"):
    """
    Creates two donut charts as subplots:
    1. Percentage of disordered regions with motifs.
    2. Proportion of predicted motif positions.
    """

    percentages_positions = [round(x / disorder_pos * 100, 2) for x in postions]

    # Plot combined subplots
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))

    plot_totals(axes[0,0], labels, percentages_positions, functional_color, text="Disorder Position Coverage", ylabel='Percent of Coverage')
    plot_totals(axes[0,1], labels, regions, functional_color, text="Number of Regions", ylabel='Number of Regions')
    plot_totals(axes[1,0], labels, pip_regions_with_motif_list, functional_color, text="Number of Regions with Motifs", ylabel='Number of Regions')
    plot_totals(axes[1,1], labels, motifs, functional_color, text="Number of Motifs", ylabel='Number of Motifs')

    # axes[0,0].set_title("Disorder Position Coverage")
    # axes[0,1].set_title("Number of Regions")
    # axes[1,0].set_title("Number of Regions with Motifs")
    # axes[1,1].set_title("Number of Motifs")


    # Add a shared title
    plt.suptitle(title)

    # Adjust layout
    plt.tight_layout()
    if fig_dir:
        plt.savefig(f"{fig_dir}/combined_disorder_donut_chart.png", dpi=300, bbox_inches='tight')
    plt.show()


def get_disordered_occurrence(protein_regions, relevant_regions_df):
    # Initialize counts for the first donut chart
    total_disordered_regions = 0
    disordered_regions_with_motifs = 0

    # Iterate over each protein
    for protein, data in tqdm(protein_regions.items()):
        disordered_regions = data['regions']
        total_disordered_regions += len(disordered_regions)

        # Get motif regions for this protein
        motif_regions = relevant_regions_df[relevant_regions_df['Protein_ID'] == protein][
            ['Start', 'End']].values.tolist()

        # Check which disordered regions contain motifs
        for d_start, d_end in disordered_regions:
            has_motif = any(m_start >= d_start and m_end <= d_end for m_start, m_end in motif_regions)
            if has_motif:
                disordered_regions_with_motifs += 1

    return total_disordered_regions, disordered_regions_with_motifs


def plot_region_metrics(percentages_positions,percentages_regions,functional_color,plot_dir=None):
    # Plot combined subplots
    fig, axes = plt.subplots(1, 2, figsize=(4, 4))

    plot_totals(axes[0], ["Positions", "Occurrence"], percentages_positions, functional_color, text="IDR Coverage",
                ylabel='Percent of Coverage')
    plot_totals(axes[1], ["Motifs", "Non-Motifs"], percentages_regions, functional_color,
                text="PIP Regions", ylabel='Number of Regions')
    axes[0].set_yticks([])
    axes[0].set_ylim(0, 100)
    axes[1].set_yticks([])

    # Add a shared title
    plt.suptitle("PIP Regions Metrics")

    # Adjust layout
    plt.tight_layout()
    if plot_dir:
        plt.savefig(os.path.join(plot_dir,"pip_region_metrics.png"),dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    base_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations'
    plot_dir = os.path.join(base_dir,'plots/fig5')
    am_disorder = os.path.join(base_dir, "processed_data/files/alphamissense/am_disorder.tsv")
    am_disorder_df = pd.read_csv(am_disorder, sep='\t')
    disorder_pos = am_disorder_df.shape[0]

    file_dir = os.path.join(base_dir, "processed_data", "files")

    main_file_path = os.path.join(base_dir, "data", "discanvis_base_files", "sequences",
                                  "loc_chrom_with_names_main_isoforms_with_seq.tsv")
    main_df = pd.read_csv(main_file_path, sep="\t")


    pips = [
        ['predicted_motif_region_by_kadane_0_15_extended.tsv','motifs_on_pip_extension_with_am_scores.tsv', 'PIP Extension', False],
    ]

    labels = []
    postions = []
    regions = []
    motifs = []

    protein_regions, region_df = get_disordered_regions(am_disorder_df)

    pip_regions_with_motif_list = []

    functional_color = "#FF9999"

    for file_name,motif, name, extension in pips:
        # Load PIP Regions
        pip_path = os.path.join(file_dir, "pip", file_name)
        pip_df = pd.read_csv(pip_path, sep="\t")
        # Filter only Kadane if that's relevant
        pip_df = pip_df[pip_df['Found_by'] == "Kadane"].copy()
        pip_df.reset_index(drop=True, inplace=True)
        pip_df['pip_region_id'] = pip_df.index  # unique ID for each region row

        # Load Motifs
        motif_path = os.path.join(file_dir, "pip", motif)
        motif_df = pd.read_csv(motif_path, sep="\t").copy()
        motif_df.reset_index(drop=True, inplace=True)
        motif_df['motif_id'] = motif_df.index  # unique ID for each motif row

        sequences_df = pip_df.merge(main_df[['Protein_ID', 'Sequence']], how="left", on="Protein_ID")
        sequences_df['Seq_Length'] = sequences_df['Sequence'].str.len()

        elm_uniqeu_positions_df = (
            sequences_df[['Protein_ID', 'Start', 'End']]
            .drop_duplicates()
            .assign(Positions=lambda df: df.apply(lambda row: range(row['Start'], row['End'] + 1), axis=1))
            .explode('Positions')
            .rename(columns={'Positions': 'Position'})
            .loc[:, ['Protein_ID', 'Position']]
            .drop_duplicates()
        )
        number_of_elm_position = elm_uniqeu_positions_df.shape[0]

        # 2) Check which PIP regions overlap at least one motif
        # Merge on Protein_ID only, then check for intervals that actually overlap
        merged = pip_df.merge(motif_df, on='Protein_ID', suffixes=('_pip', '_motif'))

        # Define an "overlap" column
        merged['overlap'] = merged.apply(
            lambda row: (
                    (row['Start_pip'] <= row['End_motif']) and
                    (row['Start_motif'] <= row['End_pip'])
            ),
            axis=1
        )

        # Which pip_region_ids actually overlap a motif?
        overlapping_pip_ids = set(merged.loc[merged['overlap'], 'pip_region_id'])
        num_pip_with_motif = len(overlapping_pip_ids)
        num_pip_without_motif = pip_df.shape[0] - num_pip_with_motif
        pip_regions_with_motif_list.append(num_pip_with_motif)

        postions.append(number_of_elm_position)
        regions.append(pip_df.shape[0])
        motifs.append(motif_df.shape[0])

        total_disordered_regions, disordered_regions_with_motifs = get_disordered_occurrence(protein_regions, motif_df)

        print(total_disordered_regions,disordered_regions_with_motifs)

        disorder_percentage = round(number_of_elm_position / disorder_pos * 100, 2)
        region_percentage = round(disordered_regions_with_motifs / total_disordered_regions * 100, 2)

        percentages_positions = [disorder_percentage,region_percentage]
        percentages_regions = [num_pip_with_motif,num_pip_without_motif]

        plot_region_metrics(percentages_positions, percentages_regions, functional_color,plot_dir=plot_dir)

        exit()


    # # Coverage of Different algorithm 1
    # plot_combined(labels, postions, disorder_pos, title=f"Disorder Position Coverage")
    # # Number of Region
    # plot_region(labels, regions, title=f"Number of Regions")
    # # Number of Motif
    # plot_region(labels, motifs, title=f"Number of Motifs")
    # # Number of Pip with Motif
    # plot_region(labels, pip_regions_with_motif_list, title=f"Number of Regions with Motifs")

    # bigplot(labels,postions,regions,motifs,pip_regions_with_motif_list,disorder_pos,title='Method Metrics')
