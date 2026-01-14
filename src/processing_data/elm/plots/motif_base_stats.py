import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from tqdm import tqdm

# import matplotlib
# matplotlib.use("Agg")

def main_for_types(elm_with_scores_df, todir, am_column='motif_am_mean_score', am_sequential_column='sequential_am_score',figsize=(8, 6)):
    elm_types = elm_with_scores_df['ELMType'].unique().tolist()

    # Add a column for the sequential difference

    # Information Gain Max
    information_gain_max = 0.20

    # Define cutoffs
    benign_cutoff = 0.34
    half_cutoff = 0.5
    pathogen_cutoff = 0.564

    # Iterate over ELM types
    for elm_type in elm_types:
        # Filter data for the current ELM type
        current_type_df = elm_with_scores_df[
            (elm_with_scores_df['ELMType'] == elm_type) & (elm_with_scores_df[am_column].notna())
        ]

        # Calculate mean scores for each ELM identifier
        mean_scores = current_type_df.groupby('ELMIdentifier')[am_column].mean().reset_index()
        mean_scores = mean_scores.sort_values(by=am_column, ascending=False)

        # Merge sorted mean scores back to the DataFrame
        current_type_df = current_type_df.merge(mean_scores, on='ELMIdentifier', suffixes=('', '_mean'))

        # Sort by mean scores
        current_type_df = current_type_df.sort_values(by=f'{am_column}_mean', ascending=False)

        # Create plots for `am_column`
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=current_type_df, x=am_column, y='ELMIdentifier')
        plt.title(f'Boxplot of AlphaMissense Scores for {elm_type}')
        plt.xlabel("AlphaMissense Mean Scores")
        plt.ylabel('ELM Identifier')
        # plt.axvline(x=pathogen_cutoff, color='red', linestyle='--', linewidth=2, label=f'Pathogen ({pathogen_cutoff})')
        # plt.legend(loc="lower right")
        plt.tight_layout()
        # plt.savefig(f'{todir}/am_score_distribution_{elm_type}.png')
        plt.show()

        # Create plots for `sequential_difference`
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=current_type_df, x='sequential_difference', y='ELMIdentifier')
        plt.title(f'Boxplot of Sequential Difference Scores for {elm_type}')
        plt.xlabel('Sequential Difference')
        plt.ylabel('ELM Identifier')
        # plt.axvline(x=0, color='grey', linestyle='--', linewidth=2, label='Zero Difference')
        # plt.axvline(x=information_gain_max, color='red', linestyle='--', linewidth=2, label='Information Gain Max')
        # plt.legend(loc="lower right")
        plt.tight_layout()
        # plt.savefig(f'{todir}/sequential_difference_distribution_{elm_type}.png')
        plt.show()

def main_for_types_combined(elm_with_scores_df, todir, am_column='motif_am_mean_score', am_sequential_column='sequential_am_score',figsize=(8, 6),choosen_elm=None,filepath=None,ylabel=True):
    elm_types = elm_with_scores_df['ELMType'].unique().tolist()

    # Add a column for the sequential difference
    elm_with_scores_df['sequential_difference'] = elm_with_scores_df[am_column] - elm_with_scores_df[am_sequential_column]

    # Cutoffs and reference values
    benign_cutoff = 0.34
    half_cutoff = 0.5
    pathogen_cutoff = 0.564
    information_gain_max = 0.20

    for elm_type in elm_types:

        if choosen_elm and elm_type != choosen_elm:
            continue

        # Filter data for the current ELM type
        current_type_df = elm_with_scores_df[
            (elm_with_scores_df['ELMType'] == elm_type) & (elm_with_scores_df[am_column].notna())
        ]

        # Calculate mean scores for each ELM identifier
        mean_scores = current_type_df.groupby('ELMIdentifier')[am_column].mean().reset_index()
        mean_scores = mean_scores.sort_values(by=am_column, ascending=False)

        # Merge sorted mean scores back to the DataFrame
        current_type_df = current_type_df.merge(mean_scores, on='ELMIdentifier', suffixes=('', '_mean'))

        # Sort by mean scores
        current_type_df = current_type_df.sort_values(by=f'{am_column}_mean', ascending=False)

        # Create combined plot
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True, )

        # Plot for `am_column`
        sns.boxplot(ax=axes[0], data=current_type_df, x=am_column, y='ELMIdentifier',color='#f08080')
        # axes[0].set_title(f'{am_column} for {elm_type}')
        axes[0].set_xlabel("AlphaMissense Mean Scores")
        # axes[0].axvline(x=pathogen_cutoff, color='red', linestyle='--', linewidth=2, label=f'Pathogen ({pathogen_cutoff})')
        # axes[0].legend(loc="lower right")

        # Plot for `sequential_difference`
        sns.boxplot(ax=axes[1], data=current_type_df, x='sequential_difference', y='ELMIdentifier',color='#f08080')
        # axes[1].set_title(f'Sequential Difference Scores for {elm_type}')
        axes[1].set_xlabel('Sequential Difference')
        # axes[1].axvline(x=0, color='grey', linestyle='--', linewidth=2, label='Zero Difference')
        # axes[1].axvline(x=information_gain_max, color='red', linestyle='--', linewidth=2, label=f'Info Gain Max')
        # axes[1].legend(loc="lower right")

        # Remove redundant y-axis labels for the second plot
        axes[1].set_ylabel('')
        axes[1].tick_params(axis='y', left=False)

        if len(mean_scores) > 50:
            # Option 1: Remove just the tick labels (keeping the axis line)
            axes[0].set_yticklabels([])
            axes[1].set_yticklabels([])

        plt.suptitle("AlphaMissense Mean Scores and Sequential Differences for " + elm_type)

        # Save the combined plot
        plt.tight_layout()
        if filepath and choosen_elm:
            plt.savefig(filepath)
        # else:
        #     pass
            # plt.savefig(f'{todir}/combined_plot_{elm_type}.png')
        plt.show()


def violinplot_individual(elm_with_scores_df_al, am_column='motif_am_mean_score', sequential_column='sequential_difference',figsize=(8, 6)):
    # Define consistent colors for ELM types
    elm_types = sorted(elm_with_scores_df_al['ELMType'].unique())
    palette = sns.color_palette("husl", len(elm_types))  # Generate a unique color palette

    # Plot for `am_column`
    plt.figure(figsize=figsize)
    sns.violinplot(data=elm_with_scores_df_al, x='ELMType', y=am_column, scale='width', inner='box', palette=palette)
    plt.title('Violin Plot of AlphaMissense Mean Scores by ELM Type')
    plt.xlabel('ELM Type')
    plt.ylabel('AlphaMissense Mean Scores')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot for `sequential_difference`
    plt.figure(figsize=figsize)
    sns.violinplot(data=elm_with_scores_df_al, x='ELMType', y=sequential_column, scale='width', inner='box', palette=palette)
    plt.title('Violin Plot of Sequential Differences by ELM Type')
    plt.xlabel('ELM Type')
    plt.ylabel('Sequential Differences')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



def violinplot_combined(elm_with_scores_df_al, am_column='motif_am_mean_score', sequential_column='sequential_difference',figsize=(8, 4),filepath=None):
    # Define consistent colors for ELM types
    elm_types = sorted(elm_with_scores_df_al['ELMType'].unique())
    palette = sns.color_palette("pastel", len(elm_types))  # Generate a unique color palette

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot for `am_column`
    sns.violinplot(ax=axes[0], data=elm_with_scores_df_al, x='ELMType', y=am_column, scale='width', inner='box', palette=palette
                   )
    # axes[0].set_title('Violin Plot of AlphaMissense Mean Scores by ELM Type')
    axes[0].set_xlabel(None)
    axes[0].set_ylabel('AlphaMissense Mean Scores')
    axes[0].tick_params(axis='x', rotation=45)

    # Plot for `sequential_difference`
    sns.violinplot(ax=axes[1], data=elm_with_scores_df_al, x='ELMType', y=sequential_column, scale='width', inner='box', palette=palette)
    # axes[1].set_title('Violin Plot of Sequential Differences by ELM Type')
    axes[1].set_xlabel(None)
    axes[1].set_ylabel('Sequential Differences')
    axes[1].tick_params(axis='x', rotation=45)

    plt.suptitle('AlphaMissense Scores and Sequential Differences by ELM Type')

    # Adjust layout
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
    plt.show()


def plot_position_based_distribution(df,figsize=(6, 4),filepath=None):
    """Plot distributions of expanded key and flanking residues scores for a more granular view."""

    # Expand the key and flanking scores into individual positions
    key_residue_scores_expanded = df['key_residue_am_scores'].dropna().str.split(", ").explode().astype(float).reset_index(drop=True)

    # This is only non key residues
    non_key_residue_scores_expanded = df['flanking_residue_am_scores'].dropna().str.split(", ").explode().astype(float).reset_index(drop=True)

    # These are Flanking Residues
    flanking_residue_scores_expanded = df['sequential_am_scores'].dropna().str.split(", ").explode().astype(
        float).reset_index(drop=True)

    # Combine into a DataFrame for plotting
    position_data = pd.DataFrame({
        'Score': pd.concat([key_residue_scores_expanded,non_key_residue_scores_expanded, flanking_residue_scores_expanded], ignore_index=True),
        'Type': ['Key Residue'] * len(key_residue_scores_expanded) + ['Non-Key Residue'] * len(non_key_residue_scores_expanded) + ['Flanking Residue'] * len(flanking_residue_scores_expanded)
    })

    # Plotting position-based distributions
    plt.figure(figsize=figsize)
    sns.violinplot(data=position_data, x='Type', y='Score',color='#f08080')
    plt.title('Distribution of Scores for Motif Residues')
    plt.ylabel('AlphaMissense Score')
    plt.xticks(rotation=10)
    plt.xlabel(None)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
    plt.show()

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
    ax.set_title(text)

    for spine in ax.spines.values():
        spine.set_visible(False)

def plot_totals_subplots(known_labels, known_total_values, known_color,
                         predicted_labels, predicted_total_values, predicted_color, figsize=(8, 6),plot_dir=None):
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
    title = f"Motif Retention"
    plt.suptitle(title)
    plt.tight_layout()
    if plot_dir:
        plt.savefig(os.path.join(plot_dir,f"{title}.png"))
    plt.show()


def make_filtered_dfs_for_base(elm_all,elm_filtered,name="PEM Motifs"):

    filters = {
        f'All Motif': elm_all,
        name: elm_filtered,
    }

    # Calculate percentage of data retained for each filter
    total_count = len(elm_all)


    percentages = {name: df.shape[0] / total_count * 100 for name, df in filters.items()}
    total_numbers = {name: df.shape[0] for name, df in filters.items()}

    return percentages,total_numbers


def find_overlapping_elms(elm_known,pip_elm,add_elm_id=True):

    # Merge only on the columns that must be identical.
    # (Here: Protein_ID and ELMIdentifier must match exactly.)
    merge_col = ["Protein_ID"]
    if add_elm_id:
        merge_col.append("ELMIdentifier")
    # else:
    #     merge_col.append('Matched_Sequence')

    merged = pd.merge(
        elm_known,
        pip_elm,
        on=merge_col,
        suffixes=("_known", "_pip")
    )

    # Keep rows for which the intervals overlap
    # i.e., Start_known <= End_pip AND Start_pip <= End_known.
    overlapping = merged[
        (merged["Start_known"] <= merged["End_pip"])
        & (merged["Start_pip"] <= merged["End_known"])
        ]
    overlapping['Start'] = overlapping['Start_pip']
    overlapping['End'] = overlapping['End_pip']

    # Print or return the overlapping regions
    return overlapping


# 2) Define a helper function to merge overlapping intervals in one group
def merge_overlapping_intervals(intervals):
    """
    intervals: DataFrame with columns ['Start','End']
    returns: the count (or list) of merged intervals for that group
    """
    # Sort the group by Start
    intervals = intervals.sort_values("Start")

    merged = []
    current_start = None
    current_end = None

    for _, row in intervals.iterrows():
        s, e = row["Start"], row["End"]
        if current_start is None:
            # first interval in this group
            current_start, current_end = s, e
        else:
            # if the next interval overlaps or touches the current one, merge them
            if s <= current_end:
                current_end = max(current_end, e)
            else:
                # no overlap => push the old one, start a new one
                merged.append((current_start, current_end))
                current_start, current_end = s, e

    # don't forget the last one
    if current_start is not None:
        merged.append((current_start, current_end))

    # Return however you want to track them:
    # a) the count of merged intervals
    return len(merged)


def get_clinvar_percentages_and_totals(clinvar_df, elm_df, columns=['Protein_ID', "Position"],file_path=None):
    # Merge to find ClinVar mutations that fall within ELM motifs
    elm_positions_df = elm_df[["Protein_ID", "Start", "End", "ELMIdentifier"]].drop_duplicates()
    elm_uniqeu_positions_df = (
        elm_df[['Protein_ID', 'Start', 'End']]
        .drop_duplicates()
        .assign(Positions=lambda df: df.apply(lambda row: range(row['Start'], row['End'] + 1), axis=1))
        .explode('Positions')
        .rename(columns={'Positions': 'Position'})
        .loc[:, ['Protein_ID', 'Position']]
        .drop_duplicates()
    )
    number_of_elm_position = elm_uniqeu_positions_df.shape[0]
    clinvar_positions_df = clinvar_df[columns].drop_duplicates()


    total_elm_motifs = len(elm_positions_df)
    total_clinvar_mutations = len(clinvar_positions_df)

    # Find mutations that occur in motif regions
    merged_df = clinvar_positions_df.merge(elm_positions_df, on="Protein_ID", how="inner")
    mutations_in_elm = merged_df[
        (merged_df["Position"] >= merged_df["Start"]) & (merged_df["Position"] <= merged_df["End"])
    ]

    # Count ELM motifs that have at least one ClinVar mutation
    grouped = mutations_in_elm.groupby(["Protein_ID", "ELMIdentifier"])
    merged_interval_counts = grouped.apply(merge_overlapping_intervals)
    nunique_elm_with_mutations = len(merged_interval_counts)
    unique_elm_with_mutations = mutations_in_elm["ELMIdentifier"].unique()

    if file_path != None:
        mutations_in_elm.to_csv(file_path, index=False,sep='\t')

    # Count ClinVar mutations in ELM regions
    total_mutated_positions = mutations_in_elm[columns].drop_duplicates()
    total_mutations_in_elm = total_mutated_positions.shape[0]

    clinvar_data_df = mutations_in_elm.merge(clinvar_df,on=columns, how="inner")

    # Compute percentages
    percent_elm_with_mutations = (nunique_elm_with_mutations / total_elm_motifs) * 100 if total_elm_motifs else 0
    percent_clinvar_in_elm = (total_mutations_in_elm / total_clinvar_mutations) * 100 if total_clinvar_mutations else 0

    return {
        "Total ELM Motifs": total_elm_motifs,
        "Total ClinVar Position": total_clinvar_mutations,
        "ELM Motifs with ClinVar Position": nunique_elm_with_mutations,
        "ELM Motifs Number of Position": number_of_elm_position,
        "Percentage of ELM Motifs with ClinVar Position": percent_elm_with_mutations,
        "Total ClinVar Position in ELM Regions": total_mutations_in_elm,
        "unique_elm_with_mutations": unique_elm_with_mutations,
        "df": clinvar_data_df,
        "Percentage of ClinVar Position in ELM Regions": percent_clinvar_in_elm,
    }

def plot_total_clinvar_positions_in_elm(known_labels, known_total_values, predicted_labels, predicted_total_values,
                                        known_color, predicted_color, figsize=(6, 6),title="Pathogenic Positions in ELM",ylabel="Number of Position",plot_dir=None):

    fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=False)

    plot_totals(axs[0], known_labels, known_total_values, known_color, text="Known",ylabel=ylabel)
    plot_totals(axs[1], predicted_labels, predicted_total_values, predicted_color, text="Predicted",ylabel=ylabel)

    axs[0].set_title("Known", pad=20)
    axs[1].set_title("Predicted" ,pad=20)
    axs[1].set_ylabel(None)

    fig.align_titles()

    plt.suptitle(title)
    plt.tight_layout()
    if plot_dir:
        plt.savefig(os.path.join(plot_dir,f"{title}.png"))
    plt.show()

def plot_motif_and_position(
        labels, motif_values,
        position_values,
        known_color,figsize=(6, 6),title="Pathogenic Mutations in ELM",plot_dir=None):

    fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=False)

    plot_totals(axs[0], labels, position_values, known_color, text="Positions",ylabel="Number of Position")
    plot_totals(axs[1], labels, motif_values, known_color, text="Motif",ylabel="Number of Motifs")

    fig.align_titles()

    plt.suptitle(title)
    plt.tight_layout()
    if plot_dir:
        plt.savefig(os.path.join(plot_dir,f"{title}.png"))
    plt.show()


def plot_up_down_bars(info_dct_known,info_dct_elm_all,info_dct_known_pip,info_dct_predicted_pip,known_total_numbers,predicted_total_numbers,choosen_motif='PIP Motifs',name="",plot_dir=None):

    clinvar_position = 'Total ClinVar Position in ELM Regions'
    elm_with_clinvar = 'ELM Motifs with ClinVar Position'
    # Just for demonstration, let's define 3 categories and random percentage values
    categories = [
        "Motif\nRetention",
        "Pathogenic Positions\n in Motifs",
        "Motif with\nPathogenic Mutations",
    ]

    print(known_total_numbers)

    motif_retention_known = known_total_numbers[choosen_motif] / known_total_numbers['All Motif'] * 100
    motif_retention_predicted = predicted_total_numbers[choosen_motif] / predicted_total_numbers['All Motif'] * 100

    clinvar_known = info_dct_known_pip[clinvar_position] / info_dct_known[clinvar_position]  * 100
    clinvar_predicted = info_dct_predicted_pip[clinvar_position] / info_dct_elm_all[clinvar_position] * 100

    elm_with_clinvar_known = info_dct_known_pip[elm_with_clinvar] / info_dct_known[elm_with_clinvar] * 100
    elm_with_clinvar_predicted = info_dct_predicted_pip[elm_with_clinvar] / info_dct_elm_all[elm_with_clinvar] * 100


    # Fake data (percentages); you will replace these with your computed values
    predicted_values = [motif_retention_predicted, clinvar_predicted, elm_with_clinvar_predicted]  # Upward bars
    known_values = [motif_retention_known, clinvar_known, elm_with_clinvar_known]  # Downward bars

    x = np.arange(len(categories))

    # We'll treat 50 as the "offset" center line
    offset = 100
    bar_width = 0.6

    fig, ax = plt.subplots(figsize=(4, 4))

    # Draw the horizontal center line at y=50
    ax.axhline(offset, color='black', linewidth=1)

    # Known bars go downward: bottom = offset - known_value, height = known_value
    ax.bar(
        x,
        predicted_values,  # bar heights
        bottom=[offset - kv for kv in predicted_values],
        color='salmon',
        width=bar_width,
        label='Predicted'
    )

    # Predicted bars go upward: bottom = offset, height = predicted_value
    ax.bar(
        x,
        known_values,
        bottom=offset,
        color='royalblue',
        width=bar_width,
        label='Known'
    )

    # X-axis ticks
    ax.set_xticks(x)
    ax.set_xticklabels(categories, ha='center')

    ax.set_ylim(0, 200)  # Force [0..100]

    ticks = np.arange(0, 201, 25)  # 0, 25, 50, ..., 200


    def custom_label(y):
        if y >= 100:
            return f"{y - 100:.0f}"
        else:
            return f"{100 - y:.0f}"

    tick_labels = [custom_label(t) for t in ticks]

    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)

    for i, kv in enumerate(known_values):
        y_label = offset + kv + 5
        ax.text(
            x[i], y_label,
            f"{kv:.1f}%",
            ha='center', va='center', color='black',
            fontsize=10,
        )

        # Predicted bars: they go from offset up to offset + v
        # Place the label in the vertical middle:
        #   y_label = offset + (v/2)

    for i, pv in enumerate(predicted_values):
        y_label = offset - pv - 5
        ax.text(
            x[i], y_label,
            f"{pv:.1f}%",
            ha='center', va='center', color='black',
            fontsize=10,
        )

    ax.set_ylabel("Percentage")
    title = f"{choosen_motif} Retention Metrics ({name})"
    ax.set_title(title)
    # ax.legend()

    plt.tight_layout()
    if plot_dir:
        plt.savefig(os.path.join(plot_dir,f"{title}.png"))
    plt.show()


def plot_percentages_bars(info_dct_known, info_dct_known_pip, known_total_numbers,
                          choosen_motif='PIP Motifs', plot_dir=None):
    clinvar_position = 'Total ClinVar Position in ELM Regions'
    elm_with_clinvar = 'ELM Motifs with ClinVar Position'

    # Categories for the plot
    categories = [
        "Motif\nRetention",
        "Pathogenic Positions\n in Motifs",
        "Motif\nwith Mutations",
    ]

    # Calculate the percentage values
    motif_retention_known = known_total_numbers[choosen_motif] / known_total_numbers['All Motif'] * 100
    clinvar_known = info_dct_known_pip[clinvar_position] / info_dct_known[clinvar_position] * 100
    elm_with_clinvar_known = info_dct_known_pip[elm_with_clinvar] / info_dct_known[elm_with_clinvar] * 100

    # Known values for the plot
    known_values = [motif_retention_known, clinvar_known, elm_with_clinvar_known]

    x = np.arange(len(categories))

    bar_width = 0.6

    fig, ax = plt.subplots(figsize=(4, 4))

    # Create the bars for known values (normal bars, no offset)
    ax.bar(
        x,
        known_values,  # bar heights
        color='royalblue',
        width=bar_width,
        label='Known'
    )

    # X-axis ticks
    ax.set_xticks(x)
    ax.set_xticklabels(categories, ha='center')

    # Adjust ylim to give extra space for text
    ax.set_ylim(0, 100)  # Increased upper limit to provide space for text labels

    ticks = np.arange(0, 110, 25)  # 0, 25, 50, ..., 120

    def custom_label(y):
        return f"{y:.0f}"

    tick_labels = [custom_label(t) for t in ticks]

    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)

    # Add the labels for known values
    for i, kv in enumerate(known_values):
        y_label = kv + 3  # Add small space above the bar for label
        ax.text(
            x[i], y_label,
            f"{kv:.1f}%",
            ha='center', va='center', color='black',
            fontsize=10,
        )

    ax.set_ylabel("Percentage")
    title = f"{choosen_motif} Retention Metrics"
    plt.suptitle(title)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    if plot_dir:
        plt.savefig(os.path.join(plot_dir, f"{title}.png"))
    plt.show()


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

def plot_combined_disorder_donut_chart(
    protein_regions, relevant_regions_df, am_disorder_df, functional_color="#FF9999", non_functional_color="#66B2FF",plot_dir=None, choosen_motif='PEM Motifs'
):
    """
    Creates two donut charts as subplots:
    1. Percentage of disordered regions with motifs.
    2. Proportion of predicted motif positions.
    """
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


    print(total_disordered_regions, disordered_regions_with_motifs)

    # Ensure counts are non-negative
    disordered_regions_without_motifs = max(total_disordered_regions - disordered_regions_with_motifs, 0)

    # Data for the first donut chart
    data1 = {
        'Regions': [choosen_motif, f'Non-{choosen_motif}'],
        'Count': [disordered_regions_with_motifs, disordered_regions_without_motifs]
    }
    percentage_df = pd.DataFrame(data1)

    # relevant_region_positions = relevant_regions_df["Length"].sum()

    elm_uniqeu_positions_df = (
        relevant_regions_df[['Protein_ID', 'Start', 'End']]
        .drop_duplicates()
        .assign(Positions=lambda df: df.apply(lambda row: range(row['Start'], row['End'] + 1), axis=1))
        .explode('Positions')
        .rename(columns={'Positions': 'Position'})
        .loc[:, ['Protein_ID', 'Position']]
        .drop_duplicates()
    )
    number_of_elm_position = elm_uniqeu_positions_df.shape[0]

    total_positions = am_disorder_df.shape[0]
    non_relevant_positions = total_positions - number_of_elm_position

    # Data for the second donut chart
    # labels2 = ["Motifs", "Non-Motifs"]
    sizes2 = [number_of_elm_position, non_relevant_positions]

    # Plot combined subplots
    fig, axes = plt.subplots(1, 2, figsize=(4, 3))

    # First donut chart
    wedges1, texts1, autotexts1 = axes[0].pie(
        percentage_df['Count'],
        labels=[None, None],
        autopct='%1.1f%%',
        startangle=140,
        colors=[functional_color, non_functional_color],
        wedgeprops=dict(width=0.3, edgecolor='w')
    )
    axes[0].set_title('Occurrences')

    # Second donut chart
    wedges2, texts2, autotexts2 = axes[1].pie(
        sizes2,
        labels=[None, None],
        autopct='%1.1f%%',
        startangle=140,
        colors=[functional_color, non_functional_color],
        wedgeprops=dict(width=0.3, edgecolor='w')
    )
    axes[1].set_title("Positions")

    # Equal axis for both charts
    for ax in axes:
        ax.axis('equal')

    # Add a shared title
    plt.suptitle(f"{choosen_motif} in IDRs")

    # Add a common legend
    fig.legend(
        handles=[
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=functional_color, markersize=10, label=choosen_motif),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=non_functional_color, markersize=10, label=f'Non-{choosen_motif}')
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=2
    )

    # Adjust layout
    plt.tight_layout()
    if plot_dir:
        plt.savefig(f"{plot_dir}/combined_disorder_donut_chart.png", dpi=300, bbox_inches='tight')
    plt.show()


def get_disordered_regions(am_disorder_df,sequential_region_difference=1):
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



def plot_number_of_elms(plot_dir):
    # Load data and compute values (assuming make_filtered_dfs_for_base and other required functions are defined)
    files = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files"

    # Regions Categories - All
    regions = {
        "All Known": 'elm/elm_known_with_am_info_all_disorder_class_filtered.tsv',
        "All Predicted": 'elm/elm_predicted_with_am_info_all_disorder_class_filtered.tsv',
        "DecisionTree": 'elm/decision_tree/elm_predicted_with_am_info_all_disorder_class_filtered.tsv',
        "SlimPrint": 'elm/decision_tree/publication_predicted_effected_norman_davey_with_am_info_all_disorder_class_filtered.tsv',
        "Kadane (0)": 'pip/motifs_on_pathogenic_region_with_am_scores.tsv',
        "Kadane (0.15)": 'pip/motifs_on_pip_with_am_scores.tsv',
        "Kadane (0.15,3e)": 'pip/motifs_on_pip_extension_with_am_scores.tsv',
    }
    # Regions Categories - High Predicted Class Filtered


    lst = [
        ["All Disorder",regions],
        # ["Class Filtered",regions_class_filtered]
    ]

    clinvar_df = pd.read_csv(f"{files}/clinvar/Pathogenic/disorder/positional_clinvar_functional_categorized_final.tsv",
                             sep='\t')

    base_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations'
    am_disorder = os.path.join(base_dir, "processed_data/files/alphamissense/am_disorder.tsv")
    am_disorder_df = pd.read_csv(am_disorder, sep='\t')
    disorder_pos = am_disorder_df.shape[0]

    protein_regions, region_df = get_disordered_regions(am_disorder_df)

    print(disorder_pos)

    method_names = [
        # "PEM",
        "DecisionTree",
        # "SlimPrint",
        # "Kadane (0)",
        # "Kadane (0.15)",
        # "Kadane (0.15,3e)",
    ]

    names = {
        "Kadane (0)":'Pathogenic Region',
        "Kadane (0.15)":'PIP Motifs',
        "Kadane (0.15,3e)":'PIP Motifs Extension',
        'PEM':'PEM Motifs',
        'DecisionTree':'AMPEM',
        "SlimPrint":"SlimPrint",
    }


    for name, regions in lst:
        for method in method_names:

            # Regions
            elm_known_disorder = pd.read_csv(f"{files}/{regions['All Known']}", sep='\t')
            elm_all_predicted_disorder = pd.read_csv(f"{files}/{regions['All Predicted']}", sep='\t')

            method_name = names[method]

            if method == "PEM":
                filtered_predicted_df = elm_all_predicted_disorder[elm_all_predicted_disorder['Sequential_Rule'] == True]
            else:
                filtered_predicted_df = pd.read_csv(f"{files}/{regions[method]}", sep='\t')

            if method == "PEM":
                filtered_predicted_overlap_with_known = elm_known_disorder[elm_known_disorder['Sequential_Rule'] == True]
            elif method == "SlimPrint":
                # filtered_predicted_df = filtered_predicted_df.drop(columns=['ELMIdentifier'])
                filtered_predicted_overlap_with_known = filtered_predicted_df[filtered_predicted_df['ELM'].notna()]
                filtered_predicted_overlap_with_known['ELMIdentifier'] = filtered_predicted_overlap_with_known['ELM'].str.split(":").str[0]
                # filtered_predicted_overlap_with_known = find_overlapping_elms(elm_known_disorder, filtered_predicted_df)
                # filtered_predicted_df['ELMIdentifier'] = filtered_predicted_df['MOTIF']
            else:
                filtered_predicted_overlap_with_known = find_overlapping_elms(elm_known_disorder, filtered_predicted_df)

            if method == "DecisionTree":
                elm_known_disorder = filtered_predicted_df[filtered_predicted_df['known'] == True]
                elm_all_predicted_disorder = filtered_predicted_df[filtered_predicted_df['known'] == False]
                filtered_predicted_overlap_with_known = elm_known_disorder[elm_known_disorder['prediction_decision_tree'] == True]
                filtered_predicted_df = elm_all_predicted_disorder[elm_all_predicted_disorder['prediction_decision_tree'] == True]

            # # Get data for known and predicted datasets
            known_percentages, known_total_numbers = make_filtered_dfs_for_base(elm_known_disorder,filtered_predicted_overlap_with_known,name=method_name)
            predicted_percentages, predicted_total_numbers = make_filtered_dfs_for_base(elm_all_predicted_disorder,filtered_predicted_df,name=method_name)

            # Remove 'All Motif' from percentages
            # known_percentages.pop('All Motif', None)
            # predicted_percentages.pop('All Motif', None)

            # print(filtered_predicted_overlap_with_known)
            # print(filtered_predicted_overlap_with_known.columns)
            #
            info_dct_known = get_clinvar_percentages_and_totals(clinvar_df, elm_known_disorder)
            info_dct_known_filtered = get_clinvar_percentages_and_totals(clinvar_df, filtered_predicted_overlap_with_known)

            info_dct_elm_all = get_clinvar_percentages_and_totals(clinvar_df, elm_all_predicted_disorder)
            info_dct_predicted_filtered = get_clinvar_percentages_and_totals(clinvar_df, filtered_predicted_df)


            # plot_combined_disorder_donut_chart(protein_regions, filtered_predicted_df, am_disorder_df,choosen_motif=method_name,plot_dir=plot_dir)
            # # exit()

            # Remove 'All Motif' from percentages
            known_percentages.pop('All Motif', None)
            predicted_percentages.pop('All Motif', None)

            # Prepare data for plotting
            known_labels = list(known_total_numbers.keys())
            known_total_values = list(known_total_numbers.values())
            predicted_labels = list(predicted_total_numbers.keys())
            predicted_total_values = list(predicted_total_numbers.values())

            # Define colors
            known_color = '#B4F8C8'
            predicted_color = '#FFB68A'

            plot_percentages_bars(info_dct_known,info_dct_known_filtered,known_total_numbers,choosen_motif=method_name,plot_dir=plot_dir)

            # Figure 5 Motif Retention
            plot_totals_subplots(known_labels, known_total_values, known_color,
                                 predicted_labels, predicted_total_values, predicted_color, figsize=(4, 4),plot_dir=plot_dir)


            motif_col = 'ELM Motifs with ClinVar Position'
            pos_col = 'Total ClinVar Position in ELM Regions'

            known_motif_values = [info_dct_known[motif_col],info_dct_known_filtered[motif_col]]
            known_pos_values = [info_dct_known[pos_col],info_dct_known_filtered[pos_col]]

            plot_motif_and_position(known_labels,known_motif_values,known_pos_values,known_color, figsize=(4, 4),plot_dir=plot_dir)
            exit()


            column = 'Total ClinVar Position in ELM Regions'

            known_total_values = [info_dct_known[column],info_dct_known_filtered[column]]
            predicted_total_values = [info_dct_elm_all[column], info_dct_predicted_filtered[column]]

            known_labels = known_labels
            predicted_labels = known_labels


            # Pathogenic Position in ELM
            plot_total_clinvar_positions_in_elm(known_labels, known_total_values,
                                                predicted_labels, predicted_total_values,
                                                known_color, predicted_color,
                                                figsize=(4, 4), title=f"Pathogenic Positions in ELM",plot_dir=plot_dir)

            column = 'ELM Motifs with ClinVar Position'

            known_total_values = [info_dct_known[column], info_dct_known_filtered[column]]
            predicted_total_values = [info_dct_elm_all[column], info_dct_predicted_filtered[column]]

            # # Plot the graph
            plot_total_clinvar_positions_in_elm(known_labels, known_total_values,
                                                predicted_labels, predicted_total_values,
                                                known_color, predicted_color,
                                                figsize=(4, 4), title=f"ELM with Pathogenic Mutations",
                                                ylabel="Number of Motifs",plot_dir=plot_dir)





if __name__ == '__main__':
    # base_dir = '/dlab/home/norbi/PycharmProjects/AlphaMissense_Stat'
    base_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations'

    all_am_elm = os.path.join(base_dir,'processed_data/files/elm/elm_known_with_am_info_all_disorder_class_filtered.tsv')
    elm_with_scores_df_al = pd.read_csv(all_am_elm, sep='\t')

    am_column = 'motif_am_mean_score'
    am_sequential_column = 'sequential_am_score'

    elm_with_scores_df_al['sequential_difference'] = elm_with_scores_df_al[am_column] - elm_with_scores_df_al[
        am_sequential_column]
    print(elm_with_scores_df_al.columns)

    plot_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/plots/fig5'

    # FIG 5 Motifs
    plot_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/plots/sm/motif/test'
    plot_number_of_elms(plot_dir)
    exit()

    # FIG 5A)
    plot_position_based_distribution(elm_with_scores_df_al, figsize=(5, 3),filepath=os.path.join(plot_dir,"A.png"))

    # FIG 5B1)
    violinplot_combined(elm_with_scores_df_al, am_column='motif_am_mean_score',
                        sequential_column='sequential_difference',figsize=(6, 3),filepath=os.path.join(plot_dir,"B.png"))

    # FIG 5B2)
    main_for_types_combined(elm_with_scores_df_al, "todir", figsize=(6, 5), choosen_elm="DEG",filepath=os.path.join(plot_dir,"B2.png"))

    for elmtype in elm_with_scores_df_al['ELMType'].unique().tolist():
        main_for_types_combined(elm_with_scores_df_al, "todir", figsize=(6, 5), choosen_elm=elmtype,ylabel=False)

    exit()

    # Individual Plots
    # violinplot_individual(elm_with_scores_df_al, am_column='motif_am_mean_score',
    #                       sequential_column='sequential_difference')

    # Combined Subplots
    violinplot_combined(elm_with_scores_df_al, am_column='motif_am_mean_score',
                        sequential_column='sequential_difference')

    # exit()
    # main_for_types(elm_with_scores_df_al, "todir",figsize=(8, 6))

    # FIG 7B)
    main_for_types_combined(elm_with_scores_df_al, "todir",figsize=(8, 4))