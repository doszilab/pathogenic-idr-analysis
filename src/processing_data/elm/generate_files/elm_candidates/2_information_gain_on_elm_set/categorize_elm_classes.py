import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

import matplotlib
# matplotlib.use("Agg")

def plot_filtered_vs_total_count_clustered_cutoff(
    df, filtered_df, cutoff_x,
    figsize=(12, 10), filepath=None, islogscale=False, df1_text="Predicted", df2_text="Predicted AM Sequential"
):
    """
    Scatter plot for total count vs filtered count based on a threshold with user-defined cutoff for clustering.

    Parameters:
    df (pd.DataFrame): DataFrame containing the full ELM data and metrics.
    filtered_df (pd.DataFrame): DataFrame containing the filtered data.
    column (str): The metric name being analyzed.
    name_mapping (dict): Mapping of metric names to display names.
    max_threshold (float): Threshold value used for filtering.
    cutoff_x (float): Cutoff value for the x-axis (Total Count) to define clusters.
    cutoff_y (float): Cutoff value for the y-axis (Filtered Count) to define clusters.
    figsize (tuple): Size of the figure.
    filepath (str): If provided, save the plot to the given file path.
    islogscale (bool): If True, apply log scale to both axes.
    df1_text (str): Label for the first dataset (default "Predicted").
    df2_text (str): Label for the second dataset (default "Predicted AM Sequential").
    """
    # Compute total and filtered counts for each ELMIdentifier
    all_counts = df.groupby('ELMIdentifier')['ELMIdentifier'].count()
    filtered_counts = filtered_df.groupby('ELMIdentifier')['ELMIdentifier'].count()

    # Merge the counts into a DataFrame
    plot_data = pd.DataFrame({
        'Total Count': all_counts,
        'Filtered Count': filtered_counts
    }).fillna(0).reset_index()

    # cluster_colors = plt.cm.get_cmap('tab10', 2)
    cluster_colors = {'Low Amount Predicted': 'lightblue', 'High Amount Predicted': 'grey'}

    # Define clusters based on cutoffs
    conditions = [
        (plot_data['Total Count'] < cutoff_x),
        (plot_data['Total Count'] >= cutoff_x),
    ]
    cluster_labels = ['Low Amount Predicted', 'High Amount Predicted']
    plot_data['Cluster'] = np.select(conditions, cluster_labels, default='Unclustered')

    # Create the figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # First subplot: Total vs Filtered Count with clusters
    ax1 = axes[0]
    ax1.grid()

    for cluster, color in cluster_colors.items():
        cluster_data = plot_data[plot_data['Cluster'] == cluster]
        proportion = len(cluster_data) / len(plot_data) * 100

        # Scatter plot for each cluster
        ax1.scatter(
            cluster_data['Total Count'],
            cluster_data['Filtered Count'],
            label=f'{cluster} ({proportion:.2f}%)',
            color=color,
            s=50,
            edgecolor='k',
            alpha=0.8,
            zorder=2
        )

    ax1.axvline(cutoff_x,color='red', linestyle='--', label=f'Cutoff (x={cutoff_x})')

    # Optional: Apply log scale if requested
    if islogscale:
        ax1.set_xscale('log')
        # ax1.set_yscale('log')

    ax1.set_xlabel(f'{df1_text} Instances')
    ax1.set_ylabel(f'{df2_text} Instances')
    ax1.set_title('All Classes')

    # Second subplot: Percentages and Counts for ELM classes
    ax2 = axes[1]
    ax2.grid()

    largest_cluster = plot_data[plot_data['Cluster'] == 'Low Amount Predicted']

    plot_data = largest_cluster.merge(df[['ELMIdentifier', 'ELMType']].drop_duplicates(),
                                      on='ELMIdentifier', how='left')

    # Assign unique colors to each ELMType
    unique_types = df['ELMType'].unique()
    # reversed_colors = sorted(list(plt.cm.tab10.colors[:len(unique_types)]))
    # type_colors = {etype: color for etype, color in zip(unique_types, reversed_colors)}
    type_colors = {etype: color for etype, color in zip(unique_types, plt.cm.tab10.colors[:len(unique_types)])}

    # Plot the percentages and counts for ELM classes
    elm_class_counts = plot_data['ELMType'].value_counts()
    elm_info = {}
    for i, (etype, count) in enumerate(elm_class_counts.items()):
        percentage = (count / len(plot_data)) * 100
        elm_info[etype] = f'{etype}: {percentage:.2f}% ({count} classes)'

    for etype, group_data in plot_data.groupby('ELMType'):
        ax2.scatter(
            group_data['Total Count'],
            group_data['Filtered Count'],
            label=elm_info[etype],
            color=type_colors[etype],
            s=50,
            edgecolor='k',
            alpha=0.8,
            zorder=2
        )

    ax2.set_xlabel(f'{df1_text} Instances')
    ax2.set_ylabel(None)
    ax2.set_title(f'Low Amount Motif Predicted Classes')

    # Move legends outside the plots and place them below each other
    fig.legend(loc='upper center', bbox_to_anchor=(1.15, 0.8), ncol=1, fontsize=8)

    # plt.suptitle("Remove ELM Classes outliers")

    # Tight layout to avoid overlapping labels
    plt.tight_layout()

    # Save the plot if a filepath is provided
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')

    plt.show()

    return largest_cluster




def find_optimal_cutoff(cutoffs, class_percentages, motif_instances_percentages):
    """
    Identifies the optimal cutoff point by analyzing the rate of change (slope) of both class retention and motif filtering.

    Parameters:
    - cutoffs (list): List of cutoff values.
    - class_percentages (list): Corresponding class percentages.
    - motif_instances_percentages (list): Corresponding motif instance percentages.

    Returns:
    - optimal_cutoff (float): The best cutoff value where class retention slows down while motif filtering remains effective.
    """
    class_slopes = np.gradient(class_percentages)  # First derivative (change in class percentage)
    motif_slopes = np.gradient(motif_instances_percentages)  # First derivative (change in motif percentage)

    # Find where class retention slows down (change in slope is small)
    class_slope_change = np.gradient(class_slopes)  # Second derivative (curvature of class retention)
    class_plateau_index = np.where(class_slope_change > -0.001)[0]  # Small negative or near zero slope

    # Find where motif filtering is still decreasing rapidly
    motif_slope_change = np.gradient(motif_slopes)  # Second derivative (curvature of motif filtering)
    motif_fast_filtering_index = np.where(motif_slope_change < -0.001)[0]  # Still decreasing strongly

    # Find the intersection of these two conditions
    intersection_indices = np.intersect1d(class_plateau_index, motif_fast_filtering_index)

    if len(intersection_indices) > 0:
        optimal_index = intersection_indices[0]  # The first occurrence where both conditions are met
        optimal_cutoff = cutoffs[optimal_index]
    else:
        optimal_cutoff = cutoffs[-1]  # Default to the highest cutoff if no intersection is found

    return optimal_cutoff

def find_optimal_cutoff_by_cuts(cutoffs, class_percentages, motif_instances_percentages, class_at_least=70, motifs_high_max=90):
    """
    Identifies the optimal cutoff point where at least 75% of the classes are retained,
    while no more than 90% of the motif instances are kept.

    Parameters:
    - cutoffs (list): List of cutoff values.
    - class_percentages (list): Corresponding class percentages.
    - motif_instances_percentages (list): Corresponding motif instance percentages.
    - class_at_least (float): Minimum percentage of classes to retain (default 75%).
    - motifs_high_max (float): Maximum percentage of motifs allowed (default 90%).

    Returns:
    - optimal_cutoff (float): The best cutoff value that satisfies both constraints.
    """


    # Find all indices where class retention is at least the required threshold
    valid_class_indices = np.where(np.array(class_percentages) >= class_at_least)[0]

    # Find all indices where motif instances are below the maximum allowed threshold
    valid_motif_indices = np.where(np.array(motif_instances_percentages) >= motifs_high_max)[0]

    # Find the intersection of both conditions
    valid_indices = np.intersect1d(valid_class_indices, valid_motif_indices)

    if len(valid_indices) > 0:
        optimal_index = valid_indices[0]  # Choose the first valid occurrence
        optimal_cutoff = cutoffs[optimal_index]
    else:
        optimal_cutoff = cutoffs[-1]  # Default to the highest cutoff if no valid point is found

    return optimal_cutoff

def plot_class_percentages_vs_instances(df, filtered_df, cutoffs, figsize=(6, 3), filepath=None):
    """
    Generates a plot showing how class percentages and predicted motif instances vary based on different cutoffs.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the full ELM data and metrics.
    - filtered_df (pd.DataFrame): DataFrame containing the filtered data.
    - cutoffs (list): List of cutoff values for filtering classes.
    - figsize (tuple): Figure size.
    - filepath (str): If provided, save the plot to the given file path.

    Returns:
    - None: Displays the plot.
    """
    class_percentages = []
    motif_instances_percentages = []

    total_elm_classes = df["ELMIdentifier"].nunique()
    total_motif_instances = len(df)

    # Compute total and filtered counts for each ELMIdentifier
    all_counts = df.groupby('ELMIdentifier')['ELMIdentifier'].count()
    filtered_counts = filtered_df.groupby('ELMIdentifier')['ELMIdentifier'].count()

    # Merge the counts into a DataFrame
    plot_data = pd.DataFrame({
        'Total Count': all_counts,
        'Filtered Count': filtered_counts
    }).fillna(0).reset_index()


    motif_instances_percentages_high = []

    for cutoff in tqdm(cutoffs):
        largest_cluster = plot_data[plot_data["Total Count"] < cutoff]
        smaller_cluster = plot_data[plot_data["Total Count"] >= cutoff]
        filtered_classes = largest_cluster["ELMIdentifier"].nunique()
        filtered_instances_high = largest_cluster['Total Count'].sum()
        filtered_instances_low = smaller_cluster['Total Count'].sum()

        class_percentages.append((filtered_classes / total_elm_classes) * 100)
        motif_instances_percentages.append((filtered_instances_low / total_motif_instances) * 100)
        motif_instances_percentages_high.append((filtered_instances_high / total_motif_instances) * 100)

    # Create dual-axis plot
    fig, ax1 = plt.subplots(figsize=figsize)

    # Run the updated plotting function with elbow point detection
    optimal_cutoff = find_optimal_cutoff_by_cuts(cutoffs, class_percentages, motif_instances_percentages,class_at_least=60)

    color1 = 'tab:blue'
    color2 = 'tab:red'

    ax1.set_xlabel("Total Predicted Motif per Class")
    ax1.set_ylabel("Class Retention (%)", color=color1)
    ax1.plot(cutoffs, class_percentages, marker='o', linestyle='-', color=color1, label="Class Percentage")
    # cutoff_line = ax1.axvline(optimal_cutoff, color='black', linestyle='--', label=f'Optimal Cutoff: {optimal_cutoff}')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 100)
    # ax1.legend(handles=[cutoff_line], loc="lower right")

    ax2 = ax1.twinx()
    ax2.set_ylabel("PEM Filtering (%)", color=color2)
    ax2.plot(cutoffs, motif_instances_percentages, marker='o', linestyle='-', color=color2, label="Motif Instances")
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0,100)

    plt.suptitle("Distinguishing Highly and Lowly Predicted Motifs")
    fig.tight_layout()

    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")

    plt.show()

    return optimal_cutoff

def determine_known_for_predicted(predicted_df, known_df, columns=['Protein_ID', 'ELMIdentifier', "Start"]):
    """
    Merges predicted_df with known_df on the specified columns and adds a Found_Known column.
    """
    merged_df = predicted_df.merge(known_df[columns].drop_duplicates(), on=columns, how='left', indicator=True)
    merged_df['Found_Known'] = merged_df['_merge'] == 'both'
    return merged_df.drop(columns=['_merge'])

def plot_elm_count_of_counts(df, fig_dir=None):
    """
    Plots the distribution of ELM motif counts with predefined bins.
    """
    # Calculate the counts of ELMIdentifier
    elm_types_count = df['ELMIdentifier'].value_counts()

    # Define bins and labels
    bins = [0, 10, 25, 50, 100, 250, 500,1000, float('inf')]
    labels = ['1-10', '10-25', '25-50', '50-100', '100-250', '250-500', '500-1000',"1000+"]

    # Bin the data
    binned_counts = pd.cut(elm_types_count, bins=bins, labels=labels, right=False)
    count_of_counts = binned_counts.value_counts().sort_index()

    # Plot the distribution
    plt.figure(figsize=(4, 3))
    bars = count_of_counts.plot(kind='bar', color=main_color)
    plt.title('Predicted Motifs Count for ELM Classes')
    plt.xlabel('Number of Motif Predicted')
    plt.ylabel('Number of Classes')
    plt.xticks(rotation=45)

    # Adjust layout
    plt.tight_layout()

    # Save the plot if a directory is provided
    if fig_dir:
        plt.savefig(f"{fig_dir}/elm_count_of_counts_range.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_elm_types(df, fig_dir=None):
    elm_types_count = df['ELMType'].value_counts()

    # Plot the distribution
    plt.figure(figsize=(4, 3))
    bars = elm_types_count.plot(kind='bar', color=main_color)
    plt.title('Predicted ELM Motif Types')
    plt.ylabel('Number of Motif Predicted')
    plt.ylim(0, elm_types_count.max() * 1.1)
    plt.xticks(rotation=0)

    # Add value annotations on top of the bars
    for i, value in enumerate(elm_types_count):
        plt.text(i, value + 0.1, str(value), ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    if fig_dir:
        plt.savefig(f"{fig_dir}/elm_type_counts.png", dpi=300, bbox_inches='tight')
    plt.show()




def categorize_elm_classes(elm_known_with_am_info_all,elm_predicted_with_am_info_all):
    # Calculate the counts of ELMIdentifier
    elm_types_count = elm_predicted_with_am_info_all['ELMIdentifier'].value_counts()
    elm_predicted_with_am_info_all['ELM_Types_Count'] = elm_predicted_with_am_info_all['ELMIdentifier'].map(elm_types_count)

    # Define bins and labels
    bins = [0, 10, 25, 50, 100, 250, 500, 1000, float('inf')]
    labels = ['1-10', '10-25', '25-50', '50-100', '100-250', '250-500', '500-1000', '1000+']

    # Bin the data
    binned_counts = pd.cut(elm_types_count, bins=bins, labels=labels, right=False)
    elm_predicted_with_am_info_all['N_Motif_Category'] = elm_predicted_with_am_info_all['ELMIdentifier'].map(binned_counts)

    filtered_df = determine_known_for_predicted(elm_predicted_with_am_info_all, elm_known_with_am_info_all)
    filtered_df = filtered_df[filtered_df['prediction_decision_tree'] == True]

    plot_elm_count_of_counts(filtered_df, fig_dir=None)
    plot_elm_types(filtered_df, fig_dir=None)
    # exit()

    cutoff = 1000

    # cutoff_values = np.arange(0, 20001, 100)

    # optimal_cutoff = plot_class_percentages_vs_instances(elm_predicted_with_am_info_all, elm_known_with_am_info_all, cutoff_values, figsize=(9, 3),
    #                                                      filepath=os.path.join(plot_dir, f"optimal_cutoff_for_classes.png"))

    # exit()
    # print(optimal_cutoff)


    # Fig SM2 - Scatterplot motif
    largest_cluster = plot_filtered_vs_total_count_clustered_cutoff(elm_predicted_with_am_info_all,elm_known_with_am_info_all, cutoff,
                                                            figsize=(6, 3), islogscale=True,
                                                             df1_text="Predicted", df2_text="Known",
                                                             filepath=os.path.join(plot_dir, f"clustering.png")
                                                             )

    filtered_df["N_Motif_Predicted"] = filtered_df["ELMIdentifier"].isin(largest_cluster["ELMIdentifier"]).map({True: "Low_Amount", False: "High_Amount"})

    high_confidence_df = filtered_df[filtered_df["N_Motif_Predicted"] == "Low_Amount"]

    print(largest_cluster)

    print(filtered_df.shape[0])
    print(high_confidence_df.shape[0])
    total_elm_classes = filtered_df["ELMIdentifier"].nunique()
    filtered_elm_classes = high_confidence_df["ELMIdentifier"].nunique()
    print(total_elm_classes)
    print(filtered_elm_classes)
    print(filtered_elm_classes / total_elm_classes * 100)

    filtered_df.to_csv(f"{files}/elm_predicted_disorder_with_confidence.tsv", sep="\t", index=False)
    high_confidence_df.to_csv(f"{files}/elm_predicted_disorder_low_predicted.tsv", sep="\t", index=False)

    # Simplified Results For Known
    known = filtered_df[filtered_df['known'] == True]
    known.to_csv(f"{files}/elm_predicted_disorder_confidence_only_known.tsv", sep="\t", index=False)



if __name__ == "__main__":
    # Load data
    core_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations'
    files = os.path.join(core_dir,"processed_data/files/elm")
    # files = r"D:\PycharmProjects\IDP_Pathogenic_Mutations\processed_data\files\elm"
    plot_dir =  os.path.join(core_dir,"plots/sm/motif")
    path_to_generate = os.path.join(core_dir,"processed_data/files/elm/prediction")

    main_color = '#f08080'


    elm_known_path = f"{files}/elm_known_with_am_info_all_disorder_class_filtered.tsv"
    elm_predicted_path = f"{files}/decision_tree/elm_predicted_with_am_info_all_disorder_class_filtered.tsv"

    elm_known_with_am_info_all = pd.read_csv(elm_known_path, sep='\t')
    elm_predicted_with_am_info_all = pd.read_csv(elm_predicted_path,sep='\t')

    # Motif Correction for High False positives
    categorize_elm_classes(elm_known_with_am_info_all,elm_predicted_with_am_info_all)
