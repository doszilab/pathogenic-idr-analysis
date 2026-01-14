import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def filter_files_high_predicted_classes(cutoff_x=10000,figsize=(6,3),filepath=None):

    files = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files"
    filtered_df = pd.read_csv(
        f"{files}/pip/motifs_on_pip_with_am_scores.tsv",
        sep='\t')

    filtered_extension_df = pd.read_csv(
        f"{files}/pip/motifs_on_pip_extension_with_am_scores.tsv",
        sep='\t')

    filtered_pathogenic_df = pd.read_csv(
        f"{files}/pip/motifs_on_pathogenic_region_with_am_scores.tsv",
        sep='\t')

    df_known = pd.read_csv(
        f"{files}/elm/elm_known_with_am_info_all_disorder_class_filtered.tsv",
        sep='\t')

    df = pd.read_csv(
        f"{files}/elm/elm_predicted_with_am_info_all_disorder_class_filtered.tsv",
        sep='\t')

    # Compute total and filtered counts for each ELMIdentifier
    all_counts = df.groupby('ELMIdentifier')['ELMIdentifier'].count()
    filtered_counts = filtered_df.groupby('ELMIdentifier')['ELMIdentifier'].count()

    # Merge the counts into a DataFrame
    plot_data = pd.DataFrame({
        'Total Count': all_counts,
        'Filtered Count': filtered_counts
    }).fillna(0).reset_index()

    # cluster_colors = plt.cm.get_cmap('tab10', 2)
    cluster_colors = {'High Confidence': 'lightblue', 'Low Confidence': 'grey'}

    # Define clusters based on cutoffs
    conditions = [
        (plot_data['Total Count'] < cutoff_x),
        (plot_data['Total Count'] >= cutoff_x),
    ]
    cluster_labels = ['High Confidence', 'Low Confidence']
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

    ax1.axvline(cutoff_x, color='red', linestyle='--', label=f'Cutoff (x={cutoff_x})')

    ax1.set_xlabel(f'All Instances')
    ax1.set_ylabel(f'Predicted Instances')
    ax1.set_title('All Classes')

    # Second subplot: Percentages and Counts for ELM classes
    ax2 = axes[1]
    ax2.grid()

    largest_cluster = plot_data[plot_data['Cluster'] == 'High Confidence']

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

    ax2.set_xlabel(f'Predicted Instances')
    ax2.set_ylabel(None)
    ax2.set_title(f'High Confidence Classes')

    # Move legends outside the plots and place them below each other
    fig.legend(loc='upper center', bbox_to_anchor=(1.15, 0.8), ncol=1, fontsize=8)

    plt.suptitle("Remove ELM Classes outliers")

    # Tight layout to avoid overlapping labels
    plt.tight_layout()

    # Save the plot if a filepath is provided
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')

    plt.show()

    final_filtered_pip = filtered_df[filtered_df["ELMIdentifier"].isin(largest_cluster["ELMIdentifier"])]
    final_filtered_df = df[df["ELMIdentifier"].isin(largest_cluster["ELMIdentifier"])]
    final_known_df = df_known[df_known["ELMIdentifier"].isin(largest_cluster["ELMIdentifier"])]
    final_filtered_pathogenic_df = filtered_pathogenic_df[filtered_pathogenic_df["ELMIdentifier"].isin(largest_cluster["ELMIdentifier"])]
    final_filtered_extension_df = filtered_extension_df[filtered_extension_df["ELMIdentifier"].isin(largest_cluster["ELMIdentifier"])]

    final_filtered_pip.to_csv(f"{files}/elm/elm_pip_disorder_corrected.tsv", sep="\t", index=False)
    final_filtered_pathogenic_df.to_csv(f"{files}/elm/elm_pathogenic_region_disorder_corrected.tsv", sep="\t", index=False)
    final_filtered_extension_df.to_csv(f"{files}/elm/elm_pip_extension_disorder_corrected.tsv", sep="\t", index=False)
    final_filtered_df.to_csv(f"{files}/elm/elm_predicted_disorder_corrected.tsv", sep="\t", index=False)
    final_known_df.to_csv(f"{files}/elm/elm_known_disorder_corrected.tsv", sep="\t", index=False)

if __name__ == "__main__":
    core_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations'
    files = os.path.join(core_dir, "processed_data/files/elm")
    # files = r"D:\PycharmProjects\IDP_Pathogenic_Mutations\processed_data\files\elm"
    plot_dir = os.path.join(core_dir, "plots/sm/motif")
    filter_files_high_predicted_classes(filepath=os.path.join(plot_dir, f"clustering.png"))