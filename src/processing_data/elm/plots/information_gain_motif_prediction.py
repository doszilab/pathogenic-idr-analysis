import pandas as pd
import numpy as np
from itertools import product

from matplotlib.pyplot import savefig
from tqdm import tqdm
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy, linregress
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

def apply_filters(df):
    """
    Pre-compute the columns required for filtering.
    """
    df['Key_vs_NonKey_Difference'] = df['key_residue_am_mean_score'] - df['flanking_residue_am_mean_score']
    df['Motif_vs_Sequential_Difference'] = df['motif_am_mean_score'] - df['sequential_am_score']
    return df


# def calculate_entropy(column):
#     """
#     Calculate entropy for a given column.
#     """
#     proportions = column.value_counts(normalize=True)
#     return -np.sum(proportions * np.log2(proportions + 1e-10))

def compute_rowwise_am_max(df):
    """
    Compute the maximum score from the `motif_am_scores` column for each row.
    """
    df['AM_Max'] = df['motif_am_scores'].apply(
        lambda x: max(map(float, x.split(', '))) if isinstance(x, str) else np.nan
    )
    return df


def calculate_information_gain(df, filter_mask, column):
    """
    Calculate information gain for a given threshold filter using scikit-learn's mutual_info_score.
    """
    binarized_split = filter_mask.astype(int)
    information_gain = mutual_info_score(binarized_split, df[column])
    return information_gain


def evaluate_single_metric_ig(df,df_predicted, column, thresholds):
    """
    Evaluate information gain for a single metric across multiple thresholds.
    """
    results = []

    df = df.dropna(subset=[column])
    df_predicted = df_predicted.dropna(subset=[column]).sample(n=len(df))
    for threshold in thresholds:

        both_df = pd.concat([df,df_predicted])

        filter_mask = both_df[column] >= threshold
        filter_mask_for_known = df[column] >= threshold
        filter_mask_predicted = df_predicted[column] >= threshold

        # Split the dataset based on the threshold
        filtered_df = both_df[filter_mask]
        non_filtered_df = both_df[~filter_mask]

        # Calculate the entropy of each subset and the full dataset
        entropy_filtered = calculate_entropy(filtered_df[column])
        entropy_non_filtered = calculate_entropy(non_filtered_df[column])
        entropy_full = calculate_entropy(both_df[column])

        # Weight the entropies of the subsets
        weight_filtered = len(filtered_df) / len(both_df)
        weight_non_filtered = len(non_filtered_df) / len(both_df)

        weighted_entropy = (weight_filtered * entropy_filtered) + (weight_non_filtered * entropy_non_filtered)

        # Calculate Information Gain
        information_gain = entropy_full - weighted_entropy

        retention = len(df[filter_mask_for_known]) / len(df)
        retention_predicted = len(df_predicted[filter_mask_predicted]) / len(df_predicted)
        results.append({
            'Threshold': threshold,
            'Retention %': round(retention * 100, 2),
            'Retention Predicted %': round(retention_predicted * 100, 2),
            'Information Gain': round(information_gain, 4),
        })

    return pd.DataFrame(results)

def calculate_entropy(series):
    """
    Calculate entropy for a given pandas Series.
    """
    value_counts = series.value_counts(normalize=True)
    return entropy(value_counts)

def evaluate_single_metric_ig_predicted_known(df, df_predicted, column, thresholds):
    """
    Evaluate information gain for a single metric across multiple thresholds.
    """
    results = []

    # Add motif labels for combined dataset
    df['Motif_Label'] = 1  # Known motifs
    df_predicted['Motif_Label'] = 0  # Predicted motifs

    # Downsample predicted motifs to match known motifs
    # df_predicted = df_predicted.dropna(subset=[column]).sample(n=len(df), random_state=42)
    df_predicted = df_predicted.dropna(subset=[column])

    # Combine datasets
    both_df = pd.concat([df, df_predicted])

    for threshold in tqdm(thresholds):
        # Apply threshold filter
        filter_mask = both_df[column] >= threshold
        filtered_df = both_df[filter_mask]
        non_filtered_df = both_df[~filter_mask]

        # Calculate entropies based on Motif_Label
        entropy_filtered = calculate_entropy(filtered_df['Motif_Label'])
        entropy_non_filtered = calculate_entropy(non_filtered_df['Motif_Label'])
        entropy_full = calculate_entropy(both_df['Motif_Label'])

        # Weighted entropy
        weight_filtered = len(filtered_df) / len(both_df)
        weight_non_filtered = len(non_filtered_df) / len(both_df)
        weighted_entropy = (weight_filtered * entropy_filtered) + (weight_non_filtered * entropy_non_filtered)

        # Information Gain
        information_gain = entropy_full - weighted_entropy

        # Calculate retention for known and predicted motifs
        # retention_known = filter_mask[:len(df)].mean()
        # retention_predicted = filter_mask[len(df):].mean()  # Second part of combined dataset
        retention_known = filtered_df[filtered_df['Motif_Label'] == 1].shape[0] / df.shape[0]
        retention_predicted = filtered_df[filtered_df['Motif_Label'] == 0].shape[0] / df_predicted.shape[0]

        results.append({
            'Threshold': threshold,
            'Retention %': round(retention_known * 100, 2),
            'Retention Predicted %': round(retention_predicted * 100, 2),
            'Information Gain': information_gain,
        })

    return pd.DataFrame(results)


def calculate_entropy_for_max(df, column, max_row):
    """
    Calculate information gain for a given column and its maximum threshold row.
    """
    # Extract the threshold from the max_row
    threshold = max_row['Threshold']

    # Split the dataset based on the threshold
    filter_mask = df[column] >= threshold
    filtered_df = df[filter_mask]
    non_filtered_df = df[~filter_mask]

    # Calculate the entropy of each subset and the full dataset
    entropy_filtered = calculate_entropy(filtered_df[column])
    entropy_non_filtered = calculate_entropy(non_filtered_df[column])
    entropy_full = calculate_entropy(df[column])

    # Weight the entropies of the subsets
    weight_filtered = len(filtered_df) / len(df)
    weight_non_filtered = len(non_filtered_df) / len(df)

    weighted_entropy = (weight_filtered * entropy_filtered) + (weight_non_filtered * entropy_non_filtered)

    # Calculate Information Gain
    information_gain = entropy_full - weighted_entropy

    print(f"Max Row: {max_row}")
    print(f"Entropy (Filtered): {entropy_filtered}")
    print(f"Entropy (Non-Filtered): {entropy_non_filtered}")
    print(f"Entropy (Full Dataset): {entropy_full}")
    print(f"Information Gain: {information_gain}")

    return information_gain


def evaluate_two_metric_ig(df, column_1, column_2, thresholds_1, thresholds_2):
    """
    Evaluate information gain for combinations of two metrics across multiple thresholds.
    """
    results = []

    df = df.dropna(subset=[column_1, column_2])

    for threshold_1, threshold_2 in tqdm(product(thresholds_1, thresholds_2)):
        filter_mask = (df[column_1] >= threshold_1) & (df[column_2] >= threshold_2)
        retention = len(df[filter_mask]) / len(df)
        information_gain = calculate_information_gain(df, filter_mask, column_1)  # Use first metric as reference
        results.append({
            'Threshold 1': threshold_1,
            'Threshold 2': threshold_2,
            'Retention %': round(retention * 100, 2),
            'Information Gain': round(information_gain, 4),
        })

    return pd.DataFrame(results)

def evaluate_two_metric_ig_predicted_known(df, df_predicted, column_1, column_2, thresholds_1, thresholds_2):
    """
    Evaluate information gain for combinations of two metrics across multiple thresholds
    for both known and predicted motifs.
    """
    results = []

    # Add motif labels for combined dataset
    df['Motif_Label'] = 1  # Known motifs
    df_predicted['Motif_Label'] = 0  # Predicted motifs

    # Downsample predicted motifs to match known motifs (optional)
    # df_predicted = df_predicted.dropna(subset=[column_1, column_2]).sample(n=len(df), random_state=42)
    df_predicted = df_predicted.dropna(subset=[column_1, column_2])
    df = df.dropna(subset=[column_1, column_2])
    print(df)
    print(df_predicted)

    # Combine datasets
    both_df = pd.concat([df, df_predicted])

    for threshold_1, threshold_2 in tqdm(product(thresholds_1, thresholds_2)):
        # Apply threshold filters for both columns
        filter_mask = (both_df[column_1] >= threshold_1) & (both_df[column_2] >= threshold_2)
        filtered_df = both_df[filter_mask]
        non_filtered_df = both_df[~filter_mask]

        print(filtered_df)
        print(non_filtered_df)
        exit()

        # Calculate entropies based on Motif_Label
        entropy_filtered = calculate_entropy(filtered_df['Motif_Label'])
        entropy_non_filtered = calculate_entropy(non_filtered_df['Motif_Label'])
        entropy_full = calculate_entropy(both_df['Motif_Label'])

        # Weighted entropy
        weight_filtered = len(filtered_df) / len(both_df)
        weight_non_filtered = len(non_filtered_df) / len(both_df)
        weighted_entropy = (weight_filtered * entropy_filtered) + (weight_non_filtered * entropy_non_filtered)

        # Information Gain
        information_gain = entropy_full - weighted_entropy

        # Calculate retention for known and predicted motifs
        retention_known = filter_mask[:len(df)].mean()  # First part of combined dataset
        retention_predicted = filter_mask[len(df):].mean()  # Second part of combined dataset

        # Store results
        results.append({
            'Threshold 1': threshold_1,
            'Threshold 2': threshold_2,
            'Retention %': round(retention_known * 100, 2),
            'Retention Predicted %': round(retention_predicted * 100, 2),
            'Information Gain': round(information_gain, 4),
        })

    return pd.DataFrame(results)



def plot_3d_ig(results, column_1, column_2):
    """
    Create a 3D plot of information gain for two metrics.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        results['Threshold 1'], results['Threshold 2'], results['Information Gain'],
        c=results['Information Gain'], cmap='viridis', s=50, alpha=0.5
    )
    ax.set_xlabel(f'Threshold: {column_1}')
    ax.set_ylabel(f'Threshold: {column_2}')
    ax.set_zlabel('Information Gain')
    ax.set_title(f'Information Gain for Two-Metric Combination ({column_1}, {column_2})')

    # Highlight the maximum information gain
    max_row = results.loc[results['Information Gain'].idxmax()]
    ax.scatter(max_row['Threshold 1'], max_row['Threshold 2'], max_row['Information Gain'],
               color='red', s=100, label=f"Max IG: {max_row['Information Gain']:.2f}")

    plt.legend(
        title=f"Max IG: {max_row['Information Gain']:.2f}"
              f"\n{column_1}: {max_row['Threshold 1']:.2f}"
              f"\n{column_2}: {max_row['Threshold 2']:.2f}"
              f"\nRetention: {max_row['Retention %']:.2f}%"
              f"\nRetention Predicted: {max_row['Retention Predicted %']:.2f}%",
        loc="upper left",
        bbox_to_anchor=(-0.2, 0),  # Shift the legend to the left outside the chart
        fontsize=8,
        title_fontsize=10
    )

    plt.tight_layout()
    plt.show()


def plot_single_metric_ig(results, column,name_mapping,figsize=(8, 6),filepath=None,title=None):
    """
    Plot Information Gain vs Threshold for a single metric.

    Parameters:
        results (pd.DataFrame): DataFrame with Threshold, Retention %, and Information Gain.
        column (str): The metric name being plotted.
    """
    plt.figure(figsize=figsize)
    plt.plot(results['Threshold'], results['Information Gain'], marker='o', label='Information Gain')
    plt.xlabel('Threshold')
    plt.ylabel('Information Gain')
    if title:
        plt.title(title)
    else:
        plt.title(f'Information Gain vs Threshold for {name_mapping[column]}')
    plt.grid(True)

    # Find max Information Gain and corresponding Threshold and Retention
    max_row = results.loc[results['Information Gain'].idxmax()]
    max_threshold = max_row['Threshold']
    max_retention = max_row['Retention %']
    max_retention_predicted = max_row['Retention Predicted %']
    max_info_gain = max_row['Information Gain']

    filtered = elm_predicted_with_am_info_all[elm_predicted_with_am_info_all[column] >= max_threshold]

    retention_for_all_predicted = len(filtered) / len(elm_predicted_with_am_info_all) * 100

    print(retention_for_all_predicted)

    # Add max point to legend
    plt.legend(title=
               # f"Max IG: {max_info_gain:.3f}\n"
               f"Threshold: {max_threshold:.2f}\n"
               f"Retentions: \n"
               f"- Known: {max_retention:.2f}%\n"
               f"- Predicted: {retention_for_all_predicted:.2f}%"
               ,
               loc="upper left")

    # Plot a red scatter point at the max Information Gain
    plt.plot(max_threshold, max_info_gain, 'ro', markersize=10)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
    plt.show()



def plot_filtered_vs_total_count_clustered(df, filtered_df, column, name_mapping, max_threshold, n_clusters=4, figsize=(12, 10), filepath=None, islogscale=False, df1_text="Predicted", df2_text="Predicted AM Sequential"):
    """
    Scatter plot for total count vs filtered count based on a threshold with clustering and percentage of points in each cluster.

    Parameters:
    df (pd.DataFrame): DataFrame containing the full ELM data and metrics.
    filtered_df (pd.DataFrame): DataFrame containing the filtered data.
    column (str): The metric name being analyzed.
    name_mapping (dict): Mapping of metric names to display names.
    max_threshold (float): Threshold value used for filtering.
    n_clusters (int): Number of clusters for KMeans clustering.
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

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    plot_data['Cluster'] = kmeans.fit_predict(plot_data[['Total Count', 'Filtered Count']])

    # Assign unique colors to each cluster
    cluster_colors = plt.cm.get_cmap('tab10', n_clusters)

    # Prepare for legend entries
    line_labels = []

    largest_cluster = None
    largest_cluster_size = 0

    # Create the figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # First subplot: Total vs Filtered Count with clusters and regression lines
    ax1 = axes[0]
    ax1.grid()

    for cluster in range(n_clusters):
        cluster_data = plot_data[plot_data['Cluster'] == cluster]
        proportion = len(cluster_data) / len(plot_data) * 100
        if proportion > largest_cluster_size:
            largest_cluster = cluster_data
            largest_cluster_size = proportion

        # Scatter plot for each cluster
        ax1.scatter(
            cluster_data['Total Count'],
            cluster_data['Filtered Count'],
            label=f'Cluster {cluster+1} ({proportion:.2f}%)',
            color=cluster_colors(cluster),
            s=100,
            edgecolor='k',
            alpha=0.8,
            zorder=2
        )

    # Optional: Apply log scale if requested
    if islogscale:
        ax1.set_xscale('log')
        ax1.set_yscale('log')

    ax1.set_xlabel(f'{df1_text} Instances')
    ax1.set_ylabel(f'{df2_text} Instances')
    ax1.set_title(f'Clustering')

    # Second subplot: Percentages and Counts for ELM classes
    ax2 = axes[1]
    ax2.grid()

    plot_data = largest_cluster.merge(df[['ELMIdentifier', 'ELMType']].drop_duplicates(),
                                      on='ELMIdentifier', how='left')

    # Assign unique colors to each ELMType
    unique_types = df['ELMType'].unique()
    # reversed_colors = sorted(list(plt.cm.tab10.colors[:len(unique_types)]))
    # type_colors = {etype: color for etype, color in zip(unique_types, reversed_colors)}
    type_colors = {etype: color for etype, color in zip(unique_types, plt.cm.tab10.colors[:len(unique_types)])}

    # Plot the percentages and counts for ELM classes
    elm_class_counts = plot_data['ELMType'].value_counts()
    elm_info = { }
    for i, (etype, count) in enumerate(elm_class_counts.items()):
        percentage = (count / len(plot_data)) * 100
        elm_info[etype] = f'{etype}: {percentage:.2f}% ({count} classes)'

    for etype, group_data in plot_data.groupby('ELMType'):
        ax2.scatter(
            group_data['Total Count'],
            group_data['Filtered Count'],
            label=elm_info[etype],
            color=type_colors[etype],
            s=100,
            edgecolor='k',
            alpha=0.8,
            zorder=2
        )

    ax2.set_xlabel(f'{df1_text} Instances')
    ax2.set_ylabel(None)
    ax2.set_title(f'Largest Cluster')


    # Move legends outside the plots and place them below each other
    fig.legend(loc='upper center', bbox_to_anchor=(1.15, 0.8), ncol=1, fontsize=8)

    plt.suptitle("Remove ELM Classes outliers based on KMeans Clustering")

    # Tight layout to avoid overlapping labels
    plt.tight_layout()

    # Save the plot if a filepath is provided
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')

    plt.show()

    return largest_cluster


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

    ax2.set_xlabel(f'{df1_text} Instances')
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

    return largest_cluster


def calculate_gain_ratio(df, column, threshold):
    """
    Calculate Gain Ratio for a given column and threshold.
    """
    filter_mask = df[column] >= threshold
    filtered_df = df[filter_mask]
    non_filtered_df = df[~filter_mask]

    # Calculate entropies
    entropy_full = calculate_entropy(df[column])
    entropy_filtered = calculate_entropy(filtered_df[column])
    entropy_non_filtered = calculate_entropy(non_filtered_df[column])

    # Weights
    weight_filtered = len(filtered_df) / len(df)
    weight_non_filtered = len(non_filtered_df) / len(df)

    # Intrinsic Value
    intrinsic_value = -(
        (weight_filtered * np.log2(weight_filtered + 1e-10)) +
        (weight_non_filtered * np.log2(weight_non_filtered + 1e-10))
    )

    # Information Gain
    weighted_entropy = (weight_filtered * entropy_filtered) + (weight_non_filtered * entropy_non_filtered)
    information_gain = entropy_full - weighted_entropy

    # Gain Ratio
    gain_ratio = information_gain / (intrinsic_value + 1e-10)  # Avoid division by zero

    return gain_ratio, information_gain, intrinsic_value


def generate_data():

    for column_of_interest in columns_to_check:
        # Single-metric evaluation
        def round_to_three(val):
            return round(val,3)

        aggregated_results = []

        for _ in range(1 ,2):
            # current_predicted_df = elm_predicted_with_am_info_all.sample(n=elm_known_with_am_info_all.shape[0] * 10)
            current_predicted_df = elm_predicted_with_am_info_all

            thresholds = list(sorted(set(map(round_to_three,current_predicted_df[column_of_interest].dropna().sort_values().unique().tolist()))))
            # thresholds = list(sorted(set(map(round_to_three,elm_known_with_am_info_all[column_of_interest].dropna().sort_values().unique().tolist()))))
            min_threshold = min(thresholds)
            max_threshold = max(thresholds)
            thresholds = np.linspace(min_threshold, max_threshold, 1000)
            # print(len(thresholds))
            # print(thresholds)
            # thresholds = elm_known_with_am_info_all[column_of_interest].dropna().sort_values().unique().tolist()

            single_metric_results = evaluate_single_metric_ig_predicted_known(elm_known_with_am_info_all,current_predicted_df, column_of_interest, thresholds)

            print("Single Metric Results:")
            print(single_metric_results)
            aggregated_results.append(single_metric_results)

        big_df = pd.concat(aggregated_results,ignore_index=True)

        print(big_df)

        big_df = big_df.sort_values("Threshold")

        big_df.to_csv(f"{path_to_generate}/{column_of_interest}.tsv", index=False,sep='\t')

def plot_the_result():
    plot_dir= "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/plots/fig5"
    # for column_of_interest in columns_to_check:
    #     big_df = pd.read_csv(f"{path_to_generate}/{column_of_interest}.tsv", sep='\t')
    #     plot_single_metric_ig(big_df, column_of_interest,name_mapping,figsize=(5, 3),filepath=os.path.join(plot_dir,f"{column_of_interest}.png"))

    columns_to_check = [
        "motif_am_mean_score",
        # "Key_vs_NonKey_Difference",
        "Motif_vs_Sequential_Difference",
        # "AM_Max",
    ]

    name_dct = {
        "motif_am_mean_score": "Mean Score",
        "AM_Max": "Maximum Score",
        "Motif_vs_Sequential_Difference": "Sequential Difference",
    }

    for column_of_interest in columns_to_check:
        # column_of_interest = 'Motif_vs_Sequential_Difference'
        big_df = pd.read_csv(f"{path_to_generate}/{column_of_interest}.tsv", sep='\t')

        # Fig 5C2
        plot_single_metric_ig(big_df, column_of_interest, name_mapping, figsize=(5, 3),
                              filepath=os.path.join(plot_dir, f"C_{column_of_interest}.png"),title=f'Information Gain For {name_dct[column_of_interest]}')


def make_correction(elm_known_with_am_info_all,elm_predicted_with_am_info_all):
    column_of_interest = "Motif_vs_Sequential_Difference"
    threshold = 0.15

    elm_predicted_with_am_info_all = elm_predicted_with_am_info_all[elm_predicted_with_am_info_all["ELMIdentifier"].isin(elm_known_with_am_info_all["ELMIdentifier"])]

    filtered_df = elm_predicted_with_am_info_all[elm_predicted_with_am_info_all[column_of_interest] >= threshold]

    # filtered_df = filtered_df[filtered_df["ELMIdentifier"].isin(elm_known_with_am_info_all["ELMIdentifier"])]

    # Fig SM2 - Scatterplot motif
    largest_cluster = plot_filtered_vs_total_count_clustered_cutoff(elm_predicted_with_am_info_all, elm_known_with_am_info_all,500,
                                                            figsize=(6, 3), islogscale=True,
                                                             df1_text="PEM", df2_text="Known",
                                                             filepath=os.path.join(plot_dir, f"clustering.png")
                                                             )

    final_filtered_df = filtered_df[filtered_df["ELMIdentifier"].isin(largest_cluster["ELMIdentifier"])]
    print(largest_cluster)

    print(filtered_df.shape[0])
    print(final_filtered_df.shape[0])

    # final_filtered_df.to_csv(f"{files}/elm_predicted_disorder_corrected.tsv", sep="\t", index=False)


def get_mostly_disorder_instances(known_df,known_disorder_df):

    known_counts = known_df['ELMIdentifier'].value_counts().to_dict()
    known_disorder_counts = known_disorder_df['ELMIdentifier'].value_counts().to_dict()

    mostly_disorder = []

    for elm_id,count in known_counts.items():
        disorder_count = known_disorder_counts.get(elm_id,0)

        if disorder_count > 0 and disorder_count / count > 0.5:
            mostly_disorder.append(elm_id)

    return mostly_disorder

if __name__ == "__main__":
    # Load data
    core_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations'
    files = os.path.join(core_dir,"processed_data/files/elm")
    # files = r"D:\PycharmProjects\IDP_Pathogenic_Mutations\processed_data\files\elm"
    plot_dir =  os.path.join(core_dir,"plots/sm/motif")
    path_to_generate = os.path.join(core_dir,"processed_data/files/elm/prediction")

    elm_known_with_am_info_all = pd.read_csv(f"{files}/elm_known_with_am_info_all_disorder.tsv", sep='\t')
    elm_known = pd.read_csv(f"{files}/elm_known_with_am_info_all.tsv", sep='\t')
    elm_predicted_with_am_info_all = pd.read_csv(f"{files}/elm_predicted_with_am_info_all_disorder_with_rules.tsv",
                                                 sep='\t',
                                                 # nrows=15111
                                                 )



    # Filter out mostly disorder instances
    mostly_disorder = get_mostly_disorder_instances(elm_known,elm_known_with_am_info_all)
    elm_known_with_am_info_all = elm_known_with_am_info_all[elm_known_with_am_info_all["ELMIdentifier"].isin(mostly_disorder)]

    # Pre-compute filtering columns
    elm_known_with_am_info_all = apply_filters(elm_known_with_am_info_all)
    elm_known_with_am_info_all = compute_rowwise_am_max(elm_known_with_am_info_all)

    # Pre-compute filtering columns
    elm_predicted_with_am_info_all = apply_filters(elm_predicted_with_am_info_all)
    elm_predicted_with_am_info_all = compute_rowwise_am_max(elm_predicted_with_am_info_all)

    name_mapping = {
        "motif_am_mean_score": 'AM Score Threshold',
        "Key_vs_NonKey_Difference": 'Key vs Non-Key Threshold',
        "Motif_vs_Sequential_Difference": 'Sequential Threshold',
        "AM_Max": 'AM Signal Threshold',
    }

    columns_to_check = [
        "motif_am_mean_score",
        "Key_vs_NonKey_Difference",
        "Motif_vs_Sequential_Difference",
        "AM_Max",
    ]


    # Information Gain data generation
    # generate_data()

    # exit()

    # Plot the results
    # plot_the_result()

    # Motif Correction for High False positives
    make_correction(elm_known_with_am_info_all,elm_predicted_with_am_info_all)



    exit()

    combinations_to_check = [
        ["motif_am_mean_score","Key_vs_NonKey_Difference"],
        ["motif_am_mean_score","Motif_vs_Sequential_Difference"],
        ["AM_Max","Motif_vs_Sequential_Difference"],
        ["Key_vs_NonKey_Difference","Motif_vs_Sequential_Difference"],
    ]

    for column_1, column_2 in combinations_to_check:
        n_bins = 100  # Adjust the number of bins as needed

        # Create binned thresholds for column_1
        thresholds_1 = pd.cut(
            elm_known_with_am_info_all[column_1], bins=n_bins, labels=False, retbins=True
        )[1]

        # Create binned thresholds for column_2
        thresholds_2 = pd.cut(
            elm_known_with_am_info_all[column_2], bins=n_bins, labels=False, retbins=True
        )[1]

        # Drop duplicates to ensure clean thresholds
        thresholds_1 = sorted(set(thresholds_1))
        thresholds_2 = sorted(set(thresholds_2))

        # Evaluate IG for both metrics
        two_metric_results = evaluate_two_metric_ig_predicted_known(
            elm_known_with_am_info_all, elm_predicted_with_am_info_all, column_1, column_2, thresholds_1, thresholds_2
        )

        print(f"Two Metric Results for {column_1} and {column_2}:")
        print(two_metric_results)

        # Plot results for two-metric evaluation
        plot_3d_ig(two_metric_results, name_mapping[column_1], name_mapping[column_2],)
