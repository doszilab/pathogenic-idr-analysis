import pandas as pd
import numpy as np
from itertools import product

from matplotlib.pyplot import savefig
from tqdm import tqdm
from scipy.stats import entropy, linregress
from scipy.signal import find_peaks
from sklearn.metrics import mutual_info_score
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

def entropy_gabor(data):
    if len(data) == 0:
        return 0.0
    p = sum(data) / len(data)
    if p == 0 or p == 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def calculate_information_gain_gabor(parent,c1,c2):
    gain = entropy_gabor(parent) - (len(c1) / len(parent) * entropy_gabor(c1) + len(c2) / len(parent) * entropy_gabor(c2))
    return gain

def calculate_information_gain_mine(both_df,filtered_df,non_filtered_df):
    # Calculate entropies based on Motif_Label
    entropy_filtered = calculate_entropy(filtered_df['Motif_Label'])
    entropy_non_filtered = calculate_entropy(non_filtered_df['Motif_Label'])
    entropy_full = calculate_entropy(both_df['Motif_Label'])

    # Weighted entropy
    weight_filtered = len(filtered_df) / len(both_df)
    weight_non_filtered = len(non_filtered_df) / len(both_df)
    weighted_entropy = (weight_filtered * entropy_filtered) + (weight_non_filtered * entropy_non_filtered)

    # print(entropy_filtered, entropy_non_filtered, entropy_full)
    # print(filtered_df, non_filtered_df, both_df)

    # Information Gain
    gain = entropy_full - weighted_entropy
    return gain

def evaluate_single_metric_ig_predicted_known(df, df_predicted, column, thresholds):
    """
    Evaluate information gain for a single metric across multiple thresholds.
    """
    results = []

    # Add motif labels for combined dataset
    df['Motif_Label'] = 1  # Known motifs
    df_predicted['Motif_Label'] = 0  # Predicted motifs

    # Downsample predicted motifs to match known motifs
    df_predicted = df_predicted.dropna(subset=[column])
    df = df.dropna(subset=[column])

    # Combine datasets
    both_df = pd.concat([df, df_predicted])

    for threshold in tqdm(thresholds):
        # Apply threshold filter
        filter_mask = both_df[column] >= threshold
        filtered_df = both_df[filter_mask]
        non_filtered_df = both_df[~filter_mask]

        information_gain = calculate_information_gain_gabor(both_df['Motif_Label'],non_filtered_df['Motif_Label'],filtered_df['Motif_Label'])

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
    if column == 'Key_vs_NonKey_Difference':
        results = results[results['Threshold'] > -0.5]

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

    # Add max point to legend
    plt.legend(title=
               # f"Max IG: {max_info_gain:.3f}\n"
               f"Threshold: {max_threshold:.2f}\n"
               # f"Retentions: \n"
               # f"- Known: {max_retention:.2f}%\n"
               # f"- Predicted: {retention_for_all_predicted:.2f}%"
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

    # plt.suptitle("Remove ELM Classes outliers")

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

        # current_predicted_df = elm_predicted_with_am_info_all.sample(n=elm_known_with_am_info_all.shape[0] * 10)
        # current_predicted_df = elm_predicted_with_am_info_all

        # thresholds = list(sorted(set(map(round_to_three,elm_predicted_with_am_info_all[column_of_interest].dropna().sort_values().unique().tolist()))))
        # min_threshold = min(thresholds)
        # max_threshold = max(thresholds)
        # thresholds = np.linspace(min_threshold, max_threshold, 100)
        # print(thresholds)
        # print(len(thresholds))
        # exit()

        single_metric_results = evaluate_single_metric_ig_predicted_known(elm_known_with_am_info_all,elm_predicted_with_am_info_all, column_of_interest, thresholds)

        big_df = single_metric_results

        print(big_df)

        big_df = big_df.sort_values("Threshold")

        big_df.to_csv(f"{path_to_generate}/{column_of_interest}.tsv", index=False,sep='\t')

def get_information_gain_max(df,column_of_interest,cls_column,resolution=1000):
    val = df[column_of_interest].to_numpy()
    cls = df[cls_column].to_numpy()

    # Sort the values
    ind = np.argsort(val)
    cls = cls[ind]
    val = val[ind]

    x, y = [], []
    max_pls, max_val = 0, -float('inf')
    for i in range(1,len(val), resolution):
        # Get split sets
        c1 = cls[:i]
        c2 = cls[i:]
        gain = calculate_information_gain_gabor(cls, c1, c2)
        x.append(val[i])
        y.append(gain)
        if gain > max_val:
            max_val = gain
            max_pls = val[i]
    return max_pls, max_val

def generate_bootstrapped_data(elm_known_with_am_info_all,elm_predicted_with_am_info_all,n_bootstrapped=1000):
    elm_known_with_am_info_all['Motif_Label'] = 1  # Known motifs
    elm_predicted_with_am_info_all['Motif_Label'] = 0  # Predicted motifs

    for column_of_interest in columns_to_check:
        def round_to_three(val):
            return round(val, 3)

        lst= []

        # Bootstrap loop
        df_known = elm_known_with_am_info_all.dropna(subset=[column_of_interest])
        df_predicted = elm_predicted_with_am_info_all.dropna(subset=[column_of_interest])
        for _ in tqdm(range(n_bootstrapped), desc=f"Bootstrapping IG calculations for {column_of_interest}"):
            sample_predicted = df_predicted.sample(n=df_known.shape[0])
            both_df = pd.concat([df_known, sample_predicted])

            max_pls, max_val = get_information_gain_max(both_df,column_of_interest,"Motif_Label",resolution=1000)
            lst.append([max_pls,max_val])

        df = pd.DataFrame(lst,columns=["Threshold",'Information Gain'])

        print(df['Threshold'].mean())
        print(df)

        plt.figure()
        df.boxplot(column="Threshold")
        plt.title(f"Box Plot of Threshold Values for {column_of_interest}")
        plt.ylabel("Threshold")
        plt.show()


def plot_the_result():
    plot_dir= "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/plots/fig5"
    # for column_of_interest in columns_to_check:
    #     big_df = pd.read_csv(f"{path_to_generate}/{column_of_interest}.tsv", sep='\t')
    #     plot_single_metric_ig(big_df, column_of_interest,name_mapping,figsize=(5, 3),filepath=os.path.join(plot_dir,f"{column_of_interest}.png"))

    columns_to_check = [
        "motif_am_mean_score",
        "Key_vs_NonKey_Difference",
        "Motif_vs_Sequential_Difference",
        "AM_Max",
    ]

    name_dct = {
        "motif_am_mean_score": "Mean Score",
        "AM_Max": "Maximum Score",
        "Motif_vs_Sequential_Difference": "Sequential Difference",
        "Key_vs_NonKey_Difference": 'Key vs Non-Key Threshold',
    }

    for column_of_interest in columns_to_check:
        # column_of_interest = 'Motif_vs_Sequential_Difference'
        big_df = pd.read_csv(f"{path_to_generate}/{column_of_interest}.tsv", sep='\t')

        # Fig 5C2
        plot_single_metric_ig(big_df, column_of_interest, name_mapping, figsize=(5, 3),
                              filepath=os.path.join(plot_dir, f"C_{column_of_interest}.png"),title=f'Information Gain For {name_dct[column_of_interest]}')


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

    ax1.set_xlabel("Cutoff Value (Total Count)")
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

def make_correction(elm_known_with_am_info_all,elm_predicted_with_am_info_all):
    column_of_interest = "Motif_vs_Sequential_Difference"
    threshold = 0.105

    elm_predicted_with_am_info_all = elm_predicted_with_am_info_all[elm_predicted_with_am_info_all["ELMIdentifier"].isin(elm_known_with_am_info_all["ELMIdentifier"].unique())]


    filtered_df = elm_predicted_with_am_info_all[elm_predicted_with_am_info_all[column_of_interest] >= threshold]
    filtered_known_df= elm_known_with_am_info_all[elm_known_with_am_info_all[column_of_interest] >= threshold]

    filtered_df = determine_known_for_predicted(filtered_df, filtered_known_df)

    cutoff_values = np.arange(0, 10001, 100)

    optimal_cutoff = plot_class_percentages_vs_instances(elm_predicted_with_am_info_all, elm_known_with_am_info_all, cutoff_values, figsize=(4, 3),
                                                         filepath=os.path.join(plot_dir, f"optimal_cutoff_for_classes.png"))

    # print(optimal_cutoff)
    # exit()



    # Fig SM2 - Scatterplot motif
    largest_cluster = plot_filtered_vs_total_count_clustered_cutoff(elm_predicted_with_am_info_all,elm_known_with_am_info_all, optimal_cutoff,
                                                            figsize=(6, 3), islogscale=True,
                                                             df1_text="Predicted", df2_text="Known",
                                                             filepath=os.path.join(plot_dir, f"clustering.png")
                                                             )

    filtered_df["Confidence"] = filtered_df["ELMIdentifier"].isin(largest_cluster["ELMIdentifier"]).map({True: "High", False: "Low"})
    high_confidence_df = filtered_df[filtered_df["Confidence"] == "High"]

    print(largest_cluster)

    print(filtered_df.shape[0])
    print(high_confidence_df.shape[0])
    total_elm_classes = filtered_df["ELMIdentifier"].nunique()
    filtered_elm_classes = high_confidence_df["ELMIdentifier"].nunique()
    print(total_elm_classes)
    print(filtered_elm_classes)
    print(filtered_elm_classes / total_elm_classes * 100)

    filtered_df.to_csv(f"{files}/elm_predicted_disorder_with_confidence.tsv", sep="\t", index=False)
    # low_confidence_df.to_csv(f"{files}/elm_predicted_disorder_low_confidence.tsv", sep="\t", index=False)


def get_mostly_disorder_instances(known_df,known_disorder_df):

    known_counts = known_df['ELMIdentifier'].value_counts().to_dict()
    known_disorder_counts = known_disorder_df['ELMIdentifier'].value_counts().to_dict()

    mostly_disorder = []

    for elm_id,count in known_counts.items():
        disorder_count = known_disorder_counts.get(elm_id,0)

        if disorder_count > 0 and disorder_count / count > 0.5:
            mostly_disorder.append(elm_id)

    return mostly_disorder

def add_rule_to_one(df,am_score_threshold = 0.48,key_vs_non_key_threshold= 0.35,sequential_threshold= 0.15,am_max=0.63):
    # Original filters
    am_score_filter = df['motif_am_mean_score'] >= am_score_threshold
    key_vs_non_key_filter = df['Key_vs_NonKey_Difference'] >= key_vs_non_key_threshold
    sequential_filter = df['Motif_vs_Sequential_Difference'] >= sequential_threshold
    am_max_filter = df['AM_Max'] >= am_max

    df['AM_Score_Rule'] = am_score_filter
    df['Key_Rule'] = key_vs_non_key_filter
    df['Sequential_Rule'] = sequential_filter
    df['AM_Max_Rule'] = am_max_filter
    df['All_Rule'] = am_score_filter & key_vs_non_key_filter & sequential_filter & am_max_filter
    return  df

def add_rule_by_information_gain(elm_known,elm_predicted):
    elm_known = add_rule_to_one(elm_known)
    elm_predicted = add_rule_to_one(elm_predicted)

    elm_known.to_csv(f"{files}/elm_known_class_filtered_with_rules.tsv",sep='\t', index=False)
    elm_predicted.to_csv(f"{files}/elm_predicted_class_filtered_with_rules.tsv",sep='\t', index=False)


if __name__ == "__main__":
    # Load data
    core_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations'
    files = os.path.join(core_dir,"processed_data/files/elm")
    # files = r"D:\PycharmProjects\IDP_Pathogenic_Mutations\processed_data\files\elm"
    plot_dir =  os.path.join(core_dir,"plots/sm/motif")
    path_to_generate = os.path.join(core_dir,"processed_data/files/elm/prediction")

    # elm_known = pd.read_csv(f"{files}/elm_known_with_am_info_all.tsv", sep='\t')
    # elm_known_with_am_info_all = pd.read_csv(f"{files}/elm_known_with_am_info_all_disorder.tsv", sep='\t')
    # elm_predicted_with_am_info_all = pd.read_csv(f"{files}/elm_predicted_with_am_info_all_disorder_with_rules.tsv",sep='\t',)
    elm_known_with_am_info_all = pd.read_csv(f"{files}/elm_known_with_am_info_all_disorder_class_filtered.tsv", sep='\t')
    elm_predicted_with_am_info_all = pd.read_csv(f"{files}/elm_predicted_with_am_info_all_disorder_class_filtered.tsv",sep='\t')

    # print(elm_known_with_am_info_all)
    # print(elm_predicted_with_am_info_all)
    # exit()

    # Filter out mostly disorder instances
    # mostly_disorder = get_mostly_disorder_instances(elm_known,elm_known_with_am_info_all)
    # elm_known_with_am_info_all = elm_known_with_am_info_all[elm_known_with_am_info_all["ELMIdentifier"].isin(mostly_disorder)]

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
        # "motif_am_mean_score",
        # "Key_vs_NonKey_Difference",
        # 'key_residue_am_mean_score',
        "Motif_vs_Sequential_Difference",
        # "AM_Max",
    ]


    # Information Gain data generation
    # generate_data()
    # generate_bootstrapped_data(elm_known_with_am_info_all,elm_predicted_with_am_info_all)

    # Plot the results
    # plot_the_result()
    # exit()

    # Add Rules
    # add_rule_by_information_gain(elm_known_with_am_info_all,elm_predicted_with_am_info_all)

    # Motif Correction for High False positives
    make_correction(elm_known_with_am_info_all,elm_predicted_with_am_info_all)
