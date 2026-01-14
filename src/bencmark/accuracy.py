import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score,accuracy_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, balanced_accuracy_score
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.utils import resample
import numpy as np
from tqdm import tqdm

def create_am_clinvar_merged_file(clinvar_file,pathogenicity_df,name=None,pathogenicity_cols=['AlphaMissense'],
                                  columns_for_final=['Protein_ID', "RCVaccession", 'protein_variant','Interpretation','ReviewStar']):
    only_binary = clinvar_file[clinvar_file['Interpretation'] != 'Uncertain'].copy()
    only_binary['protein_variant'] = only_binary['HGVSp_Short'].str[2:]
    # merged_df = only_binary.merge(am_df, on=['Protein_ID', 'protein_variant'],how='left')
    merged_df = only_binary.merge(pathogenicity_df, on=['Protein_ID', 'Position', 'protein_variant']).dropna(subset=pathogenicity_cols)
    merged_df = merged_df.drop_duplicates()
    print(f"for {name}")
    df_p = clinvar_file[clinvar_file["Interpretation"] == "Pathogenic"].shape[0]
    merged_p = merged_df[merged_df["Interpretation"] == "Pathogenic"].shape[0]
    df_b = clinvar_file[clinvar_file["Interpretation"] == "Benign"].shape[0]
    merged_b = merged_df[merged_df["Interpretation"] == "Benign"].shape[0]
    print(f"Statistics:")
    print(f"Pathogenic: {df_p} -> {merged_p}")
    print(f"Benign: {df_b} -> {merged_b}")
    df_to_save = merged_df[columns_for_final + pathogenicity_cols]
    return df_to_save

def proteingym_merged_file(proteingym_file,pathogenicity_df,pathogenicity_cols=['AlphaMissense'],
                           columns_for_final=['Protein_ID', 'protein_variant','DMS_id']):
    merged_df = proteingym_file.merge(pathogenicity_df, on=['Protein_ID', 'Position', 'protein_variant']).dropna(
        subset=pathogenicity_cols)
    df_to_save = merged_df[
        columns_for_final + pathogenicity_cols]
    return df_to_save

def create_clinvar_pathogenic_all(disorder_all,ordered_all,mutation_disorder,mutation_order,am_disorder,am_order,
                                  needed_cols=['Protein_ID', 'RCVaccession', 'Protein_position', 'HGVSp_Short', 'Interpretation', 'ReviewStar'],
                                  columns_for_final = ['Protein_ID', "RCVaccession", 'protein_variant','Interpretation','ReviewStar'],
                                  isclinvar=True
                                  ):
    clinvar_disorder_cleared = mutation_disorder[needed_cols].rename(columns={"Protein_position":"Position"})
    clinvar_order_cleared = mutation_order[needed_cols].rename(columns={"Protein_position":"Position"})

    if 'protein_variant' not in mutation_order.columns:
        clinvar_order_cleared['protein_variant'] = clinvar_disorder_cleared['HGVSp_Short'].str.split('.').str[1]
        clinvar_order_cleared['protein_variant'] = clinvar_order_cleared['HGVSp_Short'].str.split('.').str[1]

    am_all = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/data/discanvis_base_files/alphamissense/processed_alphamissense_results_mapping_new.tsv'
    prediction_dict = {
        "am": {
            'pathogenicity_cols': ['AlphaMissense'],
            'use_cols': ['Protein_ID', 'protein_variant', 'pos', 'am_pathogenicity'],
            "rename_cols": {"am_pathogenicity": "AlphaMissense", 'pos': 'Position'},
            "path": am_all,
        },
    }

    big_disordered_df = pd.DataFrame([],columns=columns_for_final)
    big_ordered_df = pd.DataFrame([],columns=columns_for_final)

    for key, info in prediction_dict.items():
        print(key)
        df_path = info['path']
        pathogenic_df = (pd.read_csv(df_path, sep='\t', usecols=info['use_cols'],
                                     low_memory=False).rename(columns=info['rename_cols'])).drop_duplicates()

        print(pathogenic_df)

        if isclinvar:
            disordered_merged_df = create_am_clinvar_merged_file(clinvar_disorder_cleared, pathogenic_df, name=f"{key} disorder",
                                                                 pathogenicity_cols=info['pathogenicity_cols'],columns_for_final=columns_for_final)
            ordered_merged_df = create_am_clinvar_merged_file(clinvar_order_cleared, pathogenic_df, name=f"{key} order",
                                                              pathogenicity_cols=info['pathogenicity_cols'],columns_for_final=columns_for_final)
        else:
            disordered_merged_df = proteingym_merged_file(clinvar_disorder_cleared, pathogenic_df,
                                                                 pathogenicity_cols=info['pathogenicity_cols'],columns_for_final=columns_for_final)
            ordered_merged_df = proteingym_merged_file(clinvar_order_cleared, pathogenic_df,
                                                              pathogenicity_cols=info['pathogenicity_cols'],columns_for_final=columns_for_final)

        big_disordered_df = big_disordered_df.merge(disordered_merged_df, how='outer',
                                                    on=columns_for_final)
        big_ordered_df = big_ordered_df.merge(ordered_merged_df, how='outer',
                                              on=columns_for_final)

    # Add AlphaMissense_Pos_based_score
    big_disordered_df = big_disordered_df.merge(am_disorder, how='left',
                                                on=['Protein_ID',"Position"])
    big_ordered_df = big_ordered_df.merge(am_order, how='left',
                                          on=['Protein_ID',"Position"])

    big_disordered_df.to_csv(disorder_all, sep='\t', index=False)
    big_ordered_df.to_csv(ordered_all, sep='\t', index=False)

def plot_roc_curves(df, title='Combined ROC Curves', file_name=None, columns=None,review_star=0):
    plt.figure(figsize=(10, 6))
    df['yTrue'] = df.apply(lambda row: 1 if row["Interpretation"] == 'Pathogenic' else 0, axis=1)

    for column in columns:
        scores = df[column].dropna()
        labels = df.loc[scores.index, 'yTrue']
        if column == 'ESM1B':
            labels = [1 if x == 0 else 0 for x in labels]

        fpr, tpr, _ = roc_curve(labels, scores)
        auc_score = roc_auc_score(labels, scores)
        plt.plot(fpr, tpr, lw=2, label=f'{column} ROC (area = {auc_score:.2f}, n = {len(scores):,d})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title + f" {review_star} star")
    plt.legend(loc="lower right")
    if file_name:
        plt.savefig(file_name)
    plt.show()


def plot_accuracy_barplot(accuracies, confidence_intervals, name, review_star):
    sorted_accuracies = {k: v for k, v in sorted(accuracies.items(), key=lambda item: item[1], reverse=True)}
    methods = list(sorted_accuracies.keys())
    scores = list(sorted_accuracies.values())
    ci_lower = [v - confidence_intervals[method][0] for method, v in sorted_accuracies.items()]
    ci_upper = [confidence_intervals[method][1] - v for method, v in sorted_accuracies.items()]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(methods, scores, xerr=[ci_lower, ci_upper], color='skyblue', capsize=5)
    plt.xlabel('Accuracy')
    plt.title(f'Accuracy of Pathogenicity Predictors ({name}, {review_star} stars)')
    plt.gca().invert_yaxis()  # Highest accuracy on top

    plt.xlim(0,1)

    # Adding accuracy scores on top of each bar
    for bar, score, lower, upper in zip(bars, scores, ci_lower, ci_upper):
        plt.text(score + upper + 0.005, bar.get_y() + bar.get_height() / 2,
                 f'{score:.3f}',
                 va='center', ha='left', fontsize=10)

    plt.show()


def plot_confusion_matrix_metrics(df, columns,name,isplot=True):
    metrics = {}
    for column in columns:
        y_true = df['yTrue']
        y_pred = df[column + '_pred']  # Adjusted to use the newly added prediction column

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        metrics[column] = {'FP': fp, 'TP': tp, 'FN': fn, 'TN': tn, 'F1': f1, 'Precision': precision, 'Recall': recall}

    metrics_df = pd.DataFrame(metrics).T

    if isplot:

        ax = metrics_df[['FP', 'TP', 'FN', 'TN']].plot(kind='bar', figsize=(14, 8))
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom')
        plt.title(f'Confusion Matrix Components for Each Predictor ({name})')
        plt.xlabel('Predictors')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()

        ax = metrics_df[['F1', 'Precision', 'Recall']].plot(kind='bar', figsize=(14, 8))
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom')
        plt.title(f'Metrics for Each Predictor ({name})')
        plt.xlabel('Predictors')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend(loc='lower right')
        plt.show()

    return metrics_df


def balance_classes(df):
    balanced_dfs = []
    for protein_id in df['Protein_ID'].unique():
        protein_df = df[df['Protein_ID'] == protein_id]
        pathogenic_df = protein_df[protein_df['Interpretation'] == 'Pathogenic']
        benign_df = protein_df[protein_df['Interpretation'] == 'Benign']

        min_class_size = min(len(pathogenic_df), len(benign_df))
        if min_class_size > 0:
            resampled_pathogenic_df = resample(pathogenic_df, n_samples=min_class_size, random_state=42)
            resampled_benign_df = resample(benign_df, n_samples=min_class_size, random_state=42)
            balanced_protein_df = pd.concat([resampled_pathogenic_df, resampled_benign_df])
            balanced_dfs.append(balanced_protein_df)

    balanced_df = pd.concat(balanced_dfs).reset_index()
    return balanced_df

def clinvar_df_setting(df,columns,review_star):
    df = df[df['ReviewStar'] >= review_star]
    df = df.dropna(subset=columns)
    # df['pos'] = df['protein_variant'].str[1:-1]
    # df = df.drop_duplicates(subset=['Protein_ID', 'pos'])
    # print(df)
    df = balance_classes(df)
    # print(df)
    # exit()
    return df


def bootstrap_accuracy(df, column, rules, n_iterations=1000):
    accuracies = []
    threshold, direction = rules[column]
    n_samples = int(round(df.shape[0] * 0.75, 0))
    print(n_samples)
    for _ in tqdm(range(n_iterations),desc="Bootstrap Accuracy"):
        # Resample with replacement
        resampled_df = resample(df, replace=True, n_samples=n_samples, random_state=None)
        resampled_df = resampled_df.dropna(subset=[column])
        if direction:
            resampled_df['yPred'] = resampled_df[column] >= threshold
        else:
            resampled_df['yPred'] = resampled_df[column] < threshold
        accuracy = balanced_accuracy_score(resampled_df['yTrue'], resampled_df['yPred'])
        accuracies.append(accuracy)

    return np.mean(accuracies), np.percentile(accuracies, 2.5), np.percentile(accuracies, 97.5)


def percentage_pathogenic_benign(df, columns=[
    'AlphaMissense', 'ESM1B', 'EVE_scores_ASM', 'REVEL',
    'VARITY_R', 'VARITY_ER', 'VARITY_R_LOO', 'VARITY_ER_LOO'
], name="Order"):
    rules = {
        'ESM1B': [-7.5, False],
        'AlphaMissense': [0.5, True],
        'AlphaMissense_pos': [0.5, True],
        'EVE_scores_ASM': [0.5, True],
        'REVEL': [0.5, True],
        'VARITY_R': [0.5, True],
        'VARITY_ER': [0.5, True],
        'VARITY_R_LOO': [0.5, True],
        'VARITY_ER_LOO': [0.5, True],
    }

    df = df.dropna(subset=columns, how="any")
    df['yTrue'] = df['Interpretation'].apply(lambda x: 1 if x == 'Pathogenic' else 0)

    results = []

    for column in columns:
        threshold, direction = rules[column]
        df[column + '_pred'] = df[column] >= threshold if direction else df[column] < threshold

        total = len(df)
        if total == 0:
            continue

        total_pathogenic = df[df['yTrue'] == 1].shape[0]
        correct_pathogenic = df[(df['yTrue'] == 1) & (df[column + '_pred'] == 1)].shape[0]
        total_benign = df[df['yTrue'] == 0].shape[0]
        correct_benign = df[(df['yTrue'] == 0) & (df[column + '_pred'] == 0)].shape[0]

        results.append({
            'Predictor': column,
            'Region': name,
            'Pathogenic (%)': (correct_pathogenic / total_pathogenic) * 100,
            'Total Pathogenic ':  total_pathogenic,
            'Benign (%)': (correct_benign / total_benign) * 100,
            'Total Benign': total_benign,
        })

    return pd.DataFrame(results)


def benchmark_plot_pathogenic_benign(disorder_metrics_df, ordered_metrics_df, name, columns=[
    'AlphaMissense', 'ESM1B', 'EVE_scores_ASM', 'REVEL',
    'VARITY_R', 'VARITY_ER'
], figsize=(7, 4), save_path=None):
    disorder_metrics_df = disorder_metrics_df[disorder_metrics_df['Predictor'].isin(columns)]
    ordered_metrics_df = ordered_metrics_df[ordered_metrics_df['Predictor'].isin(columns)]

    fig, ax = plt.subplots(figsize=figsize)

    bar_width = 0.2
    indices = np.arange(len(columns))

    colors = {
        'Disorder - Pathogenic': 'red',
        'Disorder - Benign': 'lightcoral',
        'Order - Pathogenic': 'blue',
        'Order - Benign': 'lightblue'
    }

    bars_pathogenic_disorder = ax.bar(indices - bar_width / 2, disorder_metrics_df['Pathogenic (%)'], bar_width,
                                      label='Disorder - Pathogenic', color=colors['Disorder - Pathogenic'])

    bars_pathogenic_order = ax.bar(indices + bar_width / 2, ordered_metrics_df['Pathogenic (%)'], bar_width,
                                   label='Order - Pathogenic', color=colors['Order - Pathogenic'])

    bars_benign_disorder = ax.bar(indices + 1.5 * bar_width, disorder_metrics_df['Benign (%)'], bar_width,
                                  label='Disorder - Benign', color=colors['Disorder - Benign'])

    bars_benign_order = ax.bar(indices + 2.5 * bar_width, ordered_metrics_df['Benign (%)'], bar_width,
                               label='Order - Benign', color=colors['Order - Benign'])

    for bars in [bars_pathogenic_disorder, bars_pathogenic_order, bars_benign_disorder, bars_benign_order]:
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 2, f'{yval:.1f}', ha='center', va='bottom')


    ax.set_ylabel('Percentage of Correctly Predicted')

    ax.set_xticks(indices + bar_width)

    rename_cols = {
        "AlphaMissense": "Variant",
        "AlphaMissense_pos": "Position",
    }

    labels = [rename_cols.get(x) for x in columns]
    print(labels)

    ax.set_xticklabels(labels,  ha='center')
    plt.tight_layout()
    plt.suptitle(f'Correctly Predicted ClinVar Variants')
    plt.subplots_adjust(top=0.85)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # ax.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()


def accuracy_for_each_predictor(df, name,review_star=0,
                                columns=[
                                'AlphaMissense',
                                'ESM1B',
                                'EVE_scores_ASM',
                                'REVEL',
                                'VARITY_R', 'VARITY_ER',
                                'VARITY_R_LOO', 'VARITY_ER_LOO'
                            ],isplot=True, n_bootstrap=1000
                                ):

    # df = clinvar_df_setting(df,columns,review_star)
    # print(df)
    # exit()

    df = df.dropna(subset=columns,how="any")

    rules = {
        'ESM1B': [-7.5, False],
        'AlphaMissense': [0.5, True],
        'AlphaMissense_pos': [0.5, True],
        'EVE_scores_ASM': [0.5, True],
        'REVEL': [0.5, True],
        'VARITY_R': [0.5, True],
        'VARITY_ER': [0.5, True],
        'VARITY_R_LOO': [0.5, True],
        'VARITY_ER_LOO': [0.5, True],
    }

    df['yTrue'] = df.apply(lambda row: 1 if row["Interpretation"] == 'Pathogenic' else 0, axis=1)
    accuracies = {}
    confidence_intervals = {}

    for column in columns:
        threshold, direction = rules[column]
        current_df = df.copy()
        current_df = current_df.dropna(subset=[column])
        if direction:
            current_df['yPred'] = current_df[column] >= threshold
        else:
            current_df['yPred'] = current_df[column] < threshold
        df[column + '_pred'] = current_df['yPred']

        # accuracies[column] = balanced_accuracy_score(current_df['yTrue'], current_df['yPred'])
        # accuracies[column] = accuracy_score(current_df['yTrue'], current_df['yPred'])

        mean_acc, lower_ci, upper_ci = bootstrap_accuracy(current_df, column, rules, n_iterations=n_bootstrap)
        accuracies[column] = mean_acc
        confidence_intervals[column] = (lower_ci, upper_ci)

    if isplot:
        for predictor, accuracy in accuracies.items():
            print(f'Accuracy for {predictor}: {accuracy:.2f} ({name})')

        plot_roc_curves(df, title=f'ROC Curves for {name}', columns=columns,review_star=review_star)
        plot_accuracy_barplot(accuracies,confidence_intervals, name, review_star)


    metrics_df = plot_confusion_matrix_metrics(df, columns,name,isplot=isplot)
    metrics_df['Accuracy'] = [accuracies[column] for column in metrics_df.index]
    metrics_df['95% CI Lower'] = [confidence_intervals[column][0] for column in metrics_df.index]
    metrics_df['95% CI Upper'] = [confidence_intervals[column][1] for column in metrics_df.index]
    return metrics_df


# def calculate_spearman_correlations_for_each_protein(df, columns, score_column):
#     spearman_results = {column: [] for column in columns}
#     proteins = df['Protein_ID'].unique()
#
#     for protein in proteins:
#         protein_df = df[df['Protein_ID'] == protein]
#         for column in columns:
#             if len(protein_df[column].unique()) > 1:  # Ensure there is variance in the data
#                 correlation, _ = spearmanr(protein_df[column], protein_df[score_column],nan_policy='omit')
#                 spearman_results[column].append(abs(correlation))  # Take the absolute value
#             else:
#                 spearman_results[column].append(None)  # Handle cases with no variance
#
#     return pd.DataFrame(spearman_results, index=proteins)

def calculate_spearman_correlations_for_each_protein(df, columns, score_column,ispositional=False):
    spearman_results = {column: [] for column in columns}
    proteins = df['Protein_ID'].unique()

    for protein in proteins:
        protein_df = df[df['Protein_ID'] == protein]
        dms_ids = protein_df['DMS_id'].unique()

        for column in columns:
            dms_correlations = []
            for dms_id in dms_ids:
                dms_df = protein_df[protein_df['DMS_id'] == dms_id]

                if len(dms_df[column].unique()) > 1:  # Ensure there is variance in the data
                    dms_score = dms_df[column]
                    column_score = dms_df[score_column]
                    if ispositional:
                        new_df = dms_df.copy()
                        new_df['Protein_position'] = new_df['protein_variant'].str[1:-1]
                        mean_column_df = new_df.groupby(['Protein_ID','Protein_position'])[column].mean().reset_index(name='dms_mean')
                        mean_dms_column_df = new_df.groupby(['Protein_ID','Protein_position'])[score_column].mean().reset_index(name='pathogenicity_mean')
                        dms_score = mean_column_df['dms_mean']
                        column_score = mean_dms_column_df['pathogenicity_mean']

                    if len(dms_score) < 4 or len(column_score) < 4:
                        continue

                    # print(len(dms_score))
                    # print(len(column_score))

                    correlation, _ = spearmanr(dms_score, column_score, nan_policy='omit')
                    dms_correlations.append(abs(correlation))  # Take the absolute value
            if dms_correlations:
                spearman_results[column].append(sum(dms_correlations) / len(dms_correlations))  # Average correlations
            else:
                spearman_results[column].append(None)

    spearman_df = pd.DataFrame(spearman_results, index=proteins)
    return spearman_df



def calculate_spearman_correlations(df, columns, score_column):
    spearman_results = {}

    for column in columns:
        correlation, _ = spearmanr(df[column], df[score_column],nan_policy='omit')
        spearman_results[column] = abs(correlation)  # Take the absolute value

    return spearman_results

def plot_spearman_violin(spearman_df, title):
    plt.figure(figsize=(14, 8))
    sns.violinplot(data=spearman_df, split=True, inner="quart")
    # sns.boxenplot(data=spearman_df)
    means = spearman_df.mean()

    for i, mean in enumerate(means):
        plt.text(i, mean + 0.02, f'{mean:.3f}', ha='center', va='bottom', color='black', fontsize=10)
        plt.scatter(i, mean, color='red')

    plt.title(title)
    plt.xlabel('Predictive Models')
    plt.ylabel('Spearman Correlation with DMS_score')
    # plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


def plot_spearman_scatter(spearman_df, title):
    # Determine the method with the highest mean correlation across all proteins
    best_method = spearman_df.mean().idxmax()

    # Sort the DataFrame by the Spearman correlation of the best method
    spearman_df_sorted = spearman_df.sort_values(by=best_method)

    plt.figure(figsize=(14, 8))

    for column in spearman_df_sorted.columns:
        plt.scatter(spearman_df_sorted.index, spearman_df_sorted[column], label=column, alpha=0.7)

    for protein in spearman_df_sorted.index:
        best_value = spearman_df_sorted.loc[protein, best_method]
        plt.plot([protein, protein], [0, best_value], color='gray', linestyle='-.', linewidth=0.5)

    plt.title(title)
    plt.xlabel('Proteins')
    plt.ylabel('Spearman Correlation with DMS_score')
    plt.ylim(0, 1)
    plt.legend(title='Predictive Models')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def plot_mean_spearman_vs_accuracy(spearman_df, accuracies_df, title):
    # Calculate the mean Spearman correlation for each method
    mean_spearman = spearman_df.mean()
    # best_method = spearman_df.mean().idxmax()

    # Extract accuracy values for each method
    accuracy_values = accuracies_df['Accuracy']

    # Ensure that the methods in both dataframes match
    common_methods = mean_spearman.index.intersection(accuracy_values.index)
    mean_spearman = mean_spearman[common_methods]
    accuracy_values = accuracy_values[common_methods]

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=accuracy_values, y=mean_spearman, hue=common_methods, s=100, palette="deep")

    for method in common_methods:
        plt.text(accuracy_values[method] + 0.001, mean_spearman[method], method, fontsize=12)

    plt.title(title)
    plt.legend(title='Method')
    plt.xlabel('Accuracy (ClinVar)')
    plt.ylabel('Mean Spearman Correlation')
    plt.tight_layout()
    plt.show()

def benchmark_proteingym(df,name,columns,ispositional=False):

    spearman_df = calculate_spearman_correlations_for_each_protein(df, columns, 'DMS_score',ispositional)
    # spearman_results = calculate_spearman_correlations(df, columns, 'DMS_score_bin')
    plot_spearman_violin(spearman_df, f'Spearman Correlation of Predictors with DMS_score across Proteins ({name})')
    plot_spearman_scatter(spearman_df, f'Spearman Correlation of Predictors with DMS_score for Each Protein ({name})')
    # exit()

def benchmark_proteingym_and_clinvar(disorder_proteingym_df,df,name):
    columns = [
        'AlphaMissense',
        'ESM1B',
        'EVE_scores_ASM',
        'REVEL',
        'VARITY_R', 'VARITY_ER',
        # 'VARITY_R_LOO','VARITY_ER_LOO'
    ]
    metrics_df = accuracy_for_each_predictor(df, name="Disorder", review_star=1,columns=columns,isplot=False)

    # disorder_proteingym_df = disorder_proteingym_df.dropna(subset=columns)

    spearman_df = calculate_spearman_correlations_for_each_protein(disorder_proteingym_df, columns, 'DMS_score')
    plot_mean_spearman_vs_accuracy(spearman_df, metrics_df,
                                   f'Mean Spearman Correlation vs Accuracy for ClinVar ({name})')


def benchmark_plot(metrics_df_disorder, metrics_df_order, name,
                            review_star=0, columns=[
            'AlphaMissense',
            'ESM1B',
            'EVE_scores_ASM',
            'REVEL',
            'VARITY_R', 'VARITY_ER',
        ],figsize=(18, 6),save_path=None):
    # Filter the df by columns
    metrics_df_disorder = metrics_df_disorder.loc[columns]
    metrics_df_order = metrics_df_order.loc[columns]


    # Create figure and subplots
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare the merged DataFrame for the accuracy plot
    metrics_df_order['Type'] = 'Order'
    metrics_df_disorder['Type'] = 'Disorder'

    merged_metrics_df = pd.concat([
        metrics_df_order[['Accuracy', '95% CI Lower', '95% CI Upper', 'Type']],
        metrics_df_disorder[['Accuracy', '95% CI Lower', '95% CI Upper', 'Type']],
    ], axis=0)

    # Ensure ordering of Models in merged metrics DataFrame
    merged_metrics_df['Model'] = merged_metrics_df.index
    merged_metrics_df['Model'] = pd.Categorical(merged_metrics_df['Model'], categories=columns, ordered=True)
    merged_metrics_df = merged_metrics_df.sort_values('Model')

    merged_metrics_df['Type'] = pd.Categorical(merged_metrics_df['Type'], categories=['Order', 'Disorder'],
                                               ordered=True)


    bar_width = 0.35
    indices = np.arange(len(merged_metrics_df) // 2)


    # Create bars for each type on ax2
    for i, (type_name, group) in enumerate(merged_metrics_df.groupby('Type')):
        bars = ax.bar(indices + i * bar_width, group['Accuracy'], bar_width,
                       yerr=[group['Accuracy'] - group['95% CI Lower'], group['95% CI Upper'] - group['Accuracy']],
                       label=type_name, capsize=5, color=COLORS[type_name])

        # Adding accuracy scores on top of each bar
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom')

    rename_cols = {
        "AlphaMissense": "Variant",
        "AlphaMissense_pos": "Position",
    }

    print(merged_metrics_df)

    labels = [rename_cols.get(x) for x in merged_metrics_df[merged_metrics_df['Type'] == 'Disorder']['Model'].tolist()]
    print(labels)

    # Customize ax2
    # ax2.set_xlabel('Pathogenicity Predictors')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'ClinVar Accuracy')
    ax.set_xticks(indices + bar_width / 2)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_xlim(-0.5, len(indices) - 0.5 + bar_width)
    ax.set_ylim(0, 1)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path,bbox_inches='tight',dpi=300)

    plt.show()

def bootstrap_percentage(df, column, rules, n_iterations=1000):
    threshold, direction = rules[column]

    pathogenic_accuracies = []
    benign_accuracies = []

    n_samples = int(round(df.shape[0] * 0.75, 0))
    for _ in tqdm(range(n_iterations), desc=f'Bootstrapping {column}'):
        sample = resample(df, replace=True, n_samples=n_samples)
        sample = sample.dropna(subset=[column])

        if direction:
            sample['yPred'] = sample[column] >= threshold
        else:
            sample['yPred'] = sample[column] < threshold

        sample_pathogenic = sample[sample['yTrue'] == 1]
        sample_benign = sample[sample['yTrue'] == 0]

        # Avoid division by zero
        if len(sample_pathogenic) > 0:
            correct_pathogenic = (sample_pathogenic['yPred'] == 1).sum()
            pathogenic_accuracies.append((correct_pathogenic / len(sample_pathogenic)) * 100)

        if len(sample_benign) > 0:
            correct_benign = (sample_benign['yPred'] == 0).sum()
            benign_accuracies.append((correct_benign / len(sample_benign)) * 100)

    return (
        np.mean(pathogenic_accuracies),
        np.percentile(pathogenic_accuracies, 2.5),
        np.percentile(pathogenic_accuracies, 97.5),
        np.mean(benign_accuracies),
        np.percentile(benign_accuracies, 2.5),
        np.percentile(benign_accuracies, 97.5),
    )


def benchmark_plot_pathogenic_benign_bootstrap(df_disorder, df_order, name, columns, n_bootstrap=1000, figsize=(7, 4), save_path=None):
    rules = {
        'ESM1B': [-7.5, False],
        'AlphaMissense': [0.5, True],
        'AlphaMissense_pos': [0.5, True],
        'EVE_scores_ASM': [0.5, True],
        'REVEL': [0.5, True],
        'VARITY_R': [0.5, True],
        'VARITY_ER': [0.5, True],
    }

    for df in [df_disorder, df_order]:
        df['yTrue'] = df['Interpretation'].apply(lambda x: 1 if x == 'Pathogenic' else 0)

    rows = []
    for column in columns:
        d_mean_p, d_low_p, d_high_p, d_mean_b, d_low_b, d_high_b = bootstrap_percentage(df_disorder, column, rules, n_iterations=n_bootstrap)
        o_mean_p, o_low_p, o_high_p, o_mean_b, o_low_b, o_high_b = bootstrap_percentage(df_order, column, rules, n_iterations=n_bootstrap)

        rows.append({
            'Predictor': column,
            'Region': 'Disorder',
            'Type': 'Pathogenic',
            'Mean': d_mean_p,
            'Lower': d_low_p,
            'Upper': d_high_p,
        })
        rows.append({
            'Predictor': column,
            'Region': 'Disorder',
            'Type': 'Benign',
            'Mean': d_mean_b,
            'Lower': d_low_b,
            'Upper': d_high_b,
        })
        rows.append({
            'Predictor': column,
            'Region': 'Order',
            'Type': 'Pathogenic',
            'Mean': o_mean_p,
            'Lower': o_low_p,
            'Upper': o_high_p,
        })
        rows.append({
            'Predictor': column,
            'Region': 'Order',
            'Type': 'Benign',
            'Mean': o_mean_b,
            'Lower': o_low_b,
            'Upper': o_high_b,
        })

    df_plot = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=figsize)
    bar_width = 0.2
    indices = np.arange(len(columns))

    rename_cols = {
        "AlphaMissense": "Variant",
        "AlphaMissense_pos": "Position",
    }
    labels = [rename_cols.get(x, x) for x in columns]

    type_region_colors = {
        ('Disorder', 'Pathogenic'): 'red',
        ('Disorder', 'Benign'): 'lightcoral',
        ('Order', 'Pathogenic'): 'blue',
        ('Order', 'Benign'): 'lightblue'
    }

    # Create position offsets for each bar group
    offsets = {
        ('Disorder', 'Pathogenic'): -bar_width / 2,
        ('Order', 'Pathogenic'): bar_width / 2,
        ('Disorder', 'Benign'): 1.5 * bar_width,
        ('Order', 'Benign'): 2.5 * bar_width,
    }

    # Plot bars in fixed order by looping over columns and each type
    for (region, var_type), color in type_region_colors.items():
        for i, predictor in enumerate(columns):
            subset = df_plot[
                (df_plot['Predictor'] == predictor) &
                (df_plot['Region'] == region) &
                (df_plot['Type'] == var_type)
                ]
            if subset.empty:
                continue

            mean = subset['Mean'].values[0]
            lower = subset['Lower'].values[0]
            upper = subset['Upper'].values[0]

            xpos = indices[i] + offsets[(region, var_type)]
            ax.bar(xpos, mean, bar_width,
                   yerr=[[mean - lower], [upper - mean]],
                   capsize=4,
                   label=f'{region} - {var_type}' if i == 0 else None,  # Only label first occurrence for legend
                   color=color)

            ax.text(xpos, mean + 3, f'{mean:.1f}', ha='center', va='bottom')

    # Formatting
    ax.set_xticks(indices + bar_width)
    ax.set_xticklabels(labels, ha='center')
    ax.set_ylabel('Percentage Correct')
    ax.set_ylim(0, 105)
    ax.set_title(f'ClinVar Classification Accuracy')

    # ax.legend()
    plt.tight_layout()

    for spine in ax.spines.values():
        spine.set_visible(False)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()


if __name__ == '__main__':

    COLORS = {
        "disorder": '#ffadad',
        "Disorder": '#ffadad',
        "order": '#a0c4ff',
        "Order": '#a0c4ff',
        "both": '#ffc6ff',
        "Pathogenic": '#ff686b',
        "Benign": "#b2f7ef",
        "Uncertain": "#f8edeb"
    }

    base_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations'
    files_dir = os.path.join(base_dir,'processed_data','files')
    benchmark_path = os.path.join(files_dir,'benchmark')
    fig_path = os.path.join(base_dir,'plots')

    # Clinvar
    # disorder_all = os.path.join(benchmark_path, "clinvar_disorder_all.tsv")
    # ordered_all = os.path.join(benchmark_path, "clinvar_order_all.tsv")

    disorder_all = os.path.join(benchmark_path, "clinvar_disorder_all_non_redundant.tsv")
    ordered_all = os.path.join(benchmark_path, "clinvar_order_all_non_redundant.tsv")

    # Plots
    columns = ["AlphaMissense", "AlphaMissense_pos"]

    # Clinvar
    disorder_clinvar_df = pd.read_csv(disorder_all, sep='\t')
    order_clinvar_df = pd.read_csv(ordered_all, sep='\t')


    disorder_metrics_df = percentage_pathogenic_benign(disorder_clinvar_df, name="Disorder",columns=columns)
    ordered_metrics_df = percentage_pathogenic_benign(order_clinvar_df, name="Order", columns=columns)

    print(disorder_metrics_df)
    print(ordered_metrics_df)

    # Fig 4A Accuracy for Benign and PAthogen
    # benchmark_plot_pathogenic_benign(disorder_metrics_df,ordered_metrics_df, "All", columns=columns, figsize=(4, 3),
    #                save_path=os.path.join(fig_path, 'fig4', 'ClinicalClassificationAccuracy.png')
    #                )

    benchmark_plot_pathogenic_benign_bootstrap(
        disorder_clinvar_df, order_clinvar_df, name="All",
        columns=columns, n_bootstrap=1000, figsize=(4, 3),
        save_path=os.path.join(fig_path, 'fig4', 'ClinicalClassificationAccuracy_with_CI.png')
    )

    exit()

    disorder_all = os.path.join(benchmark_path, "clinvar_disorder_all_non_redundant.tsv")
    ordered_all = os.path.join(benchmark_path, "clinvar_order_all_non_redundant.tsv")

    n_bootstrap = 1000

    # Plots

    columns = ["AlphaMissense","AlphaMissense_pos"]

    # Clinvar
    disorder_clinvar_df = pd.read_csv(disorder_all, sep='\t')
    order_clinvar_df = pd.read_csv(ordered_all, sep='\t')

    disorder_metrics_df = accuracy_for_each_predictor(disorder_clinvar_df,name="Disorder",review_star=1,columns=columns,n_bootstrap=n_bootstrap)
    ordered_metrics_df = accuracy_for_each_predictor(order_clinvar_df,name="Order",review_star=1,columns=columns,n_bootstrap=n_bootstrap)


    # Fig 4A Accuracy
    benchmark_plot(disorder_metrics_df,ordered_metrics_df,columns=columns,figsize=(4, 3),name="All",
                   save_path=os.path.join(fig_path,'fig4','A2.png')
                   )
    exit()
    accuracy_for_each_predictor(pd.concat([disorder_clinvar_df,order_clinvar_df]),name="All",review_star=1,columns=columns,n_bootstrap=n_bootstrap)
    exit()
    #
    disorder_mertics = os.path.join(benchmark_path, "clinvar_metrics_all_disorder.tsv")
    order_mertics = os.path.join(benchmark_path, "clinvar_metrics_all_order.tsv")
    #
    # disorder_metrics_df.to_csv(disorder_mertics,sep='\t',index=True)
    # ordered_metrics_df.to_csv(order_mertics,sep='\t',index=True)

    disorder_proteingym_df = pd.read_csv(disorder_proteingym_all, sep='\t')
    order_proteingym_df = pd.read_csv(ordered_proteingym_all, sep='\t')

    columns = [
        'AlphaMissense',
        'ESM1B',
        'EVE_scores_ASM',
        'REVEL',
        'VARITY_R', 'VARITY_ER',
        # 'VARITY_R_LOO','VARITY_ER_LOO'
    ]
