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
                                     # nrows=1000000,
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


    print(big_disordered_df.columns)
    print(big_ordered_df.columns)


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

    df = clinvar_df_setting(df,columns,review_star)
    # print(df)
    # exit()

    rules = {
        'ESM1B': [-7.5, False],
        'AlphaMissense': [0.5, True],
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


if __name__ == '__main__':

    files_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files'
    benchmark_path = os.path.join(files_dir,'benchmark')

    # Clinvar
    disorder_all = os.path.join(benchmark_path, "clinvar_disorder_all.tsv")
    ordered_all = os.path.join(benchmark_path, "clinvar_order_all.tsv")

    clinvar_dir = os.path.join(files_dir,'clinvar')
    clinvar_disorder = pd.read_csv(os.path.join(clinvar_dir, 'clinvar_disorder.tsv'), sep='\t').rename(columns={'Protein_position':"Position"})
    clinvar_order = pd.read_csv(os.path.join(clinvar_dir, 'clinvar_order.tsv'), sep='\t').rename(columns={'Protein_position':"Position"})

    needed_cols = ['Protein_ID', 'RCVaccession', 'Position', 'HGVSp_Short', 'Interpretation', 'ReviewStar']
    columns_for_final = ['Protein_ID', "RCVaccession", 'Position','protein_variant','Interpretation','ReviewStar']

    am_disorder = pd.read_csv(os.path.join(files_dir,'alphamissense','am_disorder.tsv'),sep='\t').rename(columns={"AlphaMissense":"AlphaMissense_pos"})
    am_order = pd.read_csv(os.path.join(files_dir,'alphamissense','am_order.tsv'),sep='\t').rename(columns={"AlphaMissense":"AlphaMissense_pos"})


    create_clinvar_pathogenic_all(disorder_all,ordered_all,clinvar_disorder,clinvar_order,am_disorder,am_order,needed_cols,columns_for_final)
    exit()

    # Plots
    # Clinvar
    disorder_clinvar_df = pd.read_csv(disorder_all, sep='\t')
    order_clinvar_df = pd.read_csv(ordered_all, sep='\t')
    # disorder_metrics_df = accuracy_for_each_predictor(disorder_clinvar_df,name="Disorder",review_star=1)
    # ordered_metrics_df = accuracy_for_each_predictor(order_clinvar_df,name="Order",review_star=1)
    # accuracy_for_each_predictor(pd.concat([disorder_clinvar_df,order_clinvar_df]),name="All",review_star=1)
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

    # benchmark_proteingym(disorder_proteingym_df,'disorder',columns=columns)
    # benchmark_proteingym(order_proteingym_df,'order',columns=columns)
    # benchmark_proteingym(pd.concat([order_proteingym_df,disorder_proteingym_df]),"all",columns=columns)

    # Positional Based
    benchmark_proteingym(disorder_proteingym_df,'disorder',columns=columns,ispositional=True)
    benchmark_proteingym(order_proteingym_df,'order',columns=columns,ispositional=True)
    benchmark_proteingym(pd.concat([order_proteingym_df,disorder_proteingym_df]),"all",columns=columns,ispositional=True)
    # exit()

    # Clinvar and Proteingym
    # benchmark_proteingym_and_clinvar(disorder_proteingym_df,disorder_clinvar_df, "disorder")
    # benchmark_proteingym_and_clinvar(order_proteingym_df,order_clinvar_df, "order")
    # benchmark_proteingym_and_clinvar(order_proteingym_df,pd.concat([order_clinvar_df,disorder_clinvar_df]), "all")


    # Clinvar and Proteingym functional
    # proteingym_disorder_functional_regions = os.path.join(benchmark_path, "proteingym_disorder_functional_regions.tsv")
    # disorder_functional_proteingym_df = pd.read_csv(proteingym_disorder_functional_regions, sep='\t')
    # columns = [
    #     "binding_info",
    #     "dibs_info",
    #     "phasepro_info",
    #     "mfib_info",
    #     "Elm_Info"
    # ]
    # disorder_functional_proteingym_df = disorder_functional_proteingym_df.dropna(subset=columns,how="all")
    # benchmark_proteingym(disorder_functional_proteingym_df,'disorder functional')