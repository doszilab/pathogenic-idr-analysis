import os.path
from typing import final

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold, ShuffleSplit, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from tqdm import tqdm

def compute_rowwise_am_max(df):
    """
    Compute the maximum score from the `motif_am_scores` column for each row.
    """
    df['AM_Max'] = df['motif_am_scores'].apply(
        lambda x: max(map(float, x.split(', '))) if isinstance(x, str) else np.nan
    )
    return df


def main_prediction(known_df,predicted_df,df_publication,df_publication_effected,df_publication_norman_davey,df_publication_effected_norman_davey,save=False):
    df = pd.concat([predicted_df, known_df])

    df = compute_rowwise_am_max(df)
    df_publication = compute_rowwise_am_max(df_publication)
    df_publication_effected = compute_rowwise_am_max(df_publication_effected)
    df_publication_norman_davey = compute_rowwise_am_max(df_publication_norman_davey)
    df_publication_effected_norman_davey = compute_rowwise_am_max(df_publication_effected_norman_davey)

    metrics_cols = [
        'motif_am_mean_score',
        'key_residue_am_mean_score',
        'flanking_residue_am_mean_score',
        'sequential_am_score',
        'Key_vs_NonKey_Difference',
        'Motif_vs_Sequential_Difference',
        'AM_Max'
    ]

    use_cols = ['known'] + metrics_cols
    # print(f"Predicted Motif - Predicted {df[(df['known'] == False)].shape[0]}")
    # print(f"Predicted Motif - Known {df[(df['known'] == True)].shape[0]}")
    df = df.dropna(subset=use_cols, axis=0, how='any')
    original_total = df[(df['known'] == False)].shape[0]
    original_known = df[(df['known'] == True)].shape[0]
    print(f"Predicted Motif - Predicted {original_total}")
    print(f"Predicted Motif - Known {original_known}")
    # exit()

    # and y to the 'known' column (1D array)
    X = df[metrics_cols].to_numpy()
    y = df['known'].astype(int).values

    idx_pos = np.where(y == 1)[0]  # indices of positives
    idx_neg = np.where(y == 0)[0]  # indices of negatives

    clf = tree.DecisionTreeClassifier(
        max_depth=3,
        criterion='entropy',
        class_weight='balanced',
        ccp_alpha=0.005  # Prune after training to remove redundant splits
    )
    # --------------------------
    # 1) Cross-Validation Phase
    # --------------------------
    # Evaluate model performance with k-fold cross-validation (e.g. 10-fold)
    cv = 10
    n_boostrap = 100
    all_scores = []
    for i in tqdm(range(1, n_boostrap)):
        sampled_neg = np.random.choice(idx_neg, size=len(idx_pos))
        sampled_indices = np.concatenate([idx_pos, sampled_neg])

        # Create the balanced "bootstrapped" X, y
        X_sample = X[sampled_indices]
        y_sample = y[sampled_indices]

        scores = cross_val_score(clf, X_sample, y_sample, cv=cv, scoring='accuracy')
        # all_scores.append(scores.mean())
        all_scores.extend(scores)

    # print(f"Cross-validation accuracy scores ({cv}-fold):", all_scores)
    print(f"Mean accuracy: {np.mean(all_scores):.3f} ± {np.std(all_scores):.3f}")

    # Create a DataFrame from scores
    df_scores = pd.DataFrame(all_scores, columns=["Accuracy"])

    # Plot as boxplot
    plt.figure(figsize=(6, 3))
    df_scores.boxplot(column="Accuracy", vert=False)
    plt.title(f"Cross-validation Accuracies ({cv}-fold, {n_boostrap} bootstrap)")
    plt.savefig("/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/plots/sm/decision_tree/accuracy.png", dpi=300)
    plt.show()
    # exit()

    # If you want a fold-by-fold confusion matrix/classification report:
    y_pred_cv = cross_val_predict(clf, X, y, cv=cv)
    print("\nConfusion Matrix from cross_val_predict:")
    print(confusion_matrix(y, y_pred_cv))
    print("\nClassification Report from cross_val_predict:")
    print(classification_report(y, y_pred_cv))

    # ----------------------------------
    # 2) Train Final Model on Full Data
    # ----------------------------------
    clf.fit(X, y)

    plt.figure(1, (15, 4))
    tree.plot_tree(clf, filled=True, feature_names=metrics_cols, fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "tree.png"), dpi=300)
    plt.show()
    exit()

    def prediction(df,X):
        predictions = clf.predict(X)

        # Add the predictions to the DataFrame
        df['predicted'] = predictions  # 0 or 1

        # (Optional) if you want a more descriptive text column:
        df['prediction_decision_tree'] = df['predicted'].map({0: False, 1: True})
        return df


    # Known
    df = prediction(df,X)

    # Publication - Ylva Ivarsson
    X = df_publication[metrics_cols].to_numpy()
    df_publication = prediction(df_publication,X)

    # Publication - Ylva Ivarsson
    X = df_publication_effected[metrics_cols].to_numpy()
    df_publication_effected = prediction(df_publication_effected, X)

    # Publication - Norman Davey
    X = df_publication_norman_davey[metrics_cols].to_numpy()
    df_publication_norman_davey = prediction(df_publication_norman_davey, X)

    # Publication - Norman Davey
    X = df_publication_effected_norman_davey[metrics_cols].to_numpy()
    df_publication_effected_norman_davey = prediction(df_publication_effected_norman_davey, X)


    total_predicted = df[(df['known'] == False) & (df['prediction_decision_tree'] == True)].shape[0]
    total_known = df[(df['known'] == True) & (df['prediction_decision_tree'] == True)].shape[0]
    total_publication = df_publication[df_publication['prediction_decision_tree'] == True].shape[0]
    total_publication_effected = df_publication_effected[df_publication_effected['prediction_decision_tree'] == True].shape[0]
    total_publication_norman_davey = df_publication_norman_davey[df_publication_norman_davey['prediction_decision_tree'] == True].shape[0]
    total_publication_effected_norman_davey = df_publication_effected_norman_davey[df_publication_effected_norman_davey['prediction_decision_tree'] == True].shape[0]

    print(f"Predicted Motif - Predicted {total_predicted}, {round(total_predicted / original_total * 100), 2}%")
    print(f"Predicted Motif - Known {total_known}, {round(total_known / original_known * 100, 2)}%")
    print(f"Predicted Motif - Publication Ylva Ivarsson {total_publication}, {round(total_publication / df_publication.shape[0] * 100, 2)}%")
    print(f"Predicted Motif - Publication Ylva Ivarsson Effected {total_publication_effected}, {round(total_publication_effected / df_publication_effected.shape[0] * 100, 2)}%")
    print(f"Predicted Motif - Publication Norman Davey Most Significant {total_publication_norman_davey}, {round(total_publication_norman_davey / df_publication_norman_davey.shape[0] * 100, 2)}%")
    print(f"Predicted Motif - Publication Norman Davey All Predicted {total_publication_effected_norman_davey}, {round(total_publication_effected_norman_davey / df_publication_effected_norman_davey.shape[0] * 100, 2)}%")


    if save:
        df.to_csv(
            os.path.join(base_dir, 'decision_tree', 'elm_predicted_with_am_info_all_disorder_class_filtered.tsv'), sep='\t',
            index=False)
        df_publication.to_csv(
            os.path.join(base_dir, 'decision_tree', 'publication_predicted_with_am_info_all_disorder_class_filtered.tsv'),
            sep='\t',
            index=False)
        df_publication_effected.to_csv(
            os.path.join(base_dir, 'decision_tree',
                         'publication_predicted_effected_with_am_info_all_disorder_class_filtered.tsv'),
            sep='\t',
            index=False)
        df_publication_norman_davey.to_csv(
            os.path.join(base_dir, 'decision_tree',
                         'publication_predicted_norman_davey_with_am_info_all_disorder_class_filtered.tsv'),
            sep='\t',
            index=False)
        df_publication_effected_norman_davey.to_csv(
            os.path.join(base_dir, 'decision_tree',
                         'publication_predicted_effected_norman_davey_with_am_info_all_disorder_class_filtered.tsv'),
            sep='\t',
            index=False)

def filter_df(df):
    df['Key_vs_NonKey_Difference'] = df['key_residue_am_mean_score'] - df['flanking_residue_am_mean_score']
    df['Motif_vs_Sequential_Difference'] = df['motif_am_mean_score'] - df['sequential_am_score']
    return df

if __name__ == "__main__":
    base_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/elm/'
    plot_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/plots/sm/decision_tree'

    df_predicted = pd.read_csv(
        os.path.join(base_dir,'elm_predicted_with_am_info_all_disorder_class_filtered.tsv'),
        sep='\t'
    )
    df_known = pd.read_csv(
        os.path.join(base_dir,'elm_known_with_am_info_all_disorder_class_filtered.tsv'),
        sep='\t'
    )
    df_publication = filter_df(pd.read_csv(
        os.path.join(base_dir, 'publication_with_am_info_all_disorder.tsv'),
        sep='\t'
    ))

    df_publication_effected = filter_df(pd.read_csv(
        os.path.join(base_dir, 'publication_effected_with_am_info_all_disorder.tsv'),
        sep='\t'
    ))

    df_publication_norman_davey = filter_df(pd.read_csv(
        os.path.join(base_dir, 'publication_norman_davey_with_am_info_all_disorder.tsv'),
        sep='\t'
    ))

    df_publication_effected_norman_davey = filter_df(pd.read_csv(
        os.path.join(base_dir, 'publication_effected_norman_davey_with_am_info_all_disorder.tsv'),
        sep='\t'
    ))

    # AlphaMissense
    main_prediction(df_known, df_predicted,df_publication,df_publication_effected,df_publication_norman_davey,df_publication_effected_norman_davey, save=True)

    # df_predicted = filter_df(pd.read_csv(
    #     os.path.join(base_dir,"conservation", 'elm_predicted.tsv'),
    #     sep='\t'
    # ))
    # df_known = filter_df(pd.read_csv(
    #     os.path.join(base_dir,"conservation", 'elm_known.tsv'),
    #     sep='\t'
    # ))
    #
    # # Conservation
    # main_prediction(df_known, df_predicted, save=False)

