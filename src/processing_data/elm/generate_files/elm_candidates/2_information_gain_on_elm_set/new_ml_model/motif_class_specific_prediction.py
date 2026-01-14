# Updated script with new logic:
# - Only input: Sequence + AlphaMissense (no class as input)
# - Outputs:
#     1. Motif probability (binary classifier)
#     2. Class prediction (multi-class classifier including 'unknown')

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
from tqdm import tqdm


class SequenceEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, max_len=15, alphabet="ACDEFGHIKLMNPQRSTVWY"):
        self.max_len = max_len
        self.alphabet = alphabet + 'X'
        self.aa_index = {aa: i for i, aa in enumerate(self.alphabet)}

    def fit(self, X, y=None):
        return self

    def transform(self, sequences):
        output = []
        for seq in sequences:
            vec = np.zeros((self.max_len, len(self.aa_index)))
            seq = seq.upper()[:self.max_len].ljust(self.max_len, 'X')
            for i, aa in enumerate(seq):
                idx = self.aa_index.get(aa, self.aa_index['X'])
                vec[i, idx] = 1
            output.append(vec.flatten())
        return np.array(output)


def prepare_features(df):
    df = df.dropna(subset=[
        'Matched_Sequence', 'ELMIdentifier',
        'motif_am_mean_score', 'key_residue_am_mean_score',
        'flanking_residue_am_mean_score', 'sequential_am_score'])

    df['Key_vs_NonKey_Difference'] = df['key_residue_am_mean_score'] - df['flanking_residue_am_mean_score']
    df['Motif_vs_Sequential_Difference'] = df['motif_am_mean_score'] - df['sequential_am_score']

    am_features = df[[
        'motif_am_mean_score', 'key_residue_am_mean_score',
        'flanking_residue_am_mean_score', 'sequential_am_score',
        'Key_vs_NonKey_Difference', 'Motif_vs_Sequential_Difference']].reset_index(drop=True)

    seq_encoder = SequenceEncoder()
    seq_features = pd.DataFrame(seq_encoder.transform(df['Matched_Sequence']))

    X = pd.concat([am_features, seq_features], axis=1)
    X.columns = X.columns.astype(str)
    return X, df


def train_binary_classifier(df):
    print("\n📌 Training binary classifier (is_motif)")
    X, df = prepare_features(df)
    y = df['known'].astype(int).values

    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print(f"✔️ Binary CV accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    clf.fit(X, y)
    df['Motif_Probability'] = clf.predict_proba(X)[:, 1]
    df['Motif_Predicted_Label'] = clf.predict(X)

    return df, clf


def train_class_classifier(df):
    print("\n📌 Training multi-class classifier (motif class)")
    # Only on known motifs
    df_known = df[df['Motif_Predicted_Label'] == 1].copy()
    X, _ = prepare_features(df_known)
    y_raw = df_known['ELMIdentifier']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    clf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1)
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print(f"✔️ Class CV accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    clf.fit(X, y)

    # Predict class for ALL motifs
    X_all, _ = prepare_features(df)
    y_pred = clf.predict(X_all)
    y_pred_label = label_encoder.inverse_transform(y_pred)
    df['Predicted_Class'] = y_pred_label

    return df, clf, label_encoder


def filter_top_pem_classes(df_known, df_pred, top_pct=0.25):
    # Count number of motifs per class in predicted set
    class_counts = df_pred['ELMIdentifier'].value_counts()

    # Identify top X% classes by number of predicted motifs
    num_top = int(len(class_counts) * top_pct)
    top_classes = class_counts.nlargest(num_top).index

    # Filter out top classes from both predicted and known
    df_known_filtered = df_known[~df_known['ELMIdentifier'].isin(top_classes)]
    df_pred_filtered = df_pred[~df_pred['ELMIdentifier'].isin(top_classes)]

    # Combine and return the result
    df_all_filtered = pd.concat([df_known_filtered, df_pred_filtered], ignore_index=True)

    print(f"📉 Removed top {top_pct*100:.0f}% most frequent classes (total {num_top})")
    print(f"🔢 Remaining motifs:")
    print(f" - Known: {df_known_filtered.shape[0]}")
    print(f" - Predicted: {df_pred_filtered.shape[0]}")
    print(f" - Total: {df_all_filtered.shape[0]}")

    return df_all_filtered, df_known_filtered, df_pred_filtered

if __name__ == "__main__":
    base_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/elm/'
    df_known = pd.read_csv(os.path.join(base_dir, 'elm_known_with_am_info_all_disorder_class_filtered.tsv'), sep='\t')
    df_pred = pd.read_csv(os.path.join(base_dir, 'elm_predicted_with_am_info_all_disorder_class_filtered.tsv'), sep='\t')

    # df_all = pd.concat([df_known, df_pred], ignore_index=True)
    df_all,df_known,df_pred = filter_top_pem_classes(df_known, df_pred, top_pct=0.25)

    df_all, motif_model = train_binary_classifier(df_all)
    df_all, class_model, class_encoder = train_class_classifier(df_all)

    only_predicted_ones = df_all[df_all['Motif_Predicted_Label'] == 1]

    only_predicted_ones.to_csv(os.path.join(base_dir, 'ml_model', 'final_motif_class_and_probability.tsv'), sep='\t', index=False)
    print("\n✅ Saved prediction file with motif probabilities and predicted classes.")