# src/motif_prediction_pipeline/model.py
import pandas as pd
import os
import joblib
from sklearn import tree
from ..config import PROCESSED_DATA_DIR


class MotifPredictor:
    def __init__(self, model_filename="decision_tree_v1.pkl"):
        """
        Initializes the predictor.
        Args:
            model_filename (str): Name of the model file in data/processed/models/
        """
        self.model = None
        self.model_path = os.path.join(PROCESSED_DATA_DIR, "models", model_filename)

        # EXACT Feature columns from original script
        self.feature_cols = [
            'motif_am_mean_score',
            'key_residue_am_mean_score',
            'flanking_residue_am_mean_score',
            'sequential_am_score',
            'Key_vs_NonKey_Difference',
            'Motif_vs_Sequential_Difference',
            'AM_Max'
        ]

        # Load model immediately if it exists
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            print(f"Warning: Model not found at {self.model_path}. You need to train it first.")

    def train(self, X, y):
        print("Training Decision Tree (Replicating original parameters)...")
        self.model = tree.DecisionTreeClassifier(
            max_depth=3,
            criterion='entropy',
            class_weight='balanced',
            ccp_alpha=0.005  # Important pruning parameter from original script
        )
        self.model.fit(X, y)
        self.save_model()
        return self.model

    def predict(self, df):
        if self.model is None:
            raise ValueError(f"Model not found or not trained. Path: {self.model_path}")

        missing_cols = [c for c in self.feature_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Input DataFrame is missing columns: {missing_cols}")

        X = df[self.feature_cols].fillna(0)

        preds = self.model.predict(X)

        try:
            probs = self.model.predict_proba(X)[:, 1]
        except:
            probs = preds

        result = df.copy()

        # --- RENAMED COLUMNS ---
        result['is_PEM'] = preds.astype(bool)
        result['PEM_probability'] = probs

        return result

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        self.model = joblib.load(self.model_path)
        print(f"Model loaded from {self.model_path}")