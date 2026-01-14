# src/config.py
import os
from pathlib import Path
import matplotlib.pyplot as plt

# --- PATH CONFIGURATION ---

# Detect the project root directory relative to this file
ROOT_DIR = Path(__file__).parent.parent

# Input directories
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

# Output directories
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")

# Ensure output directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# --- VISUALIZATION CONFIGURATION ---

# Publication Quality Settings
SHOW_TITLES = True       # Set False for final paper figures (captions are used instead)
FIG_DPI = 300            # 300 for print, 100 for screen check
SAVE_FORMAT = 'png'      # 'png', 'pdf', 'svg', 'tiff'

# Global Matplotlib Settings (Optional: applies to displayed plots too)
# plt.rcParams['figure.dpi'] = 100  # Display resolution in notebook
# plt.rcParams['savefig.dpi'] = FIG_DPI # Default save resolution

# Global color palette
COLORS = {
    "disorder": '#ffadad',
    "order": '#a0c4ff',
    "both": '#ffc6ff',
    "Pathogenic": '#ff686b',
    "Benign": "#b2f7ef",
    "Uncertain": "#8e9aaf",
    "Mutated": '#f27059',
    "Non-Mutated": '#80ed99',
}

CATEGORY_COLORS_STRUCTURE = {
    'Only Disorder': 'red',
    'Disorder Mostly': COLORS['disorder'],
    'Equal': 'green',
    'Order Mostly': COLORS['order'],
    'Only Order': 'blue',
    'Mostly Disorder': '#f27059',
    'Disorder Specific': '#f27059',
    "Non-Disorder Specific": COLORS['Uncertain'],
    "Mostly Order/Equal": COLORS['Uncertain'],
    'disorder': COLORS['disorder'],
    'order': COLORS['order'],
    "Disorder-Pathogenic": COLORS['disorder'],
    'Order-Pathogenic': COLORS['order'],
}

CATEGORY_COLORS_GENE = {
    'Monogenic': '#c9cba3',
    'Multigenic': '#ffe1a8',
    'Complex': '#A0E7E5',
}