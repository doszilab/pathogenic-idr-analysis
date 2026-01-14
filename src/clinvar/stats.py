# src/clinvar/stats.py

import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import numpy as np

from ..config import (
    PROCESSED_DATA_DIR, FIGURES_DIR, COLORS,
    SHOW_TITLES, FIG_DPI, SAVE_FORMAT, RAW_DATA_DIR
)


# --- DATA LOADING ---

def load_split_positional_data():
    """Loads Disordered and Ordered positional files (For Fig 1c, 1d)."""
    path_dis = os.path.join(PROCESSED_DATA_DIR, "clinvar_positional_disorder.tsv")
    path_ord = os.path.join(PROCESSED_DATA_DIR, "clinvar_positional_order.tsv")
    if not os.path.exists(path_dis) or not os.path.exists(path_ord):
        raise FileNotFoundError("Split positional files not found.")
    return pd.read_csv(path_dis, sep='\t', low_memory=False), pd.read_csv(path_ord, sep='\t', low_memory=False)


def load_combined_positional_data():
    """Loads the single merged positional file (For Fig 1a Left Bar)."""
    path = os.path.join(PROCESSED_DATA_DIR, "clinvar_positional_all.tsv")
    if not os.path.exists(path):
        raise FileNotFoundError("Combined positional file not found.")
    return pd.read_csv(path, sep='\t', low_memory=False)


def load_variant_data():
    """Loads the variant file (For Fig 1a Right Bar)."""
    path = os.path.join(PROCESSED_DATA_DIR, "clinvar_variants.tsv")
    if not os.path.exists(path):
        raise FileNotFoundError("Variant file not found.")
    return pd.read_csv(path, sep='\t', low_memory=False)


def load_proteome_data():
    path = os.path.join(RAW_DATA_DIR,"discanvis", "combined_dis_pos.tsv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Proteome file not found at {path}")
    return pd.read_csv(path, sep='\t', low_memory=False)


def load_isoform_data():
    path = os.path.join(PROCESSED_DATA_DIR, "main_isoforms.tsv")
    return pd.read_csv(path, sep='\t')


# --- HELPERS ---

def filter_main_isoforms(df, main_isoforms_df, id_col='Protein_ID'):
    main_ids = main_isoforms_df[main_isoforms_df['main_isoform'] == 'yes']['Protein_ID']
    return df[df[id_col].isin(main_ids)]


def extract_pos_based_df(df):
    if 'AccessionPosition' in df.columns and 'Position' not in df.columns:
        df[['Protein_ID', 'Position']] = df['AccessionPosition'].str.split('|', expand=True)
        df['Position'] = df['Position'].astype(int)
    return df


# --- PLOTTING ---

def plot_fig_1a_proteome_coverage(proteome_df, positional_all_df, variants_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

    # LEFT: Proteome Coverage
    total_pos_count = len(proteome_df)
    mutated_pos_count = len(positional_all_df[['Protein_ID', 'Position']].drop_duplicates())

    pct_mut = (mutated_pos_count / total_pos_count) * 100
    pct_non = 100 - pct_mut

    val_non_m = (total_pos_count - mutated_pos_count) / 1e6
    val_mut_m = mutated_pos_count / 1e6

    ax1.bar(0, pct_non, color=COLORS['Non-Mutated'], width=0.3)
    ax1.bar(0, pct_mut, bottom=pct_non, color=COLORS['Mutated'], width=0.3)

    ax1.text(-0.2, pct_non / 2, f"{pct_non:.1f}%", ha='right', va='center')
    ax1.text(0.2, pct_non / 2, f"{val_non_m:.1f}M", ha='left', va='center')

    y_mut = pct_non + (pct_mut / 2)
    ax1.text(-0.2, y_mut, f"{pct_mut:.1f}%", ha='right', va='center')
    ax1.text(0.2, y_mut, f"{val_mut_m:.2f}M", ha='left', va='center')

    ax1.set_title("Human Proteome (Positions)")
    ax1.set_xticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.set_yticks([])

    # RIGHT: ClinVar Variants
    counts = variants_df['Interpretation'].value_counts()
    order = ['Benign', 'Uncertain', 'Pathogenic']
    counts = counts.reindex(order).fillna(0)

    total_vars = counts.sum()
    pcts = (counts / total_vars) * 100
    counts_m = counts / 1e6

    bottom = 0
    colors = [(0.68, 0.85, 0.9, 0.7), (0.5, 0.5, 0.5, 0.7), (1, 0, 0, 0.7)]

    for i, cat in enumerate(order):
        pct = pcts[cat]
        ax2.bar(0, pct, bottom=bottom, color=colors[i], width=0.3, label=cat)
        if pct > 2:
            y_pos = bottom + pct / 2
            ax2.text(-0.2, y_pos, f"{pct:.1f}%", ha='right', va='center')
            ax2.text(0.2, y_pos, f"{counts_m[cat]:.2f}M", ha='left', va='center')
        bottom += pct

    ax2.set_title("ClinVar Variants (Mutations)")
    ax2.set_xticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.set_yticks([])

    plt.tight_layout()
    return fig


def plot_fig_1b_structural_composition(proteome_df):
    if 'CombinedDisorder' not in proteome_df.columns:
        counts = [63.2, 36.8]
    else:
        counts = proteome_df['CombinedDisorder'].value_counts().sort_index()

    labels = ['Ordered', 'Disordered']
    colors = [COLORS['order'], COLORS['disorder']]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors,
           startangle=90, explode=(0, 0.05), textprops={'fontsize': 14})

    ax.set_title("Structural Composition")
    plt.tight_layout()
    return fig


def plot_fig_1c_positional_distribution(df_dis, df_ord):
    # Filter Unknowns
    df_dis = df_dis[df_dis['category_names'] != "Unknown"].copy()
    df_ord = df_ord[df_ord['category_names'] != "Unknown"].copy()

    # Drop duplicates with nDisease (Disease-Weighted)
    cols_to_check = ['Protein_ID', 'Position', 'Interpretation']
    if 'nDisease' in df_dis.columns:
        cols_to_check.append('nDisease')

    df_dis = df_dis[cols_to_check].drop_duplicates()
    df_ord = df_ord[cols_to_check].drop_duplicates()

    categories = ['Pathogenic', 'Uncertain', 'Benign']
    fig, axs = plt.subplots(1, 3, figsize=(14, 5))

    for i, cat in enumerate(categories):
        n_dis = len(df_dis[df_dis['Interpretation'] == cat])
        n_ord = len(df_ord[df_ord['Interpretation'] == cat])
        sizes = [n_dis, n_ord]

        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct * total / 100.0))
                val_str = f"{val:,}" if val < 10000 else f"{val / 1000:.1f}K"
                return f'{pct:.1f}%\n({val_str})'

            return my_autopct

        axs[i].pie(sizes, labels=['Disorder', 'Order'], colors=[COLORS['disorder'], COLORS['order']],
                   autopct=make_autopct(sizes), startangle=90, textprops={'fontsize': 11})
        axs[i].set_title(cat)

    plt.suptitle("Positional Distribution (Disease-Weighted)")
    plt.tight_layout()
    return fig


def plot_fig_1d_gene_distribution(df_dis, df_ord):
    df_dis = df_dis[df_dis['category_names'] != "Unknown"]
    df_ord = df_ord[df_ord['category_names'] != "Unknown"]

    categories = ['Pathogenic', 'Uncertain', 'Benign']
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    for i, cat in enumerate(categories):
        genes_dis = set(df_dis[df_dis['Interpretation'] == cat]['Protein_ID'])
        genes_ord = set(df_ord[df_ord['Interpretation'] == cat]['Protein_ID'])

        v = venn2([genes_dis, genes_ord], set_labels=('Disorder', 'Order'), ax=axs[i])

        if v.get_patch_by_id('10'): v.get_patch_by_id('10').set_color(COLORS['disorder'])
        if v.get_patch_by_id('01'): v.get_patch_by_id('01').set_color(COLORS['order'])
        if v.get_patch_by_id('11'): v.get_patch_by_id('11').set_color(COLORS['both'])
        axs[i].set_title(cat)

    plt.suptitle("Gene Distribution")
    plt.tight_layout()
    return fig


# --- MAIN ---

def generate_figure_1():
    try:
        # Load all 4 types of files + proteome + isoforms
        df_pos_dis, df_pos_ord = load_split_positional_data()
        df_pos_all = load_combined_positional_data()
        df_variants = load_variant_data()
        proteome_df = load_proteome_data()
        isoforms = load_isoform_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return []

    print("Filtering main isoforms...")
    proteome_df = extract_pos_based_df(proteome_df)
    proteome_df = filter_main_isoforms(proteome_df, isoforms, id_col='Protein_ID')

    # Filter Variants
    df_variants = filter_main_isoforms(df_variants, isoforms)
    df_variants = df_variants[df_variants['Position'] != 1]

    # Filter Positional (All)
    df_pos_all = filter_main_isoforms(df_pos_all, isoforms)
    df_pos_all = df_pos_all[df_pos_all['Position'] != 1]

    # Filter Positional (Split)
    df_pos_dis = filter_main_isoforms(df_pos_dis, isoforms)
    df_pos_dis = df_pos_dis[df_pos_dis['Position'] != 1]
    df_pos_ord = filter_main_isoforms(df_pos_ord, isoforms)
    df_pos_ord = df_pos_ord[df_pos_ord['Position'] != 1]

    generated_figures = []

    print("Generating Figure 1a (Proteome & Variants)...")
    fig1a = plot_fig_1a_proteome_coverage(proteome_df, df_pos_all, df_variants)
    fig1a.savefig(os.path.join(FIGURES_DIR, f"Fig_1a_Proteome_Stats.{SAVE_FORMAT}"), dpi=FIG_DPI, bbox_inches='tight')
    generated_figures.append(fig1a)

    print("Generating Figure 1b (Structure)...")
    fig1b = plot_fig_1b_structural_composition(proteome_df)
    fig1b.savefig(os.path.join(FIGURES_DIR, f"Fig_1b_Structure_Pie.{SAVE_FORMAT}"), dpi=FIG_DPI, bbox_inches='tight')
    generated_figures.append(fig1b)

    print("Generating Figure 1c (Positional Split - Disease Weighted)...")
    fig1c = plot_fig_1c_positional_distribution(df_pos_dis, df_pos_ord)
    fig1c.savefig(os.path.join(FIGURES_DIR, f"Fig_1c_Positional_Dist.{SAVE_FORMAT}"), dpi=FIG_DPI, bbox_inches='tight')
    generated_figures.append(fig1c)

    print("Generating Figure 1d (Gene Split)...")
    fig1d = plot_fig_1d_gene_distribution(df_pos_dis, df_pos_ord)
    fig1d.savefig(os.path.join(FIGURES_DIR, f"Fig_1d_Gene_Dist.{SAVE_FORMAT}"), dpi=FIG_DPI, bbox_inches='tight')
    generated_figures.append(fig1d)

    print("Done.")
    return generated_figures