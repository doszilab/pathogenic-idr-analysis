# src/clinvar/functional.py

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..config import (
    PROCESSED_DATA_DIR, FIGURES_DIR, COLORS,
    CATEGORY_COLORS_STRUCTURE,
    SHOW_TITLES, FIG_DPI, SAVE_FORMAT, RAW_DATA_DIR
)

# --- CONFIGURATION ---
FUNCTIONAL_TAGS = {
    "Exp. Dis": "MobiDB",
    "DIBS": "dibs_info",
    "MFIB": "mfib_info",
    "ELM": "Elm_Info",
    "PhasePro": "phasepro_info",
    "UniProt Roi": "Roi",
    "PDB": "PDB"
}

PTM_TAGS = ["Phosphorylation", "Acetylation", "Sumoylation", "Ubiquitination", "Methylation"]


# --- DATA LOADING ---

def extract_pos_based_df(df):
    """Splits AccessionPosition into Protein_ID and Position if needed."""
    if 'AccessionPosition' in df.columns and 'Position' not in df.columns:
        df[['Protein_ID', 'Position']] = df['AccessionPosition'].str.split('|', expand=True)
        df['Position'] = df['Position'].astype(int)
    return df


def load_data():
    """Loads positional data."""
    path_dis = os.path.join(PROCESSED_DATA_DIR, "clinvar_positional_disorder.tsv")
    path_ord = os.path.join(PROCESSED_DATA_DIR, "clinvar_positional_order.tsv")

    if not os.path.exists(path_dis) or not os.path.exists(path_ord):
        raise FileNotFoundError("Positional files not found. Run setup_data.py first.")

    df_dis = pd.read_csv(path_dis, sep='\t', low_memory=False)
    df_ord = pd.read_csv(path_ord, sep='\t', low_memory=False)

    # Filter Unknowns
    if 'category_names' in df_dis.columns:
        df_dis = df_dis[df_dis['category_names'] != "Unknown"]
    if 'category_names' in df_ord.columns:
        df_ord = df_ord[df_ord['category_names'] != "Unknown"]

    return df_dis, df_ord


def load_proteome_data():
    """Loads background proteome (disorder positions)."""
    path = os.path.join(RAW_DATA_DIR,"discanvis" ,"combined_dis_pos.tsv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Proteome file not found at {path}")

    df = pd.read_csv(path, sep='\t', low_memory=False)
    # CRITICAL FIX: Ensure Protein_ID exists
    df = extract_pos_based_df(df)
    return df


def load_functional_regions_data():
    """Loads the functional regions file (Start-End) for background calculation."""
    path = os.path.join(PROCESSED_DATA_DIR, "functional_regions.tsv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Functional regions file not found at {path}. Run setup_data.py!")
    return pd.read_csv(path, sep='\t', low_memory=False)


# --- HELPERS ---

def get_structural_category(row):
    """Classifies IDR context."""
    disorder_cats = ["Only Disorder", "Disorder Mostly"]
    if row.get('Category') in disorder_cats:
        return "Disorder Specific"
    return "Non-Disorder Specific"


def check_overlap(info_str, tags):
    """Checks if any tag from list is in the info string."""
    if pd.isna(info_str): return False
    return any(tag in str(info_str) for tag in tags)


def expand_info_cols(df, target_info_tag):
    """
    Expands the 'info_cols' column for rows matching 'target_info_tag' in 'info'.
    This logic mimics 'calculate_specific_functional_distribution' from the original script.
    """
    # Filter rows containing the tag
    filtered = df[df['info'].str.contains(target_info_tag, na=False)].copy()

    if filtered.empty:
        return pd.Series(dtype=object)

    # Split both 'info' and 'info_cols' columns by semicolon and stack them together
    info_split = filtered['info'].str.split(';', expand=True)
    info_cols_split = filtered['info_cols'].str.split(';', expand=True)

    # Stack both and reset index to keep alignment
    stacked_info = info_split.stack().reset_index(level=1, drop=True)
    stacked_info_cols = info_cols_split.stack().reset_index(level=1, drop=True)

    # Combine into a temporary DF
    temp_df = pd.DataFrame({
        'info': stacked_info,
        'info_cols': stacked_info_cols
    })

    # Filter where the tag matches the target
    target_subset = temp_df[temp_df['info'] == target_info_tag]

    # Split values by comma (e.g. "Region A, Region B")
    expanded_values = target_subset['info_cols'].str.split(', ', expand=True).stack().reset_index(level=1, drop=True)

    return expanded_values


def categorize_roi(roi_name):
    """Categorizes UniProt ROI names."""
    if pd.isna(roi_name): return "Other"
    roi = str(roi_name)

    # Custom rules from original script
    big_categories = ['Interaction/Binding', 'Localization', 'Down-regulation-\nProtein Degradation',
                      'Linker', 'Activation', 'Oligomerization', 'Structural Element', "Head"]

    if any(x in roi for x in ['Interaction', 'interaction', 'Interacts', 'binding']):
        return 'Interaction/Binding'
    if 'localization' in roi:
        return 'Localization'
    if any(x in roi for x in ['down-regulation', 'protein degradation', 'degradation']):
        return 'Down-regulation-\nProtein Degradation'
    if 'Linker' in roi:
        return 'Linker'
    if 'activation' in roi:
        return 'Activation'
    if 'oligomerization' in roi:
        return 'Oligomerization'

    structural_elements = ['RS', 'NTAD', 'CTAD', 'Tail', 'domain', 'C', 'helical region', 'Coil']
    if roi in structural_elements or any(elem in roi for elem in structural_elements):
        return 'Structural Element'

    # If it matches none of the above specific rules, return the name itself IF it's not "Other" logic
    # But for the bar chart we want specific names if they are top counts?
    # The original script does: expanded_df.apply(lambda x: 'Other' if x not in big_categories else x)
    # This implies we ONLY want to show the Big Categories OR the specific names if they matched?
    # Actually, the old script mapped everything into big categories or 'Other'.

    # Let's return the category if matched, otherwise return the raw name (so we can see top sites that aren't categorized)
    # OR follow the old script strictly:
    return 'Other'


# --- PLOTTING FUNCTIONS ---

def plot_fig_3a_pdb_coverage(df_dis, df_ord):
    """Fig 3a: PDB Coverage."""
    interpretations = ['Pathogenic', 'Uncertain']
    data = []

    for interp in interpretations:
        sub_dis = df_dis[df_dis['Interpretation'] == interp]
        n_dis = len(sub_dis)
        n_pdb_dis = sub_dis['info'].str.contains('PDB', na=False).sum()
        pct_dis = (n_pdb_dis / n_dis * 100) if n_dis > 0 else 0
        data.append({'Interpretation': interp, 'Structure': 'Disordered', 'Percentage': pct_dis})

        sub_ord = df_ord[df_ord['Interpretation'] == interp]
        n_ord = len(sub_ord)
        n_pdb_ord = sub_ord['info'].str.contains('PDB', na=False).sum()
        pct_ord = (n_pdb_ord / n_ord * 100) if n_ord > 0 else 0
        data.append({'Interpretation': interp, 'Structure': 'Ordered', 'Percentage': pct_ord})

    df_plot = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(data=df_plot, x='Interpretation', y='Percentage', hue='Structure',
                palette=[COLORS['disorder'], COLORS['order']], ax=ax)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3)

    ax.set_ylabel("PDB Structure Coverage (%)")
    ax.set_title("PDB Coverage of Variants (Fig 3a)")
    ax.legend(title='Region Type')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"Fig_3a_PDB_Coverage.{SAVE_FORMAT}"), dpi=FIG_DPI, bbox_inches='tight')
    return fig


def plot_fig_3b_functional_donuts(df_dis):
    """Fig 3b: Donut plots."""
    func_tags_only = [v for k, v in FUNCTIONAL_TAGS.items() if k not in ["Exp. Dis", "PDB"]]
    func_tags_only.extend(PTM_TAGS)

    interpretations = ['Pathogenic', 'Uncertain', 'Benign']
    colors_donut = ["#ccd5ae", "#d4a373", "#cdb4db", "#a9def9"]
    labels = ['Func & Exp.Dis', 'Func Only', 'Exp.Dis Only', 'No Annotation']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, interp in enumerate(interpretations):
        df = df_dis[df_dis['Interpretation'] == interp].copy()

        has_mobi = df['info'].str.contains('MobiDB', na=False)
        has_func = df['info'].apply(lambda x: check_overlap(x, func_tags_only))

        sizes = [
            len(df[has_func & has_mobi]),
            len(df[has_func & ~has_mobi]),
            len(df[~has_func & has_mobi]),
            len(df[~has_func & ~has_mobi])
        ]

        axes[i].pie(
            sizes, labels=None, autopct='%1.1f%%', colors=colors_donut,
            startangle=90, pctdistance=0.85, wedgeprops=dict(width=0.3)
        )
        axes[i].set_title(f"{interp} (IDR)", y=0.48)

    fig.legend(axes[0].patches, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.05))

    if SHOW_TITLES:
        plt.suptitle("Functional Annotation Landscape (Fig 3b)", y=0.95)

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig(os.path.join(FIGURES_DIR, f"Fig_3b_Functional_Donuts.{SAVE_FORMAT}"), dpi=FIG_DPI, bbox_inches='tight')
    return fig


def prepare_stacked_data(df_dis, tags_dict, ptm_mode=False):
    df = df_dis[df_dis['Interpretation'] == 'Pathogenic'].copy()
    df['Specificity'] = df.apply(get_structural_category, axis=1)

    data = []
    for label, tag_key in tags_dict.items():
        if ptm_mode:
            mask = df['info'].str.contains(tag_key, na=False)
        else:
            mask = df['info'].str.contains(tag_key, na=False)

        subset = df[mask]
        counts = subset['Specificity'].value_counts()

        data.append({
            'Type': label if not ptm_mode else tag_key,
            'Disorder Specific': counts.get('Disorder Specific', 0),
            'Non-Disorder Specific': counts.get('Non-Disorder Specific', 0)
        })

    return pd.DataFrame(data).set_index('Type')


def plot_fig_3c_functional_types(df_dis):
    """Fig 3c: Stacked bar of functional types."""
    tags_to_plot = {
        "Exp. Dis": "MobiDB", "DIBS": "dibs_info", "MFIB": "mfib_info",
        "ELM": "Elm_Info", "PhasePro": "phasepro_info", "UniProt Roi": "Roi", "PDB": "PDB"
    }

    df_plot = prepare_stacked_data(df_dis, tags_to_plot)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [CATEGORY_COLORS_STRUCTURE['Disorder Specific'], CATEGORY_COLORS_STRUCTURE['Non-Disorder Specific']]

    df_plot.plot(kind='bar', stacked=True, color=colors, ax=ax, width=0.7)

    totals = df_plot.sum(axis=1)
    for i, total in enumerate(totals):
        ax.text(i, total + (total * 0.01), f"{int(total):,}", ha='center', va='bottom', fontsize=9)

    ax.set_ylabel("Number of Pathogenic Positions")
    ax.set_title("Pathogenic Overlap with Functional Types (Fig 3c)")
    ax.legend(title="Context Specificity")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"Fig_3c_Functional_Types.{SAVE_FORMAT}"), dpi=FIG_DPI, bbox_inches='tight')
    return fig


def plot_fig_3d_ptm_distribution(df_dis):
    """Fig 3d: PTM Distribution."""
    ptm_dict = {ptm: ptm for ptm in PTM_TAGS}
    df_plot = prepare_stacked_data(df_dis, ptm_dict, ptm_mode=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = [CATEGORY_COLORS_STRUCTURE['Disorder Specific'], CATEGORY_COLORS_STRUCTURE['Non-Disorder Specific']]

    df_plot.plot(kind='bar', stacked=True, color=colors, ax=ax, width=0.7)

    totals = df_plot.sum(axis=1)
    for i, total in enumerate(totals):
        ax.text(i, total + 5, f"{int(total):,}", ha='center', va='bottom', fontsize=9)

    ax.set_ylabel("Number of Pathogenic Positions")
    ax.set_xlabel("PTM Type")
    ax.set_title("Pathogenic Variants by PTM Type (Fig 3d)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"Fig_3d_PTM_Distribution.{SAVE_FORMAT}"), dpi=FIG_DPI, bbox_inches='tight')
    return fig


# --- SUPPLEMENTARY FIGURES ---

def plot_supp_fig_1c_top_roi(df_dis, top_n=20):
    """
    Supp Fig 1c: Top UniProt ROI sites ranked by pathogenic mutations.
    Displays raw names instead of categories to show 'sites'.
    """
    pathogenic_dis = df_dis[df_dis['Interpretation'] == 'Pathogenic'].copy()

    # 1. Expand ROI values (get raw names)
    roi_raw_values = expand_info_cols(pathogenic_dis, "Roi")

    # 2. Count top raw sites
    roi_counts = roi_raw_values.value_counts().head(top_n)

    # Note: If categorization is preferred over raw names, uncomment:
    # roi_categorized = roi_raw_values.apply(categorize_roi)
    # roi_counts = roi_categorized.value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    roi_counts.plot(kind='barh', ax=ax, color='#f08080')

    ax.set_xlabel('Number of Pathogenic Mutations')
    ax.set_title(f'Top {top_n} UniProt ROI Sites in IDRs (Supp Fig 1c)')
    ax.invert_yaxis()

    for i, v in enumerate(roi_counts):
        ax.text(v + 0.5, i, str(v), color='black', va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"Supp_Fig_1c_TopROI.{SAVE_FORMAT}"), dpi=FIG_DPI, bbox_inches='tight')
    return fig


def plot_supp_fig_1d_benign_rates(df_dis, proteome_df, functional_regions_df):
    """
    Supp Fig 1d: Benign mutation RATE calculation (Mutations / Total Residues).
    """
    # 1. Prepare Background (Denominator)
    # Ensure Protein_ID exists in proteome_df (handled by load_proteome_data)
    proteome_disorder = proteome_df[proteome_df['CombinedDisorder'] == 1.0].copy()

    relevant_proteins = proteome_disorder['Protein_ID'].unique()
    func_df = functional_regions_df[functional_regions_df['Protein_ID'].isin(relevant_proteins)].copy()

    # Expand functional regions (Start-End)
    expanded_list = []
    # Optimization: Iterate only relevant rows
    # Note: This step can be slow.
    for _, row in func_df.iterrows():
        for pos in range(row['Start'], row['End'] + 1):
            expanded_list.append((row['Protein_ID'], pos, row['Data']))

    expanded_func_df = pd.DataFrame(expanded_list, columns=['Protein_ID', 'Position', 'Category'])

    # Intersect Functional with Disorder
    annotated_disorder = expanded_func_df.merge(
        proteome_disorder[['Protein_ID', 'Position']],
        on=['Protein_ID', 'Position'],
        how='inner'
    )

    # 2. Prepare Mutations (Numerator)
    benign_muts = df_dis[df_dis['Interpretation'] == 'Benign'][['Protein_ID', 'Position']].drop_duplicates()
    benign_muts['has_mut'] = True

    # 3. Calculate Rates
    stats = []

    # Group by Category
    grouped = annotated_disorder.groupby('Category')

    for cat, group in grouped:
        total_residues = len(group)
        merged = group.merge(benign_muts, on=['Protein_ID', 'Position'], how='left')
        mutated_residues = merged['has_mut'].sum()

        rate = mutated_residues / total_residues if total_residues > 0 else 0
        stats.append({'Category': cat, 'Rate': rate})

    # 4. Unannotated Rate
    total_disorder_pos = len(proteome_disorder)
    annotated_unique = annotated_disorder[['Protein_ID', 'Position']].drop_duplicates().shape[0]
    unannotated_count = total_disorder_pos - annotated_unique

    total_benign_in_disorder = len(benign_muts)
    annotated_pos_set = annotated_disorder[['Protein_ID', 'Position']].drop_duplicates()
    benign_in_annotated = benign_muts.merge(annotated_pos_set, on=['Protein_ID', 'Position'], how='inner').shape[0]
    benign_in_unannotated = total_benign_in_disorder - benign_in_annotated

    rate_unannotated = benign_in_unannotated / unannotated_count if unannotated_count > 0 else 0
    stats.append({'Category': 'Unannotated', 'Rate': rate_unannotated})

    # 5. Plot
    df_stats = pd.DataFrame(stats).sort_values(by='Rate', ascending=False)

    target_cats = ["MobiDB", "Elm_Info", "dibs_info", "phasepro_info", "Unannotated"]
    df_plot = df_stats[df_stats['Category'].isin(target_cats)].copy()

    name_map = {"MobiDB": "MobiDB", "Elm_Info": "ELM", "dibs_info": "DIBS", "phasepro_info": "PhasePro"}
    df_plot['Category'] = df_plot['Category'].replace(name_map)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(df_plot['Category'], df_plot['Rate'], color=COLORS['Benign'])

    ax.set_ylabel("Benign Mutation Rate")
    ax.set_title("Benign Mutation Rates (Supp Fig 1d)")

    for bar, rate in zip(bars, df_plot['Rate']):
        ax.text(bar.get_x() + bar.get_width() / 2, rate, f"{rate * 100:.2f}%",
                ha='center', va='bottom', fontsize=9)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"Supp_Fig_1d_BenignRates.{SAVE_FORMAT}"), dpi=FIG_DPI, bbox_inches='tight')

    return fig


# --- MAIN WORKFLOW ---

def generate_figure_3():
    """Generates all panels for Figure 3 and Supplementary Figure 1c, 1d."""
    print("Loading data for Functional Analysis...")
    try:
        df_dis, df_ord = load_data()
        proteome_df = load_proteome_data()
        func_regions_df = load_functional_regions_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return []

    figs = []

    print("Generating Fig 3a (PDB Coverage)...")
    fig3a = plot_fig_3a_pdb_coverage(df_dis, df_ord)
    fig3a.savefig(os.path.join(FIGURES_DIR, f"Fig_3a_PDB_Coverage.{SAVE_FORMAT}"), dpi=FIG_DPI, bbox_inches='tight')
    figs.append(fig3a)

    print("Generating Fig 3b (Functional Donuts)...")
    fig3b = plot_fig_3b_functional_donuts(df_dis)
    fig3b.savefig(os.path.join(FIGURES_DIR, f"Fig_3b_Functional_Donuts.{SAVE_FORMAT}"), dpi=FIG_DPI,
                  bbox_inches='tight')
    figs.append(fig3b)

    print("Generating Fig 3c (Functional Types)...")
    fig3c = plot_fig_3c_functional_types(df_dis)
    fig3c.savefig(os.path.join(FIGURES_DIR, f"Fig_3c_Functional_Types.{SAVE_FORMAT}"), dpi=FIG_DPI, bbox_inches='tight')
    figs.append(fig3c)

    print("Generating Fig 3d (PTM Distribution)...")
    fig3d = plot_fig_3d_ptm_distribution(df_dis)
    fig3d.savefig(os.path.join(FIGURES_DIR, f"Fig_3d_PTM_Distribution.{SAVE_FORMAT}"), dpi=FIG_DPI, bbox_inches='tight')
    figs.append(fig3d)

    # Supplementary
    print("Generating Supp Fig 1c (Top ROI)...")
    supp1c = plot_supp_fig_1c_top_roi(df_dis)
    supp1c.savefig(os.path.join(FIGURES_DIR, f"Supp_Fig_1c_TopROI.{SAVE_FORMAT}"), dpi=FIG_DPI, bbox_inches='tight')
    figs.append(supp1c)

    print("Generating Supp Fig 1d (Benign Rates)...")
    supp1d = plot_supp_fig_1d_benign_rates(df_dis, proteome_df, func_regions_df)
    supp1d.savefig(os.path.join(FIGURES_DIR, f"Supp_Fig_1d_BenignRates.{SAVE_FORMAT}"), dpi=FIG_DPI,
                   bbox_inches='tight')
    figs.append(supp1d)

    print(f"Functional Analysis Figures saved to {FIGURES_DIR}")
    return figs