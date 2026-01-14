# src/clinvar/ontology.py

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import configuration
from ..config import (
    PROCESSED_DATA_DIR, FIGURES_DIR, COLORS,
    CATEGORY_COLORS_STRUCTURE, CATEGORY_COLORS_GENE,
    SHOW_TITLES, FIG_DPI, SAVE_FORMAT
)


# --- DATA PROCESSING HELPERS ---

def modify_categories(clinvar_df):
    """
    Modifies category names for better visualization grouping.
    """
    df = clinvar_df.copy()
    disorder_categories = ["Only Disorder", 'Disorder Mostly']

    # Create broad categories for IDR specificity
    df['Category'] = df['Category'].apply(
        lambda x: "Disorder Specific" if x in disorder_categories else "Non-Disorder Specific"
    )

    # Normalize disease names
    df['category_names'] = np.where(
        df['category_names'] == "Cardiovascular/Hematopoietic",
        'Cardiovascular',
        df['category_names']
    )
    return df


def get_value_counts(df, column, sum_values=False):
    """Calculates value counts or sums values for a given column."""
    if sum_values:
        return df.groupby(column)['mutation_count'].sum().sort_values(ascending=False)
    else:
        return df[column].value_counts()


def prepare_data_for_heatmap_mut_cat(clinvar_data, column, column_to_check, sum_values=False):
    """
    Prepare a DataFrame for heatmap plotting based on genic categories and mutation types.

    Args:
        clinvar_data (pd.DataFrame): ClinVar dataset.
        column (str): The column to group and count values on (e.g., 'Category').
        column_to_check (str): Grouping column (e.g., 'genic_category').

    Returns:
        pd.DataFrame: A DataFrame containing counts of categories.
    """
    # Get unique genic categories
    genic_categories = clinvar_data[column_to_check].unique()
    dfs = {}

    # Process data for each genic category
    for cat in genic_categories:
        current_df = clinvar_data[clinvar_data[column_to_check] == cat]
        top_current = get_value_counts(current_df, column, sum_values)
        dfs[cat] = top_current

    # Combine all unique categories from all subsets
    unique_categories = set()
    for top_current in dfs.values():
        unique_categories.update(top_current.index)
    unique_categories = list(unique_categories)

    # Create result DataFrame
    heatmap_data = pd.DataFrame(index=unique_categories, columns=genic_categories).fillna(0)

    for cat, top_current in dfs.items():
        heatmap_data[cat] = top_current.reindex(unique_categories, fill_value=0)

    return heatmap_data.fillna(0).astype(int)


def prepare_data_classified_structure(clinvar_order_p, clinvar_disorder_p, column='category_names'):
    """
    Prepares data for Supp Fig 1a: Comparing Pathogenic Order vs Pathogenic Disorder per Ontology.
    """
    top_disorder_p = get_value_counts(clinvar_disorder_p, column)
    top_order_p = get_value_counts(clinvar_order_p, column)

    unique_categories = list(set(top_disorder_p.index) | set(top_order_p.index))

    columns = ['Disorder-Pathogenic', 'Order-Pathogenic']
    data = pd.DataFrame(index=unique_categories, columns=columns).fillna(0)

    data['Disorder-Pathogenic'] = top_disorder_p
    data['Order-Pathogenic'] = top_order_p

    return data.fillna(0).astype(int)


# --- PLOTTING FUNCTIONS ---

def plot_disease_genic_distribution(disorder_data, order_data, max_count=15, col_to_check="nDisease", figsize=(8, 3)):
    """
    Generates Figure 2a (Left): Histogram of genes count in diseases.
    """
    all_df = pd.concat([disorder_data, order_data])
    check_df = all_df.groupby(col_to_check)['Protein_ID'].nunique().reset_index()
    check_df.columns = [col_to_check, 'Unique_Protein_ID_Count']

    # Count frequencies
    count_of_counts = check_df['Unique_Protein_ID_Count'].value_counts().reset_index()
    count_of_counts.columns = ['Unique_Protein_ID_Count', 'Count']
    count_of_counts = count_of_counts.sort_values(by='Unique_Protein_ID_Count', ascending=False)

    # Aggregate values above max_count
    above_max = count_of_counts[count_of_counts['Unique_Protein_ID_Count'] > max_count]['Count'].sum()
    below_max = count_of_counts[count_of_counts['Unique_Protein_ID_Count'] <= max_count].sort_values(
        by='Unique_Protein_ID_Count')

    aggregated_data = pd.concat([
        below_max,
        pd.DataFrame({'Unique_Protein_ID_Count': [f'>{max_count}'], 'Count': [above_max]})
    ], ignore_index=True)

    # Determine colors based on complexity
    def get_color(val):
        if isinstance(val, str): return CATEGORY_COLORS_GENE["Complex"]
        if val == 1: return CATEGORY_COLORS_GENE["Monogenic"]
        if 2 <= val < 5: return CATEGORY_COLORS_GENE["Multigenic"]
        return CATEGORY_COLORS_GENE["Complex"]

    colors = [get_color(x) for x in aggregated_data['Unique_Protein_ID_Count']]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(aggregated_data['Unique_Protein_ID_Count'].astype(str), aggregated_data['Count'], color=colors)

    ax.set_xlabel('Associated Genes per Disease')
    ax.set_ylabel('Number of Diseases')

    if SHOW_TITLES:
        ax.set_title("Distribution of Diseases by Number of Associated Genes (Fig 2a Left)")

    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    return fig


def create_piechart_for_genic_categories(clinvar_disorder_p, clinvar_order_p, figsize=(6, 3)):
    """
    Generates Figure 2a (Right): Pie chart of pathogenic positions by genetic complexity.
    """
    disorder_counts = clinvar_disorder_p['genic_category'].value_counts()
    order_counts = clinvar_order_p['genic_category'].value_counts()

    # Sort labels based on color mapping keys
    keys = list(CATEGORY_COLORS_GENE.keys())
    disorder_counts = disorder_counts.reindex(keys, fill_value=0)
    order_counts = order_counts.reindex(keys, fill_value=0)

    # Colors
    colors = [CATEGORY_COLORS_GENE.get(label, 'gray') for label in keys]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Pie chart for disorder
    axes[0].pie(disorder_counts.values, colors=colors, autopct='%1.1f%%', startangle=90, counterclock=False)
    axes[0].set_title('Disorder')

    # Pie chart for order
    axes[1].pie(order_counts.values, colors=colors, autopct='%1.1f%%', startangle=90, counterclock=False)
    axes[1].set_title('Order')

    if SHOW_TITLES:
        plt.suptitle("Mutated positions based on Genetic Complexity (Fig 2a Right)")

    plt.tight_layout()
    return fig


def create_pie_charts(df, title, category_color_mapping, figsize=(6, 4), add_legend=False):
    """
    Generates Figure 2c: Pie charts for mutation categories.
    df: Rows are the separate pie charts (e.g. 'All', 'Monogenic'), Columns are the slices.
    """
    columns_in_order = [col for col in category_color_mapping.keys() if col in df.columns]
    df = df[columns_in_order]

    nrows, ncols = 1, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i, (index, row) in enumerate(df.iterrows()):
        if i >= len(axes): break

        values = row.values
        labels = row.index
        colors = [category_color_mapping.get(cat, '#999999') for cat in labels]

        explode = [0.1 if 'Disorder' in label else 0 for label in labels]

        wedges, texts, autotexts = axes[i].pie(
            values, autopct='%1.1f%%', startangle=90, colors=colors, explode=explode
        )
        if SHOW_TITLES:
            axes[i].set_title(index)

    if add_legend:
        handles = [plt.Line2D([0], [0], color=category_color_mapping[col], lw=6) for col in columns_in_order]
        fig.legend(handles, columns_in_order, loc='lower center', bbox_to_anchor=(0.5, -0.05),
                   ncol=2, title='Category', frameon=False)

    if SHOW_TITLES:
        plt.suptitle(title)

    plt.tight_layout()
    return fig


def plot_gene_distribution(clinvar_disorder_p, col_to_check="category_names", figsize=(8, 5)):
    """
    Generates Figure 2d: Mutated Genes in IDRs by Disease Ontology.
    """
    # Count unique genes per ontology
    gene_counts = clinvar_disorder_p.drop_duplicates(subset=['Protein_ID', col_to_check])
    gene_counts = gene_counts[col_to_check].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=gene_counts.index, y=gene_counts.values, color=COLORS["disorder"], ax=ax)

    ax.set_ylabel('Number of Genes')
    ax.set_xlabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')

    if SHOW_TITLES:
        ax.set_title('Mutated Genes in IDRs by Disease Ontology (Fig 2d)')

    plt.tight_layout()
    return fig


def plot_ontology_distribution(clinvar_disorder_p, col_to_check="category_names", figsize=(8, 5)):
    """
    Generates Supp Fig 1b: Simple Bar chart of mutation counts per Ontology in IDRs.
    """
    # Count total mutations per ontology
    counts = clinvar_disorder_p[col_to_check].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=counts.index, y=counts.values, color=COLORS["disorder"], ax=ax)

    ax.set_ylabel('Number of Mutations')
    ax.set_xlabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    if SHOW_TITLES:
        ax.set_title('Pathogenic Mutations in IDRs by Disease Ontology (Supp Fig 1b)')

    plt.tight_layout()
    return fig


def create_stacked_bar_plot(df, title, xlabel='Disease Ontology', figsize=(8, 5), islegend=True):
    """
    Generates Supp Fig 1a: Stacked bar plot (Order vs Disorder).
    """
    # Create sum for sorting
    df_plot = df.copy()
    df_plot['SUM'] = df_plot.sum(axis=1)
    df_plot = df_plot.sort_values(by='SUM', ascending=False)

    max_val = df_plot['SUM'].max()
    df_final = df_plot.drop(columns=['SUM'])

    colors = [CATEGORY_COLORS_STRUCTURE.get(cat, '#999999') for cat in df_final.columns]

    fig, ax = plt.subplots(figsize=figsize)
    df_final.plot(kind='bar', stacked=True, ax=ax, color=colors)

    if SHOW_TITLES:
        ax.set_title(title)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Disease-Position Count')
    ax.set_ylim(0, max_val * 1.15)
    ax.set_xticklabels(df_final.index, rotation=45, ha='right')

    if islegend:
        ax.legend(title='Mutation Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        if ax.get_legend(): ax.get_legend().remove()

    plt.tight_layout()
    return fig


# --- MAIN GENERATION FUNCTIONS ---

def load_and_preprocess_data():
    """Loads and filters ClinVar files."""
    data = {}
    for structure in ['disorder', 'order']:
        # --- JAVÍTÁS: Új fájlnevek használata ---
        path = os.path.join(PROCESSED_DATA_DIR, f"clinvar_positional_{structure}.tsv")

        if not os.path.exists(path):
            print(f"Warning: File not found {path}")
            continue

        print(f"Loading {path}...")
        try:
            # Low_memory=False to avoid mixed type warnings
            df = pd.read_csv(path, sep='\t', low_memory=False)
        except Exception as e:
            print(f"Error reading file: {e}")
            continue

        for interpretation in ['Pathogenic', 'Uncertain', 'Benign']:
            df_filtered = df[df['Interpretation'] == interpretation].copy()

            if "Protein_position" in df_filtered.columns:
                df_filtered = df_filtered.rename(columns={"Protein_position": "Position"})

            # Exclude Unknowns
            excluded = ["Unknown", "Inborn Genetic Diseases"]
            if 'category_names' in df_filtered.columns:
                df_filtered = df_filtered[~df_filtered['category_names'].isin(excluded)]
                # Modify categories
                df_filtered = modify_categories(df_filtered)

            key = f"{structure}_{interpretation[0].lower()}"
            data[key] = df_filtered

    return data


def generate_figure_2():
    """Generates components for Figure 2 and Supplementary Figure 1."""

    print("Loading data for Figure 2...")
    data = load_and_preprocess_data()
    figs = []

    # --- FIG 2a (Left): Histogram ---
    print("Generating Fig 2a (Left)...")
    fig2a_left = plot_disease_genic_distribution(
        data['disorder_p'], data['order_p'], figsize=(7, 4)
    )
    fig2a_left.savefig(os.path.join(FIGURES_DIR, f"Figure_2a_Left_Histogram.{SAVE_FORMAT}"), dpi=FIG_DPI,
                       bbox_inches='tight')
    figs.append(fig2a_left)

    # --- FIG 2a (Right): Pie Chart (Complexity) ---
    print("Generating Fig 2a (Right)...")
    fig2a_right = create_piechart_for_genic_categories(
        data['disorder_p'], data['order_p'], figsize=(6, 3)
    )
    fig2a_right.savefig(os.path.join(FIGURES_DIR, f"Figure_2a_Right_Pie.{SAVE_FORMAT}"), dpi=FIG_DPI,
                        bbox_inches='tight')
    figs.append(fig2a_right)

    # --- FIG 2c: Pie Chart (IDR Categorization - All & Monogenic) ---
    print("Generating Fig 2c...")
    # Prepare data using the correct helper function
    heatmap_df = prepare_data_for_heatmap_mut_cat(data['disorder_p'], 'Category', 'genic_category')
    heatmap_df['All'] = heatmap_df.sum(axis=1)

    subset_df = heatmap_df[['All', 'Monogenic']].T

    fig2c = create_pie_charts(
        subset_df,
        title="Mutated IDR Positions (Fig 2c)",
        category_color_mapping=CATEGORY_COLORS_STRUCTURE,
        add_legend=True
    )
    fig2c.savefig(os.path.join(FIGURES_DIR, f"Figure_2c_Categorization.{SAVE_FORMAT}"), dpi=FIG_DPI,
                  bbox_inches='tight')
    figs.append(fig2c)

    # --- FIG 2d: Mutated GENES in IDRs by Disease Ontology ---
    print("Generating Fig 2d (Genes)...")
    fig2d = plot_gene_distribution(
        data['disorder_p'], col_to_check="category_names"
    )
    fig2d.savefig(os.path.join(FIGURES_DIR, f"Figure_2d_GeneDistribution.{SAVE_FORMAT}"), dpi=FIG_DPI,
                  bbox_inches='tight')
    figs.append(fig2d)

    # --- SUPP FIG 1a: Structural Classification (Order vs Disorder Stacked Bar) ---
    print("Generating Supp Fig 1a (Structure)...")
    heatmap_classified = prepare_data_classified_structure(
        data['order_p'], data['disorder_p'], column='category_names'
    )

    fig_supp1a = create_stacked_bar_plot(
        heatmap_classified,
        title="Structural Classification by Ontology (Supp Fig 1a)",
        xlabel="Disease Ontology"
    )
    fig_supp1a.savefig(os.path.join(FIGURES_DIR, f"Supp_Figure_1a_Structure.{SAVE_FORMAT}"), dpi=FIG_DPI,
                       bbox_inches='tight')
    figs.append(fig_supp1a)

    # --- SUPP FIG 1b: Distribution of pathogenic MUTATIONS (Simple Bar) ---
    print("Generating Supp Fig 1b (Mutations)...")
    # Using specific plot function for mutations distribution
    fig_supp1b = plot_ontology_distribution(
        data['disorder_p'], col_to_check="category_names"
    )
    fig_supp1b.savefig(os.path.join(FIGURES_DIR, f"Supp_Figure_1b_MutationDistribution.{SAVE_FORMAT}"), dpi=FIG_DPI,
                       bbox_inches='tight')
    figs.append(fig_supp1b)

    print(f"Figures saved to {FIGURES_DIR}")
    return figs