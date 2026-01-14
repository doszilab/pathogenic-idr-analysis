import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os


def plot_functional_site_distribution(df_list, column_name, tilte):
    # Combine all dataframes into one
    combined_df = pd.concat(df_list)

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=combined_df, x='fname', y=column_name, linewidth=1.5)

    benign_cutoff = 0.34
    half_cutoff = 0.5
    pathogen_cutoff = 0.564

    plt.axhline(y=benign_cutoff, color='blue', linestyle='--', linewidth=2, label=f'Benign ({benign_cutoff})')
    plt.axhline(y=half_cutoff, color='grey', linestyle='--', linewidth=2, label='0.5')
    plt.axhline(y=pathogen_cutoff, color='red', linestyle='--', linewidth=2, label=f'Pathogen ({pathogen_cutoff})')

    # Set labels and title
    plt.xlabel('Site Type')
    plt.ylabel('Distribution of Values')
    plt.title(tilte)

    plt.tight_layout()

    # Show the plot
    plt.show()


def extract_pos_based_df(df):
    df[['Protein_ID', 'Position']] = df['AccessionPosition'].str.split('|', expand=True)
    df = df.drop(columns=['AccessionPosition'])
    df['Position'] = df['Position'].astype(int)
    return df


def plot_distribution_am_secondary(disorder_df, order_df, title=None, column="secondary_structure_grouped",figsize=(14, 6)):
    # Extract unique secondary structures from both datasets
    unique_structures = sorted(set(disorder_df[column]).union(set(order_df[column])))

    # Generate a color palette based on the number of unique secondary structures
    colors = sns.color_palette("Set2", len(unique_structures))

    # Create a dictionary for the common color palette
    common_palette = dict(zip(unique_structures, colors))

    # Define the order of secondary structures based on unique values
    structure_order = unique_structures

    # Plotting pairwise distributions
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    # Order plot
    sns.boxplot(x=column, y='AlphaMissense',
                data=order_df, palette=common_palette,
                ax=axs[0], order=structure_order)
    axs[0].set_title('Ordered')
    axs[0].set_xlabel('Secondary Structure')
    axs[0].set_ylabel('AlphaMissense Score')

    # Disorder plot
    sns.boxplot(x=column, y='AlphaMissense',
                data=disorder_df, palette=common_palette,
                ax=axs[1], order=structure_order)
    axs[1].set_title('Disordered')
    axs[1].set_xlabel('Secondary Structure')
    axs[1].set_ylabel('AlphaMissense Score')

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()


def plot_distribution_secondary(disorder_df, order_df, title=None, column="secondary_structure_grouped"):
    # Extract unique secondary structures from both datasets
    unique_structures = sorted(set(disorder_df[column]).union(set(order_df[column])))

    # Generate a color palette based on the number of unique secondary structures
    colors = sns.color_palette("Set2", len(unique_structures))

    # Create a dictionary for the common color palette
    common_palette = dict(zip(unique_structures, colors))

    # Define the order of secondary structures based on unique values
    structure_order = unique_structures

    # Count occurrences for each secondary structure in both datasets
    disorder_counts = disorder_df[column].value_counts()
    order_counts = order_df[column].value_counts()

    # Create dataframes for plotting
    disorder_counts_df = pd.DataFrame(
        {column: disorder_counts.index, 'count': disorder_counts.values})
    order_counts_df = pd.DataFrame(
        {column: order_counts.index, 'count': order_counts.values})

    # Function to add count labels above bars
    def add_labels(ax, data):
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Order plot
    sns.barplot(x=column, y='count',
                data=order_counts_df, palette=common_palette,
                ax=axs[0], order=structure_order)
    axs[0].set_title('Ordered')
    axs[0].set_xlabel('Secondary Structure')
    axs[0].set_ylabel('Count')
    add_labels(axs[0], order_counts_df)

    # Disorder plot
    sns.barplot(x=column, y='count',
                data=disorder_counts_df, palette=common_palette,
                ax=axs[1], order=structure_order)
    axs[1].set_title('Disordered')
    axs[1].set_xlabel('Secondary Structure')
    axs[1].set_ylabel('Count')
    add_labels(axs[1], disorder_counts_df)

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()


def group_secondary_structure(df):
    # secondary_structure_groups = {
    #     'H': 'Helix', 'G': 'Helix', 'I': 'Helix', 'P': 'Helix',  # Group helices together
    #     'B': 'β-Structure', 'E': 'β-Structure',  # Group β-structures
    #     'T': 'Loops/Turns', 'S': 'Loops/Turns',  # Group loops/turns
    #     'C': 'Coil'  # Keep coil as a separate category
    # }

    secondary_structure_groups = {
        'H': 'Helix', 'G': 'Helix', 'I': 'Helix',  # Group helices together
        'B': 'Strand', 'E': 'Strand',  # Group β-structures
        'P': 'Loop', 'T': 'Loop', 'S': 'Loop', 'C': 'Loop'  # Group loops
    }

    df['secondary_structure_grouped'] = df['secondary_structure'].map(secondary_structure_groups)
    return df


def plot_distribution_am_secondary_4subplots(am_disorder_secondary, am_order_secondary,
                                             am_disorder_clinvar_p, am_order_clinvar_p,
                                             title="Secondary Structure AlphaMissense Score distribution",
                                             figsize=(6, 8), column='secondary_structure_grouped',fig_dir=None):
    """
    Creates a single figure with four subplots for AlphaMissense score distributions:
    Top row: Human Proteome (disorder vs order)
    Bottom row: ClinVar Pathogenic Variants (disorder vs order)

    Parameters
    ----------
    am_disorder_secondary : pd.DataFrame
        DataFrame for Human Proteome - Disorder Region variants. Must contain 'score_col' column.

    am_order_secondary : pd.DataFrame
        DataFrame for Human Proteome - Order Region variants. Must contain 'score_col' column.

    am_disorder_clinvar_p : pd.DataFrame
        DataFrame for ClinVar Pathogenic - Disorder Region variants. Must contain 'score_col' column.

    am_order_clinvar_p : pd.DataFrame
        DataFrame for ClinVar Pathogenic - Order Region variants. Must contain 'score_col' column.

    title : str
        The main title (suptitle) for the entire figure.

    figsize : tuple
        Size of the entire figure.

    score_col : str
        The column name of the AlphaMissense score in the provided DataFrames.
    """
    # Extract unique secondary structures from both datasets
    unique_structures = sorted(set(am_disorder_secondary[column]).union(set(am_order_secondary[column])))

    # Generate a color palette based on the number of unique secondary structures
    colors = sns.color_palette("Set2", len(unique_structures))

    # Create a dictionary for the common color palette
    common_palette = dict(zip(unique_structures, colors))

    # Define the order of secondary structures based on unique values
    structure_order = unique_structures

    # Create figure with 4 subplots (2 rows, 2 columns)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    fig.suptitle(title, fontsize=14)

    # Top-left: Human Proteome - Disorder
    sns.boxplot(x=column, y='AlphaMissense',
                data=am_disorder_secondary, palette=common_palette,
                ax=axes[0, 0], order=structure_order,width=0.5 )
    axes[0, 0].set_title("Disorder",fontsize=14)
    axes[0, 0].set_xlabel(None)
    axes[0, 0].set_xticklabels([])
    axes[0, 0].text(-0.35, 0.5, "Proteome", rotation=90, va='center', ha='center', transform=axes[0, 0].transAxes,
                   fontsize=14)
    axes[0, 0].set_ylabel("AlphaMissense Score")

    # Top-left: Human Proteome - Order
    sns.boxplot(x=column, y='AlphaMissense',
                data=am_order_secondary, palette=common_palette,
                ax=axes[0, 1], order=structure_order,width=0.5)
    axes[0, 1].set_title("Order")
    axes[0, 1].set_xticklabels([])
    axes[0, 1].set_xlabel(None)
    axes[0, 1].set_ylabel(None)

    # Bottom-left: ClinVar Pathogenic - Disorder
    sns.boxplot(x=column, y='AlphaMissense',
                data=am_disorder_clinvar_p, palette=common_palette,
                ax=axes[1, 0], order=structure_order,width=0.5)
    axes[1, 0].set_xlabel('Secondary Structure')
    axes[1, 0].text(-0.35, 0.5, "Pathogenic", rotation=90, va='center', ha='center', transform=axes[1, 0].transAxes,
                    fontsize=14)
    axes[1, 0].set_ylabel("AlphaMissense Score")

    # Bottom-right: ClinVar Pathogenic - Order
    sns.boxplot(x=column, y='AlphaMissense',
                data=am_order_clinvar_p, palette=common_palette,
                ax=axes[1, 1], order=structure_order,width=0.5)
    axes[1, 1].set_xlabel('Secondary Structure')
    axes[1, 1].set_ylabel(None)

    plt.tight_layout()  # adjust layout to accommodate suptitle
    if fig_dir:
        plt.savefig(os.path.join(fig_dir,'B.png'))
    plt.show()



def plot_distribution_secondary_4subplots(am_disorder_secondary, am_order_secondary,
                                             am_disorder_clinvar_p, am_order_clinvar_p,
                                             title="Secondary Structure AlphaMissense Score distribution",
                                             figsize=(6, 8), column='secondary_structure_grouped', fig_dir=None):
    """
    Creates a single figure with four subplots for Secondary Structure distributions:
    Top row: Human Proteome (disorder vs order)
    Bottom row: ClinVar Pathogenic Variants (disorder vs order)
    The plots will show percentage distribution as stacked bar plots.
    """
    # Extract unique secondary structures from both datasets
    unique_structures = sorted(set(am_disorder_secondary[column]).union(set(am_order_secondary[column])))

    # Generate a color palette based on the number of unique secondary structures
    colors = sns.color_palette("Set2", len(unique_structures))

    # Create a dictionary for the common color palette
    common_palette = dict(zip(unique_structures, colors))

    # Define the order of secondary structures based on unique values
    structure_order = unique_structures

    # Create figure with 4 subplots (2 rows, 2 columns)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)

    # Function to plot percentage stacked barplot
    def plot_percentage_stacked_bar(ax, data, column, structure_order, palette):
        # Calculate value counts for each category and normalize to get percentages
        counts = data[column].value_counts(normalize=True).sort_index()
        counts = counts.reindex(structure_order, fill_value=0)  # Ensure order matches
        counts = counts * 100  # Convert to percentages

        # Plotting the stacked barplot
        counts.plot(kind='bar', stacked=True,
                    color=[palette[structure_order.index(structure)] for structure in counts.index],
                    ax=ax, width=0.7)
        ax.set_xlabel(None)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        # Adding percentage labels on top of the bars
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10)

        ax.set_ylim(0,100)

    # Top-left: Human Proteome - Disorder
    plot_percentage_stacked_bar(axes[0, 0], am_disorder_secondary, column, structure_order, colors)
    axes[0, 0].set_title("Disorder")
    axes[0, 0].text(-0.35, 0.5, "Proteome", rotation=90, va='center', ha='center', transform=axes[0, 0].transAxes,
                    fontsize=14)
    axes[0, 0].set_ylabel('Percentage (%)')

    # Top-right: Human Proteome - Order
    plot_percentage_stacked_bar(axes[0, 1], am_order_secondary, column, structure_order, colors)
    axes[0, 1].set_title("Order")


    # Bottom-left: ClinVar Pathogenic - Disorder
    plot_percentage_stacked_bar(axes[1, 0], am_disorder_clinvar_p, column, structure_order, colors)
    axes[1, 0].text(-0.35, 0.5, "Pathogenic", rotation=90, va='center', ha='center', transform=axes[1, 0].transAxes,
                    fontsize=14)

    axes[1, 0].set_ylabel('Percentage (%)')
    axes[1, 0].set_xlabel('Secondary Structure')

    # Bottom-right: ClinVar Pathogenic - Order
    plot_percentage_stacked_bar(axes[1, 1], am_order_clinvar_p, column, structure_order, colors)
    axes[1, 1].set_xlabel('Secondary Structure')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()  # adjust layout to accommodate suptitle
    plt.subplots_adjust(top=0.85)  # adjust space for suptitle

    if fig_dir:
        plt.savefig(os.path.join(fig_dir, 'stacked_barplot.png'), dpi=300)

    plt.show()


def main_plots():

    # FIG 4b/1
    # plot_distribution_am_secondary(am_disorder_secondary, am_order_secondary,
    #                                f"Secondary Structure AlphaMissense distribution for Human Proteome",figsize=(8, 4))
    #
    # exit()


    clinvar_pathogenic_df = pd.read_csv(f'{clinvar_path}/Pathogenic/positional_clinvar_functional_categorized_final.tsv',
                                        sep='\t')
    clinvar_pathogenic_df_disorder = clinvar_pathogenic_df[clinvar_pathogenic_df['structure'] == 'disorder'][
        ['Protein_ID', 'Position']]
    clinvar_pathogenic_df_order = clinvar_pathogenic_df[clinvar_pathogenic_df['structure'] == 'order'][
        ['Protein_ID', 'Position']]

    am_disorder_clinvar_p = clinvar_pathogenic_df_disorder.merge(am_disorder_secondary,
                                                                 on=["Protein_ID", "Position"], how='inner')
    am_order_clinvar_p = clinvar_pathogenic_df_order.merge(am_order_secondary, on=["Protein_ID", "Position"],
                                                           how='inner')

    # FIG 4b
    # plot_distribution_am_secondary_4subplots(
    #     am_disorder_secondary, am_order_secondary,
    #     am_disorder_clinvar_p, am_order_clinvar_p,
    #     title="Secondary Structure AlphaMissense Score distribution",
    #     figsize=(6, 6),fig_dir=fig4
    # )


    # # FIG 4b/2
    # plot_distribution_am_secondary(am_disorder_clinvar_p, am_order_clinvar_p,
    #                                f"Secondary Structure AlphaMissense distribution for ClinVar Pathogenic Variants",figsize=(8, 4))

    # FIG 4b Percentage of distributions
    plot_distribution_secondary_4subplots(am_disorder_secondary, am_order_secondary,
                                          am_disorder_clinvar_p, am_order_clinvar_p,
                                          title="Secondary Structure Distribution",
                                          figsize=(6, 6),fig_dir=fig4)





if __name__ == "__main__":

    """
    Check The secondary Structure distribution for Human Proteome,
    Check The secondary Structure distribution for ClinVar Variants
    Check The correlation between secondary structure and Alpha Missense score
    """

    base_dir = f'/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations'

    fig4 = f'{base_dir}/plots/fig4'

    to_alphamissense_dir = os.path.join(base_dir, 'processed_data/files/alphamissense')
    clinvar_path = os.path.join(base_dir, 'processed_data/files/clinvar')
    pos_based_dir = os.path.join(base_dir, 'data/discanvis_base_files/positional_data_process')

    am_order = pd.read_csv(f"{to_alphamissense_dir}/am_order.tsv", sep='\t', )
    am_disorder = pd.read_csv(f"{to_alphamissense_dir}/am_disorder.tsv", sep='\t', )

    secondary_structure_df = extract_pos_based_df(pd.read_csv(f"{pos_based_dir}/secondary_structure_pos.tsv", sep='\t'))
    secondary_structure_df = group_secondary_structure(secondary_structure_df)

    # Human Proteome
    am_order_secondary = secondary_structure_df.merge(am_order, on=['Protein_ID', 'Position'])
    am_disorder_secondary = secondary_structure_df.merge(am_disorder, on=['Protein_ID', 'Position'])

    main_plots()
    exit()

    # print(am_order_secondary)
    # print(am_disorder_secondary)
    # plot_distribution_am_secondary(am_disorder_secondary, am_order_secondary,
    #                                f"Secondary Structure AlphaMissense distribution for Human Proteome")
    # exit()

    plot_distribution_secondary(am_disorder_secondary, am_order_secondary,
                                f"Secondary Structure distribution for Human Proteome")

    exit()

    """
    ClinVar Variants
    """



    interpretations = ['Pathogenic', 'Uncertain', 'Benign']

    for i in interpretations:

        # ClinVar Pathogenic
        clinvar_pathogenic_df = pd.read_csv(f'{clinvar_path}/{i}/positional_clinvar_functional_categorized_final.tsv',
                                            sep='\t')
        clinvar_pathogenic_df_disorder = clinvar_pathogenic_df[clinvar_pathogenic_df['structure'] == 'disorder'][
            ['Protein_ID', 'Position']]
        clinvar_pathogenic_df_order = clinvar_pathogenic_df[clinvar_pathogenic_df['structure'] == 'order'][
            ['Protein_ID', 'Position']]

        am_disorder_clinvar_p = clinvar_pathogenic_df_disorder.merge(am_disorder_secondary,
                                                                     on=["Protein_ID", "Position"], how='inner')
        am_order_clinvar_p = clinvar_pathogenic_df_order.merge(am_order_secondary, on=["Protein_ID", "Position"],
                                                               how='inner')

        plot_distribution_am_secondary(am_disorder_clinvar_p, am_order_clinvar_p,
                                       f"Secondary Structure AlphaMissense distribution for ClinVar {i} Variants")
        plot_distribution_secondary(am_disorder_clinvar_p, am_order_clinvar_p,
                                    f"Secondary Structure distribution for ClinVar {i} Variants")

        if i == "Pathogenic":
            am_wrongly_predicted_disorder = am_disorder_clinvar_p[am_disorder_clinvar_p['AlphaMissense'] <= 0.5]
            am_wrongly_predicted_order = am_order_clinvar_p[am_order_clinvar_p['AlphaMissense'] <= 0.5]
            plot_distribution_secondary(am_wrongly_predicted_disorder, am_wrongly_predicted_order,
                                        f"Secondary Structure distribution for ClinVar {i} Variants (Below 0.5 AlphaMissense score)")
