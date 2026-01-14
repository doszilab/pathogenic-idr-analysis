import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os

def create_count_for_multicategory_for_all(big_functional_df, main_category_column='Category',ptm_aggregated=True,include_non_functional=False):
    # Define the functional info categories to count

    ptms = ["Phosphorylation", "Acetylation", "Sumoylation", "Ubiquitination", "Methylation"]
    if "secondary_structure_grouped" in big_functional_df.columns:
        secondary_structures = big_functional_df["secondary_structure_grouped"].unique()

    functional_info = {
        # "binding_info":"Binding Region",
        "dibs_info":"DIBS",
        "phasepro_info":"PhasePro",
        "mfib_info":"MFIB",
        "Elm_Info":"ELM",
        "Roi":"UniProt Roi",
        "MobiDB":"Exp. Dis",
        "PDB":"PDB"
    }

    # Create a list to store the rows for pivoting
    pivot_data = []

    # Iterate over each combination of the main and genic category
    group_cols = [main_category_column]

    grouped_df = big_functional_df.groupby(group_cols)

    for group, group_df in grouped_df:
        for func_key, name in functional_info.items():
            # Use the .str.contains() method to check occurrences of each functional term in 'info' column
            count = group_df[group_df['info'].str.contains(func_key)][["Protein_ID", "Position"]].drop_duplicates().shape[0]

            # If the functional info exists in the group, append to pivot_data
            if count > 0:
                pivot_data.append({
                    main_category_column: group[0],
                    'fname': name,  # Pivot functional name into 'fname' column
                    'Count': count  # Include the count of occurrences
                })

        if "secondary_structure_grouped" in big_functional_df.columns:
            # Count secondary structures
            for ss in secondary_structures:
                count = group_df[group_df["secondary_structure_grouped"] == ss][["Protein_ID", "Position"]].drop_duplicates().shape[0]
                if count > 0:
                    pivot_data.append({
                        main_category_column: group[0],
                        'fname': ss,
                        'Count': count
                    })

        # Handle PTMs
        if ptm_aggregated:
            # Count if any PTM exists
            ptm_found = group_df[group_df['info'].astype(str).apply(lambda x: any(ptm in x for ptm in ptms))][["Protein_ID","Position"]].drop_duplicates().shape[0]
            if ptm_found > 0:
                pivot_data.append({
                    main_category_column: group[0],
                    'fname': 'PTM',
                    'Count': ptm_found
                })
        else:
            # Count each PTM individually
            for ptm in ptms:
                count = group_df[group_df['info'].str.contains(ptm, na=False)][["Protein_ID","Position"]].drop_duplicates().shape[0]
                if count > 0:
                    pivot_data.append({
                        main_category_column: group[0],
                        'fname': ptm,
                        'Count': count
                    })

        if include_non_functional:
            non_functional_count = group_df[group_df['info'].str.contains(r"-", na=False)][["Protein_ID","Position"]].drop_duplicates().shape[0]
            if non_functional_count > 0:
                pivot_data.append({
                    main_category_column: group[0],
                    'fname': 'No Annotations',
                    'Count': non_functional_count
                })

    # Convert pivot_data into a DataFrame
    pivot_df = pd.DataFrame(pivot_data)

    # # Pivot the data to get the count in the required format
    # pivoted_df = pivot_df.pivot_table(index=['Category', 'Genic_Category'],
    #                                   columns='fname', values='count', fill_value=0).reset_index()

    return pivot_df

def plot_functional_distribution_multicategory_for_all(big_functional_df, main_category_column='Category',file_name=None, category_colors={},figsize=(12, 8),islegend=True,filepath=None,xlabel="Functional Region",ptm_aggregated=True,include_non_functional=False):
    # Count the number of mutations for each function, interpretation category, and genic category
    counts_df = create_count_for_multicategory_for_all(big_functional_df,main_category_column,ptm_aggregated=ptm_aggregated,include_non_functional=include_non_functional)

    # Pivot the DataFrame to have 'fname' as index and 'Category' as columns with 'Count' as values
    counts_pivot = counts_df.pivot_table(index='fname', columns=main_category_column, values='Count', fill_value=0)

    # Reorder columns based on stack_order if provided
    stack_order = [x for x in category_colors.keys() if x in counts_pivot.columns]
    if ptm_aggregated:
        function_order = ["Exp. Dis","DIBS","MFIB","ELM","PDB","PTM","PhasePro","UniProt Roi","No Annotations"]
    else:
        function_order = ["Exp. Dis","DIBS","MFIB","ELM","PhasePro","UniProt Roi","Phosphorylation", "Acetylation", "Sumoylation", "Ubiquitination", "Methylation","No Annotations"]

    if stack_order:
        counts_pivot = counts_pivot.reindex(columns=stack_order)
        counts_pivot = counts_pivot.reindex(index=function_order)
        # counts_pivot = counts_pivot.drop(index=["PDB","UniProt Roi"])


    # Colors for each category based on stack_order
    colors = [category_colors[cat] for cat in counts_pivot.columns]

    # Plotting the stacked bar chart
    ax = counts_pivot.plot(kind='bar', stacked=True, figsize=figsize, color=colors)

    # Annotate total number at the top of each bar
    for i, total in enumerate(counts_pivot.sum(axis=1)):
        ax.text(i, total + 5, f'{int(total):,}', ha='center', va='bottom', fontsize=10)


    # Set titles and labels
    plt.suptitle("AlphaMissense Accuracy in Functional Region")
    plt.xlabel(None)
    plt.ylabel("Number of Mutations")
    plt.xticks(rotation=90)

    max_val = counts_pivot.sum(axis=1).max()
    plt.ylim(0, max_val * 1.15)

    # Add legend
    if islegend:
        plt.legend(title='Category', loc='upper left')
    else:
        # If we do not want a legend, remove it explicitly
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    plt.tight_layout()

    if filepath:
        plt.savefig(filepath,bbox_inches='tight')

    plt.show()

def extract_pos_based_df(df):
    df[['Protein_ID', 'Position']] = df['AccessionPosition'].str.split('|', expand=True)
    df = df.drop(columns=['AccessionPosition'])
    df['Position'] = df['Position'].astype(int)
    return df

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
        'P': 'Coil', 'T': 'Coil', 'S': 'Coil', 'C': 'Coil'  # Group loops
    }

    df['secondary_structure_grouped'] = df['secondary_structure'].map(secondary_structure_groups)
    return df

def create_count_for_secondary_structure(df, secondary_structure_column='secondary_structure_grouped', main_category_column='Category'):
    """
    Groups and counts mutations based on secondary structure categories.

    Args:
        df (pd.DataFrame): Input DataFrame containing secondary structure information.
        secondary_structure_column (str): The column indicating the secondary structure.
        main_category_column (str): The main category column (e.g., 'Category') to group by.

    Returns:
        pd.DataFrame: A DataFrame with counts of mutations grouped by secondary structure and main categories.
    """
    # Create a list to store the rows for pivoting
    pivot_data = []

    # Group the DataFrame by main category and secondary structure
    grouped_df = df.groupby([main_category_column, secondary_structure_column])

    for (category, structure), group_df in grouped_df:
        count = len(group_df)
        if count > 0:
            pivot_data.append({
                main_category_column: category,
                'secondary_structure': structure,
                'Count': count
            })

    # Convert pivot_data into a DataFrame
    pivot_df = pd.DataFrame(pivot_data)

    return pivot_df

def plot_secondary_structure_distribution(df, main_category_column='Category', file_name=None, category_colors={}, figsize=(12, 8), islegend=True, filepath=None):
    """
    Plots the distribution of mutations grouped by secondary structure and main category.

    Args:
        df (pd.DataFrame): DataFrame with mutation counts grouped by secondary structure.
        main_category_column (str): The column indicating main categories (e.g., 'Category').
        file_name (str): Name of the output file (if saving the plot).
        category_colors (dict): Dictionary of colors for each main category.
        figsize (tuple): Size of the figure.
        islegend (bool): Whether to display a legend.
        filepath (str): Path to save the plot.

    Returns:
        None
    """
    counts_df = create_count_for_multicategory_for_all(df, main_category_column)
    counts_df = counts_df[counts_df['fname'].isin(df['secondary_structure_grouped'])]
    # Pivot the DataFrame to have 'secondary_structure' as index and 'Category' as columns with 'Count' as values
    counts_pivot = counts_df.pivot_table(index='fname', columns=main_category_column, values='Count', fill_value=0)

    # Colors for each category based on stack_order
    colors = [category_colors[cat] for cat in counts_pivot.columns]

    # Plotting the stacked bar chart
    ax = counts_pivot.plot(kind='bar', stacked=True, figsize=figsize, color=colors)

    # Annotate total number at the top of each bar
    for i, total in enumerate(counts_pivot.sum(axis=1)):
        ax.text(i, total + 5, f'{int(total):,}', ha='center', va='bottom', fontsize=10)

    # Set titles and labels
    plt.suptitle("AlphaMissense Accuracy for Secondary Structures")
    plt.xlabel("Secondary Structure")
    plt.ylabel("Number of Mutations")
    plt.xticks(rotation=45)

    max_val = counts_pivot.sum(axis=1).max()
    plt.ylim(0, max_val * 1.15)

    # Add legend
    if islegend:
        plt.legend(title='Category', loc='upper left')
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    plt.tight_layout()

    if filepath:
        plt.savefig(filepath, bbox_inches='tight')

    plt.show()


def plot_combined_distributions(big_functional_df, main_category_column='Category',
                                category_colors={}, figsize=(6, 6), ptm_aggregated=True,
                                include_non_functional=False,filepath=None,width_ratios=[3, 2]):
    """
    Combines the functional region distribution plot and secondary structure distribution plot into a single figure with two subplots.

    Args:
        big_functional_df (pd.DataFrame): Input DataFrame with mutation and structural data.
        main_category_column (str): The main category column (e.g., 'Category').
        category_colors (dict): A dictionary mapping categories to colors.
        figsize (tuple): Figure size for the combined plot.
        ptm_aggregated (bool): Whether to aggregate PTMs.
        include_non_functional (bool): Whether to include non-functional regions.
    """
    # Create the combined figure and axes
    fig, axes = plt.subplots(1, 2, figsize=figsize,width_ratios=width_ratios,gridspec_kw={'wspace': 0.3} )

    # Plot 1: Functional Region Distribution
    counts_df = create_count_for_multicategory_for_all(big_functional_df, main_category_column, ptm_aggregated,
                                                       include_non_functional)

    counts_pivot = counts_df.pivot_table(index='fname', columns=main_category_column, values='Count', fill_value=0)



    stack_order = [x for x in category_colors.keys() if x in counts_pivot.columns]
    if ptm_aggregated:
        function_order = ["Exp. Dis","DIBS","MFIB","ELM","PDB","PTM","PhasePro","UniProt Roi","No Annotations"]
    else:
        function_order = ["Exp. Dis","DIBS","MFIB","ELM","PhasePro","UniProt Roi","Phosphorylation", "Acetylation", "Sumoylation", "Ubiquitination", "Methylation","No Annotations"]

    if stack_order:
        counts_pivot = counts_pivot.reindex(columns=stack_order)
        counts_pivot = counts_pivot.reindex(index=function_order)

    colors = [category_colors[cat] for cat in counts_pivot.columns]
    counts_pivot.plot(kind='bar', stacked=True, ax=axes[0], color=colors)

    axes[0].set_title("Functional Region Distribution")
    axes[0].set_ylabel("Number of Positions")
    axes[0].set_xlabel(None)
    axes[0].legend(loc='upper left')
    counts_pivot['Sum'] = counts_pivot['Correctly Predicted'] + counts_pivot['Incorrectly Predicted']
    axes[0].set_ylim(0,counts_pivot['Sum'].max() * 1.1)

    print(counts_pivot['Sum'])

    for i, (index, row) in enumerate(counts_pivot.iterrows()):
        total = row["Correctly Predicted"] + row["Incorrectly Predicted"]  # sum for positioning
        overlap_val = row["Correctly Predicted"]  # what we actually want to display

        # Place the text slightly above the top of the stacked bar
        axes[0].text(
            i,
            total + 5,  # position label above the total stack
            f'{int(overlap_val):,}',  # display the Overlap value
            ha='center',
            va='bottom',
            fontsize=10,
        )

    # Plot 2: Secondary Structure Distribution
    counts_df = counts_df[counts_df['fname'].isin(big_functional_df['secondary_structure_grouped'])]
    ss_counts_pivot = counts_df.pivot_table(index='fname', columns=main_category_column, values='Count', fill_value=0)


    stack_order_ss = [x for x in category_colors.keys() if x in ss_counts_pivot.columns]
    if stack_order_ss:
        ss_counts_pivot = ss_counts_pivot.reindex(columns=stack_order_ss)

    colors_ss = [category_colors[cat] for cat in ss_counts_pivot.columns]
    ss_counts_pivot.plot(kind='bar', stacked=True, ax=axes[1], color=colors_ss)

    axes[1].set_title("Secondary Structure Distribution")
    axes[1].set_ylabel(None)
    axes[1].set_xlabel(None)
    if axes[1].get_legend() is not None:
        axes[1].get_legend().remove()
    # axes[1].legend(title='Category', loc='upper left')
    ss_counts_pivot['Sum'] = ss_counts_pivot['Correctly Predicted'] + ss_counts_pivot['Incorrectly Predicted']
    axes[1].set_ylim(0, ss_counts_pivot['Sum'].max() * 1.1)

    # Annotate total number at the top of each bar
    for i, (index, row) in enumerate(ss_counts_pivot.iterrows()):
        total = row["Correctly Predicted"] + row["Incorrectly Predicted"]  # sum for positioning
        overlap_val = row["Correctly Predicted"]  # what we actually want to display

        # Place the text slightly above the top of the stacked bar
        axes[1].text(
            i,
            total + 5,  # position label above the total stack
            f'{int(overlap_val):,}',  # display the Overlap value
            ha='center',
            va='bottom',
            fontsize=10,
        )

    # for i, total in enumerate(ss_counts_pivot['Sum']):
    #     axes[1].text(i, total + 5, f'{int(total):,}', ha='center', va='bottom', fontsize=10)

    # # Common adjustments
    # for ax in axes:
    #     ax.set_xlabel(None)
    #     ax.set_xticks(range(len(counts_pivot.index)))
    #     ax.set_xticklabels(counts_pivot.index if ax == axes[0] else ss_counts_pivot.index, rotation=45, ha='right')


    # plt.suptitle("AlphaMissense Accuracy on ClinVar Pathogenic Positions")
    plt.tight_layout()

    if filepath:
        plt.savefig(filepath,bbox_inches='tight',dpi=300)

    plt.show()

def plot_combined_distributions_new(big_functional_df, main_category_column='Category',
                                category_colors={}, figsize=(6, 6), ptm_aggregated=True,
                                include_non_functional=False,filepath=None,width_ratios=[3, 2]):
    """
    Combines the functional region distribution plot and secondary structure distribution plot into a single figure with two subplots.

    Args:
        big_functional_df (pd.DataFrame): Input DataFrame with mutation and structural data.
        main_category_column (str): The main category column (e.g., 'Category').
        category_colors (dict): A dictionary mapping categories to colors.
        figsize (tuple): Figure size for the combined plot.
        ptm_aggregated (bool): Whether to aggregate PTMs.
        include_non_functional (bool): Whether to include non-functional regions.
    """
    # Create the combined figure and axes
    fig, axes = plt.subplots(1, 3, figsize=figsize,width_ratios=width_ratios)

    # Plot 1: Functional Region Distribution
    counts_df = create_count_for_multicategory_for_all(big_functional_df, main_category_column, ptm_aggregated,
                                                       include_non_functional)
    counts_pivot = counts_df.pivot_table(index='fname', columns=main_category_column, values='Count', fill_value=0)



    stack_order = [x for x in category_colors.keys() if x in counts_pivot.columns]
    if ptm_aggregated:
        function_order = ["Exp. Dis","DIBS","MFIB","ELM","PDB","PTM","PhasePro","UniProt Roi","No Annotations"]
    else:
        function_order = ["Exp. Dis","DIBS","MFIB","ELM","PhasePro","UniProt Roi","Phosphorylation", "Acetylation", "Sumoylation", "Ubiquitination", "Methylation","No Annotations"]

    if stack_order:
        counts_pivot = counts_pivot.reindex(columns=stack_order)
        counts_pivot = counts_pivot.reindex(index=function_order)

    colors = [category_colors[cat] for cat in counts_pivot.columns]
    counts_pivot.plot(kind='bar', stacked=True, ax=axes[0], color=colors)

    axes[0].set_title("Functional Region Distribution")
    axes[0].set_ylabel("Number of Positions")
    axes[0].set_xlabel(None)
    axes[0].legend(loc='upper left')
    counts_pivot['Sum'] = counts_pivot['Correctly Predicted'] + counts_pivot['Incorrectly Predicted']
    axes[0].set_ylim(0,counts_pivot['Sum'].max() * 1.1)

    print(counts_pivot['Sum'])

    # Annotate total number at the top of each bar
    for i, total in enumerate(counts_pivot['Sum']):
        axes[0].text(i, total + 5, f'{int(total):,}', ha='center', va='bottom', fontsize=10)

    # Plot 2: Secondary Structure Distribution
    counts_df = counts_df[counts_df['fname'].isin(big_functional_df['secondary_structure_grouped'])]
    ss_counts_pivot = counts_df.pivot_table(index='fname', columns=main_category_column, values='Count', fill_value=0)

    stack_order_ss = [x for x in category_colors.keys() if x in ss_counts_pivot.columns]
    if stack_order_ss:
        ss_counts_pivot = ss_counts_pivot.reindex(columns=stack_order_ss)

    colors_ss = [category_colors[cat] for cat in ss_counts_pivot.columns]
    ss_counts_pivot.plot(kind='bar', stacked=True, ax=axes[1], color=colors_ss)

    axes[1].set_title("Secondary Structure Distribution")
    axes[1].set_ylabel(None)
    axes[1].set_xlabel(None)
    if axes[1].get_legend() is not None:
        axes[1].get_legend().remove()
    # axes[1].legend(title='Category', loc='upper left')
    ss_counts_pivot['Sum'] = ss_counts_pivot['Correctly Predicted'] + ss_counts_pivot['Incorrectly Predicted']
    axes[1].set_ylim(0, ss_counts_pivot['Sum'].max() * 1.1)

    # Annotate total number at the top of each bar
    for i, total in enumerate(ss_counts_pivot['Sum']):
        axes[1].text(i, total + 5, f'{int(total):,}', ha='center', va='bottom', fontsize=10)

    plt.suptitle("AlphaMissense Accuracy on ClinVar Pathogenic Positions")
    plt.tight_layout()

    if filepath:
        plt.savefig(filepath,bbox_inches='tight',dpi=300)

    plt.show()


if __name__ == '__main__':
    core_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations'

    to_clinvar_dir = os.path.join(core_dir, 'processed_data/files/clinvar')
    plot_dir = os.path.join(core_dir, 'processed_data/plots/clinvar')

    pos_based_dir = os.path.join(core_dir, 'data/discanvis_base_files/positional_data_process')

    table_file = os.path.join(core_dir,f"data/discanvis_base_files/sequences/loc_chrom_with_names_main_isoforms_with_seq.tsv")
    tables_with_only_main_isoforms = pd.read_csv(table_file, sep='\t')

    COLORS = {
        "disorder": '#ffadad',
        "order": '#a0c4ff',
        "both": '#ffc6ff',
        "Pathogenic": '#ff686b',
        "Benign": "#b2f7ef",
        "Uncertain": "#8e9aaf",
        'Mutated': '#f27059',
        'Non-Mutated': '#80ed99',
        'Correctly Predicted': '#80ed99',
        'Incorrectly Predicted': '#f27059',
    }

    fig_path = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/plots'
    fig1 = os.path.join(fig_path, "fig1")
    fig2 = os.path.join(fig_path, "fig2")
    fig3 = os.path.join(fig_path, "fig3")
    figsm = os.path.join(fig_path, "sm", 'clinvar')
    uniprot_dir = os.path.join(fig_path, "sm", 'uniprot')
    alphamissense_dir = os.path.join(fig_path, "sm", 'alphamissense')

    # Pathogenic Positions
    pathogenic_positional_disorder_df = pd.read_csv(
        f"{to_clinvar_dir}/Pathogenic/disorder/positional_clinvar_functional_categorized_final.tsv", sep='\t')


    # AM prediction for Variants
    disorder_clinvar_with_am_score_df = pd.read_csv("/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/benchmark/clinvar_disorder_all.tsv",sep='\t')
    relevant_cols = ['Protein_ID','Position','AlphaMissense_pos']
    disorder_clinvar_with_only_scores_df = disorder_clinvar_with_am_score_df[disorder_clinvar_with_am_score_df['Interpretation'] == "Pathogenic"][relevant_cols].drop_duplicates()
    print(pathogenic_positional_disorder_df)
    print(disorder_clinvar_with_only_scores_df)
    merged_df = pathogenic_positional_disorder_df.merge(disorder_clinvar_with_only_scores_df,on=['Protein_ID','Position'])


    merged_df['Prediciton'] = np.where(merged_df['AlphaMissense_pos'] >= 0.5,"Correctly Predicted","Incorrectly Predicted")




    # # Plot By Functional
    # plot_functional_distribution_multicategory_for_all(
    #     merged_df, main_category_column='Prediciton',category_colors=COLORS,
    #     figsize=(4, 4),ptm_aggregated=True,include_non_functional=True,
    #     filepath=os.path.join(alphamissense_dir, "functional_acc.png")
    # )
    #
    # exit()

    # Plot by Secondary Structure
    secondary_structure_df = extract_pos_based_df(pd.read_csv(f"{pos_based_dir}/secondary_structure_pos.tsv", sep='\t',
                                                              # nrows=1000000
                                                              ))
    secondary_structure_df = group_secondary_structure(secondary_structure_df)

    # print(secondary_structure_df.columns)
    # print(secondary_structure_df['secondary_structure_grouped'].unique())
    # exit()
    merged_df = merged_df.merge(secondary_structure_df, on=['Protein_ID', 'Position'],how='left').drop_duplicates()

    # print(merged_df[merged_df['info'].str.contains("Elm")][["Protein_ID", "Position"]].drop_duplicates().shape[0])
    # exit()

    # print(merged_df)
    #
    # plot_secondary_structure_distribution(
    #     merged_df,
    #     main_category_column='Prediciton',
    #     category_colors=COLORS,
    #     figsize = (4, 4),
    #     islegend=False,
    #     filepath = os.path.join(alphamissense_dir, "secondary_acc.png")
    # )

    # exit()

    plot_combined_distributions(merged_df,
                                main_category_column='Prediciton',category_colors=COLORS,include_non_functional=True,
                                ptm_aggregated=True,
                                figsize=(12, 3),filepath=os.path.join(alphamissense_dir,"combined_acc.png"),width_ratios = [5, 2]
                                )

