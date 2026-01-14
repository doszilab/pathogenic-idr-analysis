import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt



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
            count = group_df[group_df['info'].str.contains(func_key)][["Protein_ID","Position"]].drop_duplicates().shape[0]

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
                count = group_df[group_df["secondary_structure_grouped"] == ss].shape[0]
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
                    'fname': 'No Annotation',
                    'Count': non_functional_count
                })

    # Convert pivot_data into a DataFrame
    pivot_df = pd.DataFrame(pivot_data)

    # # Pivot the data to get the count in the required format
    # pivoted_df = pivot_df.pivot_table(index=['Category', 'Genic_Category'],
    #                                   columns='fname', values='count', fill_value=0).reset_index()

    return pivot_df

def plot_functional_distribution_multicategory_for_all(big_functional_df, main_category_column='Category',file_name=None, category_colors={},figsize=(12, 8),islegend=True,filepath=None,xlabel="Functional Region",ptm_aggregated=True,include_non_functional=False,title="Pathogenic Positions in PIP"):
    # Count the number of mutations for each function, interpretation category, and genic category
    counts_df = create_count_for_multicategory_for_all(big_functional_df,main_category_column,ptm_aggregated=ptm_aggregated,include_non_functional=include_non_functional)


    # Pivot the DataFrame to have 'fname' as index and 'Category' as columns with 'Count' as values
    counts_pivot = counts_df.pivot_table(index='fname', columns=main_category_column, values='Count', fill_value=0)

    # Reorder columns based on stack_order if provided
    stack_order = [x for x in category_colors.keys() if x in counts_pivot.columns]
    if ptm_aggregated:
        function_order = ["Exp. Dis","DIBS","MFIB","ELM","PDB","PTM","PhasePro","UniProt Roi","No Annotation"]
    else:
        function_order = ["Exp. Dis","DIBS","MFIB","ELM","PhasePro","UniProt Roi","Phosphorylation", "Acetylation", "Sumoylation", "Ubiquitination", "Methylation","No Annotation"]

    if stack_order:
        counts_pivot = counts_pivot.reindex(columns=stack_order)
        counts_pivot = counts_pivot.reindex(index=function_order)
        # counts_pivot = counts_pivot.drop(index=["PDB","UniProt Roi"])


    # Colors for each category based on stack_order
    colors = [category_colors[cat] for cat in counts_pivot.columns]

    # Plotting the stacked bar chart
    ax = counts_pivot.plot(kind='bar', stacked=True, figsize=figsize, color=colors)

    # # Annotate total number at the top of each bar
    # for i, total in enumerate(counts_pivot.sum(axis=1)):
    #     print(i,total)
    #     print(counts_pivot)
    #     exit()
    #     ax.text(i, total + 5, f'{int(total):,}', ha='center', va='bottom', fontsize=10)

    for i, (index, row) in enumerate(counts_pivot.iterrows()):
        total = row["Correctly Predicted"] + row["Incorrectly Predicted"]  # sum for positioning
        overlap_val = row["Correctly Predicted"]  # what we actually want to display

        # Place the text slightly above the top of the stacked bar
        ax.text(
            i,
            total + 5,  # position label above the total stack
            f'{int(overlap_val):,}',  # display the Overlap value
            ha='center',
            va='bottom',
            fontsize=10
        )

    # Set titles and labels
    plt.suptitle(title)
    plt.xlabel(None)
    plt.ylabel("Number of Positions")
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

def annotation_merge(df,main_category_column,group_col):
    # Group by the category and annotation; count unique positions
    counts_df = (
        df
        .drop_duplicates(subset=['Protein_ID', 'Position'])  # ensure uniqueness
        .groupby([main_category_column, group_col])
        .size()
        .reset_index(name='Count')
    )

    # Rename the 'annotation' column for clarity
    counts_df.rename(columns={group_col: 'fname'}, inplace=True)
    return counts_df


def expand_regions_to_positions(df: pd.DataFrame,
                                pid_col: str = 'Protein_ID',
                                start_col: str = 'Start',
                                end_col: str = 'End') -> pd.DataFrame:
    """
    Expands each region (start, end) in a DataFrame into all individual positions.

    Returns
    -------
    A DataFrame with columns [pid_col, 'Position'] for every integer position in [Start, End].
    """
    df_regions = df[[pid_col, start_col, end_col]].drop_duplicates()
    positions_list = []
    for _, row in df_regions.iterrows():
        pid = row[pid_col]
        start = row[start_col]
        end = row[end_col]
        for pos in range(start, end + 1):
            positions_list.append((pid, pos))

    return pd.DataFrame(positions_list, columns=[pid_col, 'Position'])


def plot_functional_distribution_for_all(
        big_functional_df,
        main_category_column='Category',
        file_name=None,
        category_colors={},
        figsize=(8, 6),
        islegend=False,
        filepath=None,
        annotation_based=True,
        group_col=None,
        xlabel="Functional Region",
        ptm_aggregated=True,
        include_non_functional=False,
        rotation=0,
        title="Pathogenic Positions in PIP",
        ax=None  # <---- new optional Axes argument
):
    """
    Plots ONLY the 'Correctly Predicted' category as a single bar per row,
    expressed as percentages relative to the row's total (Correct + Incorrect).
    If no 'ax' is provided, this function will create its own figure.
    """

    # If no Axes object is provided, create a new figure and Axes.
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # 1) Build the DataFrame you need
    if annotation_based:
        # Mark each row as 'Has Annotation' vs 'No Annotation'
        big_functional_df['annotation'] = big_functional_df['info'].apply(
            lambda x: 'No Annotation' if isinstance(x, str) and ('-' in x) else 'Has Annotation'
        )
        counts_df = annotation_merge(big_functional_df, main_category_column, 'annotation')
    else:
        counts_df = annotation_merge(big_functional_df, main_category_column, group_col)

    # Pivot for counts
    counts_pivot = counts_df.pivot_table(
        index='fname',
        columns=main_category_column,
        values='Count',
        fill_value=0
    )

    # Ensure columns appear in the same order as in category_colors
    stack_order = [cat for cat in category_colors.keys() if cat in counts_pivot.columns]
    if stack_order:
        counts_pivot = counts_pivot.reindex(columns=stack_order)

    # Convert to percentage
    counts_pivot_perc = counts_pivot.div(counts_pivot.sum(axis=1), axis=0) * 100

    # Sort by 'Correctly Predicted'
    if "Correctly Predicted" not in counts_pivot_perc.columns:
        print("No 'Correctly Predicted' column found in data. Exiting.")
        return

    sorted_df = counts_pivot_perc.sort_values(by="Correctly Predicted", ascending=False)
    function_order = sorted_df.index.tolist()
    counts_pivot_perc = counts_pivot_perc.reindex(index=function_order)

    # Only keep the 'Correctly Predicted' column
    correctly_predicted_perc = counts_pivot_perc[["Correctly Predicted"]]
    # cp_color = category_colors.get("Correctly Predicted", "blue")
    bar_colors = [category_colors.get(fname, "gray") for fname in correctly_predicted_perc.index]

    # Plot a single bar for each row
    correctly_predicted_perc["Correctly Predicted"].plot(
        kind="bar",
        ax=ax,
        legend=False,
        color=bar_colors
    )

    # Annotate each bar
    for i, (index, value) in enumerate(correctly_predicted_perc["Correctly Predicted"].items()):
        ax.text(
            i,
            value + 1,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    # Set labels, title, etc. on this Axes
    ax.set_xlabel(None)
    ax.set_ylabel("Percentage of Positions (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
    ax.set_ylim(0, 110)
    ax.set_title(title)

    if islegend:
        ax.legend([f"Correctly Predicted"], loc="lower left")

    # If the user passed `ax` themselves, do not plt.show() inside this function.
    # If we created ax ourselves, we can safely call plt.show().
    if filepath and ax is not None:
        plt.savefig(filepath, bbox_inches='tight')
    if ax is None:
        plt.tight_layout()
        plt.show()


def plot_three_distributions_in_one_figure(merged_df, COLORS):
    """
    This function creates a single figure with 3 subplots (1 row x 3 columns),
    calling `plot_functional_distribution_for_all` with different parameters
    on each subplot.
    """
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    # 1) Subplot for "Annotations"
    plot_functional_distribution_for_all(
        big_functional_df=merged_df,
        main_category_column='Prediciton',
        category_colors=COLORS,
        annotation_based=True,  # using default group_col=None
        figsize=(4, 3),
        ptm_aggregated=True,
        include_non_functional=True,
        title="Annotations",
        rotation=0,
        ax=axes[0]  # <--- third subplot
    )

    # 2) Subplot for "Structure"
    plot_functional_distribution_for_all(
        big_functional_df=merged_df,
        main_category_column='Prediciton',
        category_colors=COLORS,
        group_col='Category',
        annotation_based=False,
        figsize=(4, 3),
        ptm_aggregated=True,
        include_non_functional=True,
        title="Structure",
        rotation=0,
        ax=axes[1]  # <--- first subplot
    )

    # 3) Subplot for "Disease"
    plot_functional_distribution_for_all(
        big_functional_df=merged_df,
        main_category_column='Prediciton',
        category_colors=COLORS,
        group_col='genic_category',
        annotation_based=False,
        figsize=(6, 5),
        ptm_aggregated=True,
        include_non_functional=True,
        title="Genic",
        rotation=0,
        ax=axes[2]  # <--- second subplot
    )



    plt.suptitle("Correctly Predicted Pathogenic Positions using AlphaMissense", fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(alphamissense_dir,"correctly_predicted_pathogenic_positions.png"), bbox_inches='tight')
    # plt.show()


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


        'PIP Region': '#80ed99',
        'PEM Region': '#80ed99',
        'Correctly Predicted': '#80ed99',
        'Overlap': '#80ed99',
        'Non-PIP': '#f27059',
        'Non-PEM': '#f27059',
        'Non-Overlap': '#f27059',
        'Incorrectly Predicted': '#f27059',

        "Non-Disorder Specific":  "#8e9aaf",
        'Disorder Specific': '#f27059',

        'Has Annotation': '#f27059',
        "No Annotation": "#8e9aaf",

        'Monogenic': '#c9cba3',
        'Multigenic': '#ffe1a8',
        'Complex': '#A0E7E5',
    }

    fig_path = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/plots'
    fig1 = os.path.join(fig_path, "fig1")
    fig2 = os.path.join(fig_path, "fig2")
    fig3 = os.path.join(fig_path, "fig3")
    figsm = os.path.join(fig_path, "sm", 'clinvar')
    uniprot_dir = os.path.join(fig_path, "sm", 'uniprot')
    alphamissense_dir = os.path.join(fig_path, "sm", 'alphamissense')
    error_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/errors'

    # Pathogenic Positions
    pathogenic_positional_disorder_df = pd.read_csv(f"{to_clinvar_dir}/Pathogenic/disorder/positional_clinvar_functional_categorized_final.tsv", sep='\t')
    common_cols = ['Protein_ID', 'Position', 'Prediciton', 'info']

    # AM prediction for Variants
    disorder_clinvar_with_am_score_df = pd.read_csv(
        "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/benchmark/clinvar_disorder_all.tsv",
        sep='\t')
    relevant_cols = ['Protein_ID', 'Position', 'AlphaMissense_pos']
    disorder_clinvar_with_only_scores_df = \
        disorder_clinvar_with_am_score_df[disorder_clinvar_with_am_score_df['Interpretation'] == "Pathogenic"][
            relevant_cols].drop_duplicates()

    am_pathogenic = pathogenic_positional_disorder_df.copy()
    merged_df = am_pathogenic.merge(disorder_clinvar_with_only_scores_df,
                                                        on=['Protein_ID', 'Position'],how='left')
    # print(merged_df)
    merged_df['Prediciton'] = np.where(merged_df['AlphaMissense_pos'] >= 0.5, "Correctly Predicted","Incorrectly Predicted")
    print(merged_df.columns)

    # # Annotation
    # plot_functional_distribution_multicategory_for_all(
    #     merged_df, main_category_column='Prediciton', category_colors=COLORS,
    #     figsize=(4, 4), ptm_aggregated=True, include_non_functional=True,
    #     # filepath=os.path.join(alphamissense_dir, "functional_acc.png")
    #     title="AM Accuracy for Pathogenic Positions"
    # )
    #
    # # Genic
    # plot_functional_distribution_for_all(merged_df, main_category_column='Prediciton', category_colors=COLORS,group_col='genic_category',annotation_based=False,
    #                                      figsize=(4, 3), ptm_aggregated=True, include_non_functional=True,
    #                                      # filepath=os.path.join(alphamissense_dir, "functional_acc.png")
    #                                      title="Correctly Predicted - Genic Category"
    #                                      )


    # Mostly Order/Disorder
    disorder_categories = ["Only Disorder", 'Disorder Mostly']
    merged_df['Category'] = merged_df['Category'].apply(
        lambda x: "Disorder Specific" if x in disorder_categories else "Non-Disorder Specific")

    plot_three_distributions_in_one_figure(merged_df, COLORS)
    exit()
    #
    # plot_functional_distribution_for_all(merged_df, main_category_column='Prediciton', category_colors=COLORS,
    #                                      group_col='Category', annotation_based=False,
    #                                      figsize=(4, 3), ptm_aggregated=True, include_non_functional=True,
    #                                      # filepath=os.path.join(alphamissense_dir, "functional_acc.png")
    #                                      title="Correctly Predicted - Structure"
    #                                      )
    #
    # plot_functional_distribution_for_all(merged_df, main_category_column='Prediciton', category_colors=COLORS,
    #                                      group_col='category_names', annotation_based=False,
    #                                      figsize=(6, 5), ptm_aggregated=True, include_non_functional=True,
    #                                      # filepath=os.path.join(alphamissense_dir, "functional_acc.png")
    #                                      title="Correctly Predicted - Disease",
    #                                      rotation=90
    #                                      )
    #
    # # Annotation
    # plot_functional_distribution_for_all(merged_df, main_category_column='Prediciton', category_colors=COLORS,
    #     figsize=(4, 3), ptm_aggregated=True, include_non_functional=True,
    #     # filepath=os.path.join(alphamissense_dir, "functional_acc.png")
    #     title="Correctly Predicted - Annotations"
    # )
    #
    # exit()

    # Pip positions
    # pip_path = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/elm/predicted_motif_region_by_am_sequential_rule.tsv"
    pip_path = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/Scripts/processing_data/pip/prediction_pipeline/predicted_motif_region_by_am_sequential_rule.tsv"
    pip_path = "/dlab/data/predicted_motif_region_by_am_sequential_rule.tsv"
    pip_df = pd.read_csv(pip_path, sep='\t')
    pip_positions = expand_regions_to_positions(pip_df, 'Protein_ID', 'Start', 'End')

    print(pip_positions[pip_positions['Protein_ID'] == 'KANSL1-202']['Position'].tolist())
    pip_positions['isPipRegion'] = True
    pip_pathogenic = pathogenic_positional_disorder_df.copy()
    merged_df_pip = pip_pathogenic.merge(pip_positions,on=['Protein_ID','Position'],how='left')
    merged_df_pip['Prediciton'] = np.where(merged_df_pip['isPipRegion'] == True,"Overlap","Non-Overlap")
    merged_df_pip = merged_df_pip[common_cols]
    # merged_df = merged_df[merged_df['Prediciton'] == "PIP Region"]

    # Plot By Functional
    plot_functional_distribution_multicategory_for_all(
        merged_df_pip, main_category_column='Prediciton',category_colors=COLORS,
        figsize=(4, 4),ptm_aggregated=True,include_non_functional=True,
        # filepath=os.path.join(alphamissense_dir, "functional_acc.png")
    )
    exit()

    # for info in ["Elm","dibs","mfib"]:
    #
    #
    #     elm_subset = merged_df[common_cols]
    #     elm_pip_subset = merged_df_pip[common_cols]
    #
    #     elm_pip = elm_pip_subset[(elm_pip_subset['Prediciton'] == "Overlap") & (elm_pip_subset['info'].str.contains(info)) ]
    #     elm = elm_subset[(elm_subset['Prediciton'] == "Overlap") & (elm_subset['info'].str.contains(info))]
    #
    #     diff_elm = (
    #         pd.concat([elm, elm_pip], ignore_index=True)
    #         .drop_duplicates(keep=False, subset=common_cols)
    #     )
    #     diff_elm.to_csv(os.path.join(error_dir, f"diff_{info}.tsv"), index=False,sep='\t')

    # exit()

    # PEM positions
    pem_path = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/elm/elm_predicted_with_am_info_all_disorder_with_rules.tsv"
    pem_df = pd.read_csv(pem_path, sep='\t')
    pem_df = pem_df[pem_df['Sequential_Rule'] == True]
    pem_positions = expand_regions_to_positions(pem_df, 'Protein_ID', 'Start', 'End')
    pem_positions['Prediction'] = "Overlap"
    merged_df = pathogenic_positional_disorder_df.merge(pem_positions, on=['Protein_ID', 'Position'], how='left')
    merged_df['Prediciton'] = np.where(merged_df['Prediction'].isna(), "Non-Overlap", merged_df['Prediction'])
    # merged_df = merged_df[merged_df['Prediciton'] == "PEM Region"]

    # # Plot By Functional
    plot_functional_distribution_multicategory_for_all(
        merged_df, main_category_column='Prediciton', category_colors=COLORS,
        figsize=(4, 4), ptm_aggregated=True, include_non_functional=True,
        # filepath=os.path.join(alphamissense_dir, "functional_acc.png")
        title="Pathogenic Positions in PEM"
    )

