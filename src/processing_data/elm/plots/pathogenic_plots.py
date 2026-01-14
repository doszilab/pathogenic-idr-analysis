import pandas as pd
import numpy as np
import os

from external_program.pyvenn import venn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import math

def plot_venn(all_mut, motif_mut, pred_motif_mut, pred_roi,
              mut_type="Pathogenic",
              names=["Predicted Pathogenic", "Known Motif", "Predicted Motif", "Predicted ROI"],save_path=None):
    """
    Draw a 4-set Venn diagram (with NO text labels on the circles),
    but add a legend that lists the original 'names' for each set.
    """

    # 1) Convert each DataFrame into a set of (Protein_ID, Position)
    full_pathogenic_set = set(zip(all_mut["Protein_ID"], all_mut["Position"]))
    known_motif_set     = set(zip(motif_mut["Protein_ID"], motif_mut["Position"]))
    predicted_motif_set = set(zip(pred_motif_mut["Protein_ID"], pred_motif_mut["Position"]))
    roi_set             = set(zip(pred_roi["Protein_ID"], pred_roi["Position"]))

    list_of_sets = [
        full_pathogenic_set,
        known_motif_set,
        predicted_motif_set,
        roi_set
    ]
    labels = venn.get_labels(list_of_sets, fill=['number'])

    # 2) Create the Venn with empty names → no text on the circles
    fig, ax = venn.venn4(labels, names=names, figsize=(6, 4))

    for text_obj in ax.texts:
        if text_obj.get_text() in names:
            text_obj.set_visible(False)

    # 5) Reposition the legend outside the figure, on the right side
    leg = ax.get_legend()
    if leg:
        # Move legend so it doesn't overlap the plot area
        leg.set_bbox_to_anchor((1.05, 0.5))  # x=1.05 (slightly outside axes), y=0.5 (center)
        leg.set_loc("center left")  # anchor the legend's left center to that point

    # 6) Title, layout, show
    plt.suptitle(f"ClinVar {mut_type} Overlaps", fontsize=16, x=0.4, y=1)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path,dpi=300,bbox_inches='tight')
    plt.show()


def plot_venn_pathogenic(motif_mut,pred_motif_mut,pred_roi,mut_type="Pathogenic",names=["Known Motif", "Predicted Motif", "Predicted ROI"],save_path=None):
    # Convert each DataFrame into a set of (Protein_ID, Position) for easy set intersection
    known_motif_set = set(
        zip(motif_mut["Protein_ID"],
            motif_mut["Position"])
    )
    predicted_motif_set = set(
        zip(pred_motif_mut["Protein_ID"],
            pred_motif_mut["Position"])
    )
    roi_set = set(
        zip(pred_roi["Protein_ID"],
            pred_roi["Position"])
    )

    # -------------------------------------------------------------------
    # 3) Prepare your list of sets (in the order you want them)
    # -------------------------------------------------------------------
    list_of_sets = [known_motif_set, predicted_motif_set, roi_set]

    # -------------------------------------------------------------------
    # 4) Calculate the labels using python-venn
    #    fill=['number', 'logic'] means:
    #       - 'number' shows the count of items in each region,
    #       - 'logic' shows the set expression (e.g., 'A ∩ B').
    # -------------------------------------------------------------------
    labels = venn.get_labels(list_of_sets, fill=['number'])

    # -------------------------------------------------------------------
    # 5) Draw the 4-set Venn diagram
    # -------------------------------------------------------------------
    fig, ax = venn.venn3(labels, names=names)

    ax.legend_.remove() if ax.get_legend() else None


    plt.suptitle(f"ClinVar {mut_type} Overlaps",fontsize=16)
    fig.set_size_inches(4, 4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path,dpi=300)
    plt.show()






def get_info_counts(df_full, df_annotated, df_pred_motif, df_pred_roi):
    """
    Given:
      - df_full        : DataFrame for the full set (e.g. all Pathogenic)
      - df_annotated   : DataFrame for annotated subset (e.g. annotated Pathogenic)
      - df_pred_motif  : DataFrame for predicted motif overlaps
      - df_pred_roi    : DataFrame for predicted ROI overlaps

    Returns:
      (with_info, no_info)
    """
    # Convert to sets of (Protein_ID, Position)
    full_set      = set(zip(df_full["Protein_ID"],      df_full["Position"]))
    annotated_set = set(zip(df_annotated["Protein_ID"], df_annotated["Position"]))
    motif_set     = set(zip(df_pred_motif["Protein_ID"], df_pred_motif["Position"]))
    roi_set       = set(zip(df_pred_roi["Protein_ID"],   df_pred_roi["Position"]))

    # Any annotation = union of the three "annotated" sets
    info_union = annotated_set.union(motif_set).union(roi_set)

    # Only count those in df_full that also appear in the annotation union
    with_info = len(full_set.intersection(info_union))
    total = len(full_set)
    no_info = total - with_info

    return with_info, no_info, info_union


def plot_annotation_bar_chart(
    df_pathogenic, df_annot_pathogenic, df_path_motif, df_path_roi,
    df_uncertain, df_annot_uncertain, df_uncertain_motif, df_uncertain_roi,
    df_pred_pathogenic, df_annot_pred_path, df_pred_path_motif, df_pred_path_roi
):
    """
    Creates a stacked bar chart for:
      1) Pathogenic
      2) Uncertain
      3) Predicted Pathogenic

    The Y-axis is percentage (0% to 100%).
    Each bar is split into 'with info' (red) and 'no info' (grey).
    Labels inside each segment show absolute counts and percentages.
    """

    # --- 1) Compute absolute counts (with_info / no_info) for each group ---
    # Pathogenic
    path_with_info, path_no_info, info_union = get_info_counts(
        df_full=df_pathogenic,
        df_annotated=df_annot_pathogenic,
        df_pred_motif=df_path_motif,
        df_pred_roi=df_path_roi
    )

    # Uncertain
    un_with_info, un_no_info, info_union = get_info_counts(
        df_full=df_uncertain,
        df_annotated=df_annot_uncertain,
        df_pred_motif=df_uncertain_motif,
        df_pred_roi=df_uncertain_roi
    )

    # Predicted Pathogenic
    predp_with_info, predp_no_info, info_union = get_info_counts(
        df_full=df_pred_pathogenic,
        df_annotated=df_annot_pred_path,
        df_pred_motif=df_pred_path_motif,
        df_pred_roi=df_pred_path_roi
    )

    # --- 2) Convert counts to percentages for plotting ---
    # We want each bar to sum to 100%.
    # We'll store both absolute counts and percentages to annotate later.
    groups = ["Pathogenic", "Uncertain", "Predicted\nPathogenic"]  # x-axis labels

    data_abs = [
        (path_with_info, path_no_info),
        (un_with_info,   un_no_info),
        (predp_with_info, predp_no_info)
    ]

    data_perc = []
    for (w_info, n_info) in data_abs:
        total = w_info + n_info
        if total > 0:
            w_info_perc = (w_info / total) * 100
            n_info_perc = (n_info / total) * 100
        else:
            w_info_perc, n_info_perc = 0, 0
        data_perc.append((w_info_perc, n_info_perc))

    # --- 3) Plot stacked bar chart with percentages ---
    x = np.arange(len(groups))  # positions for the bars
    bar_width = 0.6

    with_info_perc = [pair[0] for pair in data_perc]
    no_info_perc   = [pair[1] for pair in data_perc]

    fig, ax = plt.subplots(figsize=(4, 4))

    # The "with info" portion starts from 0 up to with_info_perc
    bars_with_info = ax.bar(
        x, with_info_perc, bar_width,
        color='red', label='PIP, PEM, Annotations'
    )

    # The "no info" portion is stacked on top of "with info"
    bars_no_info = ax.bar(
        x, no_info_perc, bar_width,
        color='grey', bottom=with_info_perc, label='Predicted Disorder'
    )

    # --- 4) Annotate each segment with absolute counts + percentages ---
    for i in range(len(groups)):
        w_abs, n_abs = data_abs[i]
        w_pct, n_pct = data_perc[i]

        # "With Info" label in the middle of the red portion
        ax.text(
            x[i], w_pct/2,  # halfway up the red portion
            f"{w_abs} \n({w_pct:.1f}%)",
            ha='center', va='center', color='black',
        )

        # "No Info" label in the middle of the grey portion
        # i.e. bottom is w_pct, halfway is w_pct + n_pct/2
        ax.text(
            x[i], w_pct + (n_pct/2),
            f"{n_abs} \n({n_pct:.1f}%)",
            ha='center', va='center', color='black',
        )

    # --- 5) Format axes ---
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel("Percentage (%)", fontsize=10)
    ax.set_ylim([0, 100])  # 0% to 100%
    fig.suptitle("Variant Position Overlap with Functional Region", fontsize=12)

    # Remove the legend from the axes
    ax.legend_.remove() if ax.get_legend() else None

    # Get legend handles/labels, and place the legend below the figure
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0),  # adjust -0.1 to move further down
    )

    # Adjust layout so there's room for the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # space at the bottom for the legend

    plt.show()


# def plot_top_genes(
#         df_pred_pathogenic,
#         top_n=10,
#         positional=False,
#         mut_type='Predicted Pathogenic',
#         rotation=45,
#         save_fig=False,
#         save_path='top_genes_plot.png'
# ):
#     """
#     FIG 6D/2 - Top Mutated Genes:
#     Show which genes have the highest mutation counts in the Predicted Pathogenic dataset.
#     --------
#     """
#
#     group_col = 'Gene_Uniprot'
#
#     # Merge to retrieve gene names (if not already present)
#     df_pred_pathogenic = df_pred_pathogenic.merge(
#         base_table[['Protein_ID', 'Gene_Uniprot']],
#         on='Protein_ID',
#         how='left'
#     )
#
#     # Depending on 'positional', sum or count
#     if positional:
#         gene_counts = (
#             df_pred_pathogenic
#             .groupby(group_col)["mutation_count"]
#             .sum()
#             .sort_values(ascending=False)
#         )
#     else:
#         gene_counts = df_pred_pathogenic[group_col].value_counts(dropna=True)
#
#     # Take top N mutated genes
#     gene_counts = gene_counts.head(top_n)
#
#     # Plot
#     fig, ax = plt.subplots(figsize=(8, 3))
#     gene_counts.plot(kind='bar', ax=ax, color='darkorange')
#
#     ax.set_title(f"Top {top_n} Mutated Genes ({mut_type})")
#     ax.set_xlabel(None)
#     ax.set_ylabel("Mutation Count")
#
#     # Rotate gene labels on X-axis
#     plt.setp(ax.get_xticklabels(), rotation=rotation, ha='right')
#
#     plt.tight_layout()
#
#     # Optionally save the figure
#     if save_fig:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Figure saved to: {save_path}")
#
#     plt.show()

def plot_top_genes(
    df_pred_pathogenic,
    top_n=20,
    positional=False,
    mut_type='Predicted Pathogenic',
    rotation=45,
    save_fig=False,
    save_path='top_genes_plot.png',
    color_dict=None,
    plot_legend_separately=False,
    simple_plot=False,
    bar_color='blue',
    figsize=(10, 2)
):
    """
    Creates a stacked bar chart showing the top N genes with the distribution of category_names.

    Parameters:
    -----------
    df_pred_pathogenic : pd.DataFrame
        DataFrame containing pathogenic mutations with 'Protein_ID', 'mutation_count', 'category_names' columns.
    base_table : pd.DataFrame
        DataFrame containing 'Protein_ID' and 'Gene_Uniprot' columns for merging.
    top_n : int, optional
        Number of top genes to display. Default is 20.
    positional : bool, optional
        If True, sum 'mutation_count' per gene and category. Else, count occurrences. Default is False.
    mut_type : str, optional
        Mutation type for the plot title. Default is 'Predicted Pathogenic'.
    rotation : int, optional
        Rotation angle for gene labels on X-axis. Default is 45.
    save_fig : bool, optional
        If True, save both the main plot and the legend (if generated) to 'save_path'. Default is False.
    save_path : str, optional
        File path to save the main plot (and legend if applicable). Default is 'top_genes_plot.png'.
    color_dict : dict, optional
        Dictionary mapping 'category_names' to colors. If None, a default palette is used.
    plot_legend_separately : bool, optional
        If True, plot the legend in a separate figure. Default is False.

    Returns:
    --------
    None
    """
    # 1. Ensure necessary columns are present
    required_columns = {'Protein_ID', 'category_names'}
    if not required_columns.issubset(df_pred_pathogenic.columns):
        missing = required_columns - set(df_pred_pathogenic.columns)
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    df_pred_pathogenic = modify_categories(df_pred_pathogenic)

    # 2. Merge with base_table to get 'Gene_Uniprot'
    df_merged = df_pred_pathogenic.merge(
        base_table[['Protein_ID', 'Gene_Uniprot']],
        on='Protein_ID',
        how='left'
    )

    if simple_plot:
        # Ensure necessary columns are present
        if not positional:
            # Ensure 'mutation_count' exists
            if 'mutation_count' not in df_merged.columns:
                raise ValueError("DataFrame must have 'mutation_count' column when positional=True")
            df_grouped = (
                df_merged
                .groupby(['Gene_Uniprot'])['mutation_count']
                .sum()
                .reset_index()
            )
            pivot_column = 'mutation_count'
            y_label = "Mutation Count"
        else:
            df_grouped = (
                df_merged
                .groupby(['Gene_Uniprot'])['Position']
                .nunique()
                .reset_index(name='count')
            )
            pivot_column = 'count'
            y_label = "Mutated Positions"

        # Sort genes by mutation count and select top_n
        df_sorted = df_grouped.sort_values(pivot_column, ascending=False).head(top_n)

        # Plotting
        plt.figure(figsize=figsize)
        sns.barplot(
            data=df_sorted,
            x='Gene_Uniprot',
            y=pivot_column,
            color=bar_color
        )

        # Labeling and formatting
        plt.xlabel(None)
        plt.ylabel(y_label)
        plt.title(f"Top {top_n} Mutated Genes ({mut_type})")
        plt.xticks(rotation=rotation, ha='right')

        # # Save the figure if required
        if save_fig:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()
        return  # Exit after simple plo


    # 3. Explode 'category_names' to have one category per row
    df_merged['category_names'] = df_merged['category_names'].str.split(', ')
    df_exploded = df_merged.explode('category_names')

    # 4. Depending on 'positional', sum 'mutation_count' or count occurrences
    if not positional:
        # Ensure 'mutation_count' exists
        if 'mutation_count' not in df_exploded.columns:
            raise ValueError("DataFrame must have 'mutation_count' column when positional=True")
        df_grouped = (
            df_exploded
            .groupby(['Gene_Uniprot', 'category_names'])['mutation_count']
            .sum()
            .reset_index()
        )
        pivot_column = 'mutation_count'
        y_label = "Mutation Count"
    else:
        df_grouped = (
            df_exploded
            .groupby(['Gene_Uniprot', 'category_names'])
            .size()
            .reset_index(name='count')
        )
        pivot_column = 'count'
        y_label = "Count"

    # 5. Pivot to get genes as rows and categories as columns
    df_pivot = df_grouped.pivot(index='Gene_Uniprot', columns='category_names', values=pivot_column).fillna(0)

    # 6. Sort genes by total counts and select top_n
    df_pivot['Total'] = df_pivot.sum(axis=1)
    df_pivot_sorted = df_pivot.sort_values('Total', ascending=False).drop('Total', axis=1).head(top_n)

    # 7. Define colors for categories
    categories = df_pivot_sorted.columns.tolist()
    num_categories = len(categories)

    if color_dict:
        # Map categories to colors using the dictionary
        colors = [color_dict.get(cat, '#999999') for cat in categories]  # Default color if not found
    else:
        # Use Seaborn's color palette
        colors = sns.color_palette("Set2", num_categories)

    # 8. Plot as a stacked bar chart without legend
    fig, ax = plt.subplots(figsize=(8, 2))  # Adjust height based on number of genes

    df_pivot_sorted.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color=colors
    )

    # 9. Labeling and formatting
    ax.set_xlabel(None)
    ax.set_ylabel(y_label)
    ax.set_title(f"Top {top_n} Mutated Genes ({mut_type})")

    # Rotate gene labels on X-axis
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha='right')

    # 10. Remove legend from plot
    if ax.get_legend():
        ax.legend_.remove()

    # 11. Optionally create a separate legend
    if plot_legend_separately:
        # Create a new figure for the legend
        fig_leg, ax_leg = plt.subplots(figsize=(3, 4))

        # Create legend patches
        legend_patches = [
            mpatches.Patch(color=colors[i], label=categories[i])
            for i in range(num_categories)
        ]

        # Add the legend to the separate axes
        ax_leg.legend(handles=legend_patches, title="Categories", loc='center', )
        ax_leg.axis('off')  # Hide the axes

        plt.tight_layout()

        # Optionally save the legend figure
        if save_fig and save_path:
            # Modify save_path to include legend if needed
            legend_save_path = save_path.replace('.png', '_legend.png')
            fig_leg.savefig(legend_save_path, dpi=300, bbox_inches='tight')
            print(f"Legend saved to: {legend_save_path}")

        plt.show()

    # 12. Optionally save the main figure
    if save_fig:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def exclude_overlap(df1,df2):
    # Exclude pathogenic positions to avoid overlap
    df1 = df1.copy()
    df2 = df2.copy()
    df = pd.merge(
        df1,
        df2[["Protein_ID", "Position"]],
        how="outer",
        on=["Protein_ID", "Position"],
        indicator=True
    )

    # Keep only those rows that came exclusively from `clinvar_pathogenic_predicted_motif`
    # (_merge == 'left_only')
    final_df = df[
        df["_merge"] == "left_only"
        ]
    final_df.drop(columns="_merge", inplace=True)
    return final_df


def ontology_plot_with_elm_types(
    df,
    sort_desc=True,
    custom_colors=None,title="Predicted Pathogenic DO across ELM Types"
):
    """
    Creates a stacked bar chart where each horizontal bar is a 'disease',
    and each segment in the bar corresponds to a different ELM_Type count.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'category_names' and 'ELMType'.
    sort_desc : bool, optional
        Whether to sort diseases by descending total mutation count.
        Default True.
    custom_colors : list, optional
        A list of colors (e.g. ['red','blue','green',...]) for each ELMType (in the order of the pivot columns).
        If None, default colors are used.

    Example usage:
        ontology_plot_with_elm_types(df, sort_desc=True, custom_colors=["#d73027","#f46d43","#fdae61","#fee08b"])
    """
    # 1) Check necessary columns
    if 'category_names' not in df.columns or 'ELMType' not in df.columns:
        raise ValueError("DataFrame must have 'category_names' and 'ELMType' columns")


    # # 1) Split into lists
    # df['ELMType'] = df['ELMType'].str.split(', ')
    #
    # # 3) Explode the ELMType column so each row has exactly one ELMType
    # df_exploded = df.explode('ELMType')

    df_exploded = modify_categories(df)

    exclude_categories = ['Unknown','Inborn genetic diseases']

    df_exploded = df_exploded[~df_exploded['category_names'].isin(exclude_categories)]

    # 4) Group by (disease, ELMType) and count occurrences
    df_count = (
        df_exploded
        .groupby(['category_names', 'Category'])
        .size()
        .unstack(fill_value=0)  # rows = category_names, cols = ELMTypes
    )

    palette = sns.color_palette("colorblind", 6)


    # 5) Sort diseases by total mutation count (sum across ELMType columns)
    row_sums = df_count.sum(axis=1)
    if sort_desc:
        df_count = df_count.loc[row_sums.sort_values(ascending=False).index]
    else:
        df_count = df_count.loc[row_sums.sort_values(ascending=True).index]


    colors = [category_colors_structure.get(cat, '#999999') for cat in df_count.columns]

    # 6) Plot as a horizontal stacked bar chart
    fig, ax = plt.subplots(figsize=(6, 3))

    # If custom_colors is provided, use it for the columns in the order they appear in df_count
    # The number of colors should match the number of ELMType columns
    if custom_colors is not None:
        df_count.plot(kind='bar', stacked=True, ax=ax, color=custom_colors)
    else:
        df_count.plot(kind='bar', stacked=True, ax=ax, color=colors)

    # 7) Labeling and formatting
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    ax.set_ylabel("Mutation Count")
    ax.set_xlabel("")
    fig.suptitle(title)

    # # Place legend to the right of the plot
    # ax.legend(title="Category",
    #           # bbox_to_anchor=(1.05, 1),
    #           loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_all_vs_mostly_disorder(
        df1, df2,
        label1="Pathogenic", label2="Predicted Pathogenic",
        title="Comparison for ELM Overlaps",
        figsize=(3, 3),
        color_o="#888888",
        color_md="#ff9999"
):
    """
    Creates a grouped bar chart to compare 'All' vs. 'Mostly Disorder'
    counts between df1 and df2.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        Each DataFrame represents a set of mutations.
        Must have a 'Category' column to identify "Mostly Disorder".
    label1, label2 : str
        Labels for the two DataFrames (used in x-axis).
    title : str
        Plot title.
    figsize : tuple
        Figure size in inches.
    color_all, color_md : str
        Bar colors for "All" and "Mostly Disorder".
    """
    # 1) Count total rows in each DataFrame


    # 2) Count how many are “Mostly Disorder”
    #    (i.e., Category in [Only Disorder, Disorder Mostly]).
    mostly_disorder_set = {"Only Disorder", "Disorder Mostly"}
    n_md_1 = len(df1[df1["Category"].isin(mostly_disorder_set)])
    n_md_2 = len(df2[df2["Category"].isin(mostly_disorder_set)])

    n_o_1 = len(df1[~df1["Category"].isin(mostly_disorder_set)])
    n_o_2 = len(df2[~df2["Category"].isin(mostly_disorder_set)])

    # 3) Prepare data for plotting
    # We'll make a grouped bar chart with 2 groups (df1 vs. df2),
    # and within each group, 2 bars (All vs. MD).
    x = np.array([0, 1])  # positions for groups
    bar_width = 0.35  # width of each bar

    # For the left group (x=0), we plot "All" at x=0 - bar_width/2, and "MD" at x=0 + bar_width/2
    # For the right group (x=1), similarly offset around x=1
    offsets = np.array([-bar_width / 2, bar_width / 2])

    # (A) All
    plt.figure(figsize=figsize)
    plt.bar(
        x[0] + offsets[0],
        n_o_1,
        width=bar_width,
        color=color_o,
        label="Mostly Order/Equal" # label shown only once
    )
    plt.bar(
        x[1] + offsets[0],
        n_o_2,
        width=bar_width,
        color=color_o,
    )

    # (B) Mostly Disorder
    plt.bar(
        x[0] + offsets[1],
        n_md_1,
        width=bar_width,
        color=color_md,
        label="Mostly Disorder"
    )
    plt.bar(
        x[1] + offsets[1],
        n_md_2,
        width=bar_width,
        color=color_md,
    )

    # 4) Format and annotate
    plt.xticks(
        ticks=x,
        labels=[label1, label2],
    )
    plt.ylabel("Mutation Count")
    plt.title(title)
    # plt.legend(loc="best")

    # Annotate each bar with the integer count
    # We know the bar positions and heights:
    bar_values = [
        (x[0] + offsets[0], n_o_1),
        (x[1] + offsets[0], n_o_2),
        (x[0] + offsets[1], n_md_1),
        (x[1] + offsets[1], n_md_2)
    ]
    for (bx, val) in bar_values:
        plt.text(
            bx, val + 0.5,  # small offset above
            str(val),
            ha='center', va='bottom',
        )

    y_max = max(n_o_1, n_o_2, n_md_1, n_md_2)
    plt.ylim(0, y_max * 1.1)

    plt.tight_layout()
    plt.show()


def plot_all_vs_md_for_all_genic_categories(
        df_pathogenic,
        df_pred_pathogenic,
        cat_col="genic_category",
        mostly_disorder_set={"Only Disorder", "Disorder Mostly"},
        color_o="#888888",
        color_md="#ff9999"
):
    """
    Create individual subplots (one subplot per genic category).
    Each subplot shows two stacked bars:
      - Left bar: Pathogenic (split into Mostly Disorder vs. Other)
      - Right bar: Pred. Pathogenic (split into Mostly Disorder vs. Other)

    Parameters
    ----------
    df_pathogenic : pd.DataFrame
        DataFrame of Pathogenic variants (must have 'Category' and 'genic_category' columns).
    df_pred_pathogenic : pd.DataFrame
        DataFrame of Predicted Pathogenic variants (must have 'Category' and 'genic_category' columns).
    cat_col : str
        Column name with genic categories to iterate over (e.g., 'genic_category').
    mostly_disorder_set : set
        Which values in 'Category' are considered "Mostly Disorder."
    """

    # 1) Identify all unique genic categories across both DataFrames
    categories = sorted(
        set(df_pathogenic[cat_col]).union(df_pred_pathogenic[cat_col])
    )

    # 2) Prepare subplots
    n_cats = len(categories)
    ncols = 3  # you can adjust how many columns of subplots
    nrows = math.ceil(n_cats / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * ncols, 3 * nrows))

    # If there's only 1 row and 1 column, axes is not a list → make it a list
    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        # Flatten the 2D array of axes into 1D for easy iteration
        axes = axes.flatten()

    # 3) For each category, plot a small bar chart in its own subplot
    for i, cat in enumerate(categories):
        ax = axes[i]

        # Subset for this category
        sub_p = df_pathogenic[df_pathogenic[cat_col] == cat]
        sub_pp = df_pred_pathogenic[df_pred_pathogenic[cat_col] == cat]

        # Count how many are "Mostly Disorder" vs. "Other"
        n_md_p = len(sub_p[sub_p["Category"].isin(mostly_disorder_set)])
        n_o_p = len(sub_p) - n_md_p

        n_md_pp = len(sub_pp[sub_pp["Category"].isin(mostly_disorder_set)])
        n_o_pp = len(sub_pp) - n_md_pp

        # We'll have two bars, x=0 (Pathogenic) and x=1 (Pred. Pathogenic)
        x_vals = [0, 1]
        bar_width = 0.8

        # Plot the "Mostly Disorder" portion on the bottom
        ax.bar(
            x=0,
            height=n_md_p,
            width=bar_width,
            color=color_md,
            label="Mostly Disorder" if i == 0 else ""  # label only on first subplot
        )
        ax.bar(
            x=1,
            height=n_md_pp,
            width=bar_width,
            color=color_md,
            label=""
        )

        # Now plot the "Other" portion on top of the "Mostly Disorder"
        ax.bar(
            x=0,
            height=n_o_p,
            width=bar_width,
            bottom=n_md_p,
            color=color_o,
            label="Mostly Order/Equal" if i == 0 else ""
        )
        ax.bar(
            x=1,
            height=n_o_pp,
            width=bar_width,
            bottom=n_md_pp,
            color=color_o,
            label=""
        )

        # Set X-axis ticks and labels
        ax.set_xticks(x_vals)
        ax.set_xticklabels(["Pathogenic", "Pred. Patho."], rotation=0)

        # Give each subplot a title with the category name
        ax.set_title(str(cat), fontsize=10)

        # Optionally show the total counts above each bar
        for (xpos, md, oth) in [(0, n_md_p, n_o_p), (1, n_md_pp, n_o_pp)]:
            total = md + oth
            if total > 0:
                ax.text(
                    xpos, md + oth + 0.2,
                    str(total),
                    ha='center', va='bottom', fontsize=8
                )

    # 4) Hide any unused subplots (if #categories < nrows*ncols)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # 5) Optional overall figure labeling
    fig.suptitle("Pathogenic vs. Predicted Pathogenic by Genic Category", fontsize=14)

    # 6) Create a single legend for the figure (instead of separate legends per subplot)
    # handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(
    #     handles, labels,
    #     loc='lower center',
    #     bbox_to_anchor=(0.5, -0.02),
    #     ncol=2
    # )

    plt.tight_layout()
    # Make space at bottom for the legend
    plt.subplots_adjust(bottom=0.15)
    plt.show()

def modify_categories(clinvar_p):
    modified_clinvar_p = clinvar_p.copy()

    disorder_categories = ["Only Disorder", 'Disorder Mostly']
    modified_clinvar_p['Category'] = modified_clinvar_p['Category'].apply(
        lambda x: "Disorder Specific" if x in disorder_categories else "Non-Disorder Specific")

    modified_clinvar_p['category_names'] = np.where(modified_clinvar_p['category_names'] == "Cardiovascular/Hematopoietic", 'Cardiovascular', modified_clinvar_p['category_names'])
    modified_clinvar_p = modified_clinvar_p[modified_clinvar_p['category_names'] != 'Inborn Genetic Diseases']
    return modified_clinvar_p



if __name__ == "__main__":

    """
    1. Motif
    - How many ClinVar Pathogenic mutations occur in predicted motif region?
    - How many ClinVar Pathogenic mutations occur in known motif region?
    - How many ClinVar Uncertain mutations occur in known motif region?
    - How many ClinVar Uncertain mutations occur in predicted motif region?
    """

    COLORS = {
        "disorder": '#ffadad',
        "order": '#a0c4ff',
        "both": '#ffc6ff',
        "Pathogenic": '#ff686b',
        "Benign": "#b2f7ef",
        "Uncertain": "#f8edeb"
    }

    category_colors_structure = {
        'Only Disorder': 'red',
        'Disorder Mostly': COLORS['disorder'],
        'Equal': 'green',
        'Order Mostly': COLORS['order'],
        'Only Order': 'blue',

        'Mostly Disorder': '#f27059',
        'Disorder Specific': '#f27059',
        "Non-Disorder Specific": "#8e9aaf",
        "Mostly Order/Equal": "#8e9aaf",

        'disorder': COLORS['disorder'],
        'order': COLORS['order'],
        "Disorder-Pathogenic": COLORS['disorder'],
        'Order-Pathogenic': COLORS['order'],

    }


    base_path = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/'
    prediction_path = os.path.join(base_path,'processed_data')
    motif_path = os.path.join(prediction_path,'files','elm','clinvar','motif')
    plot_path = os.path.join(base_path,'plots')

    base_table = pd.read_csv(os.path.join(base_path,'data','discanvis_base_files','sequences','loc_chrom_with_names_isoforms_with_seq.tsv'),sep='\t')

    clinvar_pathogenic_known_motif = pd.read_csv(os.path.join(motif_path,"Pathogenic", 'clinvar_elm_overlaps_known.tsv'),sep='\t')
    clinvar_pathogenic_predicted_motif = pd.read_csv(os.path.join(motif_path,"Pathogenic", 'clinvar_elm_overlaps_predicted.tsv'),sep='\t')

    clinvar_uncertain_known_motif = pd.read_csv(os.path.join(motif_path, "Uncertain", 'clinvar_elm_overlaps_known.tsv'),sep='\t')
    clinvar_uncertain_predicted_motif = pd.read_csv(os.path.join(motif_path, "Uncertain", 'clinvar_elm_overlaps_predicted.tsv'),sep='\t')

    # exit()

    # clinvar_predicted_pathogenic_known_motif = pd.read_csv(os.path.join(motif_path, "Predicted_Pathogenic", 'clinvar_elm_overlaps_known.tsv'), sep='\t')
    # clinvar_predicted_pathogenic_predicted_motif = pd.read_csv(os.path.join(motif_path, "Predicted_Pathogenic", 'clinvar_elm_overlaps_predicted.tsv'), sep='\t')


    # For Predicted Pathogenic
    # FIG 6C - Top Mutated Genes
    # clinvar_uncertain_predicted_motif = clinvar_uncertain_predicted_motif[clinvar_uncertain_predicted_motif['category_names'] != 'Inborn Genetic Diseases']
    # clinvar_predicted_pathogenic_predicted_motif = clinvar_predicted_pathogenic_predicted_motif[clinvar_predicted_pathogenic_predicted_motif['category_names'] != 'Inborn Genetic Diseases']
    # clinvar_pathogenic_known_motif = clinvar_pathogenic_known_motif[clinvar_pathogenic_known_motif['category_names'] != 'Inborn Genetic Diseases']
    # clinvar_pathogenic_predicted_motif = clinvar_pathogenic_predicted_motif[clinvar_pathogenic_predicted_motif['category_names'] != 'Inborn Genetic Diseases']

    # Exclude pathogenic positions to avoid overlap
    # excluded_known_pathogen_positions = exclude_overlap(clinvar_uncertain_known_motif,clinvar_pathogenic_known_motif)
    # excluded_known_motifs_positions = exclude_overlap(clinvar_pathogenic_predicted_motif,excluded_known_pathogen_positions)

    # print(clinvar_uncertain_known_motif['Protein_ID'].value_counts())
    # print(excluded_known_pathogen_positions['Protein_ID'].value_counts())
    # exit()

    # Define a custom color palette (optional)
    custom_palette = sns.color_palette("colorblind", clinvar_uncertain_predicted_motif['category_names'].nunique())  # Adjust the number based on your categories

    category_color_map = {x:custom_palette[i] for i,x in enumerate(clinvar_uncertain_predicted_motif['category_names'].unique())}

    # motif_predicted_dis_mostly_disorder = excluded_known_pathogen_positions[excluded_known_pathogen_positions['Category'].isin(["Only Disorder", 'Disorder Mostly'])]

    # exclude_cats = ['Inborn genetic diseases','not specified']
    # motif_predicted_dis_mostly_disorder = motif_predicted_dis_mostly_disorder[~motif_predicted_dis_mostly_disorder['nDisease'].isin(exclude_cats)]

    # print(motif_predicted_dis_mostly_disorder['nDisease'].value_counts())


    # exit()


    # # Plot Top Genes without Legend and with Separate Legend
    # plot_top_genes(
    #     df_pred_pathogenic=clinvar_uncertain_predicted_motif,
    #     top_n=20,
    #     positional=True,
    #     mut_type="Uncertain - PEM",
    #     rotation=45,
    #     save_fig=True,
    #     save_path=os.path.join(plot_path, 'fig6', 'C2.png'),
    #     color_dict=category_color_map,
    #     plot_legend_separately=True,  # Set to True to generate a separate legend plot
    #     simple_plot = True,
    #     bar_color = COLORS['disorder'],
    #     figsize=(6, 2)
    # )
    # # exit()
    #
    # # Another example with different data
    # plot_top_genes(
    #     df_pred_pathogenic=clinvar_pathogenic_predicted_motif,
    #     top_n=20,
    #     positional=True,
    #     mut_type="Pathogenic - PEM",
    #     rotation=45,
    #     save_fig=True,
    #     save_path=os.path.join(plot_path, 'fig6', 'C.png'),
    #     color_dict=category_color_map,
    #     plot_legend_separately=True,
    #     simple_plot=True,
    #     bar_color=COLORS['disorder'],
    #     figsize=(6, 2)
    # )
    #
    # # Plot for Pathogenic - Known ELM Motif
    # plot_top_genes(
    #     df_pred_pathogenic=clinvar_pathogenic_known_motif,
    #     top_n=20,
    #     positional=True,
    #     mut_type="Pathogenic - Known ELM Motif",
    #     rotation=45,
    #     save_fig=True,
    #     save_path=os.path.join(plot_path, 'sm', 'motif', 'top_clinvar_mut_known.png'),
    #     color_dict=category_color_map,
    #     plot_legend_separately=True,
    #     simple_plot=True,
    #     bar_color=COLORS['disorder'],
    #     figsize=(8, 2)
    # )
    # exit()

    # FIG 6D - Disease Ontology Plot

    # clinvar_predicted_pathogenic_known_motif = pd.read_csv(
    #     os.path.join(motif_path, 'clinvar_predicted_pathogenic_with_known_elm_overlaps.tsv'), sep='\t')

    ontology_plot_with_elm_types(clinvar_pathogenic_predicted_motif,title="Disease Ontology for HotspotPEMs")
    # exit()
    # print(clinvar_pathogenic_predicted_motif['Category'].unique())
    # exit()

    # ontology_plot_with_elm_types(clinvar_predicted_pathogenic_known_motif,title="Predicted Pathogenic DO (Known ELMs)")

    # Base Plots
    # 1 Predicted Pathogenic and Pathogenic for Predicted ELM -> All and Mostly Disorder
    # 2 Predicted for Known ELM -> All and Mostly Disorder
    # 3 Same plot but for Genic Categories

    # -- PLOT #1: Predicted ELM Overlaps (All vs. MD) --
    plot_all_vs_mostly_disorder(
        df1=clinvar_pathogenic_predicted_motif,
        df2=clinvar_uncertain_predicted_motif,
        label1="Pathogenic",
        label2="Uncertain",
        title="HotspotPEMs Mutations",
        color_md=category_colors_structure['Mostly Disorder'],
        color_o=category_colors_structure['Mostly Order/Equal'],
    )

    # -- PLOT #2: Known ELM Overlaps (All vs. MD) --
    plot_all_vs_mostly_disorder(
        df1=clinvar_pathogenic_known_motif,
        df2=clinvar_uncertain_known_motif,
        label1="Pathogenic",
        label2="Uncertain",
        title="Known ELM Mutations",
        color_md=category_colors_structure['Mostly Disorder'],
        color_o=category_colors_structure['Mostly Order/Equal'],
    )
    exit()


    # -- PLOT #3: Same style but for Genic Categories (example approach) --
    plot_all_vs_md_for_all_genic_categories(
        df_pathogenic=clinvar_pathogenic_known_motif,
        df_pred_pathogenic=clinvar_predicted_pathogenic_known_motif,
        cat_col="genic_category",
         color_md = category_colors_structure['Mostly Disorder'],
        color_o=category_colors_structure['Mostly Order/Equal'],
    )


