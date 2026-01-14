import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
import numpy as np
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.pyplot import title


def get_top_20_categories(df, column):
    """Get the top 20 most frequent categories for a given column."""
    return df[column].value_counts().nlargest(20)


def get_value_counts_developmental(df, column, developmental=False, filter_developmental=False):
    """Get the top 20 most frequent categories for a given column."""
    if developmental:
        df = df[df['Developmental']]
    elif filter_developmental:
        df = df[df['Developmental'] == False]
    return df[column].value_counts()


def get_value_counts(df, column, sum_values=False):
    """Get the top 20 most frequent categories for a given column."""
    if sum_values:
        return df.groupby(column)['mutation_count'].sum().sort_values(ascending=False)
    else:
        return df[column].value_counts()


def get_value_counts_genes(df, column):
    """Get the top 20 most frequent categories for a given column."""
    return df.groupby(column)['Protein_ID'].nunique()


def get_predicted_pathogenic_tables(clinvar_order, clinvar_disorder, am_disorder, am_order, disorder_cutoff,
                                    order_cutoff):
    order_uncertain = clinvar_order[clinvar_order['Interpretation'] == 'Uncertain']
    disorder_uncertain = clinvar_disorder[clinvar_disorder['Interpretation'] == 'Uncertain']
    merged_disorder = disorder_uncertain.merge(am_disorder, on=['Protein_ID', 'Position'])
    merged_order = order_uncertain.merge(am_order, on=['Protein_ID', 'Position'])

    likely_pathogenic_disorder = merged_disorder[merged_disorder['AlphaMissense'] >= disorder_cutoff]
    likely_pathogenic_order = merged_order[merged_order['AlphaMissense'] >= order_cutoff]

    return likely_pathogenic_disorder, likely_pathogenic_order


def prepare_data_for_heatmap(clinvar_order_p, clinvar_order_u, clinvar_order_b, clinvar_disorder_p, clinvar_disorder_u,
                             clinvar_disorder_b, column, sum_values=False):
    """Prepare a DataFrame for heatmap plotting."""

    # Get the top 20 categories for each subset
    top_disorder_benign = get_value_counts(clinvar_disorder_b, column, sum_values)
    top_disorder_uncertain = get_value_counts(clinvar_disorder_u, column, sum_values)
    top_disorder_pathogenic = get_value_counts(clinvar_disorder_p, column, sum_values)
    top_order_benign = get_value_counts(clinvar_order_b, column, sum_values)
    top_order_uncertain = get_value_counts(clinvar_order_u, column, sum_values)
    top_order_pathogenic = get_value_counts(clinvar_order_p, column, sum_values)

    # Combine all unique categories from each subset
    unique_categories = list(
        set(top_disorder_pathogenic.index) | set(top_disorder_uncertain.index) | set(top_disorder_benign.index) |
        set(top_order_pathogenic.index) | set(top_order_uncertain.index) | set(top_order_benign.index)
    )

    # Create an empty DataFrame to hold the combined counts
    heatmap_data = pd.DataFrame(index=unique_categories,
                                columns=['Disorder-Pathogenic', 'Disorder-Uncertain', 'Disordered-Benign',
                                         'Order-Pathogenic', 'Order-Uncertain', 'Ordered-Benign'
                                         ]).fillna(0)

    # Fill the DataFrame with counts
    heatmap_data['Disorder-Pathogenic'] = top_disorder_pathogenic
    heatmap_data['Disorder-Uncertain'] = top_disorder_uncertain
    heatmap_data['Disordered-Benign'] = top_disorder_benign
    heatmap_data['Order-Pathogenic'] = top_order_pathogenic
    heatmap_data['Order-Uncertain'] = top_order_uncertain
    heatmap_data['Ordered-Benign'] = top_order_benign

    # Convert missing values to zero and ensure values are integers
    heatmap_data = heatmap_data.fillna(0).astype(int)

    return heatmap_data


def prepare_data_for_heatmap_mut_cat(clinvar_data, column, column_to_check, sum_values=False):
    """
    Prepare a DataFrame for heatmap plotting based on genic categories and mutation types.

    Args:
        clinvar_data (pd.DataFrame): ClinVar dataset containing relevant data.
        column (str): The column to group and count values on (e.g., 'Protein_ID').

    Returns:
        pd.DataFrame: A DataFrame containing counts of categories for heatmap plotting.
    """

    # Get unique genic categories
    genic_categories = clinvar_data[column_to_check].unique()

    # Dictionary to hold dataframes for each genic category
    dfs = {}

    # Process data for each genic category
    for cat in genic_categories:
        current_df = clinvar_data[clinvar_data[column_to_check] == cat]

        # Get the top 20 values based on the column
        top_current = get_value_counts(current_df, column, sum_values)  # Assuming you have a function get_value_counts

        # Add results to dictionary
        dfs[cat] = top_current

    # Combine all unique categories from all subsets
    unique_categories = set()
    for top_current in dfs.values():
        unique_categories.update(top_current.index)

    unique_categories = list(unique_categories)

    # Create an empty DataFrame to hold the combined counts for each genic category
    heatmap_data = pd.DataFrame(index=unique_categories, columns=genic_categories).fillna(0)

    # Fill the DataFrame with counts for each genic category
    for cat, top_current in dfs.items():
        heatmap_data[cat] = top_current.reindex(unique_categories, fill_value=0)

    # Convert missing values to zero and ensure values are integers
    heatmap_data = heatmap_data.fillna(0).astype(int)

    return heatmap_data


def prepare_data_for_heatmap_percent_age(clinvar_disorder, clinvar_order, likely_pathogenic_disorder,
                                         likely_pathogenic_order, column):
    """Prepare a DataFrame for heatmap plotting."""
    # Separate by Interpretation for both ordered and disordered
    disorder_pathogenic = clinvar_disorder[clinvar_disorder['Interpretation'] == 'Pathogenic']
    disorder_uncertain = clinvar_disorder[clinvar_disorder['Interpretation'] == 'Uncertain']
    order_pathogenic = clinvar_order[clinvar_order['Interpretation'] == 'Pathogenic']
    order_uncertain = clinvar_order[clinvar_order['Interpretation'] == 'Uncertain']

    # Get the top 20 categories for each subset
    top_disorder_pathogenic_am = get_value_counts(likely_pathogenic_disorder, column)
    top_order_pathogenic_am = get_value_counts(likely_pathogenic_order, column)
    top_disorder_pathogenic = get_value_counts(disorder_pathogenic, column)
    top_disorder_uncertain = get_value_counts(disorder_uncertain, column)
    top_order_pathogenic = get_value_counts(order_pathogenic, column)
    top_order_uncertain = get_value_counts(order_uncertain, column)

    # Combine all unique categories from each subset
    unique_categories = list(
        set(top_disorder_pathogenic.index) | set(top_disorder_uncertain.index) | set(top_disorder_pathogenic_am.index) |
        set(top_order_pathogenic.index) | set(top_order_uncertain.index) | set(top_order_pathogenic_am.index)
    )

    # Create an empty DataFrame to hold the combined counts
    heatmap_data = pd.DataFrame(index=unique_categories,
                                # columns=['Disorder-Pathogenic', 'Disorder-Uncertain','Disordered-Predicted-Pathogenic',
                                #          'Order-Pathogenic', 'Order-Uncertain','Ordered-Predicted-Pathogenic'
                                #          ]
                                ).fillna(0)
    # Calculate Percentage
    heatmap_data['Pathogenic Disorder%'] = top_disorder_pathogenic / (
            top_order_pathogenic + top_disorder_pathogenic) * 100
    heatmap_data['Uncertain Disorder%'] = top_disorder_uncertain / (top_order_uncertain + top_disorder_uncertain) * 100
    heatmap_data['Pathogenic Predicted Disorder%'] = top_disorder_pathogenic_am / (
            top_order_pathogenic_am + top_disorder_pathogenic_am) * 100
    heatmap_data['Uncertain / Pathogenic Predicted %'] = top_disorder_pathogenic_am / top_disorder_uncertain * 100

    # # Fill the DataFrame with counts
    # heatmap_data['Disorder-Pathogenic'] = top_disorder_pathogenic
    # heatmap_data['Disorder-Uncertain'] = top_disorder_uncertain
    # heatmap_data['Disordered-Predicted-Pathogenic'] = top_disorder_pathogenic_am
    # heatmap_data['Order-Pathogenic'] = top_order_pathogenic
    # heatmap_data['Order-Uncertain'] = top_order_uncertain
    # heatmap_data['Ordered-Predicted-Pathogenic'] = top_order_pathogenic_am

    # Convert missing values to zero and ensure values are integers
    heatmap_data = heatmap_data.fillna(0).astype(int)

    # Add a column for the sum of all values across rows
    heatmap_data['Row_Sum'] = heatmap_data.sum(axis=1)

    # Sort the DataFrame by the sum of rows, descending
    heatmap_data = heatmap_data.sort_values('Row_Sum', ascending=False)

    # Drop the 'Row_Sum' column after sorting
    heatmap_data = heatmap_data.drop(columns=['Row_Sum'])

    return heatmap_data


def prepare_data_for_heatmap_genes(clinvar_disorder, clinvar_order, likely_pathogenic_disorder, likely_pathogenic_order,
                                   column):
    """Prepare a DataFrame for heatmap plotting."""
    # Separate by Interpretation for both ordered and disordered
    disorder_pathogenic = clinvar_disorder[clinvar_disorder['Interpretation'] == 'Pathogenic']
    disorder_uncertain = clinvar_disorder[clinvar_disorder['Interpretation'] == 'Uncertain']
    order_pathogenic = clinvar_order[clinvar_order['Interpretation'] == 'Pathogenic']
    order_uncertain = clinvar_order[clinvar_order['Interpretation'] == 'Uncertain']

    # Get the top 20 categories for each subset
    top_disorder_pathogenic_am = get_value_counts_genes(likely_pathogenic_disorder, column)
    top_order_pathogenic_am = get_value_counts_genes(likely_pathogenic_order, column)
    top_disorder_pathogenic = get_value_counts_genes(disorder_pathogenic, column)
    top_disorder_uncertain = get_value_counts_genes(disorder_uncertain, column)
    top_order_pathogenic = get_value_counts_genes(order_pathogenic, column)
    top_order_uncertain = get_value_counts_genes(order_uncertain, column)

    # Combine all unique categories from each subset
    unique_categories = list(
        set(top_disorder_pathogenic.index) | set(top_disorder_uncertain.index) | set(top_disorder_pathogenic_am.index) |
        set(top_order_pathogenic.index) | set(top_order_uncertain.index) | set(top_order_pathogenic_am.index)
    )

    # Create an empty DataFrame to hold the combined counts
    heatmap_data = pd.DataFrame(index=unique_categories,
                                columns=['Disorder-Pathogenic', 'Disorder-Uncertain', 'Disordered-Predicted-Pathogenic',
                                         'Order-Pathogenic', 'Order-Uncertain', 'Ordered-Predicted-Pathogenic'
                                         ]).fillna(0)

    # Fill the DataFrame with counts
    heatmap_data['Disorder-Pathogenic'] = top_disorder_pathogenic
    heatmap_data['Disorder-Uncertain'] = top_disorder_uncertain
    heatmap_data['Disordered-Predicted-Pathogenic'] = top_disorder_pathogenic_am
    heatmap_data['Order-Pathogenic'] = top_order_pathogenic
    heatmap_data['Order-Uncertain'] = top_order_uncertain
    heatmap_data['Ordered-Predicted-Pathogenic'] = top_order_pathogenic_am

    # Convert missing values to zero and ensure values are integers
    heatmap_data = heatmap_data.fillna(0).astype(int)

    return heatmap_data


def prepare_data_for_heatmap_disorder(clinvar_disorder, likely_pathogenic_disorder, column, column_filter=None):
    """Prepare a DataFrame for heatmap plotting."""

    if column_filter:
        for filter_column, value in column_filter.items():
            clinvar_disorder = clinvar_disorder[clinvar_disorder[filter_column] == value]
            likely_pathogenic_disorder = likely_pathogenic_disorder[likely_pathogenic_disorder[filter_column] == value]

    # Separate by Interpretation for both ordered and disordered
    disorder_pathogenic = clinvar_disorder[clinvar_disorder['Interpretation'] == 'Pathogenic']
    disorder_uncertain = clinvar_disorder[clinvar_disorder['Interpretation'] == 'Uncertain']

    # Get the top 20 categories for each subset
    top_disorder_pathogenic_am = get_top_20_categories(likely_pathogenic_disorder, column)
    top_disorder_pathogenic = get_top_20_categories(disorder_pathogenic, column)
    top_disorder_uncertain = get_top_20_categories(disorder_uncertain, column)

    # Combine all unique categories from each subset
    unique_categories = list(
        set(top_disorder_pathogenic.index) | set(top_disorder_uncertain.index) | set(top_disorder_pathogenic_am.index))

    # Create an empty DataFrame to hold the combined counts
    heatmap_data = pd.DataFrame(index=unique_categories,
                                columns=['Disorder-Pathogenic', 'Disorder-Uncertain', 'Disordered-Predicted-Pathogenic',
                                         ]).fillna(0)

    # Fill the DataFrame with counts
    heatmap_data['Disorder-Pathogenic'] = top_disorder_pathogenic
    heatmap_data['Disorder-Uncertain'] = top_disorder_uncertain
    heatmap_data['Disordered-Predicted-Pathogenic'] = top_disorder_pathogenic_am

    # Convert missing values to zero and ensure values are integers
    heatmap_data = heatmap_data.fillna(0).astype(int)

    return heatmap_data


def plot_disease_ontology_heatmap(df, additional_text='', title=None, ylabel='Disease Category',
                                  xlabel='Interpretation and Structure'):
    """Plot a heatmap from a DataFrame with disease ontology counts, normalized row-wise."""
    # Normalize each row by its sum to calculate percentages (or relative values)
    df_normalized = df.div(df.sum(axis=0), axis=1)
    print(df_normalized)

    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(df_normalized, annot=df, fmt="d", cmap="YlGnBu", linewidths=.5)
    if title:
        plt.title(title)
    else:
        plt.title(f'Top 20 Disease Ontology Categories for Structure and Pathogenicity Prediction {additional_text}')
    plt.ylabel(ylabel)
    ax.set_xticklabels(df_normalized.columns, rotation=45)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.show()


def create_stacked_bar_plot(df, title, xlabel='Disorder Category', category_color_mapping={}):
    """Create a stacked bar plot with disorder categories as bars and disease categories as stacks."""
    # Transpose to have disorder categories as columns and disease categories as rows
    df_transposed = df.T

    # Normalize each disorder category to percentages
    df_transposed_percentage = df_transposed.div(df_transposed.sum(axis=1), axis=0) * 100

    colors = [category_color_mapping.get(cat, '#999999') for cat in df.index]

    print("index", df.index)

    # Plot as a stacked bar chart
    ax = df_transposed_percentage.plot(kind='bar', stacked=True, figsize=(12, 8), color=colors)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Percentage (%)')
    ax.set_xticklabels(df.columns, rotation=45)
    plt.legend(title='Disease Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def create_bar_plot(df, title, xlabel='Disease Ontology', category_color_mapping={}, filter=None, percentage=False,
                    order_by=None, figsize=(6, 4), islegend=True,filepath=None):
    """Create a bar plot sorted by max mutations and show percentages for Disorder-Pathogenic."""

    if filter:
        df = df[filter]

        if len(filter) > 1:
            df['SUM'] = df['Order-Pathogenic'] + df['Disorder-Pathogenic']
            df['Percentage'] = (df['Disorder-Pathogenic'] / df['SUM']) * 100
        else:
            df['SUM'] = df[filter]
    else:
        df['SUM'] = df.sum(axis=1)

    max_val = df['SUM'].max()

    # Sort by SUM
    if isinstance(order_by, pd.Index):
        df = df.reindex(order_by)
    else:
        df = df.sort_values(by='SUM', ascending=False)

    # Plot the actual values
    if filter:
        colors = [category_color_mapping.get(cat, '#999999') for cat in df.columns]
        ax = df[filter].plot(kind='bar', stacked=True, figsize=figsize, color=colors)
    else:
        df = df.drop(columns=['SUM'])
        # Ensure columns follow the order of category_color_mapping keys
        if percentage:
            df = df.T
            df = df.div(df.sum(axis=1), axis=0) * 100

        columns_in_order = [col for col in category_color_mapping.keys() if col in df.columns]
        df = df[columns_in_order]
        colors = [category_color_mapping.get(cat, '#999999') for cat in df.columns]
        ax = df.plot(kind='bar', stacked=True, figsize=figsize, color=colors)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Disease-Position Count')

    plt.ylim(0, max_val * 1.1)

    # Set disease names as x-axis labels
    ax.set_xticklabels(df.index, rotation=30, ha='right')

    # Add percentage labels on top of the Disorder-Pathogenic bars
    if filter and len(filter) > 1:
        disorder_values = df['SUM'].values
        for i, value in enumerate(disorder_values):
            ax.text(i, value + 100, f'{df["Percentage"].iloc[i]:.0f}%', ha='center', color='black')

    if percentage:
        disorder_values = df['disorder'].values
        for i, value in enumerate(disorder_values):
            ax.text(i, 101, f'{value:.1f}%', ha='center', color='black')

    # Adjust legend and layout
    if islegend:
        plt.legend(title='Mutation Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # If we do not want a legend, remove it explicitly
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
    plt.show()

    return df.index


def create_pie_chart_structural_distribution(df, category_color_mapping, title='Structural Distribution'):
    # Aggregate structural categories

    print(df)
    exit()

    # Select relevant structural categories and their corresponding counts
    labels = ['Only Order', 'Order Mostly', 'Equal', 'Disorder Mostly', 'Only Disorder']
    sizes = [structural_data.get(label, 0) for label in labels]
    colors = [category_color_mapping.get(label, '#999999') for label in labels]

    # Explode to emphasize certain sections (like Disorder categories)
    explode = [0.1 if 'Disorder' in label else 0 for label in labels]

    # Create the pie chart
    fig, ax = plt.subplots(figsize=(10, 6))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,
                                      explode=explode)

    # Format the chart
    plt.title(title)
    plt.tight_layout()
    plt.show()


def create_pie_charts(df, title, category_color_mapping={}, filter=None, add_all=True, choosen_categories=[],
                      figsize=(6, 4),filepath=None,add_legend=False):
    """Create pie charts for each category in the DataFrame with a single legend and percentages on the outer edge.
       Explode the 'Disorder' categories outward and add an additional 'All' pie chart."""

    # Filter the DataFrame if needed
    if filter:
        df = df[filter]

    # Add an 'All' column, which is the sum of all categories
    if add_all:
        df.loc['All'] = df.sum(axis=0)

    # Ensure columns follow the order of category_color_mapping keys
    columns_in_order = [col for col in category_color_mapping.keys() if col in df.columns]
    df = df[columns_in_order]

    # Define subplot grid dimensions (2x2 layout for 4 pie charts)
    if add_all:
        if choosen_categories:
            nrows = 1
            ncols = 2
            df = df.loc[choosen_categories]
        else:
            nrows = 2
            ncols = 2
    else:
        nrows = 1
        ncols = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, subplot_kw=dict(aspect="equal"))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Function to calculate positions for percentage labels
    def place_labels_on_edge(wedges, ax, explode):
        for i, wedge in enumerate(wedges):
            angle = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
            x = np.cos(np.radians(angle))
            y = np.sin(np.radians(angle))

            # Adjust position based on explode value for the wedge
            explode_value = explode[i] + 0.2  # Increase this value for further placement
            ax.text(x * (0.5 +explode_value), y * (0.5 +explode_value), f'{wedge.label}',
                    va='center', fontsize=10, color='black', ha='center')

    # Create pie charts for each row in the DataFrame, including the "All" category
    for i, (index, row) in enumerate(df.iterrows()):
        values = row.values
        labels = row.index
        colors = [category_color_mapping.get(cat, '#999999') for cat in labels]

        # Set the explode parameters to lift out the 'Disorder' slices
        disorder_count = 0
        explode = []
        for label in labels:
            if 'Disorder' in label:
                explode.append(0.01 + disorder_count * 0.01)  # Incrementally increase explode for 'Disorder'
                disorder_count += 1
            else:
                explode.append(0)  # No explode for non-'Disorder' labels

        # Create the pie chart with exploded slices
        wedges, texts, autotexts = axes[i].pie(values, autopct='%1.1f%%', startangle=90, colors=colors, explode=explode)

        # Set title for each subplot (the category)
        axes[i].set_title(index)

        # Remove default labels and call function to place percentage labels on the edge
        for autotext in autotexts:
            autotext.set_text('')

        # Calculate and place percentage labels on the edge
        total = sum(values)
        for wedge, value in zip(wedges, values):
            percentage = (value / total) * 100
            wedge.label = f'{percentage:.1f}%'

        place_labels_on_edge(wedges, axes[i], explode)

    # Create a single legend with labels for the mutation categories
    # plt.legend(labels=columns_in_order, title="Mutation Categories",
    #            bbox_to_anchor=(1.5, 1.5), loc='upper right'
    #            )

    # handles = [plt.Line2D([0], [0], color=color, lw=4) for color in category_colors_gene.values()]
    # legend_labels_with_counts = [f"{key} ({legend_counts[key]})" for key in category_colors_gene.keys()]
    # plt.legend(handles, labels=columns_in_order, title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')

    if add_legend:
        handles = [plt.Line2D([0], [0], color=category_color_mapping[col], lw=6) for col in columns_in_order]
        fig.legend(handles, columns_in_order, loc='lower center', bbox_to_anchor=(0.5, -0.05),
                   ncol=len(columns_in_order), title='Category', frameon=False)

    # Add a main title
    plt.suptitle(title)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath,bbox_inches='tight')
    plt.show()

    # fig_legend, ax_legend = plt.subplots(figsize=(2, 2))
    # # fig_legend.patch.set_alpha(0.0)  # Make the background transparent if desired
    #
    # # Create handles for legend
    # handles = [plt.Line2D([0], [0], color=category_color_mapping[col], lw=4) for col in columns_in_order]
    # legend = ax_legend.legend(handles, columns_in_order, title='Category', loc='center')
    #
    # # Hide axes and just show legend
    # ax_legend.axis('off')
    # plt.show()


def create_piechart_all(disorder_df, order_df):
    disorder_df.loc['Disorder'] = disorder_df.sum(axis=0)
    order_df.loc['Order'] = order_df.sum(axis=0)

    big_df = pd.concat([disorder_df.loc[['Disorder']], order_df.loc[['Order']]]).fillna(0)
    create_pie_charts(big_df,
                      title=f'Proportion of mutated disordered positions  based on Disease Ontology (Genetic Category)',
                      category_color_mapping=category_colors_structure, add_all=False)


def get_subcategory_for_diseases():
    """Map specific disease subcategories to levels."""
    category_dictionary = {
        "cancer": ["Cancer", 'level3'],
        "cardiovascular system disease": ["Cardiovascular", 'level3'],
        "endocrine system disease": ["Endocrine", 'level3'],
        "gastrointestinal system disease": ["Gastrointestinal", 'level3'],
        "immune system disease": ["Immune", 'level3'],
        "musculoskeletal system disease": ["Musculoskeletal", 'level3'],
        "neurodegenerative disease": ["Neurodegenerative", 'level5'],
        "reproductive system disease": ["Reproductive", 'level3'],
        "respiratory system disease": ["Respiratory", 'level3'],
        "urinary system disease": ["Urinary", 'level3'],
    }

    # Create lists for columns
    subcategory_list = []
    category_names_list = []
    level_list = []

    for key, value in category_dictionary.items():
        subcategory_list.append(key)
        category_names_list.append(value[0])
        level_list.append(value[1])

    # Build the DataFrame using lists
    df = pd.DataFrame({'subcategory': subcategory_list, 'category_names': category_names_list, 'level': level_list})

    pivoted_df = df.pivot_table(
        index='category_names',
        columns='level',
        values='subcategory',
        aggfunc='first'
    ).reset_index()

    return pivoted_df


def merge_with_categories(clinvar_df, likely_df, category_df):
    """Merge clinvar and likely pathogenic data with both level3 and level5 categories."""
    # Loop over each column in the category DataFrame except 'category_names'
    for column in category_df.columns[1:]:
        if column in clinvar_df.columns:
            clinvar_df = pd.merge(clinvar_df, category_df[['category_names', column]], how='left', left_on=column,
                                  right_on=column)
            likely_df = pd.merge(likely_df, category_df[['category_names', column]], how='left', left_on=column,
                                 right_on=column)

    clinvar_df['category_names'] = np.where(clinvar_df['category_names_x'].notna(), clinvar_df['category_names_x'],
                                            clinvar_df['category_names_y'])
    likely_df['category_names'] = np.where(likely_df['category_names_x'].notna(), likely_df['category_names_x'],
                                           likely_df['category_names_y'])

    return clinvar_df, likely_df


def merge_with_categories_new(clinvar_df, likely_df, category_df):
    """Merge clinvar and likely pathogenic data with both level3 and level5 categories."""
    # Loop over each column in the category DataFrame except 'category_names'

    clinvar_df = pd.merge(clinvar_df, category_df, how='left', left_on="nDisease", right_on="Diseases").dropna(
        subset=['category'])
    likely_df = pd.merge(likely_df, category_df, how='left', left_on="nDisease", right_on="Diseases").dropna(
        subset=['category'])

    return clinvar_df, likely_df


def plot_piechart_by_category(df, outer_col='category', inner_col='Diseases', count_col='disease count'):
    # Aggregate the outer sizes by counting the number of instances for each outer category
    outer_group = df[outer_col].value_counts().reset_index(name='counts').sort_values(by=outer_col, ascending=False)
    outer_sizes = outer_group['counts'].tolist()
    outer_labels = outer_group[outer_col].tolist()

    # Group the data by outer and inner columns and sort by the outer column
    inner_group = df.groupby([outer_col, inner_col, count_col]).size().reset_index(name='counts').sort_values(
        by=outer_col, ascending=False)

    # Prepare inner labels with the count information
    inner_labels = [f"{row[count_col]} {row[inner_col]}" for _, row in inner_group.iterrows()]
    inner_sizes = inner_group['counts'].tolist()

    # Create a color map for the outer labels
    cmap = cm.get_cmap('tab20', len(outer_labels))  # Change 'tab20' to any desired colormap
    outer_colors = cmap.colors

    # Create inner colors based on outer groups
    inner_colors = []
    for outer_label in outer_labels:
        base_color = outer_colors[outer_labels.index(outer_label)]
        sub_group = inner_group[inner_group[outer_col] == outer_label]
        num_shades = len(sub_group)
        shades = [mcolors.to_rgba(base_color, alpha=0.5 + 0.5 * i / num_shades) for i in range(num_shades)]
        inner_colors.extend(shades)

    # Plot the nested pie chart
    fig, ax = plt.subplots(figsize=(18, 6))

    # Outer ring
    ax.pie(outer_sizes, labels=None, colors=outer_colors, radius=1, wedgeprops=dict(width=0.3, edgecolor='w'))

    # Inner ring
    ax.pie(inner_sizes, labels=None, colors=inner_colors, radius=0.7, wedgeprops=dict(width=0.3, edgecolor='w'))

    # Prepare legend items
    # outer_legend = [f"{outer_labels[i]} (Total: {outer_sizes[i]})" for i in range(len(outer_labels))]
    inner_legend = sorted([f"{x} (Count: {inner_sizes[i]})" for i, x in enumerate(inner_labels)])

    # Combine outer and inner legends
    # all_legends = outer_legend + inner_legend

    # Create the legend
    ax.legend(inner_legend, loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
    plt.tight_layout()

    # Display the plot
    plt.show()


def prepare_data_for_barplot(clinvar_disorder, clinvar_order, likely_pathogenic_disorder, likely_pathogenic_order,
                             column):
    """Prepare a DataFrame for grouped and stacked bar plotting."""
    # Separate by Interpretation for both ordered and disordered
    disorder_pathogenic = clinvar_disorder[clinvar_disorder['Interpretation'] == 'Pathogenic']
    disorder_uncertain = clinvar_disorder[clinvar_disorder['Interpretation'] == 'Uncertain']
    order_pathogenic = clinvar_order[clinvar_order['Interpretation'] == 'Pathogenic']
    order_uncertain = clinvar_order[clinvar_order['Interpretation'] == 'Uncertain']

    # Get the counts for each category
    top_disorder_pathogenic_am = get_value_counts(likely_pathogenic_disorder, column)
    top_order_pathogenic_am = get_value_counts(likely_pathogenic_order, column)
    top_disorder_pathogenic = get_value_counts(disorder_pathogenic, column)
    top_disorder_uncertain = get_value_counts(disorder_uncertain, column)
    top_order_pathogenic = get_value_counts(order_pathogenic, column)
    top_order_uncertain = get_value_counts(order_uncertain, column)

    # Combine all unique categories from each subset
    unique_categories = list(set(top_disorder_pathogenic.index) | set(top_disorder_uncertain.index) |
                             set(top_disorder_pathogenic_am.index) | set(top_order_pathogenic.index) |
                             set(top_order_uncertain.index) | set(top_order_pathogenic_am.index))

    # Create an empty DataFrame to hold the combined proportions
    bar_data = pd.DataFrame(index=unique_categories,
                            columns=['Pathogenic', 'Uncertain', 'Predicted Pathogenic']).fillna(0)

    # Calculate proportions for each bar
    bar_data['Pathogenic'] = (top_disorder_pathogenic / (top_disorder_pathogenic + top_order_pathogenic)) * 100
    bar_data['Uncertain'] = (top_disorder_uncertain / (top_disorder_uncertain + top_order_uncertain)) * 100
    bar_data['Predicted Pathogenic'] = (top_disorder_pathogenic_am / (
            top_disorder_pathogenic_am + top_order_pathogenic_am)) * 100

    # Fill missing values with zero
    bar_data = bar_data.fillna(0)

    return bar_data


def filter_and_aggregate_categories(bar_data, keep_categories):
    """Filter out specific categories and aggregate the rest into 'Others'.

    Args:
        bar_data (pd.DataFrame): The original DataFrame containing grouped and stacked data.
        keep_categories (list): List of categories to keep.

    Returns:
        pd.DataFrame: Modified DataFrame with desired categories and an "Others" category.
    """
    # Initialize a new DataFrame to hold the filtered data
    filtered_data = bar_data.loc[keep_categories].copy()

    # Check for categories that are not in the keep list
    all_categories = set(bar_data.index)
    other_categories = list(all_categories - set(keep_categories))  # Convert to a list for indexing

    # Aggregate the other categories into a single "Others" row
    others_row = bar_data.loc[other_categories].mean(axis=0)
    filtered_data.loc['Others'] = others_row

    return filtered_data


def plot_grouped_stacked_bar_clustered(bar_data, bar_width=0.2, intra_group_spacing=0.1,
                                       inter_group_spacing=0.5, hatch_patterns=None):
    """Create a clustered, grouped, and stacked bar plot with hatch patterns and dual legends."""
    labels = bar_data.index
    categories = ['Pathogenic', 'Uncertain', 'Predicted Pathogenic']
    n_categories = len(categories)

    # Two base colors for disorder and order
    disorder_color = 'orange'
    order_color = 'blue'

    # Define hatch patterns for each category
    hatch_patterns = hatch_patterns if hatch_patterns else ['', '/', 'x', ]

    # Calculate positions for the bars
    indices = range(len(labels))
    group_width = (bar_width * n_categories) + (
            (n_categories - 1) * intra_group_spacing)  # Total width of one group's bars
    offset_step = group_width + inter_group_spacing  # Total space between groups

    fig, ax = plt.subplots()

    # Create the bars with hatching patterns
    legend_disorder, legend_order = [], []

    for i, category in enumerate(categories):
        index_offsets = [x * offset_step + (i * (bar_width + intra_group_spacing)) for x in indices]
        hatch_pattern = hatch_patterns[i % len(hatch_patterns)]

        # Create the disorder (orange) bars with hatch pattern
        disorder_bars = ax.bar(index_offsets, bar_data[category], bar_width, color=disorder_color, hatch=hatch_pattern,
                               label=f'{category} - Disorder')
        legend_disorder.append(disorder_bars[0])

        # Create the order (blue) bars with hatch pattern
        order_bars = ax.bar(index_offsets, 100 - bar_data[category], bar_width, bottom=bar_data[category],
                            color=order_color, hatch=hatch_pattern, label=f'{category} - Order')
        legend_order.append(order_bars[0])

    # Adjust x-ticks to align with the group centers
    ax.set_xlabel('Categories')
    ax.set_ylabel('Proportion (%)')
    ax.set_title('Clustered Grouped and Stacked Bar Plot with Hatch Patterns')
    ax.set_xticks(
        [x * offset_step + ((bar_width * n_categories + intra_group_spacing * (n_categories - 1)) / 2) for x in
         indices])
    ax.set_xticklabels(labels, rotation=0, ha='right')

    # Create separate legends
    # Legend for hatch patterns (categories)
    hatch_legend = ax.legend(legend_disorder, categories, title="Categories", loc="upper right",
                             bbox_to_anchor=(1.5, 0.8))

    # Legend for the two base colors (Disorder/Order)
    color_legend = plt.legend([legend_disorder[0], legend_order[0]], ["Disorder", "Order"], title="Type",
                              loc="upper right", bbox_to_anchor=(1.3, 0.5))

    ax.add_artist(hatch_legend)  # Add the hatch legend back after replacing with the color legend

    plt.tight_layout()
    plt.show()


def plot_grouped_stacked_bar_group_spacing(bar_data, bar_width=0.2, group_spacing=0.5, hatch_patterns=None, title=None):
    """Create a grouped and stacked bar plot with only group spacing."""
    labels = bar_data.index
    categories = ['Pathogenic', 'Uncertain', 'Predicted Pathogenic']
    n_categories = len(categories)

    # Two base colors for disorder and order
    disorder_color = 'orange'
    order_color = 'lightblue'

    # Define hatch patterns for each category
    hatch_patterns = hatch_patterns if hatch_patterns else ['', '/', '.']

    fig, ax = plt.subplots()

    # Create the bars with group spacing
    legend_disorder, legend_order = [], []

    # Calculate offset step with group spacing included
    offset_step = (bar_width * n_categories) + group_spacing

    for i, category in enumerate(categories):
        index_offsets = [x * offset_step + i * bar_width for x in range(len(labels))]
        hatch_pattern = hatch_patterns[i % len(hatch_patterns)]

        # Create the disorder (orange) bars with hatch pattern
        disorder_bars = ax.bar(index_offsets, bar_data[category], bar_width, color=disorder_color, hatch=hatch_pattern,
                               label=f'{category} - Disorder')
        legend_disorder.append(disorder_bars[0])

        # Create the order (blue) bars with hatch pattern
        order_bars = ax.bar(index_offsets, 100 - bar_data[category], bar_width, bottom=bar_data[category],
                            color=order_color, hatch=hatch_pattern, label=f'{category} - Order')
        legend_order.append(order_bars[0])

    # Adjust x-ticks to align with the group centers
    ax.set_xlabel('Disease Categories')
    ax.set_ylabel('Proportion (%)')
    ax.set_title(title)
    x_tick_positions = [x * offset_step + ((bar_width * n_categories) / 2) - (bar_width / 2) for x in
                        range(len(labels))]
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(labels, rotation=15, ha='center')

    # Generate invisible bars for secondary legend
    n = [ax.bar(0, 0, color="gray", hatch=hatch_patterns[i % len(hatch_patterns)]) for i in range(n_categories)]

    # Legend for base colors (Disorder/Order)
    color_legend = ax.legend([legend_disorder[0], legend_order[0]], ["Disorder", "Order"], title="Type",
                             loc="upper right", bbox_to_anchor=(1.00, 0.7))

    # Legend for hatch patterns (Categories)
    hatch_legend = plt.legend(n, categories, title="Categories", loc="upper right", bbox_to_anchor=(1, 1))

    ax.add_artist(color_legend)

    plt.tight_layout()
    plt.show()


def plot_grouped_stacked_bar(bar_data, bar_width=0.2, intra_group_spacing=0.1, inter_group_spacing=0.5):
    """Create a grouped and stacked bar plot with spacing within and between groups, with distinct colors per category."""
    labels = bar_data.index
    categories = ['Pathogenic', 'Uncertain', 'Predicted Pathogenic']
    n_categories = len(categories)

    # Different color pairs for each category
    disorder_colors = ['red', 'yellow', 'orange']  # Colors for disorder (bottom of stack)
    order_colors = ['darkblue', 'lightblue', 'blue']  # Colors for order (top of stack)

    # Calculate positions for the bars
    indices = range(len(labels))
    group_width = (bar_width * n_categories) + (
            (n_categories - 1) * intra_group_spacing)  # Total width of one group's bars
    offset_step = group_width + inter_group_spacing  # Total space between groups

    fig, ax = plt.subplots()

    # Iterate through each category to create bars with distinct colors and both intra- and inter-group spacing
    for i, category in enumerate(categories):
        index_offsets = [x * offset_step + (i * (bar_width + intra_group_spacing)) for x in indices]

        # Create stacks for each bar using distinct colors
        ax.bar(index_offsets, bar_data[category], bar_width, label=f'{category} - Disorder', color=disorder_colors[i])
        ax.bar(index_offsets, 100 - bar_data[category], bar_width, bottom=bar_data[category],
               label=f'{category} - Order', color=order_colors[i])

    # Adjust x-ticks to align with the group centers
    ax.set_xlabel('Categories')
    ax.set_ylabel('Proportion (%)')
    ax.set_title('Grouped and Stacked Bar Plot with Internal and External Spacing')
    ax.set_xticks(
        [x * offset_step + ((bar_width * n_categories + intra_group_spacing * (n_categories - 1)) / 2) for x in
         indices])
    ax.set_xticklabels(labels, rotation=0, ha='right')
    ax.legend(title="Interpretation", loc="upper right")
    plt.tight_layout()
    plt.show()


def create_colors(df, ontology_column):
    # Define ontologies and categories
    ontologies = df[ontology_column].unique()

    # Generate a color palette for each category
    palette = sns.color_palette("tab20",
                                len(ontologies))  # You can choose a different palette like "Set1", "tab10", etc.

    # Assign a specific color to each category
    category_color_mapping = {ontology: palette[i] for i, ontology in enumerate(ontologies)}

    # Display the color mapping
    for ontology, color in category_color_mapping.items():
        print(f"Ontology: {ontology}, Color: {color}")

    return category_color_mapping


def create_piechart_for_genic_categories(clinvar_disorder_p, clinvar_order_p, category_color_mapping,
                                         title='Structural Distribution', figsize=(8, 6),filepath=None):
    # Aggregate structural categories
    disorder_counts = clinvar_disorder_p['genic_category'].value_counts()
    order_counts = clinvar_order_p['genic_category'].value_counts()

    # Sort labels based on the order in the category_color_mapping
    disorder_counts = disorder_counts.reindex(category_color_mapping.keys(), fill_value=0)
    order_counts = order_counts.reindex(category_color_mapping.keys(), fill_value=0)

    # Prepare data for plotting
    disorder_labels = disorder_counts.index
    disorder_sizes = disorder_counts.values
    order_labels = order_counts.index
    order_sizes = order_counts.values

    # Use the provided color mapping for the categories
    disorder_colors = [category_color_mapping.get(label, 'gray') for label in disorder_labels]
    order_colors = [category_color_mapping.get(label, 'gray') for label in order_labels]

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Pie chart for disorder counts
    axes[0].pie(disorder_sizes, colors=disorder_colors, autopct='%1.1f%%', startangle=90,
                counterclock=False)
    axes[0].set_title('Disorder')

    # Pie chart for order counts
    axes[1].pie(order_sizes, colors=order_colors, autopct='%1.1f%%', startangle=90,
                counterclock=False)
    axes[1].set_title('Order')

    plt.suptitle(title)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
    plt.show()


def plot_with_disease_genic_type_distribution(disorder_data, order_data, max_count=20, col_to_check="nDisease",
                                              figsize=(8, 5), tilte=None,filepath=None,legend=True):
    all_df = pd.concat([disorder_data, order_data])
    check_df = all_df.groupby(col_to_check)['Protein_ID'].nunique().reset_index()
    check_df.columns = [col_to_check, 'Unique_Protein_ID_Count']

    # Count the number of counts
    count_of_counts = check_df['Unique_Protein_ID_Count'].value_counts().reset_index()
    count_of_counts.columns = ['Unique_Protein_ID_Count', 'Count']
    count_of_counts = count_of_counts.sort_values(by='Unique_Protein_ID_Count', ascending=False)

    # Aggregate values above max_count
    above_max_count = count_of_counts[count_of_counts['Unique_Protein_ID_Count'] > max_count]
    above_max_sum = above_max_count['Count'].sum()
    below_or_equal_max_count = count_of_counts[count_of_counts['Unique_Protein_ID_Count'] <= max_count]

    # Ensure order from 1 to 20 and then the aggregated value
    below_or_equal_max_count = below_or_equal_max_count.sort_values(by='Unique_Protein_ID_Count')
    aggregated_data = pd.concat(
        [below_or_equal_max_count,
         pd.DataFrame({'Unique_Protein_ID_Count': [f'>{max_count}'], 'Count': [above_max_sum]})],
        ignore_index=True
    )

    # Define color mapping based on the number of unique proteins
    def get_color(count):
        if type(count) == str:  # Complex
            return category_colors_gene["Complex"]
        if count == 1:
            return category_colors_gene["Monogenic"]  # Monogenic
        elif 2 <= count < 5:
            return category_colors_gene["Multigenic"]  # Multigenic
        elif count >= 5:
            return category_colors_gene["Complex"]  # Complex
        return 'gray'  # Default color for any other cases

        # Apply the color mapping to each bar

    colors = [get_color(count) for count in aggregated_data['Unique_Protein_ID_Count'].astype(int, errors='ignore')]

    # Plot the distribution
    plt.figure(figsize=figsize)
    bars = plt.bar(aggregated_data['Unique_Protein_ID_Count'].astype(str), aggregated_data['Count'], color=colors)
    plt.xlabel('Unique Gene Counts')
    plt.ylabel('Number of Diseases')
    if not tilte:
        plt.title(f'Distribution Disease count in Genes for {col_to_check}')
    else:
        plt.title(tilte)
    plt.xticks(rotation=45)
    # plt.grid(axis='y')
    plt.ylim(0, aggregated_data['Count'].max() * 1.1)

    # Add count values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 1),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    # Add a legend
    def get_genic_category(row):
        count = row['Unique_Protein_ID_Count']
        if type(count) == str:  # Complex
            return 'Complex'
        if count == 1:
            return 'Monogenic'  # Monogenic
        elif 2 <= count < 5:
            return 'Multigenic'  # Polygenic
        elif count >= 5:
            return 'Complex'  # Complex
        return '-'

    aggregated_data['genic_category'] = aggregated_data.apply(get_genic_category, axis=1)

    # Calculate sums for each category
    monogenic_count = aggregated_data[aggregated_data['genic_category'] == 'Monogenic']['Count'].sum()
    multigenic_count = aggregated_data[aggregated_data['genic_category'] == 'Multigenic']['Count'].sum()
    complex_count = aggregated_data[aggregated_data['genic_category'] == 'Complex']['Count'].sum()

    # # Add a legend

    legend_counts = {
        'Monogenic': monogenic_count,
        'Multigenic': multigenic_count,
        'Complex': complex_count
    }

    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in category_colors_gene.values()]
    legend_labels_with_counts = [f"{key} ({legend_counts[key]})" for key in category_colors_gene.keys()]
    if legend:
        plt.legend(handles, legend_labels_with_counts, title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')


    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
    plt.show()

def plot_gene_distribution(clinvar_disorder_p, col_to_check="category_names", figsize=(8, 5)):
    # Count unique genes per ontology
    gene_counts = clinvar_disorder_p.drop_duplicates(subset=['Protein_ID', col_to_check])
    gene_counts = gene_counts[col_to_check].value_counts().sort_values(ascending=False)

    # Plot
    plt.figure(figsize=figsize)
    sns.barplot(x=gene_counts.index, y=gene_counts.values, color=COLORS["disorder"])
    plt.ylabel('Number of Genes')
    plt.title('Mutated Genes in IDRs by Disease Ontology')
    plt.xticks(rotation=30, ha='right')
    plt.xlabel('')
    plt.tight_layout()
    plt.show()

def plot_main_figures():
    fig_path = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/plots'
    fig1 = os.path.join(fig_path, "fig1")
    fig2 = os.path.join(fig_path, "fig2")

    # plot_gene_distribution(clinvar_disorder_p, figsize=(6, 3))
    # exit()


    # FIG 2A
    plot_with_disease_genic_type_distribution(clinvar_disorder_p, clinvar_order_p, max_count=15,
                                              col_to_check="nDisease", figsize=(8, 3),
                                              tilte="Genes count in Diseases",filepath=os.path.join(fig2,"A.png"),
                                              legend=True)

    exit()
    # FIG 2A/2
    create_piechart_for_genic_categories(clinvar_disorder_p, clinvar_order_p, category_colors_gene,
                                         title="Mutated positions based on Genetic Complexity",
                                         figsize=(5, 3),filepath=os.path.join(fig2,"A2.png"))

    exit()

    # Fig 2B
    heatmap_df_ontology_mutation_category = prepare_data_for_heatmap_mut_cat(clinvar_disorder_p, "Category",
                                                                             column_to_check='genic_category')
    #
    create_pie_charts(heatmap_df_ontology_mutation_category.T,
                      title=f'Mutated IDR positions',
                      category_color_mapping=category_colors_structure, choosen_categories=['All', 'Monogenic'],
                      figsize=(5, 3),filepath=os.path.join(fig2,"B.png"),add_legend=True)

    exit()

    # Fig 2C
    heatmap_df_ontology_classified = prepare_data_for_heatmap(clinvar_order_p, clinvar_order_u, clinvar_order_b,
                                                              clinvar_disorder_p, clinvar_disorder_u,
                                                              clinvar_disorder_b, ontology_column)
    ontology_df_index_ordered = create_bar_plot(heatmap_df_ontology_classified,
                                                xlabel=None,
                                                title=f'Proportion of mutations based on DO',
                                                category_color_mapping=category_colors_structure,
                                                filter=['Order-Pathogenic', "Disorder-Pathogenic"], islegend=False
                                                ,filepath=os.path.join(fig2,"C.png"),
                                                figsize=(7, 3)
                                                )

    exit()
    # create_bar_plot(heatmap_df_ontology_classified, title=f'Proportion of mutations based on DO (Disorder)',
    #                 category_color_mapping=category_colors_structure, filter=["Disorder-Pathogenic"],
    #                 order_by=ontology_df_index_ordered, islegend=False)

    # exit()

    # Fig 2 accumulative
    # heatmap_df_ontology_classified = prepare_data_for_heatmap(clinvar_order_p, clinvar_order_u, clinvar_order_b,
    #                                                           clinvar_disorder_p, clinvar_disorder_u,
    #                                                           clinvar_disorder_b, ontology_column,sum_values=True
    #                                                           )
    # ontology_df_index_ordered = create_bar_plot(heatmap_df_ontology_classified,
    #                                             title=f'Proportion of mutations based on Disease Ontology (Mutation Sum)',
    #                                             category_color_mapping=category_colors_structure,
    #                                             filter=['Order-Pathogenic', "Disorder-Pathogenic"])
    # create_bar_plot(heatmap_df_ontology_classified,
    #                 title=f'Proportion of mutations based on Disease Ontology (Disorder, Mutation Sum)',
    #                 category_color_mapping=category_colors_structure, filter=["Disorder-Pathogenic"],
    #                 order_by=ontology_df_index_ordered)

    # Fig 2C / 2
    heatmap_df_ontology_mutation_category = prepare_data_for_heatmap_mut_cat(clinvar_disorder_p, ontology_column,
                                                                             column_to_check='Category')
    create_bar_plot(heatmap_df_ontology_mutation_category,
                    xlabel=None,
                    title=f'DO Categories based on Structural Preference',
                    category_color_mapping=category_colors_structure, order_by=ontology_df_index_ordered,
                    islegend=False,filepath=os.path.join(fig2,"C2.png"),figsize=(6, 3))
    # exit()

    # Fig 2C / 2 accumulative
    # heatmap_df_ontology_mutation_category = prepare_data_for_heatmap_mut_cat(clinvar_disorder_p, ontology_column,
    #                                                                          column_to_check='Category',sum_values=True)
    # create_bar_plot(heatmap_df_ontology_mutation_category,
    #                 title=f'Mutation Sum Disease Ontology Categories for Structural Preference',
    #                 category_color_mapping=category_colors_structure, order_by=ontology_df_index_ordered)

    # Fig 2D
    heatmap_df_ontology_mutation_category = prepare_data_for_heatmap_mut_cat(
        clinvar_disorder_p[clinvar_disorder_p['genic_category'] == "Monogenic"], ontology_column,
        column_to_check='Category')
    create_bar_plot(heatmap_df_ontology_mutation_category,
                    xlabel=None,
                    title=f'Monogenic Diseases',
                    category_color_mapping=category_colors_structure, order_by=ontology_df_index_ordered,
                    islegend=False,filepath=os.path.join(fig2,"D.png"),figsize=(6, 3))

    # Fig 2E
    heatmap_df_ontology_mutation_category = prepare_data_for_heatmap_mut_cat(
        clinvar_disorder_p[clinvar_disorder_p['Developmental'] == True], ontology_column,
        column_to_check='Category')

    # print(heatmap_df_ontology_mutation_category)
    create_bar_plot(heatmap_df_ontology_mutation_category,
                    xlabel=None,
                    title=f'Developmental Diseases',
                    category_color_mapping=category_colors_structure, order_by=ontology_df_index_ordered,
                    islegend=False, filepath=os.path.join(fig2, "E.png"), figsize=(6, 3))


    # exit()

    # Fig 2F
    heatmap_df_ontology_mutation_category = prepare_data_for_heatmap_mut_cat(
        clinvar_disorder_p[clinvar_disorder_p['Rare'] == True], ontology_column,
        column_to_check='Category')
    # print(heatmap_df_ontology_mutation_category)
    # exit()
    create_bar_plot(heatmap_df_ontology_mutation_category,
                    xlabel=None,
                    title=f'Rare Diseases',
                    category_color_mapping=category_colors_structure, order_by=ontology_df_index_ordered,
                    islegend=False, filepath=os.path.join(fig2, "F.png"), figsize=(6, 3))

    # Fig 2D accumulative
    # heatmap_df_ontology_mutation_category = prepare_data_for_heatmap_mut_cat(
    #     clinvar_disorder_p[clinvar_disorder_p['genic_category'] == "Monogenic"], ontology_column,
    #     column_to_check='Category',sum_values=True)
    # create_bar_plot(heatmap_df_ontology_mutation_category,
    #                 title=f'Mutation Sum Disease Ontology Categories for Structural Preference (Monogenic)',
    #                 category_color_mapping=category_colors_structure, order_by=ontology_df_index_ordered)


def plot_others():
    return


def modify_categories(clinvar_p):
    modified_clinvar_p = clinvar_p.copy()
    disorder_categories = ["Only Disorder", 'Disorder Mostly']
    modified_clinvar_p['Category'] = modified_clinvar_p['Category'].apply(
        lambda x: "Disorder Specific" if x in disorder_categories else "Non-Disorder Specific")

    modified_clinvar_p['category_names'] = np.where(modified_clinvar_p['category_names'] == "Cardiovascular/Hematopoietic", 'Cardiovascular', modified_clinvar_p['category_names'])

    return modified_clinvar_p


if __name__ == '__main__':
    core_dir = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations"
    plots = os.path.join(core_dir, "processed_data", "plots")
    files = os.path.join(core_dir, "processed_data", "files")

    # analysis_type = "gene"
    analysis_type = "positional"

    clinvar_path_disorder_p = f'{files}/clinvar/Pathogenic/disorder/{analysis_type}_clinvar_functional_categorized_final.tsv'
    clinvar_path_disorder_u = f'{files}/clinvar/Uncertain/disorder/{analysis_type}_clinvar_functional_categorized_final.tsv'
    clinvar_path_disorder_b = f'{files}/clinvar/Benign/disorder/{analysis_type}_clinvar_functional_categorized_final.tsv'

    clinvar_path_order_p = f'{files}/clinvar/Pathogenic/order/{analysis_type}_clinvar_functional_categorized_final.tsv'
    clinvar_path_order_u = f'{files}/clinvar/Uncertain/order/{analysis_type}_clinvar_functional_categorized_final.tsv'
    clinvar_path_order_b = f'{files}/clinvar/Benign/order/{analysis_type}_clinvar_functional_categorized_final.tsv'

    clinvar_order_p = pd.read_csv(clinvar_path_order_p, sep='\t').rename(columns={"Protein_position": "Position"})
    clinvar_order_u = pd.read_csv(clinvar_path_order_u, sep='\t').rename(columns={"Protein_position": "Position"})
    clinvar_order_b = pd.read_csv(clinvar_path_order_b, sep='\t').rename(columns={"Protein_position": "Position"})
    clinvar_disorder_p = pd.read_csv(clinvar_path_disorder_p, sep='\t').rename(columns={"Protein_position": "Position"})
    clinvar_disorder_u = pd.read_csv(clinvar_path_disorder_u, sep='\t').rename(columns={"Protein_position": "Position"})
    clinvar_disorder_b = pd.read_csv(clinvar_path_disorder_b, sep='\t').rename(columns={"Protein_position": "Position"})

    excluded_categories = ["Unknown", "Inborn Genetic Diseases",
                           # 'Other'
                           ]

    clinvar_order_p = clinvar_order_p[~clinvar_order_p['category_names'].isin(excluded_categories)]
    clinvar_order_u = clinvar_order_u[~clinvar_order_u['category_names'].isin(excluded_categories)]
    clinvar_order_b = clinvar_order_b[~clinvar_order_b['category_names'].isin(excluded_categories)]
    clinvar_disorder_p = clinvar_disorder_p[~clinvar_disorder_p['category_names'].isin(excluded_categories)]
    clinvar_disorder_u = clinvar_disorder_u[~clinvar_disorder_u['category_names'].isin(excluded_categories)]
    clinvar_disorder_b = clinvar_disorder_b[~clinvar_disorder_b['category_names'].isin(excluded_categories)]

    clinvar_order_p = modify_categories(clinvar_order_p)
    clinvar_order_u = modify_categories(clinvar_order_u)
    clinvar_order_b = modify_categories(clinvar_order_b)
    clinvar_disorder_p = modify_categories(clinvar_disorder_p)
    clinvar_disorder_u = modify_categories(clinvar_disorder_u)
    clinvar_disorder_b = modify_categories(clinvar_disorder_b)

    ontology_column = 'category_names'

    category_color_mapping = create_colors(clinvar_order_p, ontology_column)
    ontology_order = list(category_color_mapping.keys())

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
        "Mostly Order/Equal": "#8e9aaf",
        'Disorder Specific': '#f27059',
        "Non-Disorder Specific": "#8e9aaf",

        'disorder': COLORS['disorder'],
        'order': COLORS['order'],
        "Disorder-Pathogenic": COLORS['disorder'],
        'Order-Pathogenic': COLORS['order'],

    }

    category_colors_gene = {
        'Monogenic': '#c9cba3',
        'Multigenic': '#ffe1a8',
        'Complex': '#A0E7E5',
    }

    plot_main_figures()
    exit()

    # print(clinvar_order_p['nDisease'].nunique())
    # print(clinvar_disorder_p['nDisease'].nunique())
    # pathogenic_df = pd.concat([clinvar_order_p, clinvar_disorder_p], ignore_index=True)
    # print(pathogenic_df['nDisease'].nunique())
    # exit()

    heatmap_df_ontology_classified = prepare_data_for_heatmap(clinvar_order_p, clinvar_order_u, clinvar_order_b,
                                                              clinvar_disorder_p, clinvar_disorder_u,
                                                              clinvar_disorder_b, ontology_column)
    # plot_disease_ontology_heatmap(heatmap_df_ontology_classified,title=f'Disease Ontology Categories for Structure and Pathogenicity Prediction',xlabel='Interpretation and Structure')
    # create_stacked_bar_plot(heatmap_df_ontology_classified,title=f'Proportion of mutations based on Disease Ontology',xlabel='Interpretation and Structure',category_color_mapping=category_color_mapping)
    ontology_df_index_ordered = create_bar_plot(heatmap_df_ontology_classified,
                                                title=f'Proportion of mutations based on Disease Ontology',
                                                category_color_mapping=category_colors_structure,
                                                filter=['Order-Pathogenic', "Disorder-Pathogenic"])
    # create_bar_plot(heatmap_df_ontology_classified,title=f'Proportion of mutations based on Disease Ontology (Disorder)',category_color_mapping=category_colors_structure,filter=["Disorder-Pathogenic"],order_by=ontology_df_index_ordered)
    # exit()
    # print(heatmap_df_ontology_classified)

    # Disease and Structural Preference
    # heatmap_df_ontology_mutation_category = prepare_data_for_heatmap_mut_cat(clinvar_disorder_p, ontology_column,column_to_check='Category')
    # plot_disease_ontology_heatmap(heatmap_df_ontology_mutation_category,title=f'Disease Ontology Categories for Structural Preference',xlabel='Structural Preference')
    # create_stacked_bar_plot(heatmap_df_ontology_mutation_category, title=f'Proportion of mutated positions based on Disease Ontology (Structural Preference)',xlabel='Structural Preference',category_color_mapping=category_color_mapping)
    # create_bar_plot(heatmap_df_ontology_mutation_category, title=f'Disease Ontology Categories for Structural Preference',category_color_mapping=category_colors_structure,order_by=ontology_df_index_ordered)
    # print(heatmap_df_ontology_mutation_category)
    # exit()

    # Disease and Structural Preference Monogenic (Fig2d)
    # heatmap_df_ontology_mutation_category = prepare_data_for_heatmap_mut_cat(clinvar_disorder_p[clinvar_disorder_p['genic_category'] =="Monogenic"], ontology_column,column_to_check='Category')
    # create_bar_plot(heatmap_df_ontology_mutation_category,
    #                 title=f'Disease Ontology Categories for Structural Preference (Monogenic)',
    #                 category_color_mapping=category_colors_structure,order_by=ontology_df_index_ordered)

    # exit()

    # Gene and Structure
    # heatmap_df_ontology_mutation_category = prepare_data_for_heatmap_mut_cat(pd.concat([clinvar_order_p,clinvar_disorder_p]), "structure",column_to_check='genic_category')
    # create_bar_plot(heatmap_df_ontology_mutation_category,
    #                 title=f'Proportion of mutated positions based for Structure on Disease Ontology (Genetic Category)',
    #                 category_color_mapping=category_colors_structure,percentage=True)
    # exit()

    # Disease based on Genic Category
    # heatmap_df_ontology_mutation_category = prepare_data_for_heatmap_mut_cat(clinvar_disorder_p, ontology_column,
    #                                                                          column_to_check='genic_category')

    # plot_disease_ontology_heatmap(heatmap_df_ontology_mutation_category,
    #                               title=f'Disease Ontology Categories for Genetic Category',
    #                               xlabel='Genetic Category')
    # create_stacked_bar_plot(heatmap_df_ontology_mutation_category,
    #                         title=f'Proportion of mutated positions based on Disease Ontology (Genetic Category)',
    #                         xlabel='Genetic Category',category_color_mapping=category_color_mapping)
    #
    # create_bar_plot(heatmap_df_ontology_mutation_category,
    #                 title=f'Proportion of mutated positions based on Disease Ontology (Genetic Category)',category_color_mapping=category_colors_gene )

    # exit()

    # heatmap_df_ontology_mutation_category = prepare_data_for_heatmap_mut_cat(modified_clinvar_disorder_p, "Category",
    #                                                                          column_to_check='genic_category')

    heatmap_df_ontology_mutation_category = prepare_data_for_heatmap_mut_cat(clinvar_disorder_p, "Category",
                                                                             column_to_check='genic_category')

    heatmap_df_ontology_mutation_category_o = prepare_data_for_heatmap_mut_cat(clinvar_order_p, "Category",
                                                                               column_to_check='genic_category')

    # create_bar_plot(heatmap_df_ontology_mutation_category.T,
    #                 title=f'Disease Ontology Categories for Structural Preference',
    #                 category_color_mapping=category_colors_structure)

    # exit()

    # print(heatmap_df_ontology_mutation_category)
    # create_piechart_all(heatmap_df_ontology_mutation_category.T,heatmap_df_ontology_mutation_category_o.T)
    # create_pie_charts(heatmap_df_ontology_mutation_category.T,
    #                   title=f'Proportion of mutated disordered positions  based on Disease Ontology (Genetic Category)',
    #                   category_color_mapping=category_colors_structure,choosen_categories=['All','Monogenic'],figsize=(8, 5))

    # exit()

    combined_df = heatmap_df_ontology_mutation_category.add(heatmap_df_ontology_mutation_category_o, fill_value=0).T

    # create_pie_charts(combined_df,title=f'Proportion of mutated positions based on Disease Ontology (Genetic Category)',category_color_mapping=category_colors_structure,figsize=(10, 10) )

    # exit()

    # print(heatmap_df_ontology_mutation_category)
    # print(heatmap_df_ontology_mutation_category_ordered)
    # print(combined_df)

    # Piechart for Genic
    # heatmap_df_ontology_mutation_category_d = prepare_data_for_heatmap_mut_cat(clinvar_disorder_p, "Category",
    #                                                                          column_to_check='genic_category')
    #
    # heatmap_df_ontology_mutation_category_o = prepare_data_for_heatmap_mut_cat(clinvar_order_p, "Category",
    #                                                                          column_to_check='genic_category')

    create_piechart_for_genic_categories(clinvar_disorder_p, clinvar_order_p, category_colors_gene,
                                         title="Proportion of mutated positions based on Disease Ontology (Genetic Category)")

    exit()

    """
    ------------------------------------------------------------------------------
    """
    lst_to_keep = ['Cancer', 'Neurodegenerative', 'Cardiovascular', 'Musculoskeletal']
    filtered_heatmap = filter_and_aggregate_categories(heatmap_df_new, lst_to_keep)
    # plot_grouped_stacked_bar(filtered_heatmap)
    # plot_grouped_stacked_bar_clustered(filtered_heatmap)
    plot_grouped_stacked_bar_group_spacing(filtered_heatmap, title='Distribution of Variants based on Disease Ontology')

    exit()
    #
    # heatmap_df_new_genes = prepare_data_for_heatmap_genes(merged_clinvar_disorder, merged_clinvar_order, merged_likely_disorder,merged_likely_order, ontology_column)
    # plot_disease_ontology_heatmap(heatmap_df_new_genes,
    #                               title=f'Disease Ontology Categories for Structure and Pathogenicity Prediction Genes count')
    # create_stacked_bar_plot(heatmap_df_new_genes, title=f'Proportion of Genes based on Disease Ontology')

    ontology_column = 'category_names'
    heatmap_df_percentage = prepare_data_for_heatmap_percent_age(merged_clinvar_disorder, merged_clinvar_order,
                                                                 merged_likely_disorder, merged_likely_order,
                                                                 ontology_column)
    plot_disease_ontology_heatmap(heatmap_df_percentage,
                                  title=f'Disease Ontology Categories for Structure and Pathogenicity Prediction')

    # heatmap_df_new_genes = prepare_data_for_heatmap_genes(merged_clinvar_disorder, merged_clinvar_order, merged_likely_disorder,merged_likely_order, ontology_column)
    # plot_disease_ontology_heatmap(heatmap_df_new_genes,
    #                               title=f'Disease Ontology Categories for Structure and Pathogenicity Prediction Genes count')
    # create_stacked_bar_plot(heatmap_df_new_genes, title=f'Proportion of Genes based on Disease Ontology')

    exit()

    heatmap_disorder = prepare_data_for_heatmap_disorder(clinvar_disorder, likely_pathogenic_disorder, ontology_column)
    plot_disease_ontology_heatmap(heatmap_disorder)
