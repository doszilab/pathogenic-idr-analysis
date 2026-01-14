import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np
from pandas import value_counts
from tqdm import tqdm
import re
import seaborn as sns
from matplotlib_venn import venn2, venn3

def plot_mutation_stats(n_pathogenic,n_benign,n_uncertain,n_m_disorder_pathogenic,n_m_disorder_benign,n_m_disorder_uncertain):
    # Define data for plotting
    data = {
        'Category': ['Pathogenic', 'Pathogenic', 'Uncertain', 'Uncertain', 'Benign', 'Benign'],
        'Type': ['Disordered', 'Ordered', 'Disordered', 'Ordered', 'Disordered', 'Ordered'],
        'Mutations': [
            n_m_disorder_pathogenic, n_pathogenic - n_m_disorder_pathogenic,
            n_m_disorder_uncertain, n_uncertain - n_m_disorder_uncertain,
            n_m_disorder_benign, n_benign - n_m_disorder_benign
        ]
    }

    # Convert data to DataFrame
    df = pd.DataFrame(data)

    # Create a pivot table to format the data for plotting
    df_pivot = df.pivot(index='Category', columns='Type', values='Mutations')

    # Create a stacked bar plot
    ax = df_pivot.plot(kind='bar', stacked=False, figsize=(10, 6), color=['orange','lightblue'])

    # Set labels and title
    ax.set_xlabel('Mutation Category')
    ax.set_ylabel('Number of Mutations')
    ax.set_title('Distribution of Mutations by Category and Type')

    # Add percentages as annotations on the bars
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        if height > 0:
            ax.text(x + width / 2, y + height + 100, f'{round(height):,}', ha='center', va='center', fontsize=10)

    # Display the legend
    plt.legend(title='Mutation Type')

    # Display the plot
    plt.tight_layout()
    plt.show()


def plot_pie_charts(n_pathogenic,n_benign,n_uncertain,n_m_disorder_pathogenic,n_m_disorder_benign,n_m_disorder_uncertain,positional=False,gene=False,plot_dir=None,figsize=(12, 6),filepath=None,title=None):

    colors = [COLORS['disorder'],COLORS['order']]
    # Define data for each category
    labels_pathogenic = ['Disordered', 'Ordered']
    sizes_pathogenic = [n_m_disorder_pathogenic, n_pathogenic - n_m_disorder_pathogenic]

    labels_uncertain = ['Disordered', 'Ordered']
    sizes_uncertain = [n_m_disorder_uncertain, n_uncertain - n_m_disorder_uncertain]

    labels_benign = ['Disordered', 'Ordered']
    sizes_benign = [n_m_disorder_benign, n_benign - n_m_disorder_benign]

    labels_pathogenic = None
    labels_uncertain = None
    labels_benign = None

    # Create a subplot for each category
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            # Calculate the absolute value as a rounded integer
            val = int(round(pct * total / 100.0))
            return f'{pct:.1f}%\n({val})'

        return my_autopct

    # Helper function to create pie chart with labels and colors
    def create_pie_chart(ax, sizes, labels, colors, title):
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct=lambda p: f'{p:.1f}%\n({int(p/100 * sum(sizes))})',
            startangle=90
        )

        for autotext, val in zip(autotexts, sizes):
            total = sum(sizes)
            pct = (val / total) * 100  # Calculate percentage based on actual value

            if val > 10000:
                val = f"{round(val/1000,1)}K"

            # Update the text to show both percentage and actual value
            autotext.set_text(f'{pct:.1f}%\n({val})')
            autotext.set_fontsize(12)

        # Calculate the angle for 33% reference line
        shift = 90
        disorder_percent = 360 * 0.368
        angle_disorder = disorder_percent + shift
        # print("angle")
        # print(angle_33)

        # Add the 33% reference line
        ax.plot([0, np.cos(np.radians(angle_disorder))], [0, np.sin(np.radians(angle_disorder))], color='grey', lw=2)

        ax.set_title(title)

    text = "" if positional else "Mutation"
    text = " (Genes)" if gene else text

    # Plot each pie chart with specified colors
    create_pie_chart(axs[0], sizes_pathogenic, labels_pathogenic, colors, f'Pathogenic {text}')
    create_pie_chart(axs[1], sizes_uncertain, labels_uncertain, colors, f'Uncertain {text}')
    create_pie_chart(axs[2], sizes_benign, labels_benign, colors, f'Benign {text}')

    # Adjust layout to avoid overlap
    plt.suptitle(title,fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.tight_layout()

    if filepath:
        plt.savefig(filepath)

    # Show the plot
    plt.show()


def plot_clinical_significance_distribution(clinvar_mapped_df):
    # Count occurrences of each unique value in 'ClinicalSignificance'
    value_counts = clinvar_mapped_df['ClinicalSignificance'].value_counts()

    # Convert to a DataFrame
    value_counts_df = value_counts.reset_index()
    value_counts_df.columns = ['ClinicalSignificance', 'Count']

    value_counts_df = value_counts_df.sort_values(by='Count', ascending=False)[:20]

    # Plot as a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(value_counts_df['ClinicalSignificance'], value_counts_df['Count'])

    # Label the chart
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.xlabel('Clinical Significance')
    plt.ylabel('Count')
    plt.title('Distribution of Clinical Significance in Clinvar Dataset')
    plt.tight_layout()

    plt.show()


def plot_structural_classification(mutations_in_disorder,mutations_in_ordered,positional=False,grouped=False,figsize=(12, 6),filepath=None,title=None):

    if positional:
        print(mutations_in_disorder.columns)
        if grouped:
            # Exclude Unknown
            mutations_in_disorder = mutations_in_disorder[mutations_in_disorder['category_names'] != "Unknown"]
            mutations_in_ordered = mutations_in_ordered[mutations_in_ordered['category_names'] != "Unknown"]

            mutations_in_disorder = mutations_in_disorder[['Protein_ID', 'Position', 'Interpretation','nDisease']].drop_duplicates()
            mutations_in_ordered = mutations_in_ordered[['Protein_ID', 'Position', 'Interpretation','nDisease']].drop_duplicates()
        else:
            mutations_in_disorder = mutations_in_disorder[['Protein_ID','Position','Interpretation']].drop_duplicates()
            mutations_in_ordered = mutations_in_ordered[['Protein_ID','Position','Interpretation']].drop_duplicates()


    mutations_in_disorder_pathogenic = mutations_in_disorder[mutations_in_disorder["Interpretation"] == "Pathogenic"]
    mutations_in_disorder_uncertain = mutations_in_disorder[mutations_in_disorder["Interpretation"] == "Uncertain"]
    mutations_in_disorder_benign = mutations_in_disorder[mutations_in_disorder["Interpretation"] == "Benign"]

    print(mutations_in_disorder_pathogenic)
    print(mutations_in_disorder_uncertain)
    print(mutations_in_disorder_benign)

    if positional:
        n_m_disorder_pathogenic = len(mutations_in_disorder_pathogenic)
        n_m_disorder_uncertain = len(mutations_in_disorder_uncertain)
        n_m_disorder_benign = len(mutations_in_disorder_benign)
    else:
        n_m_disorder_pathogenic = mutations_in_disorder_pathogenic['mutation_count'].sum()
        n_m_disorder_uncertain = mutations_in_disorder_uncertain['mutation_count'].sum()
        n_m_disorder_benign = mutations_in_disorder_benign['mutation_count'].sum()

    mutations_in_ordered_pathogenic = mutations_in_ordered[mutations_in_ordered["Interpretation"] == "Pathogenic"]
    mutations_in_ordered_uncertain = mutations_in_ordered[mutations_in_ordered["Interpretation"] == "Uncertain"]
    mutations_in_ordered_benign = mutations_in_ordered[mutations_in_ordered["Interpretation"] == "Benign"]

    if positional:
        n_m_order_pathogenic = len(mutations_in_ordered_pathogenic)
        n_m_order_uncertain = len(mutations_in_ordered_uncertain)
        n_m_order_benign = len(mutations_in_ordered_benign)
    else:
        n_m_order_pathogenic = mutations_in_ordered_pathogenic['mutation_count'].sum()
        n_m_order_uncertain = mutations_in_ordered_uncertain['mutation_count'].sum()
        n_m_order_benign = mutations_in_ordered_benign['mutation_count'].sum()


    n_pathogenic = n_m_order_pathogenic + n_m_disorder_pathogenic
    n_uncertain = n_m_order_uncertain + n_m_disorder_uncertain
    n_benign = n_m_order_benign + n_m_disorder_benign

    text = f"""
    Pathogenic:
        Number of mutations: {n_pathogenic:,}
            Number of mutations (Disordered): {n_m_disorder_pathogenic:,} ({round(n_m_disorder_pathogenic / n_pathogenic * 100, 2)} %)
            Number of mutations (Ordered): {n_m_order_pathogenic:,}
    Uncertain:
        Number of mutations: {n_uncertain:,}
            Number of mutations (Disordered): {n_m_disorder_uncertain:,} ({round(n_m_disorder_uncertain / n_uncertain * 100, 2)} %)
            Number of mutations (Ordered): {n_m_order_uncertain:,}
    Benign:
        Number of mutations: {n_benign:,}
            Number of mutations (Disordered): {n_m_disorder_benign:,} ({round(n_m_disorder_benign / n_benign * 100, 2)} %)
            Number of mutations (Ordered): {n_m_order_benign:,}
        """
    print(text)
    # plot_mutation_stats(n_pathogenic,n_benign,n_uncertain,n_m_disorder_pathogenic,n_m_disorder_benign,n_m_disorder_uncertain)
    plot_pie_charts(n_pathogenic,n_benign,n_uncertain,n_m_disorder_pathogenic,n_m_disorder_benign,n_m_disorder_uncertain,positional,figsize=figsize,filepath=filepath,title=title)




def plot_functional_distribution(big_functional_df):
    # Count the number of mutations for each function and interpretation category
    counts = big_functional_df.groupby(['fname', 'Interpretation'])['Protein_ID'].count().reset_index()
    counts.columns = ['fname', 'Interpretation', 'Count']

    # Pivot table for easier plotting
    counts_pivot = counts.pivot(index='fname', columns='Interpretation', values='Count').fillna(0)

    # Create a stacked bar plot
    ax = counts_pivot.plot(kind='bar', stacked=False, figsize=(12, 8), color=['lightblue', 'red', 'grey'])

    # Set labels and title
    ax.set_xlabel('Functional Category')
    ax.set_ylabel('Number of Mutations')
    ax.set_title('Distribution of Mutations by Functional Category')

    # Annotate the bars
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        if height > 0:
            # Place text above the bar
            ax.text(x + width / 2, y + height + 100, f'{round(height):,}', ha='center', va='center', fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_genes(mutations_in_disorder, mutations_in_ordered,filepath=None,figsize=(6, 3),title=None):

    mutations_in_disorder = mutations_in_disorder[['Protein_ID', 'Interpretation']].drop_duplicates()
    mutations_in_ordered = mutations_in_ordered[['Protein_ID', 'Interpretation']].drop_duplicates()

    # Extracting genes by interpretation for disordered and ordered regions
    disorder_pathogenic = set(
        mutations_in_disorder[mutations_in_disorder["Interpretation"] == "Pathogenic"]['Protein_ID'])
    disorder_uncertain = set(
        mutations_in_disorder[mutations_in_disorder["Interpretation"] == "Uncertain"]['Protein_ID'])
    disorder_benign = set(mutations_in_disorder[mutations_in_disorder["Interpretation"] == "Benign"]['Protein_ID'])

    ordered_pathogenic = set(mutations_in_ordered[mutations_in_ordered["Interpretation"] == "Pathogenic"]['Protein_ID'])
    ordered_uncertain = set(mutations_in_ordered[mutations_in_ordered["Interpretation"] == "Uncertain"]['Protein_ID'])
    ordered_benign = set(mutations_in_ordered[mutations_in_ordered["Interpretation"] == "Benign"]['Protein_ID'])

    # Creating subplots for each interpretation
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    print(len(disorder_pathogenic))

    disordered_color = COLORS['disorder']
    ordered_color = COLORS['order']
    both_color = COLORS['both']

    # Pathogenic Venn Diagram with color customization
    venn_pathogenic = venn2([disorder_pathogenic, ordered_pathogenic], set_labels=('', ''), ax=axes[0])
    axes[0].set_title('Pathogenic')

    # Modify colors
    venn_pathogenic.get_patch_by_id('10').set_color(disordered_color)
    venn_pathogenic.get_patch_by_id('01').set_color(ordered_color)
    venn_pathogenic.get_patch_by_id('11').set_color(both_color)

    # Set fontsize for numbers
    for subset_id in ['10', '01', '11']:
        label = venn_pathogenic.get_label_by_id(subset_id)
        if label is not None:
            label.set_fontsize(12)

    # Uncertain Venn Diagram
    venn_uncertain = venn2([disorder_uncertain, ordered_uncertain], set_labels=('', ''), ax=axes[1])
    axes[1].set_title('Uncertain')

    venn_uncertain.get_patch_by_id('10').set_color(disordered_color)
    venn_uncertain.get_patch_by_id('01').set_color(ordered_color)
    venn_uncertain.get_patch_by_id('11').set_color(both_color)

    for subset_id in ['10', '01', '11']:
        label = venn_uncertain.get_label_by_id(subset_id)
        if label is not None:
            label.set_fontsize(12)

    # Benign Venn Diagram
    venn_benign = venn2([disorder_benign, ordered_benign], set_labels=('', ''), ax=axes[2])
    axes[2].set_title('Benign')

    venn_benign.get_patch_by_id('10').set_color(disordered_color)
    venn_benign.get_patch_by_id('01').set_color(ordered_color)
    venn_benign.get_patch_by_id('11').set_color(both_color)

    for subset_id in ['10', '01', '11']:
        label = venn_benign.get_label_by_id(subset_id)
        if label is not None:
            label.set_fontsize(12)

    plt.suptitle(title,fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if filepath:
        plt.savefig(filepath)
    plt.show()



def create_structural_classification_for_protein(big_clinvar, disorder_df, info_df):
    # Calculate the distribution of disordered regions for each protein
    def calculate_disordered_distribution(df):
        df['Disordered_Region_Length'] = df['End'] - df['Start'] + 1
        distribution = df.groupby('Protein_ID')['Disordered_Region_Length'].sum().reset_index()
        return distribution

    # Calculate disordered region distribution
    disordered_distribution = calculate_disordered_distribution(disorder_df)
    print(disorder_df)
    print(disordered_distribution)

    # Merge big_clinvar with info_df and disordered distribution
    clinvar_info_df = info_df[info_df['Protein_ID'].isin(big_clinvar['Protein_ID'])]
    print(clinvar_info_df)
    print(info_df)

    clinvar_info_df_disorder = clinvar_info_df.merge(disordered_distribution, on='Protein_ID', how='inner')

    # Fill NaN values with 0
    clinvar_info_df_disorder['Disordered_Region_Length'] = clinvar_info_df_disorder['Disordered_Region_Length'].fillna(0)

    print("Clinvar Info with Disordered Region Lengths:")
    print(clinvar_info_df_disorder)

    clinvar_info_df_disorder['Seq_len'] = clinvar_info_df_disorder['Sequence'].str.len()
    clinvar_info_df_disorder['Ordered_Region_Length'] = clinvar_info_df_disorder['Seq_len'] - clinvar_info_df_disorder['Disordered_Region_Length']
    clinvar_info_df_disorder['Percent_Disordered_Region'] = clinvar_info_df_disorder['Disordered_Region_Length'] / clinvar_info_df_disorder['Seq_len']


    return clinvar_info_df_disorder

def lst_contain_in_series(series, lst):
    """
    Check if any of the elements in the list are present in the series.
    Returns True if any element from the list is in the series.
    """
    return series.apply(lambda x: any(item in str(x).split(';') for item in lst))


def create_functional_distribution_counts(big_functional_df):
    # Define functional info categories
    functional_info = ["binding_info", "dibs_info", "phasepro_info", "mfib_info",
                       "Elm_Info", "Roi", "Phosphorylation", "Acetylation",
                       "Sumoylation", "Ubiquitination", "Methylation"]

    # Initialize an empty list to collect all category counts
    all_counts = []

    # Iterate over each functional category and count occurrences
    for info in functional_info:
        info_df = big_functional_df[big_functional_df['info'].str.contains(info, na=False)]

        # Count occurrences for each category
        count = info_df.shape[0]
        all_counts.append([info, count])

    # Create a DataFrame from the collected counts
    functional_df = pd.DataFrame(all_counts, columns=['Functional_Category', 'Count'])

    return functional_df


def calculate_specific_functional_distribution(big_functional_df, functional_category):
    # Filter the DataFrame to include only rows relevant to the specified functional category
    filtered_df = big_functional_df[big_functional_df['info'].str.contains(functional_category, na=False)]
    print(filtered_df)

    # Split both 'info' and 'info_cols' columns by semicolon and stack them together
    info_split = filtered_df['info'].str.split(';', expand=True)
    info_cols_split = filtered_df['info_cols'].str.split(';', expand=True)

    # Stack both 'info' and 'info_cols' and reset index to keep alignment
    stacked_info = info_split.stack().reset_index(level=1, drop=True)
    stacked_info_cols = info_cols_split.stack().reset_index(level=1, drop=True)

    # Combine both the 'info' and 'info_cols' into a new DataFrame
    filtered_df = pd.DataFrame({
        'info': stacked_info,
        'info_cols': stacked_info_cols
    })
    print(filtered_df)

    # Now filter the DataFrame where the 'info' matches the 'functional_category'
    filtered_df = filtered_df[filtered_df['info'] == functional_category]

    # Split values by comma, expand into separate rows, and count occurrences
    expanded_df = filtered_df['info_cols'].str.split(', ', expand=True).stack().reset_index(level=1, drop=True)

    # Apply custom rules to categorize the functional category
    big_categories = ['Interaction/Binding','Localization','Down-regulation-\nProtein Degradation','Linker','Activation','Oligomerization','Structural Element',"Head"]

    expanded_df = expanded_df.apply(lambda x: 'Interaction/Binding' if 'Interaction' in x else x)
    expanded_df = expanded_df.apply(lambda x: 'Interaction/Binding' if 'interaction' in x else x)
    expanded_df = expanded_df.apply(lambda x: 'Interaction/Binding' if 'Interacts' in x else x)
    expanded_df = expanded_df.apply(lambda x: 'Interaction/Binding' if 'binding' in x else x)
    expanded_df = expanded_df.apply(lambda x: 'Localization' if 'localization' in x else x)
    expanded_df = expanded_df.apply(lambda x: 'Down-regulation-\nProtein Degradation' if 'down-regulation' in x else x)
    expanded_df = expanded_df.apply(lambda x: 'Down-regulation-\nProtein Degradation' if 'protein degradation' in x else x)
    expanded_df = expanded_df.apply(lambda x: 'Down-regulation-\nProtein Degradation' if 'degradation' in x else x)
    expanded_df = expanded_df.apply(lambda x: 'Linker' if 'Linker' in x else x)
    expanded_df = expanded_df.apply(lambda x: 'Activation' if 'activation' in x else x)
    expanded_df = expanded_df.apply(lambda x: 'Oligomerization' if 'oligomerization' in x else x)
    expanded_df = expanded_df.apply(lambda x: 'Structural Element' if 'RS' == x else x)
    expanded_df = expanded_df.apply(lambda x: 'Structural Element' if 'NTAD' == x else x)
    expanded_df = expanded_df.apply(lambda x: 'Structural Element' if 'CTAD' == x else x)
    expanded_df = expanded_df.apply(lambda x: 'Structural Element' if 'Tail' == x else x)
    expanded_df = expanded_df.apply(lambda x: 'Structural Element' if 'domain' in x else x)
    # expanded_df = expanded_df.apply(lambda x: 'Structural Element' if 'Head' == x else x)
    expanded_df = expanded_df.apply(lambda x: 'Structural Element' if 'C' == x else x)
    expanded_df = expanded_df.apply(lambda x: 'Structural Element' if 'helical region' in x else x)
    expanded_df = expanded_df.apply(lambda x: 'Structural Element' if 'Coil' in x else x)

    expanded_df = expanded_df.apply(lambda x: 'Other' if x not in big_categories else x)

    # Count occurrences of each unique value within the expanded DataFrame
    value_counts = expanded_df.value_counts().reset_index()
    value_counts.columns = [functional_category, 'Count']

    return value_counts


def create_functional_distribution_df_new(big_functional_df,grouped=False):


    cols = ['Protein_ID','Position','Interpretation']

    if grouped:
        cols = ['Protein_ID', 'Position', 'Interpretation','nDisease']

    info_cols = "info"
    disorder_info = ["MobiDB"]
    functional_info = ["binding_info","dibs_info","phasepro_info","mfib_info",
                       "Elm_Info","Roi",
                       "Phosphorylation","Acetylation","Sumoylation","Ubiquitination","Methylation",]

    lst = []

    # Iterate through unique interpretations
    for interpretation in big_functional_df['Interpretation'].unique():
        current_df = big_functional_df[big_functional_df['Interpretation'] == interpretation]

        info_series = current_df[info_cols]

        # 1. Mutations where disorder is NA and functional is NA
        neither_disorder_nor_functional = current_df[ ~lst_contain_in_series(info_series, disorder_info) & ~lst_contain_in_series(info_series, functional_info)]
        n_mutations_neither = neither_disorder_nor_functional[cols].drop_duplicates().shape[0]

        # 2. Mutations where disorder is NA and functional is present
        functional_not_disprot = current_df[~lst_contain_in_series(info_series, disorder_info) & lst_contain_in_series(info_series, functional_info)]
        n_functional_not_disprot = functional_not_disprot[cols].drop_duplicates().shape[0]

        # 3. Mutations where disorder is present and functional is present
        functional_and_disprot = current_df[lst_contain_in_series(info_series, disorder_info) & lst_contain_in_series(info_series, functional_info)]
        n_functional_disprot = functional_and_disprot[cols].drop_duplicates().shape[0]

        # 4. Mutations where disorder is present and functional is NA
        disorder_not_functional = current_df[lst_contain_in_series(info_series, disorder_info) & ~lst_contain_in_series(info_series, functional_info)]
        n_mutations_disprot_no_function = disorder_not_functional[cols].drop_duplicates().shape[0]

        # Append results to the list
        lst.append([interpretation, n_functional_disprot, n_functional_not_disprot,
                    n_mutations_disprot_no_function, n_mutations_neither])

    functional_df = pd.DataFrame(lst, columns=['Interpretation', 'n_functional_disprot',
                                                   'n_functional_ones_not_in_disprot',
                                                   'n_mutations_in_disorder_disprot', 'n_mutations_not_in_disprot'])

    return functional_df

def plot_individual_functional_categories(func_counts, top_n=20, plot_dir=None,functional_category='Functional_Category'):
    # Sort and filter the top functional categories by count
    top_functional_df = func_counts.sort_values(by='Count', ascending=False).head(top_n)

    main_color = '#f08080'

    # Plot
    plt.figure(figsize=(5, 4))
    bars = plt.barh(top_functional_df[functional_category], top_functional_df['Count'], color=main_color)
    plt.xlabel('Count')
    # plt.ylabel(functional_category)
    plt.xlim(1,top_functional_df['Count'].max() * 1.1)
    plt.suptitle(f'Top Uniprot Region of Interest categories')
    plt.gca().invert_yaxis()
    plt.tight_layout()


    max_number = top_functional_df['Count'].max()

    # Adding count labels at the end of each bar
    for bar in bars:
        plt.text(
            bar.get_width() + (max_number * 0.01),  # Position the text slightly beyond the bar's width
            bar.get_y() + bar.get_height() / 2,
            f'{int(bar.get_width())}',  # Convert the count to an integer and add as label
            va='center'
        )

    # Save plot if directory is provided
    if plot_dir:

        plt.savefig(f"{plot_dir}/functional_distribution_{functional_category}.png")

    plt.show()


def plot_pie_chart(interpretation_data,ispositional=False,plot_dir=None,colors = [
        "#ccd5ae",
        "#d4a373",
        "#cdb4db",
        "#a9def9"
    ]):


    # Plot pie chart for each interpretation category
    for index, row in interpretation_data.iterrows():
        labels = ['Functional and Exp. Dis', 'Functional', 'Exp. Dis',
                  'No Annotation']
        sizes = [row['n_functional_disprot'], row['n_functional_ones_not_in_disprot'],
                 row['n_mutations_in_disorder_disprot'], row['n_mutations_not_in_disprot']]
        explode = (0.1, 0, 0, 0)  # explode the first slice for emphasis

        # plt.figure()
        plt.figure(figsize=(7, 4))
        plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90,
                pctdistance=0.85, labeldistance=1.1, colors=colors,textprops={'fontsize': 14})  # Adjust the distances

        label_text = ' Mutations' if ispositional else ' Positions'

        plt.title(row['Interpretation'] + label_text, fontsize=16,pad=20)
        # plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        # plt.subplots_adjust(top=0.85, bottom=0.1, left=0.15, right=0.85)
        plt.tight_layout()
        if plot_dir:
            plt.savefig(
                f"{plot_dir}/functional_{row['Interpretation']}.png")
        plt.show()

def plot_pie_chart_three(interpretation_data, ispositional=False, colors=[
        "#ccd5ae",
        "#d4a373",
        "#cdb4db",
        "#a9def9"
    ], figsize=(10, 5), filepath=None):

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    labels = ['Functional & Exp. Dis.', 'Functional', 'Exp. Dis.', 'No Annotation']

    # Store one set of wedges for legend
    wedges_for_legend = None

    for index, row in interpretation_data.iterrows():
        sizes = [
            row['n_functional_disprot'],
            row['n_functional_ones_not_in_disprot'],
            row['n_mutations_in_disorder_disprot'],
            row['n_mutations_not_in_disprot']
        ]

        ax = axes[index]

        wedges, _, autotexts = ax.pie(
            sizes,
            labels=None, autopct='%1.1f%%',
            colors=colors, startangle=90,
            textprops={'color': "black", 'fontsize': 11},
            wedgeprops={'width': 0.3},
            pctdistance=1.4,
        )

        label_text = ' Mutations' if ispositional else ' Positions'
        ax.set_title(row['Interpretation'] + label_text, pad=20)

        # Save wedges from first chart for legend
        if wedges_for_legend is None:
            wedges_for_legend = wedges

    # Adjust layout to create space at bottom
    plt.subplots_adjust(bottom=0.25)

    # Legend below plots, single row
    fig.legend(
        handles=wedges_for_legend,
        labels=labels,
        loc='lower center',
        ncol=4,
        title="Annotation Types",
        fontsize=10,
        title_fontsize=11,
        frameon=False
    )

    plt.suptitle("Annotation Coverage of Mutated Positions in IDR", fontsize=14)
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])  # leave space for legend

    if filepath:
        plt.savefig(filepath, bbox_inches='tight')
    plt.show()


def plot_distributions(df,column,title=''):
    value_counts = df[column].value_counts()[:10]
    plt.figure(figsize=(10, 6))
    value_counts.plot(kind='bar')
    plt.title(f'Value Counts of {column}')
    plt.xlabel(column)
    plt.ylabel('Counts')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def create_count_for_multicategory(big_functional_df, main_category_column='Category', genic_category_column='genic_category',grouped=False,add_all=False):
    # Define the functional info categories to count

    ptms = ["Phosphorylation", "Acetylation", "Sumoylation", "Ubiquitination", "Methylation"]

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


    # print(big_functional_df[genic_category_column].unique())
    big_functional_df = big_functional_df[big_functional_df[genic_category_column]!= '-']



    # Iterate over each combination of the main and genic category
    group_cols = [main_category_column, genic_category_column]

    grouped_df = big_functional_df.groupby(group_cols)

    for group, group_df in grouped_df:
        for func_key, name in functional_info.items():
            # Use the .str.contains() method to check occurrences of each functional term in 'info' column
            count = group_df['info'].str.contains(func_key, na=False).sum()

            # If the functional info exists in the group, append to pivot_data
            if count > 0:
                pivot_data.append({
                    main_category_column: group[0],
                    genic_category_column: group[1],
                    'fname': name,  # Pivot functional name into 'fname' column
                    'Count': count  # Include the count of occurrences
                })

        # PTM Count: check if any PTM exists in the 'info' column (only count once if any are found)
        ptm_found = group_df['info'].astype(str).apply(lambda x: any(ptm in x for ptm in ptms)).sum()

        # If any PTM exists in the group, append a single PTM count
        if ptm_found > 0:
            pivot_data.append({
                main_category_column: group[0],
                genic_category_column: group[1],
                'fname': 'PTM',  # Group PTMs under the "PTM" label
                'Count': ptm_found  # Count occurrences, but only once per row
            })

    if add_all:
        grouped_df = big_functional_df.groupby([main_category_column])

        for group, group_df in grouped_df:
            for func_key, name in functional_info.items():
                # Use the .str.contains() method to check occurrences of each functional term in 'info' column
                count = group_df['info'].str.contains(func_key, na=False).sum()

                # If the functional info exists in the group, append to pivot_data
                if count > 0:
                    pivot_data.append({
                        main_category_column: group[0],
                        genic_category_column: "all",
                        'fname': name,  # Pivot functional name into 'fname' column
                        'Count': count  # Include the count of occurrences
                    })

            # PTM Count: check if any PTM exists in the 'info' column (only count once if any are found)
            ptm_found = group_df['info'].astype(str).apply(lambda x: any(ptm in x for ptm in ptms)).sum()

            # If any PTM exists in the group, append a single PTM count
            if ptm_found > 0:
                pivot_data.append({
                    main_category_column: group[0],
                    genic_category_column:  "all",
                    'fname': 'PTM',  # Group PTMs under the "PTM" label
                    'Count': ptm_found  # Count occurrences, but only once per row
                })

    # Convert pivot_data into a DataFrame
    pivot_df = pd.DataFrame(pivot_data)

    # # Pivot the data to get the count in the required format
    # pivoted_df = pivot_df.pivot_table(index=['Category', 'Genic_Category'],
    #                                   columns='fname', values='count', fill_value=0).reset_index()

    return pivot_df


def plot_functional_distribution_multicategory(big_functional_df, main_category_column='Category', genic_category_column='genic_category', file_name='distribution', category_colors={},add_all=False,genic_order=["Monogenic", 'Polygenic', 'Complex']):
    # Count the number of mutations for each function, interpretation category, and genic category
    counts_df = create_count_for_multicategory(big_functional_df,main_category_column,genic_category_column,add_all=True)


    # Ensure every combination of fname, main_category_column, and genic_category_column exists
    all_fnames = counts_df['fname'].unique()
    all_categories = counts_df[main_category_column].unique()
    all_genic_categories = counts_df[genic_category_column].unique()

    # Reorder
    all_categories = [x for x in category_colors.keys() if x in all_categories]
    all_genic_categories = [x for x in genic_order if x in all_genic_categories]

    max_value = counts_df.groupby(['fname', genic_category_column])['Count'].sum().max()

    num_subplots = len(all_fnames)

    fig, axes = plt.subplots(nrows=1, ncols=num_subplots, figsize=(1.5 * num_subplots, 8), sharex=True)
    axes = axes.flatten()

    if len(all_fnames) == 1:
        axes = [axes]

    naxes = 0

    for ax, fname in zip(axes, all_fnames):

        fname_df = counts_df[counts_df['fname'] == fname]
        fname_pivot = fname_df.pivot_table(index=genic_category_column, columns=main_category_column,
                                           values='Count').fillna(0)

        # Reindex to ensure all categories and genic categories are represented
        fname_pivot = fname_pivot.reindex(index=all_genic_categories, columns=all_categories, fill_value=0)

        print(fname)
        print(fname_pivot)

        # Plot the data
        bar_positions = np.arange(len(all_genic_categories))
        bar_width = 0.8

        bottoms = np.zeros(len(all_genic_categories))
        for category in all_categories:
            ax.bar(bar_positions, fname_pivot[category], width=bar_width,
                   label=category if fname == all_fnames[0] else "", bottom=bottoms,
                   color=category_colors.get(category, 'gray'))  # Set bar color
            bottoms += fname_pivot[category]

        # Set the title and labels
        ax.set_title(f'{fname}', fontsize=11)
        if naxes == 0:
            ax.set_ylabel('Number of Mutations', fontsize=12)
            naxes += 1
        else:
            ax.yaxis.set_visible(False)
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(all_genic_categories, rotation=90)
        ax.set_ylim(0, max_value * 1.05)

        # Annotate the bars with the total number of mutations on top
        for i, pos in enumerate(bar_positions):
            total = fname_pivot.iloc[i].sum()
            if total > 0:
                ax.annotate(f'{int(total):,}', xy=(pos, bottoms[i]),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points", ha='center', va='bottom', fontsize=10, color='black')

        # # Annotate the bars with the total number of mutations on top
        # for bar in ax.patches:
        #     height = bar.get_height()
        #     if height > 0:
        #         ax.annotate(f'{int(height):,}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + height),
        #                     xytext=(0, 3),  # 3 points vertical offset
        #                     textcoords="offset points", ha='center', va='bottom', fontsize=8)

    # Remove any unused subplots
    for ax in axes[num_subplots:]:
        fig.delaxes(ax)

    # Add a single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Category", title_fontsize='11', fontsize='10', loc='upper right', bbox_to_anchor=(1, 0.9))

    # Add main title and x-axis title
    fig.suptitle("Disordered Mutation Distribution on Functional Region based on the mutation occurence within the disease", fontsize=13)
    fig.text(0.5, 0.04, "Genetic Disorder Categories", ha='center', fontsize=12)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.subplots_adjust(right=0.84)

    plt.savefig(f"/dlab/home/norbi/PycharmProjects/AlphaMissense_Stat/processed_data/plots/clinvar/{file_name}.png")
    plt.show()


def create_count_for_multicategory_for_all(big_functional_df, main_category_column='Category',ptm_aggregated=True):
    # Define the functional info categories to count

    ptms = ["Phosphorylation", "Acetylation", "Sumoylation", "Ubiquitination", "Methylation"]

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
            this_df = group_df[group_df['info'].str.contains(func_key)]
            count = len(this_df)
            print(func_key,this_df,count)

            # If the functional info exists in the group, append to pivot_data
            if count > 0:
                pivot_data.append({
                    main_category_column: group[0],
                    'fname': name,  # Pivot functional name into 'fname' column
                    'Count': count  # Include the count of occurrences
                })

        # Handle PTMs
        if ptm_aggregated:
            # Count if any PTM exists
            ptm_found = group_df['info'].astype(str).apply(lambda x: any(ptm in x for ptm in ptms)).sum()
            if ptm_found > 0:
                pivot_data.append({
                    main_category_column: group[0],
                    'fname': 'PTM',
                    'Count': ptm_found
                })
        else:
            # Count each PTM individually
            for ptm in ptms:
                count = group_df['info'].str.contains(ptm, na=False).sum()
                if count > 0:
                    pivot_data.append({
                        main_category_column: group[0],
                        'fname': ptm,
                        'Count': count
                    })

    # Convert pivot_data into a DataFrame
    pivot_df = pd.DataFrame(pivot_data)

    # # Pivot the data to get the count in the required format
    # pivoted_df = pivot_df.pivot_table(index=['Category', 'Genic_Category'],
    #                                   columns='fname', values='count', fill_value=0).reset_index()

    return pivot_df

def plot_functional_distribution_multicategory_for_all(big_functional_df, main_category_column='Category',file_name=None, category_colors={},figsize=(12, 8),islegend=True,filepath=None,xlabel="Functional Region",ptm_aggregated=True):
    # Exclude Unknowns
    big_functional_df = big_functional_df[big_functional_df['category_names'] != "Unknown"]

    # Count the number of mutations for each function, interpretation category, and genic category
    counts_df = create_count_for_multicategory_for_all(big_functional_df,main_category_column,ptm_aggregated=ptm_aggregated)

    # Pivot the DataFrame to have 'fname' as index and 'Category' as columns with 'Count' as values
    counts_pivot = counts_df.pivot_table(index='fname', columns=main_category_column, values='Count', fill_value=0)

    # Reorder columns based on stack_order if provided
    stack_order = [x for x in category_colors.keys() if x in counts_pivot.columns]
    if ptm_aggregated:
        function_order = ["Exp. Dis","DIBS","MFIB","ELM","PDB","PTM","PhasePro","UniProt Roi"]
    else:
        function_order = ["Exp. Dis","DIBS","MFIB","ELM","PhasePro","UniProt Roi","PDB","Phosphorylation", "Acetylation", "Sumoylation", "Ubiquitination", "Methylation"]

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
    plt.suptitle("Pathogenic Positions in Functional Regions")
    plt.xlabel(xlabel)
    plt.ylabel("Disease-Position Count")
    plt.xticks(rotation=90)

    max_val = counts_pivot.sum(axis=1).max()
    plt.ylim(0, max_val * 1.15)

    plt.tight_layout()

    # Add legend
    if islegend:
        plt.legend(title='Category', bbox_to_anchor=(0.5, -0.8), loc='lower center',frameon=False,ncol=2)
    else:
        # If we do not want a legend, remove it explicitly
        if ax.get_legend() is not None:
            ax.get_legend().remove()


    if filepath:
        plt.savefig(filepath,bbox_inches='tight')

    plt.show()

def plot_functional_distribution_multicategory_for_ptm(big_functional_df, main_category_column='Category',file_name=None, category_colors={},figsize=(12, 8),islegend=True,filepath=None,xlabel="Functional Region",ptm_aggregated=False):
    # Exclude Unkowns
    big_functional_df = big_functional_df[big_functional_df['category_names'] != "Unknown"]
    # Count the number of mutations for each function, interpretation category, and genic category
    counts_df = create_count_for_multicategory_for_all(big_functional_df,main_category_column,ptm_aggregated=False)

    # Pivot the DataFrame to have 'fname' as index and 'Category' as columns with 'Count' as values
    counts_pivot = counts_df.pivot_table(index='fname', columns=main_category_column, values='Count', fill_value=0)

    # Reorder columns based on stack_order if provided
    stack_order = [x for x in category_colors.keys() if x in counts_pivot.columns]
    function_order = ["Phosphorylation", "Acetylation", "Sumoylation", "Ubiquitination", "Methylation"]

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
    plt.suptitle("Pathogenic Positions in PTMs")
    plt.xlabel(xlabel)
    plt.ylabel("Disease-Position Count")
    plt.xticks(rotation=45)

    max_val = counts_pivot.sum(axis=1).max()
    plt.ylim(0, max_val * 1.15)
    plt.tight_layout()

    # Add legend
    if islegend:
        plt.legend(title='Category', bbox_to_anchor=(0.5, -1), loc='lower center', frameon=False, ncol=2)
    else:
        # If we do not want a legend, remove it explicitly
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    if filepath:
        plt.savefig(filepath,bbox_inches='tight')

    plt.show()


def plot_with_disease_genic_type_distribution(disorder_data, order_data,max_count=20, col_to_check="nDisease"):
    all_df = pd.concat([disorder_data,order_data])
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
        if type(count) == str: # Complex
            return 'lightcoral'
        if count == 1:
            return 'lightgreen'  # Monogenic
        elif 2 <= count < 5:
            return 'lightblue'  # Polygenic
        elif count >= 5:
            return 'lightcoral'  # Complex
        return 'gray'  # Default color for any other cases

        # Apply the color mapping to each bar

    colors = [get_color(count) for count in aggregated_data['Unique_Protein_ID_Count'].astype(int, errors='ignore')]


    # Plot the distribution
    plt.figure(figsize=(10, 7))
    bars = plt.bar(aggregated_data['Unique_Protein_ID_Count'].astype(str), aggregated_data['Count'], color=colors)
    plt.xlabel('Unique Gene Counts')
    plt.ylabel('Number of Diseases')
    plt.title(f'Distribution Disease count in Genes for {col_to_check}')
    plt.xticks(rotation=45)
    # plt.grid(axis='y')

    # Add count values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')


    # Add a legend
    print(aggregated_data)
    def get_genic_category(row):
        count = row['Unique_Protein_ID_Count']
        if type(count) == str: # Complex
            return 'Complex'
        if count == 1:
            return 'Monogenic'  # Monogenic
        elif 2 <= count < 5:
            return 'Polygenic'  # Polygenic
        elif count >= 5:
            return 'Complex'  # Complex
        return '-'

    aggregated_data['genic_category'] = aggregated_data.apply(get_genic_category,axis=1)

    # Calculate sums for each category
    monogenic_count = aggregated_data[aggregated_data['genic_category'] == 'Monogenic']['Count'].sum()
    polygenic_count = aggregated_data[aggregated_data['genic_category'] == 'Polygenic' ]['Count'].sum()
    complex_count = aggregated_data[aggregated_data['genic_category'] == 'Complex']['Count'].sum()

    # Add a legend
    legend_labels = {
        'Monogenic': 'lightgreen',
        'Polygenic': 'lightblue',
        'Complex': 'lightcoral'
    }

    legend_counts = {
        'Monogenic': monogenic_count,
        'Polygenic': polygenic_count,
        'Complex': complex_count
    }

    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in legend_labels.values()]
    legend_labels_with_counts = [f"{key} ({legend_counts[key]})" for key in legend_labels.keys()]
    plt.legend(handles, legend_labels_with_counts, title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')


    plt.tight_layout()

    plt.show()




def plot_with_gene_disease_count_distribution(disorder_data, order_data, max_count=20, col_to_check="nDisease"):
    all_df = pd.concat([disorder_data, order_data])

    # Group by Protein_ID and count the number of unique diseases
    gene_disease_counts = all_df.groupby('Protein_ID')[col_to_check].nunique().reset_index()
    gene_disease_counts.columns = ['Protein_ID', 'Disease_Count']

    # Count how many genes have each number of diseases
    count_of_diseases_per_gene = gene_disease_counts['Disease_Count'].value_counts().reset_index()
    count_of_diseases_per_gene.columns = ['Disease_Count', 'Gene_Count']
    count_of_diseases_per_gene = count_of_diseases_per_gene.sort_values(by='Disease_Count', ascending=False)

    # Aggregate values above max_count
    above_max_count = count_of_diseases_per_gene[count_of_diseases_per_gene['Disease_Count'] > max_count]
    above_max_sum = above_max_count['Gene_Count'].sum()
    below_or_equal_max_count = count_of_diseases_per_gene[count_of_diseases_per_gene['Disease_Count'] <= max_count]

    # Ensure order from 1 to max_count and then the aggregated value
    below_or_equal_max_count = below_or_equal_max_count.sort_values(by='Disease_Count')
    aggregated_data = pd.concat(
        [below_or_equal_max_count,
         pd.DataFrame({'Disease_Count': [f'>{max_count}'], 'Gene_Count': [above_max_sum]})],
        ignore_index=True
    )

    # Plot without coloring
    plt.figure(figsize=(10, 7))
    bars = plt.bar(aggregated_data['Disease_Count'].astype(str), aggregated_data['Gene_Count'])
    plt.xlabel('Number of Diseases per Gene')
    plt.ylabel('Number of Genes')
    plt.title('Distribution of Disease Counts Across Genes')
    plt.xticks(rotation=45)

    # Add count values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def plot_monogenic_threeplot(df,file=None, title="Distribution of Genes/Disease based on Mutation occurrence for Monogenic Diseases",
                   xlabel="Structural Classification", ylabel="Number of Genes"):

    # Prepare data for plotting
    # categories = ["Disorder Only", "Order Only", "Both", 'Disorder Mostly', 'Order Mostly', "Equal"]
    categories = df['Category'].unique()

    bar_width = 0.2
    x = range(len(categories))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Gene
    values = [df[df['Category'] == category]['Protein_ID'].nunique() for category in categories]
    ax.bar([p + bar_width * 0 for p in x], values, width=bar_width, label="Gene")
    # Disease
    values = [df[df['Category'] == category]['nDisease'].nunique() for category in categories]
    ax.bar([p + bar_width * 1 for p in x], values, width=bar_width, label="Disease")

    for category in categories:
        cat = df[df['Category'] == category]
        print(category)
        print(cat['nDisease'].unique())
    # Position
    values = [df[df['Category'] == category].shape[0] for category in categories]
    ax.bar([p + bar_width * 2 for p in x], values, width=bar_width, label="Position")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks([p + 1.5 * bar_width for p in x])
    ax.set_xticklabels(categories)
    ax.bar_label(ax.containers[0], fmt='{:,.0f}')
    ax.bar_label(ax.containers[1], fmt='{:,.0f}')
    ax.bar_label(ax.containers[2], fmt='{:,.0f}')
    ax.legend()
    plt.tight_layout()

    if file:
        plt.savefig(file)
    plt.show()



def plot_monogenic_position_plot(df, file=None, title="Distribution of Genes/Disease based on Mutation Occurrence for Monogenic Diseases",
                                 xlabel="Structural Classification", ylabel="Number of Genes", category_colors={}):

    # Prepare data for plotting
    categories = df['Category'].unique()
    bar_width = 0.4
    x = range(len(categories))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Values for the categories
    values = [df[df['Category'] == category].shape[0] for category in categories]

    # Plotting bars with the specific colors for each category
    bars = ax.bar(x, values, width=bar_width, color=[category_colors[cat] for cat in categories])

    # Customize the plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")

    # Add value labels above the bars
    ax.bar_label(bars, fmt='{:,.0f}', padding=3)


    # Tight layout for better spacing
    plt.tight_layout()

    # Save the plot to a file if specified
    if file:
        plt.savefig(file)

    # Show the plot
    plt.show()


def extract_pos_based_df(df):
    df[['Protein_ID', 'Position']] = df['AccessionPosition'].str.split('|', expand=True)
    df = df.drop(columns=['AccessionPosition'])
    df['Position'] = df['Position'].astype(int)
    return df

def plot_disordered_distribution(df,figsize=(6, 6),filepath=None):
    # df['CombinedDisorder'] = df['ValidRegionStart'].astype(int)

    values = df['CombinedDisorder'].value_counts()

    # Prepare labels for pie chart
    labels = ['Ordered','Disordered']

    # Plot pie chart
    plt.figure(figsize=figsize)
    plt.pie(
        values,
        labels=labels,
        autopct=lambda p: f'{p:.1f}% ',
        colors=[COLORS['order'],COLORS['disorder']],
        startangle=90,
        explode=(0.1, 0),
        textprops={'fontsize': 12}
    )
    plt.title('Structural Distribution of Human Proteome')
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
    plt.show()

def create_base_table(df,tables_with_only_main_isoforms):
    """
    Creates a summary table with the following statistics for both Disorder and Order regions:
    - Total number of unique protein-position pairs
    - Percentage of unique protein-position pairs in each sub-category (Disorder, Order)
    - Number of mutations
    - Number of unique genes
    - Number of unique diseases

    For each of the categories: Pathogenic, Uncertain, Benign.

    :param df: DataFrame containing protein information with columns such as 'Interpretation', 'Position', 'Protein_ID', 'Disease_ID', and 'Structure' (Disorder/Order)
    :return: DataFrame containing the summarized statistics
    """
    # Initialize an empty list to store the results
    result_list = []

    # Define the categories and structure types we are interested in
    categories = df['Interpretation'].unique()
    structures = df['structure'].unique()

    # Calculate the total number of unique protein-position pairs for percentage calculation (ClinVar)
    total_unique_protein_positions = df[['Protein_ID', 'Position']].drop_duplicates().shape[0]

    # Calculate the total number of unique protein-position pairs for percentage calculation (Human Proteome)
    total_unique_protein_positions_human_proteome = tables_with_only_main_isoforms['Sequence'].astype(str).str.len().sum()

    # Loop through each category and structure, and calculate the required statistics
    for category in categories:
        category_df = df[(df['Interpretation'] == category)]
        n_unique_protein_positions = category_df[['Protein_ID', 'Position']].drop_duplicates().shape[0]
        percent_protein_positions = (n_unique_protein_positions / total_unique_protein_positions) * 100
        percent_protein_positions_proteome = (
                                                         n_unique_protein_positions / total_unique_protein_positions_human_proteome) * 100
        n_mutations = category_df.shape[0]
        n_genes = category_df['Protein_ID'].nunique()
        n_diseases = category_df['nDisease'].nunique()

        # Append the results for this category and structure to the result list
        result_list.append({
            'Category': category,
            'Structure': "All",
            'Unique_Protein_Position_Count': n_unique_protein_positions,
            'ClinVar_Residue_Percentage': f"{round(percent_protein_positions, 2)} %",
            'Human_Proteome_Residue_Percentage': f"{round(percent_protein_positions_proteome, 2)} %",
            'Mutation_Count': n_mutations,
            'Gene_Count': n_genes,
            'Disease_Count': n_diseases
        })
        for structure in structures:
            # Filter the DataFrame by the current category and structure type
            category_structure_df = df[(df['Interpretation'] == category) & (df['structure'] == structure)]

            # Calculate statistics
            n_unique_protein_positions = category_structure_df[['Protein_ID', 'Position']].drop_duplicates().shape[0]
            percent_protein_positions = (n_unique_protein_positions / total_unique_protein_positions) * 100
            percent_protein_positions_proteome = (n_unique_protein_positions / total_unique_protein_positions_human_proteome) * 100
            n_mutations = category_structure_df.shape[0]
            n_genes = category_structure_df['Protein_ID'].nunique()
            n_diseases = category_structure_df['nDisease'].nunique()

            # Append the results for this category and structure to the result list
            result_list.append({
                'Category': category,
                'Structure': structure,
                'Unique_Protein_Position_Count': n_unique_protein_positions,
                'ClinVar_Residue_Percentage': f"{round(percent_protein_positions, 2)} %",
                'Human_Proteome_Residue_Percentage': f"{round(percent_protein_positions_proteome, 2)} %",
                'Mutation_Count': n_mutations,
                'Gene_Count': n_genes,
                'Disease_Count': n_diseases
            })

    # Create a DataFrame from the result list
    table_df = pd.DataFrame(result_list)

    return table_df

def clear_base_clinvar(clinvar_df,tables_with_only_main_isoforms):
    main_isoforms = tables_with_only_main_isoforms[tables_with_only_main_isoforms['main_isoform']=='yes']
    clinvar_df = clinvar_df[clinvar_df['Protein_ID'].isin(main_isoforms['Protein_ID'])]

    # # Exclude not provided or not specified mutations
    # clinvar_df = clinvar_df[clinvar_df['category_names'] != "Unknown"]

    clinvar_df = clinvar_df[clinvar_df['Position'] != 1]
    return clinvar_df


def plot_structural_distribution_based_on_all(disoder_df,order_df,info='PDB',info_col='info'):
    info_disorder = disoder_df[disoder_df[info_col].str.contains(info)]
    info_order = order_df[order_df[info_col].str.contains(info)]

    other_disorder = disoder_df.shape[0] - info_disorder.shape[0]
    other_order = order_df.shape[0] - info_order.shape[0]

    # Labels for the pie chart
    labels = [f'{info}', 'Other']

    # Create the two pie charts
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Pie chart for disorder
    axes[0].pie([info_disorder.shape[0], other_disorder], labels=labels, autopct='%1.1f%%',
                colors=['orange', 'lightgray'], startangle=90)
    axes[0].set_title(f'Disordered {info} Distribution')

    # Pie chart for order
    axes[1].pie([info_order.shape[0], other_order], labels=labels, autopct='%1.1f%%', colors=['lightblue', 'lightgray'],
                startangle=90)
    axes[1].set_title(f'Ordered {info} Distribution')

    # Equal aspect ratio ensures that pie is drawn as a circle.
    for ax in axes:
        ax.axis('equal')

    plt.tight_layout()
    plt.show()


# def plot_structural_distribution_with_interpretations(disoder_df, order_df, info='PDB', info_col='info', interpretation_col='Interpretation', category_order=['Pathogenic', 'Uncertain', 'Benign'],alternative_name=None):
#     # Filter data for the specified info
#     info_disorder = disoder_df[disoder_df[info_col].str.contains(info)]
#     info_order = order_df[order_df[info_col].str.contains(info)]
#
#     # Group by interpretation and count occurrences for both disorder and order
#     disorder_grouped = disoder_df.groupby([interpretation_col]).size()
#     disorder_info_grouped = info_disorder.groupby([interpretation_col]).size()
#
#     order_grouped = order_df.groupby([interpretation_col]).size()
#     order_info_grouped = info_order.groupby([interpretation_col]).size()
#
#     # Create subplots, one for each interpretation (Pathogenic, Uncertain, Benign)
#     num_categories = len(category_order)
#     fig, axes = plt.subplots(1, num_categories, figsize=(5, 5), sharey=True)
#
#     if num_categories == 1:
#         axes = [axes]  # Ensure axes is always a list
#
#     label_added = False
#
#     # Loop through each interpretation and create subplots
#     for ax, interpretation in zip(axes, category_order):
#         # Get the total counts for the interpretation for disorder and order
#         disorder_total = disorder_grouped.get(interpretation, 0)
#         disorder_info = disorder_info_grouped.get(interpretation, 0)
#         disorder_info_percentage = (disorder_info / disorder_total) * 100 if disorder_total > 0 else 0
#         disorder_other_percentage = 100 - disorder_info_percentage
#
#         order_total = order_grouped.get(interpretation, 0)
#         order_info = order_info_grouped.get(interpretation, 0)
#         order_info_percentage = (order_info / order_total) * 100 if order_total > 0 else 0
#         order_other_percentage = 100 - order_info_percentage
#
#         # Bar positions for the two structures (disordered and ordered)
#         bar_positions = np.arange(2)
#         bar_width = 0.5
#
#         # Plot disorder and order bars for the current interpretation
#         ax.bar(bar_positions[0], disorder_info_percentage, width=bar_width, label=f'Disordered {info}', color='orange')
#         # ax.bar(bar_positions[0], disorder_other_percentage, width=bar_width, bottom=disorder_info_percentage, label='Disordered Other', color='lightgray')
#
#         ax.bar(bar_positions[1], order_info_percentage, width=bar_width, label=f'Ordered {info}', color='lightblue')
#         # ax.bar(bar_positions[1], order_other_percentage, width=bar_width, bottom=order_info_percentage, label='Ordered Other', color='lightgray')
#
#         # Set titles and labels
#         ax.set_title(f'{interpretation}')
#         ax.set_xticks(bar_positions)
#         ax.set_xticklabels(['Disordered', 'Ordered'])
#         if not label_added:
#             ax.set_ylabel('Percentage (%)')
#             label_added = True
#
#     # Ensure only one legend is shown (on the first subplot)
#     handles, labels = axes[0].get_legend_handles_labels()
#     fig.legend(handles, labels, title="Category", title_fontsize='11', fontsize='10', loc='upper right', bbox_to_anchor=(1, 0.9))
#
#     if alternative_name:
#         info = alternative_name
#
#     plt.suptitle( f"Structural distribution of {info} for ClinVar Variants (Percentage)" )
#
#     # Adjust layout and show the plot
#     plt.tight_layout(rect=[0, 0, 1, 1])
#     plt.subplots_adjust(right=0.75)
#
#     plt.show()


def plot_structural_distribution_with_interpretations_new(disorder_df, order_df, info='PDB', info_col='info', interpretation_col='Interpretation',
                                                          category_order=['Pathogenic', 'Uncertain', 'Benign'], alternative_name=None,
                                                          filter_interpretation=[], figsize=(4, 4), ax=None,islegend=True):
    # Filter data for the specified info
    info_disorder = disorder_df[disorder_df[info_col].str.contains(info, case=False, na=False)]
    info_order = order_df[order_df[info_col].str.contains(info, case=False, na=False)]

    if filter_interpretation:
        info_disorder  = info_disorder[info_disorder[interpretation_col].isin(filter_interpretation)]
        info_order  = info_order[info_order[interpretation_col].isin(filter_interpretation)]

    # Group by interpretation and count occurrences for both disorder and order
    disorder_grouped = disorder_df.groupby([interpretation_col]).size()
    disorder_info_grouped = info_disorder.groupby([interpretation_col]).size()

    order_grouped = order_df.groupby([interpretation_col]).size()
    order_info_grouped = info_order.groupby([interpretation_col]).size()

    # Initialize the data for plotting
    interpretation_labels = category_order
    disordered_percentages = []
    ordered_percentages = []

    for interpretation in category_order:
        if filter_interpretation and interpretation not in filter_interpretation:
            continue
        # Calculate disorder percentages
        disorder_total = disorder_grouped.get(interpretation, 0)
        disorder_info = disorder_info_grouped.get(interpretation, 0)
        disorder_info_percentage = (disorder_info / disorder_total * 100) if disorder_total > 0 else 0
        disordered_percentages.append(disorder_info_percentage)

        # Calculate order percentages
        order_total = order_grouped.get(interpretation, 0)
        order_info = order_info_grouped.get(interpretation, 0)
        order_info_percentage = (order_info / order_total * 100) if order_total > 0 else 0
        ordered_percentages.append(order_info_percentage)

    # Determine x locations and labels
    if not filter_interpretation:
        x = np.arange(len(interpretation_labels))  # label locations
    else:
        interpretation_labels = filter_interpretation
        x = np.arange(len(filter_interpretation))

    bar_width = 0.35  # Width of the bars

    # If no ax is provided, create a figure and axis
    created_figure = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_figure = True

    # Plot the bars using original colors
    ax.bar(x - bar_width / 2, disordered_percentages, width=bar_width,
           label=f'Disordered {alternative_name if alternative_name else info}', color=COLORS['disorder'])
    ax.bar(x + bar_width / 2, ordered_percentages, width=bar_width,
           label=f'Ordered {alternative_name if alternative_name else info}', color=COLORS['order'])

    # Add labels and custom x-axis tick labels
    # ax.set_xlabel('Interpretation')
    ax.set_ylabel('Percentage (%)')
    # Each subplot title is just the annotation (alternative_name or info)
    ax.set_title(alternative_name if alternative_name else info)
    ax.set_xticks(x)
    ax.set_xticklabels(interpretation_labels)
    if islegend:
        ax.legend(title='Category', loc='upper left')
    else:
        # If we do not want a legend, remove it explicitly
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Adjust layout if this plot is standalone
    if created_figure:
        plt.tight_layout()
        plt.show()


# def plot_base_distribution(comined_disorder_df, big_clinvar_final_df,figsize=(8, 5),filepath=None):
#     # Create unique pairs of Protein_ID and Position in combined disorder dataset
#     total_unique_positions = comined_disorder_df[['Protein_ID', 'Position']].drop_duplicates()
#
#     # Find unique pairs in ClinVar dataset
#     clinvar_unique_positions = big_clinvar_final_df[['Protein_ID', 'Position']].drop_duplicates()
#
#     # Calculate percentage of unique positions in the combined disorder dataset that have mutations in ClinVar
#     mutated_positions_count = pd.merge(total_unique_positions, clinvar_unique_positions, on=['Protein_ID', 'Position']).shape[0]
#     total_positions_count = total_unique_positions.shape[0]
#     percentage_mutated_positions = (mutated_positions_count / total_positions_count) * 100
#     non_mutated_percentage = 100 - percentage_mutated_positions
#
#     # Convert counts to millions
#     total_positions_million = total_positions_count / 1_000_000
#     mutated_positions_million = mutated_positions_count / 1_000_000
#     non_mutated_positions_million = total_positions_million - mutated_positions_million
#
#     # # Remove any exact duplicate rows before counting
#     # big_clinvar_final_df = big_clinvar_final_df.drop_duplicates(subset=['Protein_ID', 'Position', 'Interpretation','Mutations'])
#
#     # Count mutations by interpretation for the distribution plot
#     interpretation_counts = big_clinvar_final_df['Interpretation'].value_counts()
#     desired_order = ['Pathogenic', 'Uncertain', 'Benign']
#     interpretation_counts = interpretation_counts.reindex(desired_order).fillna(0)
#     interpretation_counts_million = interpretation_counts / 1_000_000  # Convert to millions
#
#     # Plotting
#     fig, axes = plt.subplots(1, 2, figsize=figsize)
#
#     # Subplot 1: Bar plot for proportion of mutated vs. non-mutated positions
#     # Plot the non-mutated portion
#     bars_non_mutated = axes[0].bar(
#         ['Positions'],  # X-tick label
#         [non_mutated_percentage],
#         color=COLORS["Non-Mutated"]
#     )
#
#     # Plot the mutated portion on top of the non-mutated portion
#     bars_mutated = axes[0].bar(
#         ['Positions'],
#         [percentage_mutated_positions],
#         bottom=[non_mutated_percentage],
#         color=COLORS["Mutated"]
#     )
#     # axes[0].set_ylim(0, 100)
#     # axes[0].set_ylabel('Proportion (%)')
#     axes[0].set_title('Human Proteome', pad=20,fontsize=14)
#
#     for spine in axes[0].spines.values():
#         spine.set_visible(False)
#
#     # Set title
#     axes[0].set_title('Human Proteome', pad=20, fontsize=14)
#
#     # Add labels inside the bars, centered vertically within each segment
#     # For the non-mutated segment
#     axes[0].text(
#         0,  # same x-position as the bar; 'All' is at index 0
#         non_mutated_percentage / 2,  # halfway through the non_mutated segment
#         f'{non_mutated_percentage:.1f}% ({non_mutated_positions_million:.2f}M)',
#         ha='center', va='center', fontsize=12, color='black'
#     )
#
#     # For the mutated segment
#     axes[0].text(
#         0,  # same x-position as the bar
#         non_mutated_percentage + (percentage_mutated_positions / 2),
#         # halfway through the mutated segment from the top of non_mutated
#         f'{percentage_mutated_positions:.1f}% ({mutated_positions_million:.2f}M)',
#         ha='center', va='center', fontsize=12, color='black'
#     )
#
#     colors = [(1, 0, 0, 0.7),  # red with alpha
#               (0.5, 0.5, 0.5, 0.7),  # gray with alpha
#               (0.68, 0.85, 0.9, 0.7)]  # light blue with alpha
#     explode = (0.2, 0, 0)  # Offset the "Pathogenic" slice
#
#     # Function to format pie chart labels with both percentage and actual value
#     def pie_label_format(pct, all_vals):
#         absolute = int(round(pct / 100. * sum(all_vals)))
#         return f"{pct:.1f}% ({absolute / 1_000_000:.2f}M)"
#
#     # Subplot 2: Pie chart for mutation distribution by interpretation with actual values
#     axes[1].pie(
#         interpretation_counts,
#         labels=interpretation_counts.index,
#         autopct=lambda pct: pie_label_format(pct, interpretation_counts),
#         startangle=90,
#         colors=colors,
#         explode=explode,
#         textprops={'fontsize': 12},
#         labeldistance=1.2,
#         pctdistance=0.5
#     )
#     axes[1].set_title('Mutations', pad=20,fontsize=14)
#
#     # Main title
#     plt.suptitle('Distribution of Mutations in ClinVar',fontsize=16)
#
#     plt.tight_layout()
#     plt.subplots_adjust(hspace=0.5)
#     if filepath:
#         plt.savefig(filepath)
#     plt.show()

def plot_base_distribution(comined_disorder_df, big_clinvar_final_df, figsize=(8, 5), filepath=None):
    # Create unique pairs of Protein_ID and Position in combined disorder dataset
    total_unique_positions = comined_disorder_df[['Protein_ID', 'Position']].drop_duplicates()

    # Find unique pairs in ClinVar dataset
    clinvar_unique_positions = big_clinvar_final_df[['Protein_ID', 'Position']].drop_duplicates()

    # Calculate percentage of unique positions in the combined disorder dataset that have mutations in ClinVar
    mutated_positions_count = pd.merge(total_unique_positions, clinvar_unique_positions, on=['Protein_ID', 'Position']).shape[0]
    total_positions_count = total_unique_positions.shape[0]
    percentage_mutated_positions = (mutated_positions_count / total_positions_count) * 100
    non_mutated_percentage = 100 - percentage_mutated_positions

    # Convert counts to millions
    total_positions_million = total_positions_count / 1_000_000
    mutated_positions_million = mutated_positions_count / 1_000_000
    non_mutated_positions_million = total_positions_million - mutated_positions_million

    # Count mutations by interpretation
    interpretation_counts = big_clinvar_final_df['Interpretation'].value_counts()
    desired_order = ['Benign', 'Uncertain', 'Pathogenic']
    interpretation_counts = interpretation_counts.reindex(desired_order).fillna(0)
    interpretation_counts_million = interpretation_counts / 1_000_000  # Convert to millions

    # Calculate percentages for interpretation categories
    total_mutations = interpretation_counts.sum()
    interpretation_percentages = (interpretation_counts / total_mutations) * 100

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Use numeric positions for bars
    # Left plot: single column at x=0
    x_left = 0
    # Right plot: single column at x=0
    x_right = 0

    # --- Left subplot: Stacked bar for non-mutated vs. mutated positions ---
    # Non-mutated portion
    axes[0].bar(
        x_left,
        non_mutated_percentage,
        color=COLORS["Non-Mutated"],
        width=0.3
    )

    # Mutated portion on top
    axes[0].bar(
        x_left,
        percentage_mutated_positions,
        bottom=non_mutated_percentage,
        color=COLORS["Mutated"],
        width=0.3
    )

    # Remove spines
    for spine in axes[0].spines.values():
        spine.set_visible(False)

    axes[0].set_title('Human Proteome', pad=10)

    # Add labels next to the bars
    # For non-mutated segment
    y_mid_non_mutated = non_mutated_percentage / 2
    # Percentage (left)
    axes[0].text(
        x_left - 0.2, y_mid_non_mutated,
        f'{non_mutated_percentage:.1f}%',
        ha='right', va='center', color='black'
    )
    # Actual value (right)
    axes[0].text(
        x_left + 0.2, y_mid_non_mutated,
        f'{non_mutated_positions_million:.2f}M',
        ha='left', va='center', color='black'
    )

    # For mutated segment
    y_mid_mutated = non_mutated_percentage + (percentage_mutated_positions / 2)
    # Percentage (left)
    axes[0].text(
        x_left - 0.2, y_mid_mutated,
        f'{percentage_mutated_positions:.1f}%',
        ha='right', va='center', color='black'
    )
    # Actual value (right)
    axes[0].text(
        x_left + 0.2, y_mid_mutated,
        f'{mutated_positions_million:.2f}M',
        ha='left', va='center', color='black'
    )

    # Set x-tick
    axes[0].set_yticklabels([])
    axes[0].tick_params(axis='y', which='both', length=0)
    axes[0].set_xticks([x_left])
    axes[0].set_xticklabels(['Positions'])

    # --- Right subplot: Stacked bar for Pathogenic, Uncertain, Benign mutations ---
    colors = [(0.68, 0.85, 0.9, 0.7),# Benign (light blue)
              (0.5, 0.5, 0.5, 0.7), # Uncertain (gray)
              (1, 0, 0, 0.7),# Pathogenic (red)
              ]

    categories = interpretation_counts.index
    bottom_val = 0

    # Stacking each category
    for (cat, pct, val_million, c) in zip(categories, interpretation_percentages, interpretation_counts_million, colors):
        axes[1].bar(
            x_right,
            pct,
            bottom=bottom_val,
            color=c,
            width=0.3
        )
        # Vertical midpoint for this segment
        y_mid = bottom_val + (pct / 2)
        # Percentage (left)
        axes[1].text(
            x_right - 0.2, y_mid,
            f'{pct:.1f}%',
            ha='right', va='center', color='black'
        )
        # Actual value (right)
        axes[1].text(
            x_right + 0.2, y_mid,
            f'{val_million:.2f}M',
            ha='left', va='center', color='black'
        )
        bottom_val += pct

    # Remove Y-tick labels
    axes[1].set_yticklabels([])
    axes[1].tick_params(axis='y', which='both', length=0,)

    # Remove spines
    for spine in axes[1].spines.values():
        spine.set_visible(False)

    axes[1].set_title('ClinVar Variants', pad=10)
    axes[1].set_xticks([x_right])
    axes[1].set_xticklabels(['Mutations'])

    # Adjust x-limits to create space
    # This ensures the width=0.3 is visible as a narrower bar
    axes[0].set_xlim(-0.5, 0.5)
    axes[1].set_xlim(-0.5, 0.5)

    # Main title
    plt.suptitle('Distribution of Mutations')

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
    plt.show()


def plot_interpretation_distribution(big_clinvar_final_df):
    # Count mutations by interpretation for the distribution plot
    interpretation_counts = big_clinvar_final_df['Interpretation'].value_counts()
    desired_order = ['Pathogenic', 'Uncertain', 'Benign']

    all = big_clinvar_final_df.shape[0]
    interpretation_counts = interpretation_counts.reindex(desired_order).fillna(0)
    interpretation_counts_million = interpretation_counts / 1_000_000  # Convert to millions

    # Plotting
    fig, ax = plt.subplots(figsize=(4, 4))

    # Set x-axis limits to move the bar left
    ax.set_xlim(-0.1, 1)  # Adjust x-axis limits to create more space on the right

    # Stacked bar plot for mutation distribution by interpretation
    bar = ax.bar(
        ['Mutation Distribution'],  # Keep the label "Mutation Distribution"
        [interpretation_counts_million.sum()],
        color='white', edgecolor='black', linewidth=0, width=0.2
    )

    bottom = 0
    colors = [(1, 0, 0, 0.7),  # red with alpha
              (0.5, 0.5, 0.5, 0.7),  # gray with alpha
              (0.68, 0.85, 0.9, 0.7)]  # light blue with alpha

    for i, (label, count) in enumerate(interpretation_counts_million.items()):
        ax.bar(
            ['Mutation Distribution'],  # Keep the label consistent
            [count],
            bottom=bottom,
            label=f'{label} ({count:.2f}M)',
            color=colors[i],
            width=0.15
        )
        percentage = count / (all / 1_000_000 ) * 100
        # Position the label with an arrow
        ax.annotate(
            f'{label}: {count:.2f}M ({percentage:.2f}%) ',
            xy=(0.1, bottom + count / 2),
            xytext=(0.2, bottom + count / 2),  # Position text further to the right
            ha='left',  # Align text to the left of the annotation point
            va='center',
            arrowprops=dict(facecolor='black', arrowstyle='->')
        )
        bottom += count

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Customize x-tick to keep only the label and move bar to the left
    ax.tick_params(axis='x', which='both', bottom=False)
    ax.set_xticks([-0.4])
    # ax.set_xticklabels(['Mutation Distribution'], ha='right', position=(-0.3, 0))

    ax.set_ylabel('Mutations (in Millions)')
    ax.set_title('Distribution of Mutations by Interpretation')
    # ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)

    plt.tight_layout()
    plt.savefig(
        "/dlab/home/norbi/PycharmProjects/AlphaMissense_Stat/processed_data/plots/clinvar/interpretation_distribution_stacked_bar.png")
    plt.show()

def plot_genes_proteome(mutations_in_disorder, mutations_in_ordered, tables_with_only_main_isoforms,filepath=None,figsize=(8, 6)):
    # Filter mutations to only include genes in the human proteome
    main_isoforms = tables_with_only_main_isoforms[tables_with_only_main_isoforms['main_isoform'] == 'yes']
    human_proteome_genes = set(main_isoforms['Protein_ID'])

    mutations_in_disorder = mutations_in_disorder[['Protein_ID', 'Interpretation']].drop_duplicates()
    mutations_in_ordered = mutations_in_ordered[['Protein_ID', 'Interpretation']].drop_duplicates()

    # Extracting mutated genes by interpretation within the human proteome
    pathogenic_genes = set(
        mutations_in_disorder[mutations_in_disorder["Interpretation"] == "Pathogenic"]['Protein_ID']
    ).union(set(mutations_in_ordered[mutations_in_ordered["Interpretation"] == "Pathogenic"][
                    'Protein_ID'])) & human_proteome_genes

    uncertain_genes = set(
        mutations_in_disorder[mutations_in_disorder["Interpretation"] == "Uncertain"]['Protein_ID']
    ).union(set(mutations_in_ordered[mutations_in_ordered["Interpretation"] == "Uncertain"][
                    'Protein_ID'])) & human_proteome_genes

    benign_genes = set(
        mutations_in_disorder[mutations_in_disorder["Interpretation"] == "Benign"]['Protein_ID']
    ).union(set(
        mutations_in_ordered[mutations_in_ordered["Interpretation"] == "Benign"]['Protein_ID'])) & human_proteome_genes

    # Calculate counts of mutated and non-mutated genes for each interpretation
    data = {
        'Interpretation': ['Pathogenic', 'Uncertain', 'Benign'],
        'Mutated': [
            len(pathogenic_genes),
            len(uncertain_genes),
            len(benign_genes)
        ],
        'Non-Mutated': [
            len(human_proteome_genes - pathogenic_genes),
            len(human_proteome_genes - uncertain_genes),
            len(human_proteome_genes - benign_genes)
        ]
    }

    # Calculate total counts for each interpretation for percentages
    total_counts = [mutated + non_mutated for mutated, non_mutated in zip(data['Mutated'], data['Non-Mutated'])]

    # Plotting stacked bar chart
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the bars
    ax.bar(data['Interpretation'], data['Non-Mutated'], label='Non-Mutated', color=COLORS["Non-Mutated"], alpha=0.7)
    ax.bar(data['Interpretation'], data['Mutated'], bottom=data['Non-Mutated'], label='Mutated', color=COLORS["Mutated"], alpha=0.7)

    # Add labels and title
    ax.set_ylabel('Number of Genes')

    # Move the legend outside of the plot
    # ax.legend(title="Gene Status", loc='center left', bbox_to_anchor=(1, 0.5))

    # Display the values and percentages on each segment
    for i, (mutated, non_mutated, total) in enumerate(zip(data['Mutated'], data['Non-Mutated'], total_counts)):
        mutated_pct = mutated / total * 100
        non_mutated_pct = non_mutated / total * 100

        # Show Mutated counts and percentages
        ax.text(i, non_mutated + mutated / 2, f'{mutated}\n ({mutated_pct:.1f}%)', ha='center', color='black',
                )

        # Show Non-Mutated counts and percentages
        ax.text(i, non_mutated / 2, f'{non_mutated}\n ({non_mutated_pct:.1f}%)', ha='center', color='black',
                )

    # Set the title
    plt.title('Distribution of Mutated and Non-Mutated Genes')
    plt.tight_layout()  # Adjust layout to fit the legend
    if filepath:
        plt.savefig(filepath)
    plt.show()

def plot_pdb_and_roi(big_disorder_clinvar,big_order_clinvar,filepath=None,figsize=(6, 3)):
    filter_interpretation = ['Pathogenic', 'Uncertain']

    fig, axes = plt.subplots(1, 1, figsize=figsize)

    # Plot for PDB
    plot_structural_distribution_with_interpretations_new(big_disorder_clinvar, big_order_clinvar,
                                                          info='PDB',
                                                          filter_interpretation=filter_interpretation,
                                                          figsize=(4, 3),
                                                          ax=axes, islegend=False)

    # # Plot for UniProt ROI
    # plot_structural_distribution_with_interpretations_new(big_disorder_clinvar, big_order_clinvar,
    #                                                       info="Roi",
    #                                                       alternative_name="UniProt ROI",
    #                                                       filter_interpretation=filter_interpretation,
    #                                                       figsize=(3, 3),
    #                                                       ax=axes[1], islegend=False)

    # Add a suptitle above both subplots
    plt.suptitle("Structural Coverage of ClinVar Variants")

    plt.tight_layout()  # Adjust layout so suptitle is visible
    if filepath:
        plt.savefig(filepath)
    plt.show()


def plot_main_figures():
    fig_path = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/plots'
    fig1 = os.path.join(fig_path,"fig1")
    fig2 = os.path.join(fig_path,"fig2")
    fig3 = os.path.join(fig_path,"fig3")
    figsm = os.path.join(fig_path,"sm",'clinvar')
    uniprot_dir = os.path.join(fig_path,"sm",'uniprot')

    # # Fig 1A
    # combined_disorder_df = extract_pos_based_df(
    #     pd.read_csv(os.path.join(pos_based_dir, 'combined_dis_pos.tsv'), sep='\t'))
    # # print(combined_disorder_df)
    # combined_disorder_df = combined_disorder_df[
    #     combined_disorder_df['Protein_ID'].isin(tables_with_only_main_isoforms['Protein_ID'])]
    # # plot_disordered_distribution(combined_disorder_df, figsize=(4, 3),filepath=os.path.join(fig1,"A.png"))
    # # exit()
    #
    # # # Fig 1B
    # big_clinvar_final_df = pd.read_csv(f"{to_clinvar_dir}/clinvar_mutations_with_annotations_merged.tsv", sep='\t')
    # big_clinvar_final_df = clear_base_clinvar(big_clinvar_final_df, tables_with_only_main_isoforms)
    # plot_base_distribution(combined_disorder_df, big_clinvar_final_df, figsize=(4, 3),filepath=os.path.join(fig1,"B.png"))
    # exit()
    #
    # mutations_in_disorder = big_clinvar_final_df[big_clinvar_final_df['structure'] == 'disorder']
    # mutations_in_ordered = big_clinvar_final_df[big_clinvar_final_df['structure'] == 'order']
    # #
    # base_table_df = create_base_table(pd.concat([mutations_in_disorder,mutations_in_ordered]),tables_with_only_main_isoforms)
    # print(base_table_df)
    # base_table_df.to_csv(f"{to_clinvar_dir}/base_table.tsv",sep='\t',index=False)
    # exit()

    # Positional Merged Data

    pathogenic_positional_disorder_df = pd.read_csv(
        f"{to_clinvar_dir}/Pathogenic/disorder/positional_clinvar_functional_categorized_final.tsv", sep='\t')
    uncertain_positional_disorder_df = pd.read_csv(
        f"{to_clinvar_dir}/Uncertain/disorder/positional_clinvar_functional_categorized_final.tsv", sep='\t')
    benign_positional_disorder_df = pd.read_csv(
        f"{to_clinvar_dir}/Benign/disorder/positional_clinvar_functional_categorized_final.tsv", sep='\t')
    pathogenic_positional_order_df = pd.read_csv(
        f"{to_clinvar_dir}/Pathogenic/order/positional_clinvar_functional_categorized_final.tsv", sep='\t')
    uncertain_positional_order_df = pd.read_csv(
        f"{to_clinvar_dir}/Uncertain/order/positional_clinvar_functional_categorized_final.tsv", sep='\t')
    benign_positional_order_df = pd.read_csv(
        f"{to_clinvar_dir}/Benign/order/positional_clinvar_functional_categorized_final.tsv", sep='\t')

    pathogenic_positional_disorder_df = clear_base_clinvar(pathogenic_positional_disorder_df,tables_with_only_main_isoforms)
    uncertain_positional_disorder_df = clear_base_clinvar(uncertain_positional_disorder_df,tables_with_only_main_isoforms)
    benign_positional_disorder_df = clear_base_clinvar(benign_positional_disorder_df,tables_with_only_main_isoforms)
    pathogenic_positional_order_df = clear_base_clinvar(pathogenic_positional_order_df,tables_with_only_main_isoforms)
    uncertain_positional_order_df = clear_base_clinvar(uncertain_positional_order_df,tables_with_only_main_isoforms)
    benign_positional_order_df = clear_base_clinvar(benign_positional_order_df,tables_with_only_main_isoforms)

    print(pathogenic_positional_disorder_df.columns)

    # Fig UniprotSM
    # roi_count = calculate_specific_functional_distribution(pathogenic_positional_disorder_df, "Roi")
    # plot_individual_functional_categories(roi_count, plot_dir=uniprot_dir, functional_category="Roi")
    # exit()


    modified_clinvar_disorder_p = pathogenic_positional_disorder_df.copy()
    disorder_categories = ["Only Disorder", 'Disorder Mostly']
    modified_clinvar_disorder_p['Category'] = modified_clinvar_disorder_p['Category'].apply(
        lambda x: "Disorder Specific" if x in disorder_categories else "Non-Disorder Specific")

    # # Fig 3B
    plot_functional_distribution_multicategory_for_all(modified_clinvar_disorder_p, file_name='categorized_functional',
                                                       xlabel=None,
                                                       category_colors=category_colors_structure, figsize=(6, 3),islegend=True,
                                                       filepath=os.path.join(fig3,"B.png"),
                                                       ptm_aggregated=True
                                                       )
    #
    plot_functional_distribution_multicategory_for_ptm(modified_clinvar_disorder_p, file_name='categorized_functional',
                                                       xlabel=None,
                                                       category_colors=category_colors_structure, figsize=(4, 3),islegend=True,
                                                       filepath=os.path.join(figsm,"ptm.png"),
                                                       )

    exit()
    big_disorder_clinvar = pd.concat(
        [pathogenic_positional_disorder_df, uncertain_positional_disorder_df, benign_positional_disorder_df])
    big_order_clinvar = pd.concat(
        [pathogenic_positional_order_df, uncertain_positional_order_df, benign_positional_order_df])


    # # Fig 1B
    # plot_structural_classification(big_disorder_clinvar, big_order_clinvar, positional=False,filepath=os.path.join(figsm,"distribution_mutational.png"), figsize=(8, 3),title="Mutational Distribution")
    # plot_structural_classification(big_disorder_clinvar, big_order_clinvar, positional=True, grouped=False,filepath=os.path.join(figsm,"distribution_positional.png"), figsize=(8, 3),title="Positional Distribution")
    plot_structural_classification(big_disorder_clinvar, big_order_clinvar, positional=True, grouped=True,filepath=os.path.join(fig1,"B1.png"), figsize=(8, 3),title="Positions")

    # exit()

    # Fig 1C
    # plot_genes_proteome(big_disorder_clinvar, big_order_clinvar, tables_with_only_main_isoforms,filepath=os.path.join(fig1,"C.png"), figsize=(5, 4))
    # plot_genes(big_disorder_clinvar, big_order_clinvar,filepath=os.path.join(fig1,"C1.png"), figsize=(8, 3),title="Genes")
    # exit()

    # Fig 3A
    functional_df = create_functional_distribution_df_new(big_disorder_clinvar,grouped=True)
    print(functional_df)
    colors = [
        "#ccd5ae",
        "#cdb4db",
        "#d4a373",
        COLORS['Uncertain']
    ]
    # plot_pie_chart(functional_df,plot_dir=plot_dir,colors=colors)
    plot_pie_chart_three(functional_df,colors=colors,filepath=os.path.join(fig3,"A.png"), figsize=(6, 3))

    # exit()

    # Fig 3C
    plot_pdb_and_roi(big_disorder_clinvar,big_order_clinvar,filepath=os.path.join(fig3,"C.png"),figsize=(4,3))


if __name__ == "__main__":
    """
    ClinVar Base Stats
    """

    core_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations'

    to_clinvar_dir = os.path.join(core_dir,'processed_data/files/clinvar')
    plot_dir = os.path.join(core_dir,'processed_data/plots/clinvar')

    pos_based_dir = os.path.join(core_dir,'data/discanvis_base_files/positional_data_process')

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
    }

    category_colors_structure = {
        'Only Disorder': 'red',
        'Disorder Mostly': COLORS['disorder'],
        'Equal': 'green',
        'Order Mostly': COLORS['order'],
        'Only Order': 'blue',

        'Mostly Disorder': '#f27059',
        'Disorder Specific': '#f27059',
        "Non-Disorder Specific":  COLORS['Uncertain'],
        "Mostly Order/Equal":  COLORS['Uncertain'],

        'disorder': COLORS['disorder'],
        'order': COLORS['order'],
        "Disorder-Pathogenic": COLORS['disorder'],
        'Order-Pathogenic': COLORS['order'],



    }

    category_colors_gene = {
        'Monogenic': '#c9cba3',
        'Multigenic': '#ffe1a8',
        'Complex': '#e26d5c',
    }

    genic_order = [
        "Monogenic", 'Multigenic', 'Complex'
    ]

    plot_main_figures()
    exit()

    # Base Combined-Disorder Stat
    combined_disorder_df = extract_pos_based_df(pd.read_csv(os.path.join(pos_based_dir,'combined_dis_pos.tsv'),sep='\t'))
    # comined_disorder_df = pd.read_csv(os.path.join(pos_based_dir,'final_disorder_pos.tsv'),sep='\t')
    # print(combined_disorder_df)
    # combined_disorder_df = combined_disorder_df[combined_disorder_df['Protein_ID'].isin(tables_with_only_main_isoforms['Protein_ID'])]
    # plot_disordered_distribution(combined_disorder_df)
    # exit()

    # exit()

    # Base Clinvar-Pos Stat
    # big_clinvar_final_df = pd.read_csv(f"{to_clinvar_dir}/clinvar_mutations_with_annotations_merged.tsv",sep='\t')
    # big_clinvar_final_df = clear_base_clinvar(big_clinvar_final_df,tables_with_only_main_isoforms)
    # print(big_clinvar_final_df)

    # print(big_clinvar_final_df)
    # plot_interpretation_distribution(big_clinvar_final_df)
    # plot_base_distribution(combined_disorder_df,big_clinvar_final_df,figsize=(9, 5))

    # exit()

    # mutations_in_disorder = big_clinvar_final_df[big_clinvar_final_df['structure'] == 'disorder']
    # mutations_in_ordered = big_clinvar_final_df[big_clinvar_final_df['structure'] == 'order']

    # base_table_df = create_base_table(pd.concat([mutations_in_disorder,mutations_in_ordered]),tables_with_only_main_isoforms)
    # print(base_table_df)
    # base_table_df.to_csv(f"{to_clinvar_dir}/base_table.tsv",sep='\t',index=False)
    # exit()

    # exit()

    # For Positions

    # mutations_in_disorder = big_clinvar_final_df[(big_clinvar_final_df['structure'] == 'disorder') & (big_clinvar_final_df['category_names'] != 'Unknown')]
    # mutations_in_ordered = big_clinvar_final_df[(big_clinvar_final_df['structure'] == 'order') & (big_clinvar_final_df['category_names'] != 'Unknown')]

    # plot_structural_classification(mutations_in_disorder, mutations_in_ordered, positional=False)
    # plot_structural_classification(mutations_in_disorder, mutations_in_ordered, positional=True, grouped=False)
    # plot_structural_classification(mutations_in_disorder, mutations_in_ordered, positional=True, grouped=True)

    # exit()

    # For Genes

    # mutations_in_disorder = big_clinvar_final_df[big_clinvar_final_df['structure'] == 'disorder']
    # mutations_in_ordered = big_clinvar_final_df[big_clinvar_final_df['structure'] == 'order']

    # plot_genes_proteome(mutations_in_disorder, mutations_in_ordered, tables_with_only_main_isoforms)
    # plot_genes(mutations_in_disorder, mutations_in_ordered)
    # exit()


    # Plot Functional Distributions
    # Base - Disprot - Functional
    # functional_df = create_functional_distribution_df_new(mutations_in_disorder,grouped=True)
    # print(functional_df)
    # plot_pie_chart(functional_df,plot_dir=plot_dir)
    # plot_pie_chart_three(functional_df,plot_dir=plot_dir)

    # exit()

    # Apply the process to create data and plot
    # mutations_in_disordered_pathogenic = mutations_in_disorder[mutations_in_disorder["Interpretation"] == "Pathogenic"]

    # func_counts = create_functional_distribution_counts(mutations_in_disordered_pathogenic)
    # plot_individual_functional_categories(func_counts,plot_dir=plot_dir)
    #
    # roi_count = calculate_specific_functional_distribution(mutations_in_disordered_pathogenic,"Roi")
    # plot_individual_functional_categories(roi_count, plot_dir=plot_dir,functional_category="Roi")
    # exit()

    # Detailed Functional plot



    pathogenic_positional_disorder_df = pd.read_csv(f"{to_clinvar_dir}/Pathogenic/disorder/positional_clinvar_functional_categorized_final.tsv",sep='\t')
    uncertain_positional_disorder_df = pd.read_csv(f"{to_clinvar_dir}/Uncertain/disorder/positional_clinvar_functional_categorized_final.tsv",sep='\t')
    benign_positional_disorder_df = pd.read_csv(f"{to_clinvar_dir}/Benign/disorder/positional_clinvar_functional_categorized_final.tsv",sep='\t')
    pathogenic_positional_order_df = pd.read_csv(f"{to_clinvar_dir}/Pathogenic/order/positional_clinvar_functional_categorized_final.tsv",sep='\t')
    uncertain_positional_order_df = pd.read_csv(f"{to_clinvar_dir}/Uncertain/order/positional_clinvar_functional_categorized_final.tsv",sep='\t')
    benign_positional_order_df = pd.read_csv(f"{to_clinvar_dir}/Benign/order/positional_clinvar_functional_categorized_final.tsv",sep='\t')
    # #
    big_disorder_clinvar = pd.concat([pathogenic_positional_disorder_df,uncertain_positional_disorder_df,benign_positional_disorder_df])
    big_order_clinvar = pd.concat([pathogenic_positional_order_df,uncertain_positional_order_df,benign_positional_order_df])
    # big_disorder_clinvar = big_clinvar_final_df[big_clinvar_final_df['structure'] == "disorder"]
    # big_order_clinvar = big_clinvar_final_df[big_clinvar_final_df['structure'] == "order"]
    big_order_clinvar = clear_base_clinvar(big_order_clinvar, tables_with_only_main_isoforms)
    big_disorder_clinvar = clear_base_clinvar(big_disorder_clinvar, tables_with_only_main_isoforms)

    # PDB
    # plot_structural_distribution_based_on_all(pathogenic_positional_disorder_df,pathogenic_positional_order_df)
    filter_interpretation=['Pathogenic', 'Uncertain']

    # plot_structural_distribution_with_interpretations_new(big_disorder_clinvar,big_order_clinvar,filter_interpretation=filter_interpretation,figsize=(8, 6))
    # plot_structural_distribution_with_interpretations_new(big_disorder_clinvar,big_order_clinvar,info="Roi",alternative_name="UniProt ROI",filter_interpretation=filter_interpretation,figsize=(8, 6))
    # exit()

    # pathogenic_positional_df = pd.read_csv(f"{to_clinvar_dir}/Pathogenic/positional_clinvar_functional_categorized_final.tsv",sep='\t')

    # pathogenic = mutations_in_disorder[mutations_in_disorder['Interpretation'] == "Pathogenic"]

    # plot_functional_distribution_multicategory(pathogenic, file_name='categorized',category_colors=category_colors)
    plot_functional_distribution_multicategory(pathogenic_positional_disorder_df, file_name='categorized',category_colors=category_colors)

    modified_clinvar_disorder_p = pathogenic_positional_disorder_df.copy()
    disorder_categories = ["Only Disorder", 'Disorder Mostly']
    modified_clinvar_disorder_p['Category'] = modified_clinvar_disorder_p['Category'].apply(lambda x: "Mostly Disorder" if x in disorder_categories else "Mostly Order/Equal")


    plot_functional_distribution_multicategory_for_all(modified_clinvar_disorder_p, file_name='categorized_functional',category_colors=category_colors,figsize=(8,5))
    # exit()

    plot_with_disease_genic_type_distribution(pathogenic_positional_disorder_df, pathogenic_positional_order_df, max_count=20, col_to_check="nDisease")
    plot_with_gene_disease_count_distribution(pathogenic_positional_disorder_df, pathogenic_positional_order_df, max_count=10, col_to_check="nDisease")
    exit()


    # plot_monogenic_threeplot(pathogenic_positional_disorder_df,title=f"Distribution of Genes/Disease/Mutations based on Mutation occurence for Diseases (Disorder)")

    # plot_monogenic_threeplot(pathogenic_positional_df[pathogenic_positional_df['category_names'] == 'Unknown'],title=f"Distribution of Genes/Disease/Mutations based on Mutation occurence for Diseases (Unknown)")
    # plot_monogenic_threeplot(pathogenic_positional_df[pathogenic_positional_df['category_names'] == 'Unknown'],title=f"Distribution of Genes/Disease/Mutations based on Mutation occurence for Diseases (Unknown Excluded)")
    # plot_monogenic_threeplot(pathogenic_positional_disorder_df[pathogenic_positional_disorder_df['category_names'] == 'Unknown'],title=f"Distribution of Genes/Disease/Mutations based on Mutation occurence for Diseases (Unknown Disorder)")

    # plot_monogenic_threeplot(pathogenic_positional_df,
    #                          title=f"Distribution of Genes/Disease/Mutations based on Mutation occurence for Diseases")
    exit()

    plot_monogenic_position_plot(pathogenic_positional_df,
                             title=f"Distribution of Genes/Disease/Mutations based on Mutation occurence for Diseases",category_colors=category_colors)

    exit()

    plot_monogenic_threeplot(pathogenic_positional_disorder_df,
                             title=f"Distribution of Genes/Disease/Mutations based on Mutation occurence for Diseases")

    plot_monogenic_position_plot(pathogenic_positional_disorder_df,
                                 title=f"Distribution of Genes/Disease/Mutations based on Mutation occurence for Diseases",
                                 category_colors=category_colors)

    exit()

    for i in pathogenic_positional_disorder_df['genic_category'].unique():
        current_df = pathogenic_positional_disorder_df[pathogenic_positional_disorder_df['genic_category'] == i]
        plot_monogenic_threeplot(current_df,
                                 title=f"Distribution of Genes/Disease/Mutations based on Mutation occurence for {i} Diseases")

        plot_monogenic_position_plot(current_df,
                                     title=f"Distribution of Genes/Disease/Mutations based on Mutation occurence for {i} Diseases",
                                     category_colors=category_colors)



