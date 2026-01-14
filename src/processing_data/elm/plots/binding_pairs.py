import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function 1: Binding pairs with mutations in motif and binding domain (pathogenic and predicted pathogenic)

def get_group_sizes(df):
    group_sizes = (
        df.groupby(['Disease', 'ELM_ID', 'Protein_ID_Motif'])
        .size()
        .reset_index(name='Group_Size')
    )
    bins = [0, 1, 2, 5, 10, 20, 50, 100, float('inf')]
    bin_labels = ['1', '2', '3-5', '6-10', '11-20', '21-50', '51-100', '100+']
    group_sizes['Group_Size_Binned'] = pd.cut(group_sizes['Group_Size'], bins=bins, labels=bin_labels, right=True)
    grouped = group_sizes['Group_Size_Binned'].value_counts().reset_index()
    grouped.columns = ['Group_Size_Binned', 'Count']
    return grouped.sort_values('Group_Size_Binned')


def plot_binding_pairs_and_group_mut_distribution(
    df_known_pathogenic,
    df_known_predicted_pathogenic,
    df_predicted_pathogenic,
    df_predicted_predicted_pathogenic,
    output_dir
):
    """
    Generate a bar plot showing the count of ELM-Domain pairs per category,
    with side-by-side comparison of Mutation-Based and PPI-Based approaches.
    """

    # Assign categories
    datasets = [
        (df_known_pathogenic, "Known Pathogenic"),
        (df_known_predicted_pathogenic, "Known PP"),
        (df_predicted_pathogenic, "PEM Pathogenic"),
        (df_predicted_predicted_pathogenic, "PEM PP")
    ]

    all_data = []
    for df, category in datasets:
        for ppi_based in [True, False]:
            count = df[df["PPI_Based"] == ppi_based].shape[0]
            all_data.append({"Category": category, "Count": count, "Type": "PPI" if ppi_based else "Mutation-Based"})

    df_plot = pd.DataFrame(all_data)

    # Plot
    plt.figure(figsize=(4, 4))
    ax = sns.barplot(data=df_plot, x="Category", y="Count", hue="Type", palette="Set2")

    # Add numbers on top of bars
    for p in ax.containers:
        ax.bar_label(p, fmt='%.0f', label_type='edge', padding=3)

    # Increase ylim dynamically
    max_count = df_plot["Count"].max()
    plt.ylim(0, max_count * 1.15)

    plt.title("ELM-Domain Pairs")
    plt.xlabel("")
    plt.ylabel("Number of Cases")
    plt.xticks(rotation=45)
    plt.legend(title="Approach")

    # Save and show
    output_path = os.path.join(output_dir, "binding_pairs_summary.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def plot_ppi_vs_non_ppi_binding_group_distribution(df, output_dir, category="Known Pathogenic"):
    """
    Generate a bar plot showing the Binding Partner Group Size Distribution
    for PPI-Based and Non-PPI-Based approaches.

    :param df: DataFrame containing ELM-Domain pairs with a 'PPI_Based' column.
    :param output_dir: Directory to save the output plot.
    :param category: Category name for labeling.
    """

    df['Category'] = category

    # Separate the data based on PPI_Based column
    df_ppi = df[df["PPI_Based"] == True]
    df_non_ppi = df[df["PPI_Based"] == False]

    # Get partner group distributions
    grouped_ppi = get_group_sizes(df_ppi)
    grouped_ppi['Type'] = 'PPI'

    grouped_non_ppi = get_group_sizes(df_non_ppi)
    grouped_non_ppi['Type'] = 'Mutation-Pair'

    # Combine both distributions
    grouped_data = pd.concat([grouped_ppi, grouped_non_ppi])

    # Create a bar plot
    plt.figure(figsize=(4, 4))
    sns.barplot(data=grouped_data, x='Group_Size_Binned', y='Count', hue='Type', palette="Set2")

    # Formatting
    plt.title(f'{category} Binding Partners')
    plt.xlabel('Number of Binding Partners')
    plt.ylabel('Number of Cases')
    plt.xticks(rotation=90)
    plt.legend(title="Approach")

    # Save and show
    output_path = os.path.join(output_dir, f'ppi_vs_nonppi_binding_group_distribution_{category}.png')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

# Function 2: Top mutated binding pairs
def plot_top_mutated_binding_pairs(df_known, df_predicted, output_dir,mutated_type='Pathogenic'):
    """
    Plot two independent subplots showing the top mutated binding pairs:
    - Left: Known binding pairs.
    - Right: Predicted binding pairs.
    """

    # Add a combined mutation count column for Known Data
    # df_known['Total_Mutations'] = df_known['Mutations_in_Motif'] + df_known['Mutations_in_Domain']
    df_known['Binding_Pair'] = df_known['Protein_ID_Motif'] + ' - ' + df_known['Protein_ID_Domain']
    grouped_known = df_known.groupby('Binding_Pair')['Mutations_in_Motif'].sum().reset_index().drop_duplicates()
    top_known = grouped_known.nlargest(10, 'Mutations_in_Motif')

    # Add a combined mutation count column for Predicted Data
    # df_predicted['Total_Mutations'] = df_predicted['Mutations_in_Motif'] + df_predicted['Mutations_in_Domain']
    df_predicted['Binding_Pair'] = df_predicted['Protein_ID_Motif'] + ' - ' + df_predicted['Protein_ID_Domain']
    grouped_predicted = df_predicted.groupby('Binding_Pair')['Mutations_in_Motif'].sum().reset_index()
    top_predicted = grouped_predicted.nlargest(10, 'Mutations_in_Motif')

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)

    # Left Subplot: Known
    sns.barplot(data=top_known, x='Mutations_in_Motif', y='Binding_Pair', ax=axes[0], palette="Blues_r")
    axes[0].set_title('Known')
    axes[0].set_xlabel('Total Number of Mutations')
    axes[0].set_ylabel('')

    # Right Subplot: Predicted
    sns.barplot(data=top_predicted, x='Mutations_in_Motif', y='Binding_Pair', ax=axes[1], palette="Greens_r")
    axes[1].set_title('PEM')
    axes[1].set_xlabel('Total Number of Mutations')
    axes[1].set_ylabel('')  # Remove duplicate ylabel for clean display

    # Set overall title and layout
    plt.suptitle(f"Top Mutated Binding Pairs ({mutated_type})")
    plt.tight_layout()

    # Save and show the plot
    output_path = os.path.join(output_dir, f'top_mutated_binding_pairs_{mutated_type}.png')
    plt.savefig(output_path)
    plt.show()




# Function 3: Top mutated ELMs and top mutated binding domains

def plot_top_mutated_elms_and_domains(df, output_dir, category="Known Pathogenic", ntop=10):
    df['Category'] = category

    # Separate the data based on PPI_Based column
    df_ppi = df[df["PPI_Based"] == True].copy()
    df_ppi['Type'] = 'PPI'
    df_non_ppi = df[df["PPI_Based"] == False].copy()
    df_non_ppi['Type'] = 'Mutation-Based'

    ## ---- TOP MUTATED ELMS ---- ##
    elm_grouped_ppi = df_ppi.groupby(['ELM_ID']).size().reset_index(name='Count')
    elm_grouped_ppi['Type'] = 'PPI'

    elm_grouped_mut = df_non_ppi.groupby(['ELM_ID']).size().reset_index(name='Count')
    elm_grouped_mut['Type'] = 'Mutation-Based'

    top_elms_ppi = elm_grouped_ppi.nlargest(ntop, 'Count')
    top_elms_mut = elm_grouped_mut.nlargest(ntop, 'Count')

    top_elms = pd.concat([top_elms_ppi, top_elms_mut])

    ## ---- TOP MUTATED BINDING DOMAINS ---- ##
    domain_grouped_ppi = df_ppi.groupby(['Pfam_ID', 'Domain_name']).size().reset_index(name='Count')
    domain_grouped_ppi['Type'] = 'PPI'

    domain_grouped_mut = df_non_ppi.groupby(['Pfam_ID', 'Domain_name']).size().reset_index(name='Count')
    domain_grouped_mut['Type'] = 'Mutation-Based'

    top_domains_ppi = domain_grouped_ppi.nlargest(ntop, 'Count')
    top_domains_mut = domain_grouped_mut.nlargest(ntop, 'Count')

    top_domains = pd.concat([top_domains_ppi, top_domains_mut])

    # Create a 2x2 subplot layout
    fig, ax = plt.subplots(2, 2, figsize=(6, 4), sharex='col', gridspec_kw={'height_ratios': [1, 1]})

    # ELM PPI-Based (Top)
    sns.barplot(data=top_elms_ppi, x='Count', y='ELM_ID', hue='Type', palette="Blues", ax=ax[0, 0])
    ax[0, 0].set_title('PPI-Based')
    ax[0, 0].set_ylabel('')
    ax[0, 0].set_xlabel('')
    ax[0, 0].legend([], [], frameon=False)
    for p in ax[0, 0].containers:
        ax[0, 0].bar_label(p, fmt='%d', padding=3)

    # ELM Mutation-Based (Top)
    sns.barplot(data=top_elms_mut, x='Count', y='ELM_ID', hue='Type', palette="Reds", ax=ax[0, 1])
    ax[0, 1].set_title('Mutation-Based')
    ax[0, 1].set_ylabel('')
    ax[0, 1].set_xlabel('')
    ax[0, 1].legend([], [], frameon=False)
    for p in ax[0, 1].containers:
        ax[0, 1].bar_label(p, fmt='%d', padding=3)

    # Domain PPI-Based (Top)
    sns.barplot(data=top_domains_ppi, x='Count', y='Domain_name', hue='Type', palette="Blues", ax=ax[1, 0])
    ax[1, 0].set_title('')
    ax[1, 0].set_ylabel('')
    ax[1, 0].set_xlabel('Number of Cases')
    ax[1, 0].legend([], [], frameon=False)
    for p in ax[1, 0].containers:
        ax[1, 0].bar_label(p, fmt='%d', padding=3)


    # Domain Mutation-Based (Top)
    sns.barplot(data=top_domains_mut, x='Count', y='Domain_name', hue='Type', palette="Reds", ax=ax[1, 1])
    ax[1, 1].set_title('')
    ax[1, 1].set_ylabel('')
    ax[1, 1].set_xlabel('Number of Cases')
    ax[1, 1].legend([], [], frameon=False)
    for p in ax[1, 1].containers:
        ax[1, 1].bar_label(p, fmt='%d', padding=3)

    ax[1, 0].set_xlim(0, top_domains_ppi['Count'].max() * 1.3)
    ax[1, 1].set_xlim(0, top_domains_mut['Count'].max() * 1.3)


    plt.suptitle(f"Top Mutated ELMs and Binding Domains ({category})")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the title position
    plt.savefig(os.path.join(output_dir, 'top_mutated_elms_and_domains.png'))
    plt.show()



def plot_binding_pairs_and_group_distribution(
        df_known_pathogenic,
        df_known_predicted_pathogenic,
        df_predicted_pathogenic,
        df_predicted_predicted_pathogenic,
        output_dir
):
    """
    Generate four subplots comparing Binding Partner Group Size Distribution
    for PPI-Based and Mutation-Pair-Based approaches across four categories.
    """
    datasets = {
        "Known Pathogenic": df_known_pathogenic,
        "Known PP": df_known_predicted_pathogenic,
        "PEM Pathogenic": df_predicted_pathogenic,
        "PEM PP": df_predicted_predicted_pathogenic,
    }

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    i = 0
    legend = True
    for (title, df), ax in zip(datasets.items(), axes.flatten()):
        df_ppi = df[df["PPI_Based"] == True]
        df_non_ppi = df[df["PPI_Based"] == False]

        grouped_ppi = get_group_sizes(df_ppi)
        grouped_ppi['Type'] = 'PPI-Based'

        grouped_non_ppi = get_group_sizes(df_non_ppi)
        grouped_non_ppi['Type'] = 'Mutation-Pair-Based'

        grouped_data = pd.concat([grouped_ppi, grouped_non_ppi])

        sns.barplot(data=grouped_data, x='Group_Size_Binned', y='Count', hue='Type', palette="Set2",
                    ax=ax)
        ax.set_title(title)
        row, col = divmod(i, 2)
        if legend:
            ax.legend(title="Approach")
        else:
            ax.get_legend().set_visible(False)

        if row == 0:
            axes[row, col].set_xlabel("")
            if col == 0:
                legend = False
                axes[row, col].set_ylabel("Number of Partners")
            else:
                axes[row, col].set_ylabel("")
        else:
            if col == 1:
                axes[row, col].set_ylabel("")
            axes[row, col].set_xlabel('Binding Partner Group Size')


        i +=1
        # ax.set_xlabel('Binding Partner Group Size')
        # ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=90)






    plt.suptitle("Binding Partner Group Size Distribution")
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'binding_pairs_group_distribution.png')
    plt.savefig(output_path)
    plt.show()


if __name__ == '__main__':

    base_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations'

    binding_pairs_dir = f'{base_dir}/processed_data/files/elm/ppi_based'

    output_dir = f'{base_dir}/plots/figexample'

    # Load the binding pairs
    elm_pfam_disease_pairs_elm_known_mut_pathogenic = pd.read_csv(f'{binding_pairs_dir}/elm_pfam_disease_pairs_elm_known_mut_pathogenic.tsv', sep='\t')
    elm_pfam_disease_pairs_elm_known_mut_predicted_pathogenic = pd.read_csv(f'{binding_pairs_dir}/elm_pfam_disease_pairs_elm_known_mut_predicted_pathogenic.tsv', sep='\t')
    elm_pfam_disease_pairs_elm_predicted_mut_pathogenic = pd.read_csv(f'{binding_pairs_dir}/elm_pfam_disease_pairs_elm_predicted_mut_pathogenic.tsv', sep='\t')
    elm_pfam_disease_pairs_elm_predicted_mut_predicted_pathogenic = pd.read_csv(f'{binding_pairs_dir}/elm_pfam_disease_pairs_elm_predicted_mut_predicted_pathogenic.tsv', sep='\t')


    # # Generate plots
    # plot_binding_pairs_and_group_mut_distribution(
    #     elm_pfam_disease_pairs_elm_known_mut_pathogenic,
    #     elm_pfam_disease_pairs_elm_known_mut_predicted_pathogenic,
    #     elm_pfam_disease_pairs_elm_predicted_mut_pathogenic,
    #     elm_pfam_disease_pairs_elm_predicted_mut_predicted_pathogenic,
    #     output_dir
    # )
    #
    # plot_binding_pairs_and_group_distribution(
    #     elm_pfam_disease_pairs_elm_known_mut_pathogenic,
    #     elm_pfam_disease_pairs_elm_known_mut_predicted_pathogenic,
    #     elm_pfam_disease_pairs_elm_predicted_mut_pathogenic,
    #     elm_pfam_disease_pairs_elm_predicted_mut_predicted_pathogenic,
    #     output_dir
    # )
    #
    # plot_ppi_vs_non_ppi_binding_group_distribution(elm_pfam_disease_pairs_elm_known_mut_pathogenic,output_dir)
    # plot_ppi_vs_non_ppi_binding_group_distribution(elm_pfam_disease_pairs_elm_predicted_mut_pathogenic,output_dir,"PEM Pathogenic")

    # exit()

    # plot_top_mutated_binding_pairs(
    #     elm_pfam_disease_pairs_elm_known_mut_pathogenic,
    #     elm_pfam_disease_pairs_elm_predicted_mut_pathogenic,
    #     output_dir
    # )
    # # exit()
    # plot_top_mutated_binding_pairs(
    #     elm_pfam_disease_pairs_elm_known_mut_predicted_pathogenic,
    #     elm_pfam_disease_pairs_elm_predicted_mut_predicted_pathogenic,
    #     output_dir, mutated_type='Predicted Pathogenic'
    # )
    #
    # exit()
    plot_top_mutated_elms_and_domains( elm_pfam_disease_pairs_elm_known_mut_pathogenic,output_dir,ntop=5)
    # plot_top_mutated_elms_and_domains( elm_pfam_disease_pairs_elm_predicted_mut_predicted_pathogenic,output_dir,ntop=5, category="PEM Pathogenic")
