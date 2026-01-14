import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_protein_id_and_position(path):
    df = pd.read_csv(path, sep='\t')
    return df


def plot_the_count(df,title=""):
    # Plot the distribution for 'ambiguous', 'benign', and 'pathogenic'
    plt.figure(figsize=(10, 6))

    # Plot each class with different colors and labels
    plt.hist(df['ambiguous'], bins=range(0, 20), alpha=0.5, color='grey', label='Ambiguous')
    plt.hist(df['benign'], bins=range(0, 20), alpha=0.5, color='blue', label='Benign')
    plt.hist(df['pathogenic'], bins=range(0, 20), alpha=0.5, color='red', label='Pathogenic')

    # Add titles and labels
    plt.title(f'Distribution of Ambiguous, Benign, and Pathogenic Counts {title}')
    plt.xlabel('Count')
    plt.ylabel('Frequency')
    plt.legend()

    # Show the plot
    plt.show()


def get_stat_for_position(df,path,file_path):
    proteins_pos = df.groupby("Protein_ID")['Position'].apply(list).to_dict()
    # print(proteins_pos)

    lst = []

    # Open the file and filter based on positions
    with open(path, 'r') as f:
        next(f)
        for line in tqdm(f):
            line_data = line.strip().split('\t')  # Assuming tab-separated values in the file
            protein_id = line_data[0]
            position = int(line_data[3])
            variant = line_data[2]
            am_pathogenicity = line_data[4]

            # Check if the current protein_id exists in the dictionary and if the position is in the list
            if protein_id in proteins_pos and position in proteins_pos[protein_id]:
                lst.append([protein_id, position,variant,am_pathogenicity])

            # if len(lst) > 500000:
            #     break


    am_df = pd.DataFrame(lst, columns=['Protein_ID','Position','variant','am_pathogenicity']).drop_duplicates()
    am_df = am_df.drop(columns=['variant'])

    am_df['am_class'] = np.where(am_df['am_pathogenicity'].astype(float) >= 0.5,"pathogenic","benign")

    print(am_df)

    # Group by 'Protein_ID', 'Position', and 'am_class' and count occurrences
    count_df = am_df.groupby(['Protein_ID', 'Position', 'am_class']).size().reset_index(name='count')

    # Pivot the table to have 'am_class' as columns and counts as values
    pivot_df = count_df.pivot_table(index=['Protein_ID', 'Position'], columns='am_class', values='count', fill_value=0)

    # Reset the index to make 'Protein_ID' and 'Position' regular columns
    pivot_df = pivot_df.reset_index()

    final_df = pivot_df.merge(df, on=['Protein_ID', 'Position'], how='inner')

    print(final_df)
    final_df.to_csv(file_path, sep='\t',index=False)
    return final_df

def generate_files(disorder_class_count_path,order_class_count_path):
    am_disorder_path = os.path.join(am_dir, 'am_disorder.tsv')
    am_order_path = os.path.join(am_dir, 'am_order.tsv')

    # Count the number of lines (residues) and unique genes in each file
    disordered_df = get_protein_id_and_position(am_disorder_path)

    disorder_class_count_df = get_stat_for_position(disordered_df, am_full_path, disorder_class_count_path)

    ordered_df = get_protein_id_and_position(am_order_path)
    order_class_count_df = get_stat_for_position(ordered_df, am_full_path, order_class_count_path)


def plot_distributions(disorder_class_count_path,order_class_count_path):
    disorder_class_count_df = pd.read_csv(disorder_class_count_path, sep='\t')
    plot_the_count(disorder_class_count_df, "Disorder")

    order_class_count_df = pd.read_csv(order_class_count_path, sep='\t')

    plot_the_count(order_class_count_df, "Order")


def create_plot_count(df, ax, title="",legend=True,show_xlabel=True,show_ylabel=True):
    # Plot the distribution for 'ambiguous', 'benign', and 'pathogenic' on a single subplot
    # ax.hist(df['ambiguous'], bins=range(0, 21), alpha=0.5, color='grey', label='Ambiguous')
    # ax.hist(df['benign'], bins=range(0, 21), alpha=0.5, color='blue', label='Benign')
    ax.hist(df['pathogenic'], bins=range(0, 21), alpha=0.5, color='red', label='Pathogenic')

    ax.set_title(title,fontsize=14)

    if show_xlabel:
        ax.set_xlabel('Count')
    else:
        ax.set_xticklabels([])

    if show_ylabel:
        ax.set_ylabel('Frequency')
        ax.yaxis.set_label_coords(-0.35, 0.5)
    if legend:
        ax.legend()


def plot_patogenic_distributions(disorder_class_count_path,order_class_count_path,disorder_clinvar_path,order_clinvar_path):
    disorder_class_count_df = pd.read_csv(disorder_class_count_path, sep='\t')
    clinvar_df_disorder = pd.read_csv(disorder_clinvar_path, sep='\t').rename(columns={'Protein_position':'Position'})


    print(clinvar_df_disorder)

    for i in clinvar_df_disorder['Interpretation'].unique():
        pathogenic = clinvar_df_disorder[clinvar_df_disorder['Interpretation'] == i]
        filtered_df = pathogenic[['Protein_ID', 'Position']].merge(disorder_class_count_df,
                                                    on=['Protein_ID', 'Position'],
                                                    how='inner')
        title = f"{i} Disorder"
        plot_the_count(filtered_df,title)

    order_class_count_df = pd.read_csv(order_class_count_path, sep='\t')
    clinvar_df_order = pd.read_csv(order_clinvar_path, sep='\t').rename(columns={'Protein_position': 'Position'})


    for i in clinvar_df_order['Interpretation'].unique():
        pathogenic = clinvar_df_order[clinvar_df_order['Interpretation'] == i]
        filtered_df = pathogenic[['Protein_ID', 'Position']].merge(order_class_count_df,
                                                    on=['Protein_ID', 'Position'],
                                                    how='inner')
        title = f"{i} Order"
        plot_the_count(filtered_df,title)


def plot_patogenic_distributions_structural(disorder_class_count_path, order_class_count_path, figsize=(9, 6)):
    disorder_class_count_df = pd.read_csv(disorder_class_count_path, sep='\t')
    order_class_count_df = pd.read_csv(order_class_count_path, sep='\t')

    # Create a single figure with 6 subplots
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    ax = axs[0]
    create_plot_count(disorder_class_count_df, ax, f"Disorder")

    ax = axs[1]
    create_plot_count(order_class_count_df, ax, f"Order",legend=False)

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_patogenic_distributions_one_fig(disorder_class_count_path, order_class_count_path,
                                         disorder_clinvar_path, order_clinvar_path,
                                         figsize=(8, 12), legend=True,dir_path=None):
    # Read data
    disorder_class_count_df = pd.read_csv(disorder_class_count_path, sep='\t')
    order_class_count_df = pd.read_csv(order_class_count_path, sep='\t')
    clinvar_df_disorder = pd.read_csv(disorder_clinvar_path, sep='\t').rename(columns={'Protein_position': 'Position'})
    clinvar_df_order = pd.read_csv(order_clinvar_path, sep='\t').rename(columns={'Protein_position': 'Position'})

    # Desired order of categories:
    row_labels = ["Proteome", "Pathogenic", "Uncertain", "Benign"]
    # Desired interpretations for rows 1-3:
    desired_interps = ["Pathogenic", "Uncertain", "Benign"]

    # Extract interpretations and filter to desired order
    disorder_interps = clinvar_df_disorder['Interpretation'].unique()
    order_interps = clinvar_df_order['Interpretation'].unique()
    interpretations = [interp for interp in desired_interps if interp in disorder_interps and interp in order_interps]

    # Create a figure with 4 rows and 2 columns = 8 subplots total
    # Column 0 = Order, Column 1 = Disorder
    # Row 0 = Proteome
    # Rows 1-3 = Pathogenic, Uncertain, Benign (if present)
    fig, axs = plt.subplots(4, 2, figsize=figsize)
    fig.suptitle("AlphaMissense Classification Positional Distribution", fontsize=14)

    # ------------------ Proteome Row (Row 0) ------------------
    # Left = Proteome Disorder, Right = Proteome Order
    # Show titles only here
    create_plot_count(disorder_class_count_df, axs[0, 0], "Disorder",
                      legend=legend, show_xlabel=False)
    axs[0, 0].text(-0.55, 0.5, row_labels[0], rotation=90, va='center', ha='center', transform=axs[0, 0].transAxes,fontsize=14)

    create_plot_count(order_class_count_df, axs[0, 1], "Order",
                      legend=legend, show_xlabel=False, show_ylabel=False)
    if legend:
        legend = False  # Turn off legend after the first subplot


    # ------------------ ClinVar Rows (Rows 1-3) ------------------
    # For each interpretation, plot order on the left and disorder on the right
    # No subplot titles in these rows, just the data and row labels on the left y-axis
    for i, interpretation in enumerate(interpretations, start=1):
        # Disorder (right column is actually index 0 or 1?
        # According to previous code, left column = order, right column = disorder or vice versa?
        # The user stated: "the left should be order the right should be disorder"
        # So column 0 = order, column 1 = disorder

        # Disorder
        pathogenic_disorder = clinvar_df_disorder[clinvar_df_disorder['Interpretation'] == interpretation]
        filtered_disorder_df = pathogenic_disorder[['Protein_ID', 'Position']].merge(
            disorder_class_count_df, on=['Protein_ID', 'Position'], how='inner'
        )
        create_plot_count(filtered_disorder_df, axs[i, 0], "",
                          legend=(i == 3), show_xlabel=(i == 3), show_ylabel=True)

        # Order
        pathogenic_order = clinvar_df_order[clinvar_df_order['Interpretation'] == interpretation]
        filtered_order_df = pathogenic_order[['Protein_ID', 'Position']].merge(
            order_class_count_df, on=['Protein_ID', 'Position'], how='inner'
        )
        create_plot_count(filtered_order_df, axs[i, 1], "",
                          legend=legend, show_xlabel=(i == 3), show_ylabel=False)



        # Set the y-axis label for this row on the left subplot
        # axs[i, 0].set_ylabel(row_labels[i])
        axs[i, 0].text(-0.55, 0.5, row_labels[i], rotation=90, va='center', ha='center', transform=axs[i, 0].transAxes,fontsize=14)

    # For any interpretation rows that don't exist (e.g., if we had fewer than 3),
    # they remain blank. If needed, you could hide them or label them differently.

    # Adjust layout
    plt.tight_layout()
    if dir_path:
        plt.savefig(os.path.join(dir_path,"A.png"))
    plt.show()

if __name__ == "__main__":
    # Define the directory and file paths
    am_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/alphamissense'
    am_disorder_path = os.path.join(am_dir, 'am_disorder.tsv')
    am_order_path = os.path.join(am_dir, 'am_order.tsv')

    am_full_path = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/data/discanvis_base_files/alphamissense/processed_alphamissense_results_mapping_new.tsv'

    disorder_class_count_path = f'{am_dir}/am_disorder_class_count.tsv'
    order_class_count_path = f'{am_dir}/am_order_class_count.tsv'

    # generate_files(disorder_class_count_path, order_class_count_path)

    clinvar_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/clinvar'
    disorder_clinvar_path = os.path.join(clinvar_dir, 'clinvar_disorder.tsv')
    order_clinvar_path = os.path.join(clinvar_dir, 'clinvar_order.tsv')

    # Fig 5A
    # plot_patogenic_distributions_structural(disorder_class_count_path, order_class_count_path, figsize=(8, 4))
    # exit()

    # Fig 5B
    fig4 = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/plots/fig4'
    plot_patogenic_distributions_one_fig(disorder_class_count_path, order_class_count_path, disorder_clinvar_path,order_clinvar_path,
                                         figsize=(6, 6),legend=False,dir_path=fig4)


    # plot_patogenic_distributions(disorder_class_count_path, order_class_count_path, disorder_clinvar_path,order_clinvar_path)



