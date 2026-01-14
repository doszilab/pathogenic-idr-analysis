import pandas as pd
import os
from  matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm


def create_structural_classification(big_clinvar, disorder_df):
    # Expand disordered regions to individual positions
    # def expand_disordered_regions(df):
    #     positions = []
    #     for _, row in tqdm(df.iterrows(),total=df.shape[0]):
    #         positions.extend([(row['Protein_ID'], pos) for pos in range(row['Start'], row['End'] + 1)])
    #     return pd.DataFrame(positions, columns=['Protein_ID', 'Position'])

    # Expand disordered regions
    # expanded_disorder = expand_disordered_regions(disorder_df)
    expanded_disorder = disorder_df[disorder_df['CombinedDisorder'] == 1].drop(columns=['CombinedDisorder'])
    expanded_order = disorder_df[disorder_df['CombinedDisorder'] == 0].drop(columns=['CombinedDisorder'])

    # Merge the big_clinvar DataFrame with expanded disorder positions to identify disordered mutations
    merged_disordered = pd.merge(big_clinvar, expanded_disorder, on=['Protein_ID', 'Position'], how='inner')
    merged_disordered = merged_disordered.drop_duplicates()

    # Find ordered mutations (not in disordered regions)
    merged_ordered = pd.merge(big_clinvar, expanded_order, on=['Protein_ID', 'Position'], how='inner')
    merged_ordered = merged_ordered.drop_duplicates()

    print("Mutations in Ordered Regions:")
    print(merged_ordered)
    print("\nMutations in Disordered Regions:")
    print(merged_disordered)

    return merged_disordered, merged_ordered





def extract_pos_based_df(df):
    df[['Protein_ID', 'Position']] = df['AccessionPosition'].str.split('|', expand=True)
    df = df.drop(columns=['AccessionPosition'])
    df['Position'] = df['Position'].astype(int)
    return df


def plot_functional_site_distribution(df_list, column_name,tilte):
    # Combine all dataframes into one
    combined_df = pd.concat(df_list)

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=combined_df, x='fname', y=column_name, linewidth=1.5)

    # benign_cutoff = 0.34
    # half_cutoff = 0.5
    # pathogen_cutoff = 0.564
    #
    # plt.axhline(y=benign_cutoff, color='blue', linestyle='--', linewidth=2, label=f'Benign ({benign_cutoff})')
    # plt.axhline(y=half_cutoff, color='grey', linestyle='--', linewidth=2, label='0.5')
    # plt.axhline(y=pathogen_cutoff, color='red', linestyle='--', linewidth=2, label=f'Pathogen ({pathogen_cutoff})')

    # Set labels and title
    plt.xlabel('Site Type')
    plt.ylabel('Distribution of Values')
    plt.title(tilte)

    plt.tight_layout()

    # Show the plot
    plt.show()




def alphamissense_main():
    to_alphamissense_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/alphamissense'

    pos_based_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/data/discanvis_base_files/positional_data_process'
    am_df = extract_pos_based_df(pd.read_csv(f"{pos_based_dir}/alphamissense_pos.tsv", sep='\t'))
    print(am_df)
    # exit()

    # Combined Disorder
    disorder_df = pd.read_csv(os.path.join(pos_based_dir,"CombinedDisorderNew_Pos.tsv"),sep="\t")
    print(disorder_df)

    am_disorder, am_order = create_structural_classification(am_df, disorder_df)
    am_disorder.to_csv(f"{to_alphamissense_dir}/am_disorder.tsv", sep='\t', index=False)
    am_order.to_csv(f"{to_alphamissense_dir}/am_order.tsv", sep='\t', index=False)

    # am_order = pd.read_csv(f"{to_alphamissense_dir}/am_order.tsv", sep='\t', )
    # am_disorder = pd.read_csv(f"{to_alphamissense_dir}/am_disorder.tsv", sep='\t', )

    am_order['fname'] = "Ordered"
    am_disorder['fname'] = "Disordered"

    df_list = [am_order, am_disorder]
    plot_functional_site_distribution(df_list, "AlphaMissense", f'Distribution of AM score')



if __name__ == "__main__":
    """
    Clinvar + Rendezetlen
        Ez hány százaléka az összeshez képest?
        Hány százalékára van funkció?
        Hány százaléka van benne a disportba? (Kisérletesen igazolt rendezetlen regió
    """

    alphamissense_main()

