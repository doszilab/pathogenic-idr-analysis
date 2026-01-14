import pandas as pd
import os

if __name__ == "__main__":
    core_dir = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations"
    files = os.path.join(core_dir, "processed_data", "files")

    to_clinvar_dir = os.path.join(files, 'clinvar')
    # final_df = pd.read_csv(os.path.join(to_clinvar_dir, 'clinvar_final.tsv'), sep='\t')
    final_df = pd.read_csv(os.path.join(to_clinvar_dir, 'clinvar_with_do_categories.tsv'), sep='\t')

    disease_related_cols = ['PhenotypeIDS','PhenotypeList',  'ID','Disease', 'nDisease', 'disease_group',
                            'DO Subset', 'disease_category', 'Rare','Developmental',
                            'Mixed', 'Final_Category', 'Result_Found_by', 'category_names',]

    # ----- 1) Create a groupby object to count mutations per disease category -----
    # If you want the total number of rows for each category:
    category_counts = (final_df.groupby('nDisease')['ID'].count().reset_index(name='mutation_count'))

    # ----- 2) Filter final_df to disease-related columns and drop duplicates -----
    disease_df = final_df[disease_related_cols].drop_duplicates(keep='first')

    # ----- 3) Merge the count information back into disease_df -----
    disease_df = disease_df.merge(category_counts, on='nDisease', how='left')

    # ----- 4) Save result to disk -----
    disease_df.to_csv(os.path.join(to_clinvar_dir, 'clinvar_diseases.tsv'),
                      sep='\t', index=False)