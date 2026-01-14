# TODO
# Filter by Poly?
# Generate tables:
# Positional, Gene

import os
import pandas as pd



def group_for_position(df):

    group_lst = [
        'Protein_ID','Position',
        # 'DOID',
       # 'DO Subset',
        'nDisease',
        "disease_category", "Rare","Developmental",
        "category_names", 'genic_category','structure',
       'Interpretation', 'Category', 'info', 'info_cols',
    ]
    positional_df = df.groupby(group_lst).size().reset_index(name='mutation_count')
    return positional_df

def group_for_gene(df):

    group_lst = [
        'Protein_ID',
        # 'DOID',
       # 'DO Subset',
        'nDisease',
        "disease_category", "Rare","Developmental",
        "category_names", 'genic_category','structure',
       'Interpretation', 'Category',
    ]

    def aggregate_non_empty(series):
        combined_values = set()
        for val in series.dropna().astype(str):
            # Split each value by commas and update the set
            if val:
                combined_values.update(val.split(','))
        # Join the unique values with a comma
        return ', '.join(sorted(combined_values))

    # Aggregate info and info_cols
    positional_df = df.groupby(group_lst).agg({
        'info': aggregate_non_empty,  # Aggregate info
        'info_cols': aggregate_non_empty  # Aggregate info_cols
    }).reset_index()



    # Add mutation_count by counting the occurrences within each group
    positional_df['mutation_count'] = df.groupby(group_lst).size().values


    return positional_df

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':

    cols_to_keep = [
        'Protein_ID','GeneSymbol','Name', 'Position','HGVSp_Short',
       'ClinicalSignificance', 'RS# (dbSNP)', 'RCVaccession',
       'PositionVCF',
        # 'DOID',
        'ID',
        'Disease',
       'DO Subset', 'nDisease',
        "disease_category","Rare", "Developmental",
        "category_names",
        'genic_category','structure',
       'Interpretation', 'Category', 'info', 'info_cols',
    ]

    generate_path = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/clinvar'


    full_file_path = f'{generate_path}/clinvar_mutations_with_annotations_merged.tsv'
    df = pd.read_csv(full_file_path,sep='\t')

    final_df = df[cols_to_keep].drop_duplicates()

    for interpretation in df['Interpretation'].unique():

        interpretation_path = f'{generate_path}/{interpretation}'
        create_dir(interpretation_path)

        current_interpret_df = final_df[final_df['Interpretation'] == interpretation]

        current_interpret_df = current_interpret_df.fillna("-")

        to_file_name = f"clinvar_functional_categorized_final.tsv"
        current_interpret_df.to_csv(f'{interpretation_path}/{to_file_name}',sep='\t',index=False)


        positional_df = group_for_position(current_interpret_df)
        positional_df.to_csv(f'{interpretation_path}/positional_{to_file_name}', sep='\t', index=False)

        gene_df = group_for_gene(current_interpret_df)
        gene_df.to_csv(f'{interpretation_path}/gene_{to_file_name}', sep='\t', index=False)

        for structure in current_interpret_df['structure'].unique():
            structure_path = f'{interpretation_path}/{structure}'
            create_dir(structure_path)

            structure_df = current_interpret_df[current_interpret_df['structure'] == structure]
            structure_pos_df = group_for_position(structure_df)
            structure_gene_df = group_for_gene(structure_df)

            structure_df.to_csv(f'{structure_path}/{to_file_name}', sep='\t', index=False)
            structure_pos_df.to_csv(f'{structure_path}/positional_{to_file_name}', sep='\t', index=False)
            structure_gene_df.to_csv(f'{structure_path}/gene_{to_file_name}', sep='\t', index=False)