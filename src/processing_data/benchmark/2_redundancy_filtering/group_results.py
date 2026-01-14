import os
import pandas as pd
from sklearn.utils import resample

def drop_first_ones(df):
    df = df[~((df['protein_variant'].str.contains("M1") )& (df['protein_variant'].str.len() == 3))]
    return df

def balance_classes(df):
    balanced_dfs = []
    for protein_id in df['Protein_ID'].unique():
        protein_df = df[df['Protein_ID'] == protein_id]
        pathogenic_df = protein_df[protein_df['Interpretation'] == 'Pathogenic']
        benign_df = protein_df[protein_df['Interpretation'] == 'Benign']

        min_class_size = min(len(pathogenic_df), len(benign_df))
        if min_class_size > 0:
            resampled_pathogenic_df = resample(pathogenic_df, n_samples=min_class_size, random_state=42)
            resampled_benign_df = resample(benign_df, n_samples=min_class_size, random_state=42)
            balanced_protein_df = pd.concat([resampled_pathogenic_df, resampled_benign_df])
            balanced_dfs.append(balanced_protein_df)

    balanced_df = pd.concat(balanced_dfs).reset_index()
    return balanced_df

def make_balance_dataset(clinvar_df):
    clinvar_df = drop_first_ones(clinvar_df)
    clinvar_df_nonempty = clinvar_df.dropna(subset=['AlphaMissense'],how='any')
    balanced_df = balance_classes(clinvar_df_nonempty)
    return balanced_df


def count_variant_scores(df, columns):
    # Count how many scores (non-NaN) are present for each row
    df['score_count'] = df[columns].notna().sum(axis=1)

    # Count how many methods (columns) have no NaN for the entire row
    df['method_count'] = (df[columns].notna().all(axis=0)).sum()

    return df[['Protein_ID', 'score_count', 'method_count']]

def create_groups(matches_df,balanced_df,mut_df,columns = [
        "AlphaMissense","AlphaMissense_pos"]):

    balanced_count = balanced_df["Protein_ID"].value_counts().reset_index().rename(columns={'count':'balanced_count'})
    all_count = mut_df["Protein_ID"].value_counts().reset_index().rename(columns={'count':'mut_count'})
    protein_df = matches_df.copy()

    counted_df = count_variant_scores(balanced_df, columns)

    passed_proteins = []

    for prot in protein_df['Protein_ID'].unique():
        current_prot = protein_df[protein_df['Protein_ID'] == prot]
        if current_prot.shape[0] == 1:
            passed_proteins.append([prot,'Group'])
            continue

        proteins = list(set(current_prot['Protein_ID']) | set(current_prot['Protein_ID_match']))
        current_prot_df = pd.DataFrame(proteins, columns=['Protein_ID'])


        current_prot_df = current_prot_df.merge(balanced_count, on='Protein_ID', how='left')
        current_prot_df = current_prot_df.merge(all_count, on='Protein_ID', how='left')

        current_prot_df = current_prot_df.dropna(subset=['balanced_count'])
        if current_prot_df.empty:
            continue
        elif current_prot_df.shape[0] == 1:
            passed_proteins.append([prot, 'Balanced Count'])
            continue

        current_prot_df = current_prot_df.merge(counted_df, on='Protein_ID', how='left')

        current_prot_df = current_prot_df.sort_values(by=['balanced_count','mut_count','method_count','score_count', ], ascending=False)
        final_protein = current_prot_df['Protein_ID']
        passed_proteins.append([final_protein,'Mutation Count'])

    passed_protein_df = pd.DataFrame(passed_proteins, columns=['Protein_ID','Passed_by'])

    non_redundant_df = balanced_df[balanced_df['Protein_ID'].isin(passed_protein_df['Protein_ID'])].drop(columns=["index"])
    non_redundant_df = non_redundant_df.drop(columns=['method_count','score_count'])

    print("Balanced")
    print(balanced_df)
    print("Balanced non Redundant")
    print(non_redundant_df)
    non_redundant_df.drop_duplicates(subset=['Protein_ID', 'protein_variant'], inplace=True)

    return non_redundant_df


def main(fasta_files_path,file_name, mut_df,output_path):

    print(file_name)
    matches_path = os.path.join(fasta_files_path, f"{file_name}_matches.tsv")
    matches_df = pd.read_csv(matches_path, sep='\t', header=None)
    matches_df.columns = ['Protein_ID', 'Protein_ID_match']

    balanced_df = make_balance_dataset(mut_df)
    print(balanced_df)
    non_redundant_df = create_groups(matches_df, balanced_df, mut_df)
    print(non_redundant_df)
    non_redundant_df.to_csv(output_path, sep='\t', index=False)

if __name__ == "__main__":

    file_path = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/'
    benchmark_path = os.path.join(file_path,'benchmark')
    clinvar_order = os.path.join(benchmark_path,'clinvar_order_all.tsv')
    clinvar_disorder = os.path.join(benchmark_path,'clinvar_disorder_all.tsv')

    clinvar_disorder_df = pd.read_csv(clinvar_disorder, sep='\t').drop(columns=['RCVaccession']).drop_duplicates()
    clinvar_order_df = pd.read_csv(clinvar_order, sep='\t').drop(columns=['RCVaccession']).drop_duplicates()


    # clinvar_disorder_df.drop_duplicates(subset=['Protein_ID','protein_variant'],inplace=True)
    # clinvar_order_df.drop_duplicates(subset=['Protein_ID','protein_variant'],inplace=True)


    clinvar_disorder_df_unique = set(clinvar_disorder_df['Protein_ID'])
    clinvar_order_df_unique = set(clinvar_order_df['Protein_ID'])

    fasta_files_path = os.path.join(benchmark_path, 'fasta_files')

    file_dct = {
        "disorder_set": clinvar_disorder_df,
        "order_set": clinvar_order_df,
    }

    main(fasta_files_path,"disorder_set",clinvar_disorder_df,os.path.join(benchmark_path,'clinvar_disorder_all_non_redundant.tsv'))
    main(fasta_files_path,"order_set",clinvar_order_df,os.path.join(benchmark_path,'clinvar_order_all_non_redundant.tsv'))