import os
import subprocess
import pandas as pd


def make_fasta_file(df):
    fasta = ''
    for index, row in df.iterrows():
        fasta += f'>{row["Protein_ID"]}\n{row["Sequence"]}\n'
    return fasta

if __name__ == "__main__":

    base_dir_path = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations'

    file_path = os.path.join(base_dir_path,'processed_data','files')
    benchmark_path = os.path.join(file_path,'benchmark')
    clinvar_order = os.path.join(benchmark_path,'clinvar_order_all.tsv')
    clinvar_disorder = os.path.join(benchmark_path,'clinvar_disorder_all.tsv')

    clinvar_disorder_df_unique = set(pd.read_csv(clinvar_disorder, sep='\t',usecols=['Protein_ID'])['Protein_ID'])
    clinvar_order_df_unique = set(pd.read_csv(clinvar_order, sep='\t',usecols=['Protein_ID'])['Protein_ID'])

    proteins = clinvar_disorder_df_unique | clinvar_order_df_unique
    print(len(proteins))
    print(len(clinvar_disorder_df_unique))
    print(len(clinvar_order_df_unique))

    table_file = os.path.join(base_dir_path,'data','discanvis_base_files','sequences','loc_chrom_with_names_main_isoforms_with_seq.tsv')
    tables_with_only_main_isoforms = pd.read_csv(table_file, sep='\t',usecols=['Protein_ID','Sequence'])

    disorder_set_df = tables_with_only_main_isoforms[tables_with_only_main_isoforms['Protein_ID'].isin(clinvar_disorder_df_unique)]
    order_set_df = tables_with_only_main_isoforms[tables_with_only_main_isoforms['Protein_ID'].isin(clinvar_order_df_unique)]
    all_set_df = tables_with_only_main_isoforms[tables_with_only_main_isoforms['Protein_ID'].isin(proteins)]

    fasta_files_path = os.path.join(benchmark_path, 'fasta_files')

    file_dct = {
        "disorder_set": disorder_set_df,
        "order_set": order_set_df,
        "all_set": all_set_df,
    }

    for file_name, df in file_dct.items():
        fasta = make_fasta_file(df)
        fasta_path = os.path.join(fasta_files_path, f"{file_name}.fasta", )
        with open(fasta_path, 'w') as f:
            f.write(fasta)

        make_db_command = f"/home/nosyfire/diamond makedb --in {fasta_path} -d {file_name}"
        subprocess.call(make_db_command, shell=True)

        matches_path = os.path.join(fasta_files_path, f"{file_name}_matches.tsv")
        run_blast_command = f'/home/nosyfire/diamond cluster -d {file_name} --approx-id {40} -o {matches_path} -M 32G'
        subprocess.call(run_blast_command, shell=True)

