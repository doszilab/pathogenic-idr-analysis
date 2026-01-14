import pandas as pd


def process_data(df):
    # Remove rows where 'nDisease' is "-"
    df = df[df['nDisease'] != "-"]

    disorder = df['structure'] == 'disorder'
    order = df['structure'] == 'order'

    only_disorder = disorder & ~order
    only_order = ~disorder & order

    # Grouping and aggregating data
    grouped_df = df.groupby(['nDisease', 'category_names', 'genic_category']).agg(
        structures_with_mutation=('structure', lambda x: ','.join(list(sorted(x.unique())))),
        disorder_only_proteins=('Protein_ID', lambda x: ','.join(x[only_disorder].unique())),
        order_only_proteins=('Protein_ID', lambda x: ','.join(x[only_order].unique())),
        both_disorder_order_proteins=('Protein_ID', lambda x: ','.join(x.unique())),
        total_number_of_proteins=('Protein_ID', lambda x: x.nunique()),
        disorder_number_of_proteins=('Protein_ID', lambda x: x[only_disorder].nunique()),
        order_number_of_proteins=('Protein_ID', lambda x: x[only_order].nunique()),
        disorder_mutation_count=('mutation_count', lambda x: x[df.loc[x.index, 'structure'] == 'disorder'].sum()),
        order_mutation_count=('mutation_count', lambda x: x[df.loc[x.index, 'structure'] == 'order'].sum()),
        disorder_info=('info', lambda x: '/'.join(x[df['structure'] == 'disorder'].unique())),
        disorder_info_cols=('info_cols', lambda x: '/'.join(x[df['structure'] == 'disorder'].unique())),
        order_info=('info', lambda x: '/'.join(x[df['structure'] == 'order'].unique())),
        order_info_cols=('info_cols', lambda x: '/'.join(x[df['structure'] == 'order'].unique()))
    ).reset_index()

    return grouped_df


if __name__ == "__main__":

    base_path = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations"
    base_clinvar = f"{base_path}/processed_data/files/clinvar"

    interpretations = ['Benign','Uncertain','Pathogenic']

    for interpretation in interpretations:
        df = pd.read_csv(f"{base_clinvar}/{interpretation}/gene_clinvar_functional_categorized_final.tsv",sep='\t')
        gene_df = process_data(df)
        gene_df.to_csv(f"{base_clinvar}/{interpretation}/clinvar_disease_pairs.tsv",sep='\t',index=False)

        multigenic_df = gene_df[(gene_df['genic_category'] == "Multigenic") & (gene_df['structures_with_mutation'].str.contains("disorder"))]

        multigenic_df.to_csv(f"{base_clinvar}/{interpretation}/clinvar_disease_pairs_multigenic.tsv",sep='\t',index=False)