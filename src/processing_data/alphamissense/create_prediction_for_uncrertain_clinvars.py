import pandas as pd
import os


def get_predicted_pathogenic_tables_pos_based(clinvar_disorder,clinvar_order, am_order, am_disorder, disorder_cutoff, order_cutoff):
    order_uncertain = clinvar_order[clinvar_order['Interpretation'] == 'Uncertain']
    disorder_uncertain = clinvar_disorder[clinvar_disorder['Interpretation'] == 'Uncertain']

    merged_disorder = disorder_uncertain.merge(am_disorder, on=['Protein_ID', 'Position'])
    merged_order = order_uncertain.merge(am_order, on=['Protein_ID', 'Position'])

    likely_pathogenic_disorder = merged_disorder[merged_disorder['AlphaMissense'] >= disorder_cutoff].drop_duplicates()
    likely_pathogenic_order = merged_order[merged_order['AlphaMissense'] >= order_cutoff].drop_duplicates()

    return likely_pathogenic_disorder, likely_pathogenic_order


def get_predicted_pathogenic_tables(clinvar_order,clinvar_disorder,am_df,disorder_cutoff,order_cutoff):
    order_uncertain = clinvar_order[clinvar_order['Interpretation'] == 'Uncertain']
    disorder_uncertain = clinvar_disorder[clinvar_disorder['Interpretation'] == 'Uncertain']

    order_uncertain['protein_variant'] = order_uncertain['HGVSp_Short'].str[2:]
    disorder_uncertain['protein_variant'] = disorder_uncertain['HGVSp_Short'].str[2:]

    merged_ordered = order_uncertain.merge(am_df, on=['Protein_ID', 'Position', 'protein_variant']).dropna(subset=['AlphaMissense'])
    merged_disordered = disorder_uncertain.merge(am_df, on=['Protein_ID', 'Position', 'protein_variant']).dropna(subset=['AlphaMissense'])

    likely_pathogenic_disorder = merged_disordered[merged_disordered['AlphaMissense'] >= disorder_cutoff].drop_duplicates()
    likely_pathogenic_order = merged_ordered[merged_ordered['AlphaMissense'] >= order_cutoff].drop_duplicates()

    return likely_pathogenic_disorder, likely_pathogenic_order



if __name__ == '__main__':
    core_dir = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations"
    plots = os.path.join(core_dir, "processed_data", "plots")
    files = os.path.join(core_dir, "processed_data", "files")

    clinvar_path_disorder = f'{files}/clinvar/Uncertain/disorder/clinvar_functional_categorized_final.tsv'
    clinvar_path_order = f'{files}/clinvar/Uncertain/order/clinvar_functional_categorized_final.tsv'


    # disorder_cutoff = 0.469
    disorder_cutoff = 0.564
    disorder_cutoff_pos = 0.5
    # new_disorder_cutoff = 0.429

    # order_cutoff = 0.564
    order_cutoff = 0.564
    order_cutoff_pos = 0.5
    # new_order_cutoff = 0.592

    clinvar_order = pd.read_csv(clinvar_path_order, sep='\t').rename(columns={"Protein_position": "Position"})
    clinvar_disorder = pd.read_csv(clinvar_path_disorder, sep='\t').rename(columns={"Protein_position": "Position"})

    print(clinvar_order.columns)
    print(clinvar_disorder.columns)

    am_disorder = pd.read_csv(f'{files}/alphamissense/am_disorder.tsv', sep='\t')
    am_order = pd.read_csv(f'{files}/alphamissense/am_order.tsv',sep='\t')
    #
    likely_pathogenic_disorder, likely_pathogenic_order = get_predicted_pathogenic_tables_pos_based(clinvar_disorder,clinvar_order, am_order, am_disorder, disorder_cutoff_pos, order_cutoff_pos)
    #
    likely_pathogenic_disorder.to_csv(f'{files}/alphamissense/clinvar/likely_pathogenic_disorder_pos_based.tsv',sep='\t', index=False)
    likely_pathogenic_order.to_csv(f'{files}/alphamissense/clinvar/likely_pathogenic_order_pos_based.tsv',sep='\t', index=False)

    exit()


    am_all = os.path.join(core_dir, "data", "discanvis_base_files",'alphamissense','processed_alphamissense_results_mapping_new.tsv')

    am_df = (pd.read_csv(am_all, sep='\t', usecols=['Protein_ID', 'protein_variant', 'pos', 'am_pathogenicity'],
                         low_memory=False)
             .rename(columns={"am_pathogenicity": "AlphaMissense", 'pos': 'Position'}))


    likely_pathogenic_disorder, likely_pathogenic_order = get_predicted_pathogenic_tables(clinvar_order,clinvar_disorder,am_df,disorder_cutoff,order_cutoff)
    #
    likely_pathogenic_disorder.to_csv(f'{files}/alphamissense/clinvar/likely_pathogenic_disorder.tsv',sep='\t', index=False)
    likely_pathogenic_order.to_csv(f'{files}/alphamissense/clinvar/likely_pathogenic_order.tsv',sep='\t', index=False)


    #
    # print(am_df)
    #
    # likely_pathogenic_disorder, likely_pathogenic_order = get_predicted_pathogenic_tables(clinvar_order, clinvar_disorder, am_df, new_disorder_cutoff, new_order_cutoff)
    # print(likely_pathogenic_disorder)
    # print(likely_pathogenic_order)

    # likely_pathogenic_disorder.to_csv(f'{files}/alphamissense/clinvar/likely_pathogenic_disorder.tsv', sep='\t', index=False)
    # likely_pathogenic_order.to_csv(f'{files}/alphamissense/clinvar/likely_pathogenic_order.tsv',sep='\t', index=False)
