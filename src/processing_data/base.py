import os

"""
In this script I define the base files and functions for the projects

We use for the analysis the Gencode 44 version transcripts
We use AlphaMissense scores from AlphaMissense paper
"""

discanvis_base_files = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/data/discanvis_base_files"
MAIN_ISOFORM_TABLE = os.path.join(discanvis_base_files,"sequences", "loc_chrom_with_names_main_isoform")
MAIN_ISOFORM_TABLE_SEQ = os.path.join(discanvis_base_files,"sequences", "loc_chrom_with_names_main_isoforms_with_seq.tsv")


pos_data = os.path.join(discanvis_base_files,"positional_data_process")

# from : /dlab/home/norbi/PycharmProjects/DisCanVis_Data_Process/Processed_Data/gencode_process/positional_data_process
binding_mfib_phasepro_dibs_pos = os.path.join(pos_data,"binding_mfib_phasepro_dibs_pos.tsv")
elm_pos = os.path.join(pos_data,"elm_pos.tsv")
mobidb_pos = os.path.join(pos_data,"mobidb_pos.tsv")
pdb_ids_disorder = os.path.join(pos_data,"pdb_ids_disorder.tsv")
ptm_pos = os.path.join(pos_data,"ptm_pos.tsv")
roi_pos = os.path.join(pos_data,"roi_pos.tsv")
alphamissense_pos = os.path.join(pos_data,"alphamissense_pos.tsv")


clinvar_path = os.path.join(pos_data,"clinvar")

# ClinVar
# from : /dlab/home/norbi/PycharmProjects/Cancer_Visualization_Data_Process/18_ClinVar/data/gencode_process/
clinvar_mapped_with_ontology = os.path.join(clinvar_path,"clinvar_mapped_with_ontology.tsv")


# Benchmarking

# from : /dlab/home/norbi/PycharmProjects/AlphaMissense_Stat/processed_data/files/benchmark
dbNFSP_cleaned = os.path.join(discanvis_base_files,"benchmark","dbNFSP_cleaned.tsv")
dbNFSP_pos = os.path.join(discanvis_base_files,"benchmark","dbNFSP_pos.tsv")


# from : /dlab/home/norbi/PycharmProjects/Mapping/pythonProject/Processed_Data/mave_experiments
proteingym = os.path.join(discanvis_base_files,"benchmark","processed_proteingym_results_hg38_mapping_isoforms.tsv")




"""
Columns
"""

protein_column = 'Protein_ID'


"""
Base Functions
"""

def extract_pos_based_df(df):
    df[['Protein_ID', 'Position']] = df['AccessionPosition'].str.split('|', expand=True)
    df = df.drop(columns=['AccessionPosition'])
    df['Position'] = df['Position'].astype(int)
    return df


