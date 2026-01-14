import pandas as pd
import os
from  matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm



def extract_pos_based_df(df):
    df[['Protein_ID', 'Position']] = df['AccessionPosition'].str.split('|', expand=True)
    df = df.drop(columns=['AccessionPosition'])
    df['Position'] = df['Position'].astype(int)
    return df


def modify_ptm(df):
    # PTM

    # Step 1: Expand the 'ptm' column by splitting on ';'
    df_expanded = df.assign(ptm=df['ptm'].str.split('; ')).explode('ptm')

    # Step 2: Further split each entry by '|'
    df_expanded[['Type', 'Source']] = df_expanded['ptm'].str.split('|', expand=True)
    df_expanded.drop(columns=['ptm'], inplace=True)

    # Step 3: Pivot the DataFrame
    df_pivot = df_expanded.pivot_table(index=['Protein_ID', 'Position'], columns='Type', values='Source',
                                       aggfunc=lambda x: ', '.join(x.unique())).reset_index()

    # Flatten the columns
    df_pivot.columns.name = None
    df_pivot.columns = [col if isinstance(col, str) else col for col in df_pivot.columns]

    return df_pivot

def aggragate_cols(row):
    full_list_val = []
    cols = []
    for col in cols_to_aggregate:
        val = row[col]
        if pd.notna(val):
            full_list_val.append(str(val))
            cols.append(col)
    str_cols = ';'.join(cols)
    str_vals = ';'.join(full_list_val)

    return str_cols, str_vals


def clean_roi(df):
    notusefullinfo = [
        'Disordered', 'Globular',
        "Tail",
        "Triple-helical region",
        "Nonhelical region",
          "Mucin-like stalk",
          "A","B",
          "Coil 1A",
          "Coil 1B",
          "Coil 1b",
          "Coil 2A",
          "Coil 2B",
          "Coil 2",
                      ]
    contains_not_useful_info = ['repeat']
    df = df[~df['Roi'].isin(notusefullinfo)]
    df = df[~df['Roi'].str.contains('|'.join(contains_not_useful_info))]
    return df


def create_disordered_functional_df_new(pos_based_dir,clinvar_final_df):
    # Load and process each data source
    dibs_mfib_phasepro_binding = extract_pos_based_df(
        pd.read_csv(f"{pos_based_dir}/binding_mfib_phasepro_dibs_pos.tsv", sep='\t'))

    # Load and merge other datasets (ELM and MobiDB)
    elm_df = extract_pos_based_df(pd.read_csv(f"{pos_based_dir}/elm_pos.tsv", sep='\t'))

    print("ELM")
    print(elm_df)

    # Experimental disorder region data (MobiDB)
    mobidb = extract_pos_based_df(pd.read_csv(f"{pos_based_dir}/mobidb_pos.tsv", sep='\t'))

    # PTM
    ptm = modify_ptm(extract_pos_based_df(pd.read_csv(f"{pos_based_dir}/ptm_pos.tsv", sep='\t')))

    # Polymorphism
    # snp = extract_pos_based_df(pd.read_csv(f"{pos_based_dir}/polymorphism_pos.tsv", sep='\t'))
    # common_snp = snp[snp['Polymorphism'] == 'Common Polymorphisms']
    #
    # print("SNP")
    # print(common_snp)

    # PDB
    pdb = pd.read_csv("/dlab/home/norbi/PycharmProjects/AlphaMissense_Stat/processed_data/files/functional_regions/pdb.tsv", sep='\t')
    pdb = pdb[pdb['isdisorder'] == False].rename(columns={'pdb_ids':"PDB"})

    # UniProt Roi
    uni_roi = extract_pos_based_df(pd.read_csv(f"{pos_based_dir}/roi_pos.tsv", sep='\t'))
    uni_roi = clean_roi(uni_roi)

    print("UniProt")
    print(uni_roi)

    # Merge disorder mutations
    mutations_in_disorder_with_annotations = clinvar_final_df.merge(pdb, on=['Protein_ID', 'Position'],how="left")

    mutations_in_disorder_with_annotations = mutations_in_disorder_with_annotations.merge(dibs_mfib_phasepro_binding, on=['Protein_ID', 'Position'],how="left")

    mutations_in_disorder_with_annotations = mutations_in_disorder_with_annotations.merge(elm_df, on=['Protein_ID', 'Position'],how="left")

    mutations_in_disorder_with_annotations = mutations_in_disorder_with_annotations.merge(ptm,
                                                                                          on=['Protein_ID', 'Position'],
                                                                                          how="left")

    # mutations_in_disorder_with_annotations = mutations_in_disorder_with_annotations.merge(common_snp,
    #                                                                                       on=['Protein_ID', 'Position'],
    #                                                                                       how="left")

    mutations_in_disorder_with_annotations = mutations_in_disorder_with_annotations.merge(uni_roi,
                                                                                          on=['Protein_ID', 'Position'],
                                                                                          how="left")

    big_functional_df = mutations_in_disorder_with_annotations.merge(mobidb, on=['Protein_ID', 'Position'],how="left")

    print("Big functional")
    print(big_functional_df)

    return big_functional_df


if __name__ == "__main__":
    """
    Clinvar + Rendezetlen
        Ez hány százaléka az összeshez képest?
        Hány százalékára van funkció?
        Hány százaléka van benne a disportba? (Kisérletesen igazolt rendezetlen regió
    """
    origin = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations"
    to_clinvar_dir = os.path.join(origin,'processed_data/files/clinvar')
    to_am_dir = os.path.join(origin,'processed_data/files/alphamissense')

    # Clinvar

    clinvar_final_df = pd.read_csv(f"{to_clinvar_dir}/clinvar_final.tsv", sep='\t').rename(columns={"Protein_position":"Position"})

    pos_based_dir = os.path.join(origin,'data/discanvis_base_files/positional_data_process')


    # How much percentage got function?

    # Disorder:
        # DIBS, MFIB, PhasePro, ELM, PTM, Polymorphism, UniProt ROI

    big_functional_df = create_disordered_functional_df_new(pos_based_dir,clinvar_final_df)

    # Add info to the cols

    cols_to_aggregate = ["binding_info", "dibs_info", "phasepro_info",
                         "mfib_info", "Elm_Info", "MobiDB", "Acetylation",
                         "Methylation", "Phosphorylation", "Sumoylation",
                         "Ubiquitination",
                         # "Polymorphism",
                         "Roi",
                         'PDB',
                         ]


    big_functional_df[['info', 'info_cols']] = big_functional_df.apply(lambda row: aggragate_cols(row), axis=1, result_type='expand')


    disorder_functional = big_functional_df[big_functional_df['structure'] == "disorder"]
    print("Disorder functional")
    print(disorder_functional)

    # Save the final merged dataframe
    big_functional_df.to_csv(f"{to_clinvar_dir}/clinvar_mutations_with_annotations_merged.tsv", sep='\t', index=False)

    disorder_functional.to_csv(f"{to_clinvar_dir}/disordered_mutations_with_annotations_merged.tsv", sep='\t', index=False)