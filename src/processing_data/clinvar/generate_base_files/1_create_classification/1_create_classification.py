import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import re



def apply_one_star(df):
    df = df[df['ReviewStar']>=1]
    return df


def filter_clinvar_pathogen(clinvar_mapped_df):
    new_clinvar_mapped = clinvar_mapped_df[
        (clinvar_mapped_df["ClinicalSignificance"] == "Pathogenic") |
        (clinvar_mapped_df["ClinicalSignificance"] == "Likely pathogenic") |
        (clinvar_mapped_df["ClinicalSignificance"] == "Pathogenic/Likely pathogenic") |
        (clinvar_mapped_df["ClinicalSignificance"] == 'Pathogenic; risk factor') |
        (clinvar_mapped_df["ClinicalSignificance"] == 'Likely pathogenic/Likely risk allele') |
        (clinvar_mapped_df["ClinicalSignificance"] == 'Pathogenic; protective') |
        (clinvar_mapped_df["ClinicalSignificance"] == 'Pathogenic/Likely pathogenic/Likely risk allele')
    ]
    print(clinvar_mapped_df['ClinicalSignificance'].unique().tolist())
    print(new_clinvar_mapped['ClinicalSignificance'].unique().tolist())
    return new_clinvar_mapped

def filter_clinvar_uncertain(clinvar_mapped_df):
    new_clinvar_mapped = clinvar_mapped_df[
        (clinvar_mapped_df["ClinicalSignificance"].str.contains("Uncertain")) |
        (clinvar_mapped_df["ClinicalSignificance"].str.contains("Conflicting interpretations of pathogenicity")) |
        (clinvar_mapped_df["ClinicalSignificance"].str.contains("not provided"))
    ]
    print(clinvar_mapped_df['ClinicalSignificance'].unique().tolist())
    print(new_clinvar_mapped['ClinicalSignificance'].unique().tolist())
    return new_clinvar_mapped

def filter_clinvar_benign(clinvar_mapped_df):
    new_clinvar_mapped = clinvar_mapped_df[
        (clinvar_mapped_df["ClinicalSignificance"].str.contains("benign")) |
        (clinvar_mapped_df["ClinicalSignificance"].str.contains("Benign"))
                                          ]
    print(clinvar_mapped_df['ClinicalSignificance'].unique().tolist())
    print(new_clinvar_mapped['ClinicalSignificance'].unique().tolist())
    return new_clinvar_mapped


def exclude_coiled_coils(cleaned_clinvar,coiled_coil_df):

    coiled_coil_df_positions = coiled_coil_df[['Protein_ID',"Position"]].rename(columns={
        "Position":"Protein_position"
    })

    final_clinvar = (cleaned_clinvar.merge(coiled_coil_df_positions, on=['Protein_ID',"Protein_position"], how='left', indicator=True)
     .query('_merge == "left_only"')
     .drop('_merge', axis=1))


    return final_clinvar

def create_clinvar_df(clinvar_mapped_df):

    clinvar_pathogen = filter_clinvar_pathogen(clinvar_mapped_df)
    clinvar_uncertain = filter_clinvar_uncertain(clinvar_mapped_df)
    clinvar_benign = filter_clinvar_benign(clinvar_mapped_df)


    clinvar_pathogen = apply_one_star(clinvar_pathogen)
    clinvar_benign = apply_one_star(clinvar_benign)

    clinvar_pathogen["Interpretation"] = "Pathogenic"
    clinvar_uncertain["Interpretation"] = "Uncertain"
    clinvar_benign["Interpretation"] = "Benign"

    big_clinvar = pd.concat([clinvar_pathogen, clinvar_uncertain, clinvar_benign])
    return big_clinvar, clinvar_pathogen, clinvar_uncertain, clinvar_benign

def create_structural_classification(big_clinvar, disorder_df):
    # Expand disordered regions to individual positions
    def expand_disordered_regions(df):
        positions = []
        for _, row in tqdm(df.iterrows(),total=df.shape[0]):
            positions.extend([(row['Protein_ID'], pos) for pos in range(row['Start'], row['End'] + 1)])
        return pd.DataFrame(positions, columns=['Protein_ID', 'Protein_position'])

    # Expand disordered regions
    # expanded_disorder = expand_disordered_regions(disorder_df)
    expanded_disorder = disorder_df[disorder_df['CombinedDisorder'] == 1].drop(columns=['CombinedDisorder']).rename(columns={
        "Position":"Protein_position"
    })
    expanded_order = disorder_df[disorder_df['CombinedDisorder'] == 0].drop(columns=['CombinedDisorder']).rename(
        columns={
            "Position": "Protein_position"
        })


    # Merge the big_clinvar DataFrame with expanded disorder positions to identify disordered mutations
    merged_disordered = pd.merge(big_clinvar, expanded_disorder, on=['Protein_ID', 'Protein_position'], how='inner')
    merged_disordered = merged_disordered.drop_duplicates()

    # Find ordered mutations (not in disordered regions)
    merged_ordered = pd.merge(big_clinvar, expanded_order, on=['Protein_ID', 'Protein_position'], how='inner')
    merged_ordered = merged_ordered.drop_duplicates()

    print("All Mutations:")
    print(big_clinvar)

    print("Mutations in Ordered Regions:")
    print(merged_ordered)
    print("\nMutations in Disordered Regions:")
    print(merged_disordered)

    # Merge with the expanded disorder DataFrame and add 'structure' column
    big_clinvar['structure'] = 'order'  # Default to 'order'

    # Identify disordered mutations
    disordered_mutations = pd.merge(big_clinvar, expanded_disorder, on=['Protein_ID', 'Protein_position'], how='inner')
    disordered_indices = disordered_mutations.index

    # Update the 'structure' column to 'disorder' for the identified disordered mutations
    big_clinvar.loc[big_clinvar.set_index(['Protein_ID', 'Protein_position']).index.isin(disordered_mutations.set_index(['Protein_ID', 'Protein_position']).index), 'structure'] = 'disorder'

    print("All Mutations with Structure Classification:")
    print(big_clinvar[['Protein_ID', 'Protein_position', 'structure']])

    return merged_disordered, merged_ordered, big_clinvar





def extract_pos_based_df(df):
    df[['Protein_ID', 'Position']] = df['AccessionPosition'].str.split('|', expand=True)
    df = df.drop(columns=['AccessionPosition'])
    df['Position'] = df['Position'].astype(int)
    return df



def compute_functional_distribution(mutations_in_disorder, big_functional_df):
    # Merging functional data with disordered mutations

    merged_df = mutations_in_disorder.merge(big_functional_df, on=['Protein_ID', 'Position'], how='left', suffixes=('', '_func'))

    # Filtering out functional mutations from the disordered set
    disordered_only = merged_df[merged_df['fname'].isna()]

    # Aggregating counts for functional mutations by function and interpretation
    functional_distribution = big_functional_df.groupby(['fname', 'Interpretation'])['Protein_ID'].count().reset_index()
    functional_distribution.columns = ['Function', 'Interpretation', 'Count']

    # Aggregating counts for disordered mutations by interpretation
    disordered_distribution = disordered_only.groupby('Interpretation')['Protein_ID'].count().reset_index()
    disordered_distribution.columns = ['Interpretation', 'Disordered_Count']

    # Merging functional and disordered counts by interpretation
    combined_distribution = pd.merge(functional_distribution, disordered_distribution, on='Interpretation', how='left')

    return combined_distribution



def create_functional_distribtuion_df(big_functional_df):

    disprot = big_functional_df[big_functional_df['fname'] == "DisProt"][
        ['Protein_ID', 'Position', 'Interpretation']].drop_duplicates()
    functional_ones = big_functional_df[big_functional_df['fname'] != "DisProt"][
        ['Protein_ID', 'Position', 'Interpretation']].drop_duplicates()

    functional_disprot = functional_ones.merge(disprot, on=['Protein_ID', 'Position', 'Interpretation'])
    merged = functional_ones.merge(disprot, on=['Protein_ID', 'Position', 'Interpretation'], how='left',
                                   indicator=True)
    functional_ones_not_in_disprot = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

    # Merge disordered mutations with DisProt mutations
    mutations_in_disorder_disprot = mutations_in_disorder.merge(disprot,
                                                                on=['Protein_ID', 'Position', 'Interpretation'],
                                                                how='inner')

    # Find mutations not in DisProt
    merged = mutations_in_disorder.merge(disprot, on=['Protein_ID', 'Position', 'Interpretation'], how='left',
                                         indicator=True)
    mutations_not_in_disprot = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

    lst = []
    for i in mutations_in_disorder_disprot['Interpretation'].unique():
        n_functional_disprot = functional_disprot[functional_disprot['Interpretation'] == i].shape[0]
        n_functional_ones_not_in_disprot = functional_ones[functional_ones['Interpretation'] == i].shape[0]
        n_mutations_in_disorder_disprot = \
        mutations_in_disorder_disprot[mutations_in_disorder_disprot['Interpretation'] == i].shape[0]
        n_mutations_not_in_disprot = \
        mutations_not_in_disprot[mutations_not_in_disprot['Interpretation'] == i].shape[0]
        lst.append([i, n_functional_disprot, n_functional_ones_not_in_disprot, n_mutations_in_disorder_disprot,
                    n_mutations_not_in_disprot])

    functional_df = pd.DataFrame(lst, columns=['Interpretation', 'n_functional_disprot',
                                               'n_functional_ones_not_in_disprot',
                                               'n_mutations_in_disorder_disprot', 'n_mutations_not_in_disprot'])

    return functional_df


def create_disordered_functional_df():
    dibs_mfib_phasepro_binding = extract_pos_based_df(
        pd.read_csv(f"{pos_based_dir}/binding_mfib_phasepro_dibs_pos.tsv", sep='\t'))
    mutations_in_disorder_with_annotations = mutations_in_disorder.merge(dibs_mfib_phasepro_binding,on=['Protein_ID', 'Position'])
    mfib_df = mutations_in_disorder_with_annotations[mutations_in_disorder_with_annotations['mfib_info'].notna()]
    mfib_df['fname'] = "MFIB"
    dibs_df = mutations_in_disorder_with_annotations[mutations_in_disorder_with_annotations['dibs_info'].notna()]
    dibs_df['fname'] = "DIBS"
    phasepro_df = mutations_in_disorder_with_annotations[mutations_in_disorder_with_annotations['phasepro_info'].notna()]
    phasepro_df['fname'] = "PhasePro"
    elm_df = extract_pos_based_df(pd.read_csv(f"{pos_based_dir}/elm_pos.tsv", sep='\t'))
    elm_with_mutations = mutations_in_disorder.merge(elm_df,on=['Protein_ID', 'Position'])
    elm_with_mutations['fname'] = "ELM"

    # How much percentage under experimental disordered region?
        # MobiDB
    mobidb = extract_pos_based_df(pd.read_csv(f"{pos_based_dir}/mobidb_pos.tsv", sep='\t'))
    mutations_in_disorder_with_experimental_disorder = mutations_in_disorder.merge(mobidb,on=['Protein_ID', 'Position'])
    print(mutations_in_disorder_with_experimental_disorder)
    mutations_in_disorder_with_experimental_disorder['fname'] = "DisProt"
    big_functional_df = pd.concat([mfib_df,dibs_df,phasepro_df,elm_with_mutations,mutations_in_disorder_with_experimental_disorder])
    big_functional_df.to_csv(f"{to_clinvar_dir}/disordered_mutations_with_annotations.tsv",sep='\t',index=False)

def create_structural_classification_for_protein(big_clinvar, disorder_df, info_df):
    # Calculate the distribution of disordered regions for each protein
    def calculate_disordered_distribution(df):
        df['Disordered_Region_Length'] = df['End'] - df['Start'] + 1
        distribution = df.groupby('Protein_ID')['Disordered_Region_Length'].sum().reset_index()
        return distribution

    # Calculate disordered region distribution
    disordered_distribution = calculate_disordered_distribution(disorder_df)
    print(disorder_df)
    print(disordered_distribution)

    # Merge big_clinvar with info_df and disordered distribution
    clinvar_info_df = info_df[info_df['Protein_ID'].isin(big_clinvar['Protein_ID'])]
    print(clinvar_info_df)
    print(info_df)
    exit()
    clinvar_info_df_disorder = clinvar_info_df.merge(disordered_distribution, on='Protein_ID', how='inner')

    # Fill NaN values with 0
    clinvar_info_df_disorder['Disordered_Region_Length'] = clinvar_info_df_disorder['Disordered_Region_Length'].fillna(0)

    print("Clinvar Info with Disordered Region Lengths:")
    print(clinvar_info_df_disorder)

    clinvar_info_df_disorder['Seq_len'] = clinvar_info_df_disorder['Sequence'].str.len()
    clinvar_info_df_disorder['Ordered_Region_Length'] = clinvar_info_df_disorder['Seq_len'] - clinvar_info_df_disorder['Disordered_Region_Length']
    clinvar_info_df_disorder['Percent_Disordered_Region'] = clinvar_info_df_disorder['Disordered_Region_Length'] / clinvar_info_df_disorder['Seq_len']


    return clinvar_info_df_disorder



def clinvar_add_stars_to_df(clinvar_mapped_df):
    """
    four	practice guideline
    three	reviewed by expert panel
    two	criteria provided, multiple submitters, no conflicts
    one	criteria provided, conflicting classifications
    one	criteria provided, single submitter
    none	no assertion criteria provided
    none	no classification provided
    none	no classification for the individual variant
    :param clinvar_mapped_df:
    :return:
    """
    review_column = 'ReviewStatus'
    stars = [
        [0, "no assertion criteria provided"],
        [1, "criteria provided, single submitter"],
        [1, "criteria provided, conflicting interpretations"],
        [2, "criteria provided, multiple submitters, no conflicts"],
        [0,"no classification for the individual variant"],
        [0,"no classification provided"],
        [0,"no assertion provided"],
        [3,"reviewed by expert panel"],
        [4,"practice guideline"],
    ]
    star_df = pd.DataFrame(stars, columns=['ReviewStar',review_column])
    clinvar_mapped_df_with_star = pd.merge(clinvar_mapped_df, star_df,on=review_column)
    return clinvar_mapped_df_with_star



def categorize_mutations(df):
    # Define regex patterns for each mutation category
    patterns = {
        'Missense': r'p\.[A-Z]\d+[A-Z]',
        'Nonsense': r'p\.[A-Z]\d+\*',
        'Indel': r'p\.[A-Z]\d+(ins|del)[A-Z]*',
        'Frameshift': r'p\.[A-Z]\d+[A-Z]*fs\*\d+',
        'Splice_site': r'c\.\d+[+-]\d+[A-Z]+',
        'Samesense': r'p\.[A-Z]\d+=',
    }

    # Function to categorize a single mutation
    def categorize_mutation(hgvsp_short):
        for category, pattern in patterns.items():
            if re.search(pattern, hgvsp_short):
                return category
        return 'other'

    # Apply the categorization function to the DataFrame
    df['Mutation_Category'] = df['HGVSp_Short'].apply(categorize_mutation)
    return df

def exclude_collagenes(tables_with_only_main_isoforms):

    col_definitions = 'COL'
    collagenes = ['COL10A1-205', 'COL11A1-203', 'COL11A2-201', 'COL12A1-201', 'COL13A1-205',
                  'COL14A1-201', 'COL15A1-201', 'COL16A1-203', 'COL17A1-208', 'COL18A1-203',
                  'COL19A1-205', 'COL1A1-201', 'COL1A2-201', 'COL20A1-201', 'COL21A1-201',
                  'COL22A1-201', 'COL23A1-201', 'COL24A1-201', 'COL25A1-203', 'COL26A1-201',
                  'COL27A1-201', 'COL28A1-201', 'COL2A1-202', 'COL3A1-201', 'COL4A1-201',
                  'COL4A2-201', 'COL4A3-202', 'COL4A4-201', 'COL4A5-202', 'COL4A6-202',
                  'COL5A1-201', 'COL5A2-201', 'COL5A3-201', 'COL6A1-201', 'COL6A2-201',
                  'COL6A3-201', 'COL6A5-201', 'COL6A6-201', 'COL7A1-211', 'COL8A1-209',
                  'COL8A2-202', 'COL9A1-202', 'COL9A2-202', 'COL9A3-214',
                  'COLEC10-201',
                  'COLEC11-202', 'COLEC12-201', 'COLGALT1-201', 'COLGALT2-201', 'COLQ-203'
                  ]


    cols = tables_with_only_main_isoforms[tables_with_only_main_isoforms['Protein_ID'].str.startswith(col_definitions)]
    print(cols['Protein_ID'].unique().tolist())

    non_cols = tables_with_only_main_isoforms[~tables_with_only_main_isoforms['Protein_ID'].isin(collagenes)]

    return non_cols


if __name__ == "__main__":
    """
    Clinvar + Rendezetlen
        Ez hány százaléka az összeshez képest?
        Hány százalékára van funkció?
        Hány százaléka van benne a disportba? (Kisérletesen igazolt rendezetlen regió
    """

    to_clinvar_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/clinvar'

    discanvis_base_file = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/data/discanvis_base_files'

    # Clinvar
    clinvar_mapped_df = pd.read_csv(os.path.join(discanvis_base_file,'clinvar',"clinvar_mapped_with_ontology.tsv"),sep="\t")
    clinvar_mapped_df = clinvar_add_stars_to_df(clinvar_mapped_df)
    clinvar_mapped_df = categorize_mutations(clinvar_mapped_df)

    table_file = os.path.join(discanvis_base_file,'sequences',"loc_chrom_with_names_main_isoforms_with_seq.tsv")
    tables_with_only_main_isoforms = pd.read_csv(table_file,sep='\t')

    tables_with_only_main_isoforms = exclude_collagenes(tables_with_only_main_isoforms)


    clinvar_mapped_df_reduced = clinvar_mapped_df[clinvar_mapped_df['Accession'].isin(tables_with_only_main_isoforms["Protein_ID"])]


    searchfor = ["=",'*']
    searchfor = [re.escape(pattern) for pattern in searchfor if pattern]
    pattern = '|'.join(searchfor)

    cleaned_clinvar = clinvar_mapped_df_reduced[~clinvar_mapped_df_reduced["HGVSp_Short"].str.contains(pattern, regex=True)]
    # cleaned_clinvar = clinvar_mapped_df_reduced
    cleaned_clinvar = cleaned_clinvar[cleaned_clinvar["Type"] == "single nucleotide variant"]
    # cleaned_clinvar = cleaned_clinvar[cleaned_clinvar["Mutation_Category"] == "Missense"]
    print(clinvar_mapped_df_reduced)
    print(cleaned_clinvar)

    cleaned_clinvar["RCVaccession"] = cleaned_clinvar["RCVaccession"].str.split("|")
    cleaned_clinvar = cleaned_clinvar.explode("RCVaccession", ignore_index=True).rename(columns={"Accession": "Protein_ID"})

    # Exclude Coiled Coils
    coiled_coil_df = extract_pos_based_df(pd.read_csv(
        os.path.join(discanvis_base_file, 'positional_data_process', "coiled_coil_pos.tsv"), sep="\t"))

    print(coiled_coil_df)
    # exit()

    print(cleaned_clinvar)

    cleaned_clinvar = exclude_coiled_coils(cleaned_clinvar,coiled_coil_df)
    print(cleaned_clinvar)

    # exit()

    big_clinvar, clinvar, clinvar_uncertain, clinvar_benign = create_clinvar_df(cleaned_clinvar)

    # Combined Disorder
    disorder_df = pd.read_csv(os.path.join(discanvis_base_file,'positional_data_process',"CombinedDisorderNew_Pos.tsv"),sep="\t")

    mutations_in_disorder, mutations_in_ordered, big_clinvar_with_structure = create_structural_classification(big_clinvar, disorder_df)
    mutations_in_disorder.to_csv(f"{to_clinvar_dir}/clinvar_disorder.tsv", sep='\t', index=False)
    mutations_in_ordered.to_csv(f"{to_clinvar_dir}/clinvar_order.tsv", sep='\t', index=False)
    big_clinvar_with_structure.to_csv(f"{to_clinvar_dir}/clinvar_with_classification.tsv", sep='\t', index=False)




