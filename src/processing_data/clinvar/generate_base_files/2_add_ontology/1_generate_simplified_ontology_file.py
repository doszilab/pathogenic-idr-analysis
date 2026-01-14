import pandas as pd
import numpy as np




def merge_with_categories(clinvar_df, category_df):
    """Merge clinvar and likely pathogenic data with both level3 and level5 categories."""
    # Loop over each column in the category DataFrame except 'category_names'
    for column in category_df.columns[1:]:
        if column in clinvar_df.columns:
            clinvar_df = pd.merge(clinvar_df, category_df[['category_names', column]], how='left', left_on=column, right_on=column)

    clinvar_df['category_names'] = np.where(clinvar_df['category_names_x'].notna(),clinvar_df['category_names_x'],clinvar_df['category_names_y'])

    return clinvar_df

def aggragate_cols(row):
    full_list_val = []
    cols = []
    for col in cols_to_aggregate:
        val = row[col]
        if pd.notna(val):
            full_list_val.append(str(val))
            cols.append(col)
    str_cols = ','.join(cols)
    str_vals = ','.join(full_list_val)

    return str_cols, str_vals

def get_subcategory_for_diseases():
    """Map specific disease subcategories to levels."""
    category_dictionary = {
        "cancer": ["Cancer", 'level3'],
        "cardiovascular system disease": ["Cardiovascular", 'level3'],
        "endocrine system disease": ["Endocrine", 'level3'],
        "gastrointestinal system disease": ["Gastrointestinal", 'level3'],
        "immune system disease": ["Immune", 'level3'],
        "musculoskeletal system disease": ["Musculoskeletal", 'level3'],
        "neurodegenerative disease": ["Neurodegenerative", 'level5'],
        "reproductive system disease": ["Reproductive", 'level3'],
        "respiratory system disease": ["Respiratory", 'level3'],
        "urinary system disease": ["Urinary", 'level3'],
    }

    # Create lists for columns
    subcategory_list = []
    category_names_list = []
    level_list = []

    for key, value in category_dictionary.items():
        subcategory_list.append(key)
        category_names_list.append(value[0])
        level_list.append(value[1])

    # Build the DataFrame using lists
    df = pd.DataFrame({'subcategory': subcategory_list, 'category_names':category_names_list,'level': level_list})

    pivoted_df = df.pivot_table(
        index='category_names',
        columns='level',
        values='subcategory',
        aggfunc='first'
    ).reset_index()

    return pivoted_df



def get_dictonary_of_categories(df):
    # Initialize new columns to store the meaningful category and the level used
    df['meaningful_category'] = None
    df['level'] = None

    # Step 1: Set 'meaningful_category' to 'level3' if it's not '-'
    df.loc[df['level3'] != '-', ['meaningful_category', 'level']] = df.loc[df['level3'] != '-', ['level3']].assign(level='level3').values

    # Step 2: For rows where 'level3' is '-', check 'level2'
    mask = df['level3'] == '-'
    df.loc[mask & ~df['level2'].isin(['-', 'syndrome']), ['meaningful_category', 'level']] = df.loc[mask & ~df['level2'].isin(['-', 'syndrome']), ['level2']].assign(level='level2').values

    # Step 3: If both 'level3' is '-' and 'level2' is '-' or 'syndrome', set the category to the disease name (nDisease)
    df.loc[mask & df['level2'].isin(['-', 'syndrome']), ['meaningful_category', 'level']] = df.loc[mask & df['level2'].isin(['-', 'syndrome']), ['nDisease']].assign(level='nDisease').values

    # Select the relevant columns for the output
    result_df = df[['DOID', 'nDisease', 'meaningful_category', 'level']]

    print(result_df)
    simplified = result_df[['meaningful_category', 'level']].drop_duplicates()

    return simplified


if __name__ == '__main__':


    cols_to_keep = [
        'DOID', 'Disease',
       'synonyms', 'DO Subset', 'xref_source', 'reference_id', 'level1',
       'level2', 'level3', 'level4', 'level5', 'level6', 'level7', 'level8',
       'level9', 'level10', 'level11', 'level12', 'level13', 'nDisease',
    ]

    file_names = [
        'all', 'complex', 'monogenic', 'polygenic'
    ]

    do_path = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/disease_ontology'

    disease_ontology_normalized = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/data/discanvis_base_files/clinvar/disease_ontology_normalized.tsv"

    df = pd.read_csv(disease_ontology_normalized,sep='\t').drop_duplicates()

    df_of_categories  =get_dictonary_of_categories(df)
    df_of_categories.to_csv(f'{do_path}/simplified_categorizes.tsv',sep='\t',index=False)