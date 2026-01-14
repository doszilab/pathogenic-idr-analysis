import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def merge_with_categories(clinvar_df, category_df):
    """Merge clinvar and likely pathogenic data with both level3 and level5 categories."""
    # Loop over each column in the category DataFrame except 'category_names'
    for column in category_df.columns[1:]:
        if column in clinvar_df.columns:
            clinvar_df = pd.merge(clinvar_df, category_df[['category_names', column]], how='left', left_on=column, right_on=column)

    clinvar_df['category_names'] = np.where(clinvar_df['category_names_x'].notna(),clinvar_df['category_names_x'],clinvar_df['category_names_y'])

    return clinvar_df


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
    result_df = df
    # result_df = df[['DOID', 'nDisease', 'meaningful_category', 'level']]

    return result_df

def final_categorization_based_on_gpt(not_categorized_df,df_with_category_gpt):
    res_df = get_dictonary_of_categories(not_categorized_df)
    print(res_df.columns)
    # level_ids = {
    #
    # }
    # for level in res_df['level'].unique():
    #     current_level = res_df[res_df['level'] == level]
    #     level_ids[level] = current_level['DOID'].tolist()
    #     print(level)
    #     print(current_level)

    lst = []
    for index, row in df_with_category_gpt.iterrows():
        dct = {
            row['level']:row['meaningful_category'],
            'category_names': row['final_category']
        }
        lst.append(dct)


    final_df = pd.DataFrame(lst)
    final_df = final_df[['category_names','level3','level2','nDisease']]
    print(final_df)
    print(final_df.columns)
    # exit()

    res_df = res_df.drop(columns=['category_names_x', 'category_names_y','category_names'])

    lst_of_dfs = []
    for column in final_df.columns[1:]:
        if column in res_df.columns:
            current_categorized_df = final_df[final_df[column].notna()]
            current_df = res_df[res_df['level'] == column]
            final_current = pd.merge(current_df, current_categorized_df[[column,'category_names']], how='inner', on=column)

            lst_of_dfs.append(final_current)

    final_df = pd.concat(lst_of_dfs)

    return final_df

if __name__ == '__main__':

    do_path = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/disease_ontology'

    disease_ontology_normalized = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/data/discanvis_base_files/clinvar/disease_ontology_normalized.tsv"

    df = pd.read_csv(disease_ontology_normalized,
        sep='\t').drop_duplicates()

    category_df = get_subcategory_for_diseases()

    df_with_category = merge_with_categories(df, category_df)
    not_categorized_df = df_with_category[df_with_category['category_names'].isna()]
    categorized_df = df_with_category[df_with_category['category_names'].notna()]
    # print(not_categorized_df)
    # print(categorized_df)
    # exit()

    simplified_df = pd.read_csv(os.path.join(do_path , 'simplified_categorizes.tsv'),sep='\t').drop_duplicates()
    gpt_categorized = pd.read_csv(os.path.join(do_path , 'category.tsv'),sep=';').drop_duplicates()

    print(df)
    print(simplified_df)
    print(gpt_categorized)
    print(gpt_categorized.columns)
    print(simplified_df.columns)

    df_with_category_gpt = simplified_df.merge(gpt_categorized,on='meaningful_category',how='inner').drop_duplicates()

    print(df_with_category_gpt)

    final_categorized_df = final_categorization_based_on_gpt(not_categorized_df, df_with_category_gpt)
    print(final_categorized_df)

    final_df = pd.concat([final_categorized_df,categorized_df])
    final_df = final_df.drop(columns=['category_names_x', 'category_names_y','level'])
    print(final_df)
    final_df.to_csv(os.path.join(do_path , 'big_categorized_do.tsv'),sep='\t',index=False)
    # final_df['category_names'].value_counts().plot(kind='pie')
    # plt.show()