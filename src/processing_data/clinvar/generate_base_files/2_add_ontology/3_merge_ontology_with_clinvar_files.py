import pandas as pd
import numpy as np
from tqdm import tqdm
import os



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

def clean_data(df):
    # Prioritize more specific categories over 'Other' for the same DOID
    df = df.sort_values(by='category_names', key=lambda col: col == 'Other')

    # Drop duplicates, keeping the more specific category where it exists
    df = df.drop_duplicates(subset=['DOID'], keep='first')
    return df

def group_for_position(df):

    group_lst = [
        'Protein_ID','Position','DOID',
       'DO Subset', 'nDisease',"category_names",
       'Interpretation', 'Category', 'info', 'info_cols',
    ]
    positional_df = df.groupby(group_lst).size().reset_index(name='mutation_count')
    return positional_df

def group_for_gene(df):

    group_lst = [
        'Protein_ID','DOID',
       'DO Subset', 'nDisease',"category_names",
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


def define_developmental(df):
    tqdm.pandas()
    developmental = "developmental"
    def row_function(row):
        category_names = row['category_names']
        dname = row['PhenotypeList']
        if type(dname) == str and dname.lower() == "Inborn genetic diseases".lower():
            return dname
        elif category_names == "Neurodegenerative":
            is_developmental = False
            if row["level2"] == "Developmental Disease":
                is_developmental = True
            if any([developmental in x.strip().lower() for x in row.values if type(x) == str]):
                is_developmental = developmental

            if is_developmental:
                category_names = "Neurodevelopmental"
            return category_names
        else:
            return category_names

    df['category_names'] = df.progress_apply(row_function,axis=1)
    return df




if __name__ == '__main__':

    to_clinvar_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/clinvar'

    do_path = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/disease_ontology'

    category_df = pd.read_csv(os.path.join(do_path,'big_categorized_do.tsv'),sep='\t')
    category_df = category_df[['DOID','category_names']]
    category_df = clean_data(category_df)

    print(category_df)

    big_clinvar = pd.read_csv(f"{to_clinvar_dir}/clinvar_with_classification.tsv", sep='\t')
    # big_clinvar = pd.read_csv(f"{to_clinvar_dir}/clinvar_mistakes.tsv", sep='\t')

    print(big_clinvar)

    categorized_mut = big_clinvar.merge(category_df, on='DOID',how='left')
    print(categorized_mut.columns)
    print(categorized_mut)
    categorized_mut['category_names'] = categorized_mut['category_names'].apply(lambda x: x if pd.notna(x) else "Other")
    print(categorized_mut)
    categorized_mut['category_names'] = np.where(categorized_mut["nDisease"] == "-", "Unknown", categorized_mut["category_names"])

    categorized_mut = define_developmental(categorized_mut)
    print(categorized_mut)
    print(categorized_mut.nDisease)
    final_df = categorized_mut.drop_duplicates()
    print(final_df)
    to_file_name = f"clinvar_with_do_categories.tsv"

    final_df.to_csv(f'{to_clinvar_dir}/{to_file_name}', sep='\t', index=False)