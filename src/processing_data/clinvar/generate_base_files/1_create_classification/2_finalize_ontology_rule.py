import pandas as pd
import numpy as np
from tqdm import tqdm



def define_neurodevelopmental(df):
    df['Final_Category'] = np.where((df['Final_Category'] == "Neurodegenerative") & (df['Developmental'] == True),"Neurodevelopmental",df['Final_Category'])
    return df

def modify_contains(df,category,lst,check_col='disease_group'):
    possible_categories = ["Other", "Mixed"]
    for value in lst:
        df['Final_Category'] = np.where((df[check_col].str.contains(value, case=False)) & (df["Final_Category"].isin(possible_categories)),
                                        category, df["Final_Category"])
    return df

def modify_exact(df,category,lst,check_col='disease_group'):
    possible_categories = ["Other","Mixed"]
    df['Final_Category'] = np.where((df[check_col].isin(lst)) & (df["Final_Category"].isin(possible_categories)),
                                             category, df["Final_Category"])
    return df

def categorize_not_found_ones(big_clinvar):
    desired_categories = [
        'Cancer', 'Cardiovascular/Hematopoietic', 'Developmental', 'Endocrine',
        'Gastrointestinal', 'Immune', 'Integumentary', 'Metabolic', 'Musculoskeletal',
        'Neurodegenerative', 'Reproductive', 'Respiratory', 'Urinary'
    ]

    # Musculoskeletal
    musculoskeletal = ["Jeune syndrome", "Spastic paraplegia", 'syndromic craniosynostosis',
                       'autosomal recessive limb-girdle muscular dystrophy', 'Marfan syndrome',
                       'disease of bone structure', 'distal arthrogryposis',
                       "autosomal dominant titinopathy","autosomal recessive titinopathy",
                       "muscular dystrophy",
                       "muscular dystrophy-dystroglycanopathy, type A",
                       "muscular dystrophy-dystroglycanopathy, type C",
                       "muscular dystrophy-dystroglycanopathy, type B",
                       ]
    musculoskeletal_contains = ["spastic paraplegia", " myopathy"]
    big_clinvar = modify_contains(big_clinvar, "Musculoskeletal", musculoskeletal_contains, check_col='disease_group')
    big_clinvar = modify_exact(big_clinvar, "Musculoskeletal", musculoskeletal, check_col='disease_group')

    # CardioVascular/Hematopoietic
    cardiovascular = ["Cardiovascular phenotype","Cardiac arrhythmia","familial thoracic aortic aneurysm and aortic dissection",
                      "Seizure","long QT syndrome","Brugada syndrome",'left ventricular noncompaction','Fanconi anemia',
                      "cardiomyopathy",'hereditary hemorrhagic telangiectasia'
                      ]
    big_clinvar =  modify_exact(big_clinvar,"Cardiovascular/Hematopoietic",cardiovascular,check_col='disease_group')
    cardiovascular_contains = ["Cardiomyopathy","thoracic aortic aneurysm"]
    big_clinvar =  modify_contains(big_clinvar,"Cardiovascular/Hematopoietic",cardiovascular_contains,check_col='disease_group')

    # Cancer
    cancer = ["Hereditary nonpolyposis colorectal neoplasms","hereditary breast carcinoma","hereditary neoplastic syndrome"]
    big_clinvar = modify_exact(big_clinvar, "Cancer", cancer, check_col='disease_group')
    cancer_contains = ["cancer", "tumor", "malignant","carcinoma","sarcoma"]
    big_clinvar = modify_contains(big_clinvar, "Cancer", cancer_contains,check_col='disease_group')

    # Neuro
    neurodegenerative = ["Epileptic encephalopathy","gastrointestinal stromal tumor",'West syndrome',"Meckel syndrome"]
    neurodevelopmental = ['West syndrome','dyneinopathy']
    neurodevelopmental_contains = ["Neurodevelopmental"]
    big_clinvar = modify_contains(big_clinvar, "Neurodevelopmental", neurodevelopmental_contains, check_col='disease_group')
    big_clinvar = modify_exact(big_clinvar, "Neurodevelopmental", neurodevelopmental, check_col='disease_group')

    neurodegenerative_contains = ["retinitis pigmentosa", "deafness", "Usher syndrome","encephalopathy",'neurodegeneration']
    big_clinvar = modify_contains(big_clinvar, "Neurodegenerative", neurodegenerative_contains,check_col='disease_group')
    big_clinvar = modify_exact(big_clinvar, "Neurodegenerative", neurodegenerative,check_col='disease_group')

    # Integumentary
    integumentary = ["epidermolysis bullosa simplex",'skin vascular disease']
    big_clinvar = modify_exact(big_clinvar, "Integumentary", integumentary, check_col='disease_group')
    # Urinary
    urinary = ["combined oxidative phosphorylation deficiency",'Senior-Loken syndrome',"nephronophthisis"]
    urinary_contains = ["kidney disease",'renal disease']
    big_clinvar = modify_exact(big_clinvar, "Urinary", urinary, check_col='disease_group')
    big_clinvar = modify_contains(big_clinvar, "Urinary", urinary_contains,check_col='disease_group')

    # Metabolic
    metabolic = ["combined oxidative phosphorylation deficiency"]
    big_clinvar = modify_exact(big_clinvar, "Metabolic", metabolic, check_col='disease_group')
    metabolic_contains = ["mitochondrial complex","mucopolysaccharidosis"]
    big_clinvar = modify_contains(big_clinvar, "Metabolic", metabolic_contains, check_col='disease_group')

    # Reproductive
    Reproductive = []
    Reproductive_contains = ["spermatogenic failure"]
    big_clinvar = modify_contains(big_clinvar, "Reproductive", Reproductive_contains, check_col='disease_group')

    # Endocrine
    endocrine = ["autosomal dominant hypocalcemia"]
    big_clinvar = modify_contains(big_clinvar, "Endocrine", endocrine, check_col='disease_group')

    # Developmental
    # Developmental = ["CFTR-related disorders"]
    # big_clinvar = modify_exact(big_clinvar, "Developmental", Developmental, check_col='disease_group')

    # Inborn Genetic Diseases
    inborn = ["Inborn genetic diseases;Nephrolithiasis/nephrocalcinosis"]
    big_clinvar = modify_exact(big_clinvar, "Inborn Genetic Diseases", inborn, check_col='disease_group')

    return big_clinvar


if __name__ == '__main__':

    to_clinvar_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/clinvar'

    big_clinvar = pd.read_csv(f"{to_clinvar_dir}/clinvar_with_classification.tsv", sep='\t')
    # big_clinvar = pd.read_csv(f"{to_clinvar_dir}/clinvar_mistakes.tsv", sep='\t')


    # big_clinvar['category_names'] = big_clinvar['category_names'].apply(lambda x: x if pd.notna(x) else "Other")
    # print(big_clinvar)
    # 1. Disease Name cleaning
    # Handle not found
    big_clinvar['Disease'] = np.where(big_clinvar["Result_Found_by"] == "Not found", big_clinvar["PhenotypeList"], big_clinvar["Disease"])
    big_clinvar['disease_group'] = np.where(big_clinvar["Result_Found_by"] == "Not found", big_clinvar["PhenotypeList"], big_clinvar["disease_group"])
    # Handle disease groups
    big_clinvar['disease_group'] = np.where(big_clinvar["disease_group"] == '-', big_clinvar["PhenotypeList"],
                                            big_clinvar["disease_group"])
    big_clinvar['disease_group'] = np.where(big_clinvar["disease_group"] == "syndromic disease", big_clinvar["Disease"], big_clinvar["disease_group"])
    big_clinvar['disease_group'] = np.where(big_clinvar["disease_group"] == "hereditary disease", big_clinvar["Disease"], big_clinvar["disease_group"])
    big_clinvar['disease_group'] = np.where(big_clinvar["disease_group"] == "syndromic intellectual disability", big_clinvar["Disease"], big_clinvar["disease_group"])
    # exit()

    big_clinvar['disease_group'] = np.where(big_clinvar["disease_group"] == 'See cases', "not specified", big_clinvar["disease_group"])
    big_clinvar['disease_group'] = np.where(big_clinvar["disease_group"].str.contains("conditions"),"not specified", big_clinvar["disease_group"])


    # 1. Determine Category
    unknown_lst = ["not provided","not specified","-"]
    big_clinvar['Final_Category'] = np.where(big_clinvar["disease_group"].isin(unknown_lst), "Unknown", big_clinvar["Final_Category"])
    big_clinvar['Final_Category'] = np.where(big_clinvar["disease_group"] == "Inborn genetic diseases", "Inborn Genetic Diseases", big_clinvar["Final_Category"])
    big_clinvar['Final_Category'] = np.where(big_clinvar["Final_Category"] == "-", "Other", big_clinvar["Final_Category"])


    categorized_mut = categorize_not_found_ones(big_clinvar)


    categorized_mut['Rare'] = np.where(categorized_mut['DO Subset'].str.contains("rare"),True,False)
    mapping = {
        '-': False,
        'False': False,
        'True': True
    }

    categorized_mut['Developmental'] = categorized_mut['Developmental'].replace(mapping)
    # categorized_mut['Developmental'] = np.where(categorized_mut['Final_Category'] == "Neurodevelopmental",True,categorized_mut['Developmental'])
    print(categorized_mut['Developmental'].value_counts())
    categorized_mut['Developmental'] = categorized_mut['Developmental'].astype(bool)
    print(categorized_mut['Developmental'].value_counts())

    categorized_mut = define_neurodevelopmental(categorized_mut)

    # categorized_mut = unify_disease_groups(categorized_mut)
    # print(categorized_mut)
    # exit()

    categorized_mut["category_names"] = categorized_mut['Final_Category']
    categorized_mut["nDisease"] = categorized_mut['disease_group']
    print(categorized_mut)
    print(categorized_mut.nDisease)
    final_df = categorized_mut.drop_duplicates()



    print(final_df)
    to_file_name = f"clinvar_with_do_categories.tsv"

    final_df.to_csv(f'{to_clinvar_dir}/{to_file_name}', sep='\t', index=False)