import pandas as pd
import os
from tqdm import tqdm
tqdm.pandas()

def generate_summary_tsv(final_df,motif_type,mutation_type):
    """
    Generates a summary TSV file from the final dataframe.

    The summary file has the following columns:
      - Category: Taken from the 'Dataset' column (e.g., Known, Predicted).
      - Number of Motifs: Unique count of motifs defined by the combination
        of Protein_ID_Motif and ELM_ID.
      - Number of Pairs: Unique count of motif-domain pairs defined by the
        combination of Protein_ID_Motif and Protein_ID_Domain.
      - Number of Domains: Unique count of domains (Protein_ID_Domain).
      - One column per ELM type (ELM_ID) reporting the number of unique motifs
        for that ELM type.

    Parameters:
      final_df (pd.DataFrame): The aggregated DataFrame after pair generation and filtering.
      output_file (str): Path for the TSV output file.
    """
    # Identify the unique ELM types in the data
    all_elm_types = sorted(final_df['ELMType'].unique())

    summary_rows = []

    # Use the 'Dataset' column as the category (e.g., Known or Predicted)
    for cat in final_df['Dataset'].unique():
        for mut_set in final_df['Mutationset'].unique():
            df_cat = final_df[(final_df['Dataset'] == cat) & (final_df['Mutationset'] == mut_set)]

            # Unique motifs: unique combination of Protein_ID_Motif and ELM_ID
            num_motifs = df_cat['ELM_ID'].nunique()
            num_domains = df_cat['Pfam_ID'].nunique()
            proteins_motifs = df_cat['Protein_ID_Motif'].nunique()
            proteins_domains = df_cat['Protein_ID_Domain'].nunique()
            diseases = df_cat['Disease'].nunique()

            # Unique pairs: unique combination of Protein_ID_Motif and Protein_ID_Domain
            number_of_prot_pairs = df_cat[['Protein_ID_Motif', 'Protein_ID_Domain']].drop_duplicates().shape[0]
            number_of_elm_domain_pairs = df_cat[['ELM_ID', 'Pfam_ID']].drop_duplicates().shape[0]

            row = {
                'Dataset': cat,
                'Mutationset': mut_set,
                'Number of Rows': df_cat.shape[0],
                'Number of Diseases': diseases,
                'Number of Motifs': num_motifs,
                'Number of Domains': num_domains,
                'Number of Protein Pairs': number_of_prot_pairs,
                'Number of Motif - Domain Pairs': number_of_elm_domain_pairs,
                'Number of Proteins for Motifs': proteins_motifs,
                'Number of Proteins for Domains': proteins_domains,

            }

            # For each ELM type, count the number of unique motifs (as defined above)
            for elm in all_elm_types:
                count = df_cat[df_cat['ELMType'] == elm]['ELM_ID'].nunique()
                row[f'Number of {elm}'] = count

            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df['Motif_Type'] = motif_type
    summary_df['Mutation_Type'] = mutation_type
    return summary_df

if __name__ == '__main__':
    core_dir = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations"
    plots = os.path.join(core_dir, "processed_data", "plots")
    files = os.path.join(core_dir, "processed_data", "files")

    elm_path = f"{files}/elm/elm_predicted_disorder_with_confidence.tsv"

    elm_predicted_with_am_info_all = pd.read_csv(
        elm_path,
        sep='\t',
        # nrows=1000,
    )

    if 'N_Motif_Predicted' in elm_predicted_with_am_info_all.columns:
        elm_predicted_with_am_info_all = elm_predicted_with_am_info_all[
            elm_predicted_with_am_info_all['N_Motif_Predicted'] == "Low_Amount"]

        elm_predicted_with_am_info_all.to_csv(f"{files}/elm/predicted_elm_dataset.tsv", sep='\t', index=False)

        elm_predicted_with_am_info_all = elm_predicted_with_am_info_all[
            elm_predicted_with_am_info_all['Found_Known'] == False]



    cut_off = 0.5

    disorder_cutoff = cut_off
    order_cutoff = cut_off

    mutation_types = [
        "Pathogenic",
        "Uncertain",
    ]
    elm_type = [
        "predicted",
        'known',
    ]


    summary_rows = []

    needed_cols = ['Protein_ID', 'Motif_Start', 'Motif_End', 'ELMIdentifier', 'ELMType',
       'N_Motif_Predicted', 'N_Motif_Category', 'ELM_Types_Count',
       'Matched_Sequence', 'found_regex_pattern',  'diseases', 'number_of_diseases',
       'categories', 'genic_category', 'Intepretation', 'Number_of_Mutations',
       'Mutated_Positions_Count']


    for mut in mutation_types:

        mut_df = pd.DataFrame()

        for elm in elm_type:
            elm_path = f"{files}/elm/clinvar/motif/{mut}/motif_with_clinvar_{elm}.tsv"
            clinvar_path = f"{files}/elm/clinvar/motif/{mut}/clinvar_elm_overlaps_{elm}.tsv"

            elm_df = pd.read_csv(
                elm_path,
                sep='\t',
            )

            modified_elm = elm_df[needed_cols]
            modified_elm['known'] = True if elm == "known" else False
            mut_df = pd.concat([mut_df, modified_elm])

            clinvar_df = pd.read_csv(
                clinvar_path,
                sep='\t',
            )

            # Unique motifs: unique combination of Protein_ID_Motif and ELM_ID
            all_elm_types = elm_df['ELMType'].unique()
            num_motifs = elm_df[['Protein_ID','ELMIdentifier']].drop_duplicates().shape[0]
            unique_motifs = elm_df['ELMIdentifier'].nunique()
            proteins_motifs = elm_df['Protein_ID'].nunique()
            diseases = clinvar_df['nDisease'].nunique()

            row = {
                'Motif_Type': elm,
                'Mutation_Type': mut,
                'Number of Rows': elm_df.shape[0],
                'Number of Diseases': diseases,
                'Number of Mutations': clinvar_df.shape[0],
                'Number of Motif Classes': unique_motifs,
                'Number of Motif Instances': num_motifs,
                'Number of Proteins': proteins_motifs,
            }

            # For each ELM type, count the number of unique motifs (as defined above)
            for elm in all_elm_types:
                count = elm_df[elm_df['ELMType'] == elm]['ELMIdentifier'].nunique()
                row[f'Number of {elm}'] = count

            summary_rows.append(row)

        mut_df.to_csv(f"{files}/elm/clinvar/motif/{mut}/motif_summary.tsv",sep='\t',index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.fillna(0)
    float_cols = summary_df.select_dtypes(include='float').columns
    summary_df[float_cols] = summary_df[float_cols].astype(int)
    print(summary_df)
    summary_df.to_csv(f"{files}/elm/summary.tsv",sep='\t',index=False)