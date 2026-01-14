import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

def generate_pairs(ppi_df,order_table_df,elm_df,elm_interactor_domains,pfam_table,region,disease_based=True):
    # Keep only relevant columns
    relevant_cols = [
        'Protein_ID', 'Position', 'nDisease', 'category_names',
        'genic_category', 'ELMIdentifier', 'Motif_Region','mutation_count', 'Matched_Sequence'
    ]
    if region == 'Motif_gain':
        relevant_cols += ["Original_Sequence",'HGVSp_Short']

    filtered_df = elm_df[relevant_cols]

    disease_col = "nDisease"
    if not disease_based:
        disease_col = 'category_names'
        filtered_df = filtered_df[filtered_df['nDisease'] != '-']
    else:
        filtered_df = filtered_df[filtered_df['nDisease'] != '-']
        filtered_df = filtered_df[filtered_df['category_names'] != 'Unknown']

    # Split columns by comma where needed
    columns_to_explode = ['ELMIdentifier', 'Motif_Region','Matched_Sequence']

    # Explode the relevant columns
    for col in columns_to_explode:
        filtered_df[col] = filtered_df[col].str.split(', ')
    elm_known_pathogenic_df_filtered = filtered_df.explode(columns_to_explode).reset_index(drop=True)

    elm_ids = elm_known_pathogenic_df_filtered['ELMIdentifier'].unique().tolist()

    pfam_table["Region"] = pfam_table['Start'].astype(str) + "-" + pfam_table['End'].astype(str)

    result_rows = []

    for elm in tqdm(elm_ids):
        elm_with_mutations = elm_known_pathogenic_df_filtered[elm_known_pathogenic_df_filtered['ELMIdentifier'] == elm]
        # mut_with_positions = elm_with_mutations['Position'].unique().astype(str).tolist()
        if region == "Motif_gain":
            elm_with_mutations = elm_with_mutations
        else:
            elm_with_mutations = (elm_with_mutations.groupby(["Protein_ID","nDisease","category_names","Motif_Region","genic_category","ELMIdentifier",'Matched_Sequence'])
                                  .agg({"mutation_count": "sum",'Position': lambda x: ', '.join(sorted(x.astype(str).unique()))}).reset_index())
        domains = elm_interactor_domains[elm_interactor_domains['ELMIdentifier'] == elm]
        domain_regions = pfam_table[pfam_table['Interaction Domain Id'].isin(domains['Interaction Domain Id'])]

        # Merge domain regions with the order_table_df on Protein_ID to ensure alignment
        relevant_domains = order_table_df.merge(
            domain_regions[['Protein_ID', 'Start', 'End',"Region", 'hmm_name', 'Interaction Domain Id']],
            on='Protein_ID',
            how='inner'
        )

        # Include PPI filtering
        ppi_interactions = ppi_df[ppi_df['Accession A'].isin(elm_with_mutations['Protein_ID'])]

        ppi_domains = ppi_interactions.merge(
            domain_regions[['Protein_ID', 'Start', 'End', "Region", 'hmm_name', 'Interaction Domain Id']],
            left_on='Accession B',
            right_on='Protein_ID',
            how='inner'
        )

        relevant_ppi_domains = relevant_domains[
            relevant_domains['Protein_ID'].isin(ppi_interactions['Accession B'])
        ]

        # Filter mutations that fall within the relevant regions
        filtered_mutations = relevant_domains[
            (relevant_domains['Position'] >= relevant_domains['Start']) &
            (relevant_domains['Position'] <= relevant_domains['End'])
            ]

        if not relevant_ppi_domains.empty:
            filtered_mutations = pd.concat([filtered_mutations, relevant_ppi_domains]).drop_duplicates()

        if disease_based:
            filtered_mutations = filtered_mutations[filtered_mutations['nDisease'].isin(elm_with_mutations['nDisease'].unique())]
        else:
            filtered_mutations = filtered_mutations[filtered_mutations['category_names'].isin(elm_with_mutations['category_names'].unique())]


        if not filtered_mutations.empty or not ppi_domains.empty:
            for index, motifrow in elm_with_mutations.iterrows():
                motif_region = motifrow['Motif_Region']
                motif_sequence = motifrow['Matched_Sequence']
                disease = motifrow['nDisease']
                genic_category = motifrow['genic_category']
                category_names = motifrow['category_names']
                protein_id_1 = motifrow['Protein_ID']
                positons = motifrow['Position']
                search_for = disease if disease_based else category_names
                disease_mutations = filtered_mutations[filtered_mutations[disease_col] == search_for]
                ppi_domains_current = ppi_domains[ppi_domains["Accession A"] == protein_id_1]

                if not disease_mutations.empty:
                    proteins = disease_mutations['Protein_ID'].unique()

                    for protein_id_2 in proteins:
                        current_domain = disease_mutations[disease_mutations['Protein_ID'] == protein_id_2]
                        mutation_count_1 = motifrow['mutation_count']

                        result_for_this_row = {
                            'Disease': disease,
                            'genic_category': genic_category,
                            'category_names': category_names,
                            'Protein_ID_Motif': protein_id_1,
                            'ELM_ID': elm,
                            'Matched_Sequence': motif_sequence,
                            'Region_Motif': motif_region,
                            'Protein_ID_Domain': protein_id_2,
                            'Pfam_ID': current_domain['Interaction Domain Id'].iloc[0],
                            'Domain_name': current_domain['hmm_name'].iloc[0],
                            'Region_Domain': ", ".join(current_domain["Region"].unique()),
                            'Mutated_Positions_Motif': positons,
                            'Mutated_Positions_Domain': ", ".join(current_domain["Position"].unique().astype(str)),
                            'Mutations_in_Motif': mutation_count_1,
                            'Mutations_in_Domain': current_domain.shape[0],
                            'PPI_Based': False,
                            'Interacted': protein_id_2 in ppi_interactions['Accession B'].unique()
                        }

                        if not disease_based:
                            diseases = current_domain['nDisease'].unique()
                            result_for_this_row['Disease_Domain'] = ", ".join(diseases)
                            result_for_this_row['Same_Disease'] = disease in diseases

                        if region == 'Motif_gain':
                            result_for_this_row["Original_Sequence"] = motifrow['Matched_Sequence']
                            result_for_this_row["HGVSp_Short"] = motifrow['HGVSp_Short']

                        # Append aggregated row to results
                        result_rows.append(result_for_this_row)
                else:
                    # ppi_domains_current = ppi_domains_current[ppi_domains_current['hmm_name'] != 'Pkinase']

                    proteins = ppi_domains_current['Protein_ID'].unique()

                    for protein_id_2 in proteins:
                        current_domain = ppi_domains_current[ppi_domains_current['Protein_ID'] == protein_id_2]
                        mutation_count_1 = motifrow['mutation_count']

                        result_for_this_row = {
                            'Disease': disease,
                            'genic_category': genic_category,
                            'category_names': category_names,
                            'Protein_ID_Motif': protein_id_1,
                            'ELM_ID': elm,
                            'Matched_Sequence': motif_sequence,
                            'Region_Motif': motif_region,
                            'Protein_ID_Domain': protein_id_2,
                            'Pfam_ID': current_domain['Interaction Domain Id'].iloc[0],
                            'Domain_name': current_domain['hmm_name'].iloc[0],
                            'Region_Domain': ", ".join(current_domain["Region"].unique()),
                            'Mutated_Positions_Motif': positons,
                            'Mutated_Positions_Domain': 0,
                            'Mutations_in_Motif': mutation_count_1,
                            'Mutations_in_Domain': 0,
                            'PPI_Based': True,
                            'Interacted': True
                        }

                        if not disease_based:
                            diseases = current_domain['nDisease'].unique()
                            result_for_this_row['Disease_Domain'] = ", ".join(diseases)
                            result_for_this_row['Same_Disease'] = disease in diseases

                        if region == 'Motif_gain':
                            result_for_this_row["Original_Sequence"] = motifrow['Matched_Sequence']
                            result_for_this_row["HGVSp_Short"] = motifrow['HGVSp_Short']

                        # Append aggregated row to results
                        result_rows.append(result_for_this_row)

    # Convert results to a DataFrame
    result_df = pd.DataFrame(result_rows).drop_duplicates()

    print(result_df)
    return result_df


def check_overlaps_with_pathogenic(final_df):
    print(final_df.columns)
    # Define the columns to use as the key (exclude the mutation type columns and optionally Dataset)
    key_cols = [col for col in final_df.columns if
                col not in ["Motif_Mutation_Type", "Domain_Mutation_Type", "Dataset"]]

    # Subset the pathogenic-pathogenic rows
    mask_of_pathogenic = (final_df['Motif_Mutation_Type'] == "Pathogenic") & (final_df['Domain_Mutation_Type'] == "Pathogenic")
    pathogenic_df = final_df[mask_of_pathogenic]

    # Create a set of keys from the pathogenic rows
    # Each key is a tuple of the values from key_cols
    pathogenic_keys = set(tuple(row) for row in pathogenic_df[key_cols].to_numpy())

    # Define a function to check if a given row (from an uncertain dataset) overlaps with a pathogenic row
    def check_overlap(row):
        key = tuple(row[col] for col in key_cols)
        return key in pathogenic_keys

    # Apply the check to uncertain rows (assuming these are the ones not labeled as Pathogenic-Pathogenic)
    mask_uncertain = ~mask_of_pathogenic
    final_df.loc[mask_uncertain, 'Overlapping_Pathogen'] = final_df[mask_uncertain].apply(check_overlap, axis=1)

    final_df = final_df[(mask_of_pathogenic) | ((mask_uncertain) & (final_df['Overlapping_Pathogen'] == False))]
    # Now you have an additional column 'Overlapping' in final_df for the uncertain rows.
    print(final_df)
    final_df = final_df.drop(columns=["Domain_Sequence",'Overlapping_Pathogen'])
    return final_df


def check_overlaps_with_known(final_df):
    print(final_df.columns)
    # Define the columns to use as the key (exclude the mutation type columns and optionally Dataset)
    key_cols = [col for col in final_df.columns if
                col not in ["Motif_Mutation_Type", "Domain_Mutation_Type", "Dataset"]]

    # Subset the pathogenic-pathogenic rows
    mask_of_known = final_df['Dataset'] == "Known"
    known_df = final_df[mask_of_known]

    # Create a set of keys from the pathogenic rows
    # Each key is a tuple of the values from key_cols
    pathogenic_keys = set(tuple(row) for row in known_df[key_cols].to_numpy())

    # Define a function to check if a given row (from an uncertain dataset) overlaps with a pathogenic row
    def check_overlap(row):
        key = tuple(row[col] for col in key_cols)
        return key in pathogenic_keys

    # Apply the check to uncertain rows (assuming these are the ones not labeled as Pathogenic-Pathogenic)
    mask_uncertain = ~mask_of_known
    final_df.loc[mask_uncertain, 'Overlapping_Known'] = final_df[mask_uncertain].apply(check_overlap, axis=1)

    final_df = final_df[(mask_of_known) | ((mask_uncertain) & (final_df['Overlapping_Known'] == False))]

    # Now you have an additional column 'Overlapping' in final_df for the uncertain rows.
    print(final_df)
    final_df = final_df.drop(columns=["Overlapping_Known"])
    return final_df


def generate_summary_tsv(final_df,pair_type):
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
    summary_df['Pair_Type'] = pair_type
    return summary_df



if __name__ == '__main__':

    base_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations'

    base_mut_dir = os.path.join(base_dir,'processed_data/files/clinvar')
    base_elm_dir = os.path.join(base_dir,'processed_data/files/elm')
    base_merged_dir = os.path.join(base_elm_dir,'clinvar','merged')
    ppi_base_dir = os.path.join(base_dir,'data/discanvis_base_files/ppi')

    # Import PPI df
    base_table = pd.read_csv(
        "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/data/discanvis_base_files/sequences/loc_chrom_with_names_main_isoforms_with_seq.tsv",
        sep='\t')


    output_file = f'/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/elm/binding_pairs'

    # Generate Tables
    region_categories = [
        # "Motif_gain",
        "Known",
        "Predicted",
    ]

    cols_needed = ["Disease","genic_category","category_names",
                    'Protein_ID_Motif',
                   'ELM_ID', 'ELMType',
                   'Matched_Sequence',
                   'Region_Motif', 'Protein_ID_Domain','Pfam_ID',
                   'Domain_name', 'Region_Domain',
                   "Motif_Mutation_Type",
                   "Domain_Mutation_Type",
                   "Mutated_Positions_Motif",
                   "Mutations_in_Motif",
                   "Interacted",
                   ]

    # base_table has columns ["Protein_ID", "Sequence"]
    # Drop duplicates, then build a dictionary
    base_table_unique = base_table[["Protein_ID", "Sequence"]].drop_duplicates()
    seq_dict = dict(zip(base_table_unique["Protein_ID"], base_table_unique["Sequence"]))

    pair_types = {
        "Mutation_Pair":"elm_pfam_disease_pairs.tsv",
        "PPI_Pair":"elm_pfam_disease_pairs_ppi.tsv",
    }

    summary_df = pd.DataFrame()

    for pair_type, file_name in pair_types.items():

        all_df = pd.DataFrame()

        for region in region_categories:
            current_cols = cols_needed.copy()
            if region == 'Motif_gain':
                current_cols += ['Original_Sequence']

            # Save or display results
            pathogenic_path = f'{output_file}/Pathogenic-Pathogenic/{region}'
            pathogenic_res_df = pd.read_csv(f'{pathogenic_path}/{file_name}', sep='\t')
            print(pathogenic_res_df.columns)
            pathogenic_res_df = pathogenic_res_df[current_cols].drop_duplicates()
            pathogenic_res_df['Dataset'] = region
            pathogenic_res_df['Mutationset'] = "Pathogenic-Pathogenic"
            pathogenic_res_df["Domain_Sequence"] = pathogenic_res_df["Protein_ID_Domain"].map(seq_dict)

            # Save or display results
            uncertain_path = f'{output_file}/Uncertain-Pathogenic/{region}'
            uncertain_res_df= pd.read_csv(f'{uncertain_path}/{file_name}', sep='\t')
            uncertain_res_df = uncertain_res_df[current_cols].drop_duplicates()
            uncertain_res_df['Dataset'] = region
            uncertain_res_df['Mutationset'] = "Uncertain-Pathogenic"
            uncertain_res_df["Domain_Sequence"] = uncertain_res_df["Protein_ID_Domain"].map(seq_dict)

            # Save or display results
            uncertain_path = f'{output_file}/Uncertain-Uncertain/{region}'
            uncertain2_res_df = pd.read_csv(f'{uncertain_path}/{file_name}', sep='\t')
            uncertain2_res_df = uncertain2_res_df[current_cols].drop_duplicates()
            uncertain2_res_df['Dataset'] = region
            uncertain2_res_df['Mutationset'] = "Uncertain-Uncertain"
            uncertain2_res_df["Domain_Sequence"] = uncertain_res_df["Protein_ID_Domain"].map(seq_dict)

            all_df = pd.concat([all_df, pathogenic_res_df, uncertain_res_df,uncertain2_res_df])

        final_df = all_df.drop_duplicates()

        print(final_df)

        filtered_final_df = check_overlaps_with_pathogenic(final_df)
        filtered_final_df = check_overlaps_with_known(filtered_final_df)

        known_df = filtered_final_df[filtered_final_df['Dataset'] == 'Known']
        known_df.to_csv(os.path.join(output_file, f"Known_{pair_type}.tsv"),sep='\t',index=False)

        predicted_df = filtered_final_df[filtered_final_df['Dataset'] == 'Predicted']
        predicted_df.to_csv(os.path.join(output_file, f"Predicted_{pair_type}.tsv"),sep='\t',index=False)

        summary_df = pd.concat([summary_df, generate_summary_tsv(final_df, pair_type)])

    summary_df.to_csv(os.path.join(output_file, f"summary.tsv"), sep='\t', index=False)
