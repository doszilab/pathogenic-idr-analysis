import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

def generate_pairs(ppi_df,order_table_df,elm_df,elm_interactor_domains,pfam_table,elm_type="known",mutation_type='pathogenic',disease_based=True,output_file=None):
    # Keep only relevant columns
    relevant_cols = [
        'Protein_ID', 'Position', 'nDisease', 'category_names',
        'genic_category', 'ELMIdentifier', 'Motif_Region','mutation_count', 'Matched_Sequence'
    ]
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

                        # Append aggregated row to results
                        result_rows.append(result_for_this_row)

    # Convert results to a DataFrame
    result_df = pd.DataFrame(result_rows).drop_duplicates()

    print(result_df)

    # Save or display results
    class_based = "_class_based" if not disease_based else ""
    file_name = f'{output_file}/elm_pfam_disease_pairs_elm_{elm_type}_mut_{mutation_type}{class_based}.tsv'
    result_df.to_csv(file_name, sep='\t', index=False)


if __name__ == '__main__':

    base_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations'

    base_mut_dir = os.path.join(base_dir,'processed_data/files/clinvar')
    elm_base_dir = os.path.join(base_dir,'processed_data/files/elm/clinvar/motif')
    ppi_base_dir = os.path.join(base_dir,'data/discanvis_base_files/ppi')

    mutation_table_order_pathogenic = os.path.join(base_mut_dir, 'Pathogenic', 'order',
                                                   'positional_clinvar_functional_categorized_final.tsv')
    mutation_table_order_uncertain = os.path.join(base_mut_dir, 'Uncertain', 'order',
                                                  'positional_clinvar_functional_categorized_final.tsv')
    mutation_table_ordern_likely_pathogenic = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/alphamissense/clinvar/likely_pathogenic_order.tsv'

    mutation_table_order_pathogenic_df = pd.read_csv(mutation_table_order_pathogenic, sep='\t')
    # mutation_table_order_uncertain_df = pd.read_csv(mutation_table_order_uncertain, sep='\t')
    mutation_table_ordern_likely_pathogenic_df = pd.read_csv(mutation_table_ordern_likely_pathogenic, sep='\t')

    elm_interactor_domains = pd.read_csv(
        f"{base_dir}/data/discanvis_base_files/elm/elm_interaction_domains.tsv", sep='\t')
    elm_interactor_domains = elm_interactor_domains.rename(columns={"ELM identifier": "ELMIdentifier"})

    pfam_table = pd.read_csv(
        f"{base_dir}/data/discanvis_base_files/pfam/pfamtable.tsv",
        sep='\t')
    pfam_table['Interaction Domain Id'] = pfam_table['hmm_acc'].str.split('.').str[0]
    pfam_table = pfam_table.rename(columns={'envelope_start': 'Start', 'envelope_end': 'End'})

    output_file = f'/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/elm/ppi_based'

    # Import PPI df
    base_table = pd.read_csv(
        "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/data/discanvis_base_files/sequences/loc_chrom_with_names_main_isoforms_with_seq.tsv",
        sep='\t')

    ppi_path = f"{ppi_base_dir}/Interactions.tsv"

    ppi_df = pd.read_csv(ppi_path, sep='\t')
    ppi_df = ppi_df[ppi_df['Accession A'].isin(base_table['Protein_ID'])]
    ppi_df = ppi_df[ppi_df['Accession B'].isin(base_table['Protein_ID'])]


    # Import ELM list
    elm_known_pathogenic = f"{elm_base_dir}/clinvar_pathogenic_with_known_elm_overlaps.tsv"
    elm_predicted_pathogenic = f"{elm_base_dir}/clinvar_pathogenic_with_predicted_elm_overlaps.tsv"
    elm_known_uncertain = f"{elm_base_dir}/clinvar_uncertain_with_known_elm_overlaps.tsv"
    elm_predicted_uncertain = f"{elm_base_dir}/clinvar_uncertain_with_predicted_elm_overlaps.tsv"
    elm_known_predicted_pathogenic = f"{elm_base_dir}/clinvar_predicted_pathogenic_with_known_elm_overlaps.tsv"
    elm_predicted_predicted_pathogenic = f"{elm_base_dir}/clinvar_predicted_pathogenic_with_predicted_elm_overlaps.tsv"

    elm_known_pathogenic_df = pd.read_csv(elm_known_pathogenic, sep='\t')
    elm_predicted_pathogenic_df = pd.read_csv(elm_predicted_pathogenic, sep='\t')
    elm_known_uncertain_df = pd.read_csv(elm_known_uncertain, sep='\t')
    elm_predicted_uncertain_df = pd.read_csv(elm_predicted_uncertain, sep='\t')
    elm_known_predicted_pathogenic_df = pd.read_csv(elm_known_predicted_pathogenic, sep='\t')
    elm_predicted_predicted_pathogenic_df = pd.read_csv(elm_predicted_predicted_pathogenic, sep='\t')



    # Generate Tables
    # Pathogenic
    # generate_pairs(ppi_df,mutation_table_order_pathogenic_df, elm_known_pathogenic_df, elm_interactor_domains, pfam_table, elm_type="known",
    #                mutation_type='pathogenic',output_file=output_file)
    # generate_pairs(ppi_df,mutation_table_order_pathogenic_df, elm_predicted_pathogenic_df, elm_interactor_domains, pfam_table, elm_type="predicted",
    #                mutation_type='pathogenic',output_file=output_file)


    # Predicted Pathogenic (Ordered Predicted Pathogenic + Pathogenic)
    order_predicted_pathogenic_and_pathogenic = pd.concat([mutation_table_order_pathogenic_df, mutation_table_ordern_likely_pathogenic_df],ignore_index=True)

    generate_pairs(ppi_df,order_predicted_pathogenic_and_pathogenic, elm_known_predicted_pathogenic_df, elm_interactor_domains, pfam_table, elm_type="known",
                   mutation_type='predicted_pathogenic',output_file=output_file)
    generate_pairs(ppi_df,order_predicted_pathogenic_and_pathogenic, elm_predicted_predicted_pathogenic_df, elm_interactor_domains, pfam_table,
                   elm_type="predicted",
                   mutation_type='predicted_pathogenic',output_file=output_file)

    # generate_pairs(order_predicted_pathogenic_and_pathogenic, elm_known_predicted_pathogenic_df, elm_interactor_domains, pfam_table,
    #                elm_type="known",
    #                mutation_type='predicted_pathogenic',disease_based=False)
    # generate_pairs(order_predicted_pathogenic_and_pathogenic, elm_predicted_predicted_pathogenic_df, elm_interactor_domains, pfam_table,
    #                elm_type="predicted",
    #                mutation_type='predicted_pathogenic',disease_based=False)