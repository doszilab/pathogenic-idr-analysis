import pandas as pd


def create_lst_of_genes(mutations_in_disorder, mutations_in_ordered):
    # Extracting genes by interpretation for disordered and ordered regions
    disorder_pathogenic = set(
        mutations_in_disorder[mutations_in_disorder["Interpretation"] == "Pathogenic"]['Protein_ID'])
    disorder_uncertain = set(
        mutations_in_disorder[mutations_in_disorder["Interpretation"] == "Uncertain"]['Protein_ID'])
    disorder_benign = set(mutations_in_disorder[mutations_in_disorder["Interpretation"] == "Benign"]['Protein_ID'])

    ordered_pathogenic = set(mutations_in_ordered[mutations_in_ordered["Interpretation"] == "Pathogenic"]['Protein_ID'])
    ordered_uncertain = set(mutations_in_ordered[mutations_in_ordered["Interpretation"] == "Uncertain"]['Protein_ID'])
    ordered_benign = set(mutations_in_ordered[mutations_in_ordered["Interpretation"] == "Benign"]['Protein_ID'])

    # Get the union of all genes across disorder and ordered regions
    all_genes = disorder_pathogenic.union(disorder_uncertain).union(disorder_benign) \
        .union(ordered_pathogenic).union(ordered_uncertain).union(ordered_benign)

    # Create a dictionary to hold the values for Pathogenic, Uncertain, and Benign
    gene_classification = {}

    # Classify each gene based on the presence in disorder and order sets
    for gene in all_genes:
        # Classify Pathogenic
        if gene in disorder_pathogenic and gene in ordered_pathogenic:
            pathogenic_status = "both"
        elif gene in disorder_pathogenic:
            pathogenic_status = "disordered"
        elif gene in ordered_pathogenic:
            pathogenic_status = "ordered"
        else:
            pathogenic_status = None

        # Classify Uncertain
        if gene in disorder_uncertain and gene in ordered_uncertain:
            uncertain_status = "both"
        elif gene in disorder_uncertain:
            uncertain_status = "disordered"
        elif gene in ordered_uncertain:
            uncertain_status = "ordered"
        else:
            uncertain_status = None

        # Classify Benign
        if gene in disorder_benign and gene in ordered_benign:
            benign_status = "both"
        elif gene in disorder_benign:
            benign_status = "disordered"
        elif gene in ordered_benign:
            benign_status = "ordered"
        else:
            benign_status = None

        # Store classification in the dictionary
        gene_classification[gene] = {
            'Pathogenic': pathogenic_status,
            'Uncertain': uncertain_status,
            'Benign': benign_status
        }

    # Convert dictionary to DataFrame
    gene_df = pd.DataFrame.from_dict(gene_classification, orient='index').reset_index()
    gene_df.columns = ['Gene', 'Pathogenic', 'Uncertain', 'Benign']

    return gene_df



if __name__ == "__main__":
    to_clinvar_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/clinvar'

    big_clinvar_final_df = pd.read_csv(f"{to_clinvar_dir}/clinvar_mutations_with_annotations_merged.tsv", sep='\t')

    mutations_in_disorder = big_clinvar_final_df[big_clinvar_final_df['structure'] == 'disorder']
    mutations_in_ordered = big_clinvar_final_df[big_clinvar_final_df['structure'] == 'order']

    mutations_in_disorder = mutations_in_disorder[
        (mutations_in_disorder['category_names'] != 'Unknown') & (mutations_in_disorder['Position'] != 1)]
    mutations_in_ordered = mutations_in_ordered[
        (mutations_in_ordered['category_names'] != 'Unknown') & (mutations_in_ordered['Position'] != 1)]

    mutations_in_disorder = mutations_in_disorder[['Protein_ID', 'Interpretation']].drop_duplicates()
    mutations_in_ordered = mutations_in_ordered[['Protein_ID', 'Interpretation']].drop_duplicates()

    gene_df = create_lst_of_genes(mutations_in_disorder,mutations_in_ordered)
    print(gene_df)
    gene_df.to_csv(f"{to_clinvar_dir}/clinvar_genes_structural_distribution_for_mutations.tsv",sep='\t', index=False)