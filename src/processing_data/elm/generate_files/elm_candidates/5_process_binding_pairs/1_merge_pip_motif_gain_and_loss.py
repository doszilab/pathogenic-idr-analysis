import os
import pandas as pd



def generate_summary_tsv_elm(big_df, output_file):
    """
    Generates a summary TSV file from an ELM-only merged dataset.

    The summary file has the following columns:
      - Category: Taken from the 'Region_category' column (e.g., Known, Predicted, PIP, Motif_gain).
      - Number of Rows: Total number of rows in that category.
      - Number of Diseases: Unique count of diseases (from the 'nDisease' column).
      - Number of Motifs: Unique count of motifs defined as the combination of Protein_ID and ELMIdentifier.
      - Number of Proteins: Unique count of proteins (Protein_ID).
      - One column per ELM type (from ELMIdentifier) reporting the number of unique motifs for that type.

    Parameters:
      big_df (pd.DataFrame): The merged ELM dataset.
      output_file (str): Path for the TSV output file.
    """
    # Split columns by comma where needed
    columns_to_explode = ['ELMIdentifier', 'ELMType', 'Motif_Region', 'Matched_Sequence']

    new_df = big_df.copy()

    # Explode the relevant columns
    for col in columns_to_explode:
        new_df[col] = new_df[col].str.split(', ')
    new_df = new_df.explode(columns_to_explode).reset_index(drop=True)

    print("Running generate_summary_tsv_elm...")
    # Determine all unique ELM types in the data
    all_elm_types = new_df['ELMType'].unique()

    summary_rows = []

    # Group by Region_category
    for mutationset in new_df['Mutationset'].unique():
        for category in new_df['Region_category'].unique():
            print(category,mutationset)
            df_cat = new_df[(new_df['Region_category'] == category) & (new_df['Mutationset'] == mutationset)]
            df_cat_mut = big_df[(big_df['Region_category'] == category) & (big_df['Mutationset'] == mutationset)]

            # Total rows in this category
            num_rows = df_cat_mut.shape[0]

            # Unique diseases count
            num_diseases = df_cat['nDisease'].nunique()

            # Unique motifs defined by Protein_ID and ELMIdentifier
            # (if a protein has more than one occurrence of an ELMIdentifier, count it once)
            num_motifs =  df_cat['ELMIdentifier'].nunique()

            # Unique proteins
            num_proteins = df_cat['Protein_ID'].nunique()

            # Prepare the summary row for this category
            row = {
                'Category': category,
                'Mutationset': mutationset,
                'Number of Rows': num_rows,
                'Number of Diseases': num_diseases,
                'Number of Motifs': num_motifs,
                'Number of Proteins': num_proteins,
            }

            # For each ELM type, count unique motifs (Protein_ID + ELMIdentifier) with that ELM type
            for elm in all_elm_types:
                if pd.isna(elm):
                    continue
                count = df_cat[df_cat['ELMType'] == elm]['ELMIdentifier'].nunique()
                row[f'Number of {elm}'] = count

            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_file, sep='\t', index=False)
    print(f"Summary TSV written to {output_file}")




if "__main__" == __name__:
    """
    Here there will be 8 total category:
    
    Motif loss
    1. Mutations with both Motif and Domain (add PPI)
    2. Mutations with mutation in Motif (Binding Domain only with PPI)
    
    3. Mutations with PIP and PPI Domain
    4. Mutations with both Motif and Domain (add PPI) -> PEM
    5. Mutations with mutation in Motif (Binding Domain only with PPI) -> PEM
    
    Motif gain
    7. Mutations with both Motif and Domain (add PPI)
    8. Mutations with mutation in Motif (Binding Domain only with PPI)
    
    We should check for both pathogenic and uncertain mutations
    Maybe check for only predicted pathogenic?
    """

    base_dir = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations"
    file_dir = os.path.join(base_dir, "processed_data", "files")
    elm_dir = os.path.join(file_dir, "elm", "clinvar")

    mutation_types = [
        "Pathogenic",
        "Uncertain",
    ]

    summary_df = pd.DataFrame()

    for mut in mutation_types:
        print(mut)
        # motif loss mutations
        known_elm_motifs = pd.read_csv(os.path.join(elm_dir,'motif',mut, "clinvar_elm_overlaps_known.tsv"), sep="\t")
        predicted_elm_motifs = pd.read_csv(os.path.join(elm_dir,'motif',mut, "clinvar_elm_overlaps_predicted.tsv"), sep="\t")

        known_elm_motifs['Region_category'] = 'Known'
        predicted_elm_motifs['Region_category'] = 'Predicted'

        # motif gain mutations
        motif_gain_mutations = pd.read_csv(os.path.join(elm_dir,'motif_gain',mut, "motif_gain_mutations.tsv"), sep="\t")
        motif_gain_mutations['Region_category'] = 'Motif_gain'

        # pip_regions = pd.read_csv(os.path.join(elm_dir,'pip',mut, "clinvar_with_roi_overlaps.tsv"), sep="\t")
        # pip_regions['Region_category'] = 'PIP'

        big_df = pd.concat(
            [
                known_elm_motifs,
                predicted_elm_motifs,
                 # motif_gain_mutations,
             ],
            ignore_index=True).reset_index()

        to_dir = os.path.join(elm_dir,"merged",mut)
        os.makedirs(to_dir,exist_ok=True)
        big_df.to_csv(os.path.join(to_dir,"clinvar_regions_merged_overlaps.tsv"),index=False,sep="\t")

        big_df['Mutationset'] = mut
        summary_df = pd.concat([summary_df, big_df], ignore_index=True)

    output_summary = os.path.join(elm_dir, "summary_elm.tsv")
    generate_summary_tsv_elm(summary_df, output_summary)