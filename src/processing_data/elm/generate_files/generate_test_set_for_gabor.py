import pandas as pd
import os

if __name__ == "__main__":
    base_dir = r"/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files"
    ml_test_set = os.path.join(base_dir,"elm", "ml_test_set")

    known_motif = pd.read_csv(os.path.join(base_dir, "elm", "elm_known_with_am_info_all_disorder_class_filtered.tsv"),
                              sep="\t")
    predicted_motif = pd.read_csv(
        os.path.join(base_dir, "elm", "elm_predicted_with_am_info_all_disorder_class_filtered.tsv"), sep="\t")

    columns = ['Protein_ID', 'ELM_Accession', 'ELMType', 'ELMIdentifier',
               'Start','End', 'known',
               'motif_am_mean_score', 'motif_am_scores',
               'key_residue_am_mean_score', 'flanking_residue_am_mean_score',
               'is_terminated', 'sequential_am_score',
               'Key_vs_NonKey_Difference','Motif_vs_Sequential_Difference']

    known_motif = known_motif[columns]
    predicted_motif = predicted_motif[columns]

    print(known_motif.columns)

    big_df = pd.concat([known_motif, predicted_motif]).reset_index().drop(columns=['index'])
    # print(big_df[big_df['motif_am_mean_score'].isna()].shape[0])
    # print(big_df[big_df['Key_vs_NonKey_Difference'].isna()].shape[0])
    # print(big_df[big_df['Motif_vs_Sequential_Difference'].isna()].shape[0])
    # print(big_df)
    # print(big_df.columns)
    # exit()

    big_df.to_csv(os.path.join(ml_test_set, "disordered_class_filtered_motif_set.tsv"), sep="\t",index=False)