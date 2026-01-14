import os
import pandas as pd


if __name__ == "__main__":

    motif_path = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/elm/clinvar/motif"
    to_dir_path = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/elm/simplified_data"

    mut_types = ['Pathogenic',
                 # 'Predicted_Pathogenic',
                 'Uncertain'
                 ]
    elm_types = ['known','predicted']
    needed_cols = ['Protein_ID', 'Motif_Start', 'Motif_End',
                   'ELMIdentifier', 'ELMType',
                   'N_Motif_Predicted', 'N_Motif_Category', 'ELM_Types_Count',
                   'diseases', 'number_of_diseases', 'Number_of_Mutations',
                   ]

    for mut_type in mut_types:
        for elm_type in elm_types:
            this_dir = os.path.join(motif_path,mut_type)
            to_dir = os.path.join(to_dir_path,mut_type)
            motif = pd.read_csv(os.path.join(this_dir,f"motif_with_clinvar_{elm_type}.tsv"),sep='\t')

            if not os.path.exists(to_dir):
                os.mkdir(to_dir)

            sorted_motif = motif[needed_cols].sort_values(by='ELM_Types_Count')
            sorted_motif.to_csv(os.path.join(to_dir,f"motif_with_clinvar_{elm_type}_simplified.tsv"),sep='\t',index=False)
    pass