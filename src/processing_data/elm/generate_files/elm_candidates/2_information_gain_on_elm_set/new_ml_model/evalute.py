import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":

    ml_model_dir = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/elm/ml_model/"
    df_all = pd.read_csv(os.path.join(ml_model_dir,"final_motif_class_and_probability.tsv"),sep='\t')

    higher_prob_check = df_all[df_all['Motif_Probability'] > 0.1]
    sns.histplot(higher_prob_check[higher_prob_check['known'] == 0]['Motif_Probability'], bins=100)
    plt.show()
    sns.histplot(higher_prob_check[higher_prob_check['known'] == 1]['Motif_Probability'], bins=100)
    plt.show()

    predicted_df = df_all[df_all['Motif_Probability'] >= 0.5]
    predicted_df = predicted_df.sort_values(by=['ELMIdentifier','Protein_ID','Start'])
    predicted_df.to_csv(os.path.join(ml_model_dir,"predicted_motifs.tsv"),sep='\t',index=False)
