import os.path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_score_by_position(df,fig_path,protein,figsize=(30, 1)):
    """
    Plot a histogram based on scores and positions in a DataFrame.
    Scores above 0.5 are red, and scores below or equal to 0.5 are blue.

    Parameters:
    df (pd.DataFrame): A DataFrame with two columns: 'Position' and 'AlphaMissense'.
    """
    df = df[df['Protein_ID'] == protein]

    if not {'Position', 'AlphaMissense'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'Position' and 'AlphaMissense' columns.")

    # Sort by position to ensure proper visualization
    df = df.sort_values(by='Position')

    # Create histogram with positions on x-axis and scores on y-axis
    plt.figure(figsize=figsize)  # Larger plot for better readability

    # Plot scores above 0.5 in red
    above_threshold = df[df['AlphaMissense'] > 0.5]
    plt.bar(above_threshold['Position'], above_threshold['AlphaMissense'], color='red', label='Above 0.5',alpha=0.8)

    # Plot scores below or equal to 0.5 in blue
    below_threshold = df[df['AlphaMissense'] <= 0.5]
    plt.bar(below_threshold['Position'], below_threshold['AlphaMissense'], color='blue', label='Below or Equal to 0.5',alpha=0.8)

    # Customize the plot
    # plt.grid(alpha=0.3)

    # Remove axis labels
    # plt.xticks([])  # Remove x-axis ticks (positions)
    plt.yticks([])  # Remove y-axis ticks (scores)

    # Remove the box around the plot
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    # Show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path,f'{protein}.png'),dpi=300)
    plt.show()

def extract_pos_based_df(df):
    df[['Protein_ID', 'Position']] = df['AccessionPosition'].str.split('|', expand=True)
    df = df.drop(columns=['AccessionPosition'])
    df['Position'] = df['Position'].astype(int)
    return df

if __name__ == "__main__":

    fig_path = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/plots/figexample/alphamissense_scores"


    df = extract_pos_based_df(pd.read_csv("/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/data/discanvis_base_files/positional_data_process/alphamissense_pos.tsv",sep='\t',
                                          # nrows=1000
                                          ))


    # WNK1 - main example
    # plot_score_by_position(df,fig_path,'WNK1-201',figsize=(30, 1))
    # POLK1 - main example
    plot_score_by_position(df,fig_path,'POLK-201',figsize=(30, 1))
    # LMOD3 - main example
    # plot_score_by_position(df,fig_path,'LMOD3-201',figsize=(30, 1))
