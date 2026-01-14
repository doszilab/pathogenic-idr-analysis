import pandas as pd
import pandas as pd
import os
from  matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

def create_sequential(df,shifting=0):

    lst =[]
    df['length'] = df['End'] - df['Start'] +1

    for index,row in df.iterrows():

        length = int(row['length'])
        if length > 20:
            length = 20

        fstart = int(row['Start']) + length + shifting
        fend = int(row['End']) + length + shifting
        bstart = int(row['Start']) - length - shifting
        bend = int(row['End']) - length - shifting
        forward_positions = [x for x in range(fstart, fend +1)]
        backward_positions = [x for x in range(bstart, bend +1)]
        for i,position in enumerate(forward_positions):
            lst.append([row['Protein_ID'],position])
            lst.append([row['Protein_ID'],backward_positions[i]])

    new_df = pd.DataFrame(lst,columns=['Protein_ID', 'Position'])
    return new_df

def plot_functional_site_distribution(df_list, column_name,tilte,figsize=(8, 4),fig_path=None):
    # Combine all dataframes into one
    combined_df = pd.concat(df_list)
    cleared_df = combined_df.dropna(subset=[column_name])

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(data=cleared_df, x='fname', y=column_name,color='#f08080', ax=ax)

    # plt.axhline(y=benign_cutoff, color='blue', linestyle='--', linewidth=2, label=f'Benign ({benign_cutoff})')
    # plt.axhline(y=half_cutoff, color='grey', linestyle='--', linewidth=2, label='0.5')
    # plt.axhline(y=pathogen_cutoff, color='red', linestyle='--', linewidth=2, label=f'Pathogen ({pathogen_cutoff})')

    # Set labels and title
    # plt.xlabel('Site Type')
    ax.set_xlabel(None)
    ax.set_ylabel('AlphaMissense Mean Score')
    ax.set_title(tilte)

    plt.tight_layout()

    for spine in ax.spines.values():
        spine.set_visible(False)

    if fig_path is not None:
        plt.savefig(fig_path,dpi=300)

    # Show the plot
    plt.show()


def plot_functional_site_distribution_two_subplot(df_list, column_name,tilte,figsize=(8, 4)):
    # Combine all dataframes into one
    combined_df = pd.concat(df_list)
    cleared_df = combined_df.dropna(subset=[column_name])

    # Create the plot
    plt.figure(figsize=figsize)
    sns.violinplot(data=cleared_df, x='fname', y=column_name, linewidth=1.5)

    # plt.axhline(y=benign_cutoff, color='blue', linestyle='--', linewidth=2, label=f'Benign ({benign_cutoff})')
    # plt.axhline(y=half_cutoff, color='grey', linestyle='--', linewidth=2, label='0.5')
    # plt.axhline(y=pathogen_cutoff, color='red', linestyle='--', linewidth=2, label=f'Pathogen ({pathogen_cutoff})')

    # Set labels and title
    # plt.xlabel('Site Type')
    plt.ylabel('AlphaMissense Score Distribution')
    plt.title(tilte)

    plt.tight_layout()

    # Show the plot
    plt.show()

def sequential_processing(df_regions,am_df,functional_am,type,main_col='AlphaMissense'):
    elm_sequential = create_sequential(df_regions, shifting=0)
    elm_sequential_plus_10 = create_sequential(df_regions,shifting=10)

    elm_sequential_am = elm_sequential.merge(am_df, on=['Protein_ID', 'Position'])

    elm_sequential_am['fname'] = f"{type} Sequential"

    df_list = [functional_am, elm_sequential_am ]
    plot_functional_site_distribution(df_list, main_col,
                                      f'Distribution of {main_col} score {type} and Sequential Environment')

def extract_pos_based_df(df):
    df[['Protein_ID', 'Position']] = df['AccessionPosition'].str.split('|', expand=True)
    df = df.drop(columns=['AccessionPosition'])
    df['Position'] = df['Position'].astype(int)
    return df

def sequential_processing_final(df_elm,df_pfam,am_df,functional_am_pfam,functional_am_elm,figsize=(8, 4)):
    elm_sequential = create_sequential(df_elm, shifting=0)
    elm_sequential_am = elm_sequential.merge(am_df, on=['Protein_ID', 'Position'])
    elm_sequential_am['fname'] = f"ELM Sequential"

    pfam_sequential = create_sequential(df_pfam, shifting=0)
    pfam_sequential_am = pfam_sequential.merge(am_df, on=['Protein_ID', 'Position'])
    pfam_sequential_am['fname'] = f"Pfam Sequential"

    df_list = [functional_am_elm, elm_sequential_am,  functional_am_pfam, pfam_sequential_am]
    plot_functional_site_distribution(df_list, "AlphaMissense",
                                      f'Distribution of AlphaMissense scores for Non and Sequential Environment',
                                      figsize=figsize)

def main_plots(df_list,sequential_df_list,figsize=(8, 4),column_name='AlphaMissense',fig_dir=None):
    functional_df = pd.concat(df_list)
    cleared_functional_df_df = functional_df.dropna(subset=[column_name])

    sequential_df = pd.concat(sequential_df_list)
    sequential_functional_df_df = sequential_df.dropna(subset=[column_name])

    # Create the plot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    sns.violinplot(data=cleared_functional_df_df, x='fname', y=column_name,ax=axes[0],color='#f08080')
    axes[0].set_title("Disorder Functional")
    axes[0].set_xlabel(None)
    axes[1].set_ylabel("AlphaMissense Score")


    sns.violinplot(data=sequential_functional_df_df, x='fname', y=column_name,ax=axes[1],color='#f08080')
    axes[1].set_title("Short Linear Motif Specificity")
    axes[1].set_ylabel(None)
    axes[1].set_xlabel(None)
    axes[1].set_yticklabels([])

    plt.suptitle("Disorder Specific Distribution of AlphaMissense Scores")
    plt.tight_layout()

    # Show the plot
    if fig_dir is not None:
        plt.savefig(os.path.join(fig_dir, "C.png"), bbox_inches='tight')
    plt.show()


def get_binding_regions(am_disorder):
    pos_based_dir = f'{core_dir}/data/discanvis_base_files/positional_data_process'
    aiupred_binding_df = extract_pos_based_df(pd.read_csv(f"{pos_based_dir}/aiupred_binding_pos.tsv", sep='\t'))
    aiupred_binding_df = aiupred_binding_df[aiupred_binding_df["Protein_ID"].isin(am_disorder["Protein_ID"])]
    aiupred_binding_df = aiupred_binding_df[aiupred_binding_df["AIUPredBinding"] > 0.5]

    # Include only CAID scores
    caid_path = os.path.join(core_dir, 'processed_data', 'files', 'caid', 'binding_position_table.tsv')
    binding_position_df = pd.read_csv(caid_path, sep='\t')
    non_binding_df = binding_position_df[binding_position_df['Binding'] == 0]
    binding_position_df = binding_position_df[binding_position_df['Binding'] == 1]


    aiupred_binding_df = aiupred_binding_df[aiupred_binding_df['Protein_ID'].isin(binding_position_df["Protein_ID"])]

    aiupred_binding_am = aiupred_binding_df.merge(am_disorder, on=['Protein_ID', 'Position'])
    aiupred_binding_am['fname'] = "AIUPred\nBinding"

    anchor_df = extract_pos_based_df(pd.read_csv(f"{pos_based_dir}/anchor_pos.tsv", sep='\t'))
    anchor_df = anchor_df[anchor_df["Protein_ID"].isin(am_disorder["Protein_ID"])]
    anchor_df = anchor_df[anchor_df["AnchorScore"] > 0.5]

    anchor_df = anchor_df[anchor_df['Protein_ID'].isin(binding_position_df["Protein_ID"])]

    anchor_am = anchor_df.merge(am_disorder, on=['Protein_ID', 'Position'])
    anchor_am['fname'] = "Anchor"

    caid_am = binding_position_df.merge(am_disorder, on=['Protein_ID', 'Position'])
    caid_am['fname'] = "Binding"
    caid_non_am = non_binding_df.merge(am_disorder, on=['Protein_ID', 'Position'])
    caid_non_am['fname'] = "Non Binding"

    df_list = [caid_am, caid_non_am, aiupred_binding_am, anchor_am]
    fig_path = f'{fig4}/binding_region.png'
    plot_functional_site_distribution(df_list, "AlphaMissense", f'Binding Region Distribution',
                                      figsize=(4, 3), fig_path=fig_path)

    return

if __name__ == "__main__":
    core_dir  = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations'

    fig4 = f'{core_dir}/plots/fig4'

    to_alphamissense_dir = '/dlab/home/norbi/PycharmProjects/AlphaMissense_Stat/processed_data/files/alphamissense'

    # Fig 2
    # am_order = pd.read_csv(f"{to_alphamissense_dir}/am_order.tsv", sep='\t', )
    # am_disorder = pd.read_csv(f"{to_alphamissense_dir}/am_disorder.tsv", sep='\t', )

    # print("Mean for Order",am_order["AlphaMissense"].mean())
    # print("Mean for Disorder",am_disorder["AlphaMissense"].mean())

    # am_order['fname'] = "Ordered"
    # am_disorder['fname'] = "Disordered"

    # df_list = [am_order, am_disorder]
    # fig_path = f'{fig4}/A3.png'
    # plot_functional_site_distribution(df_list, "AlphaMissense", f'Structural Distribution',
    #                                   figsize=(4,3),fig_path=fig_path)

    # get_binding_regions(am_disorder)

    # exit()


    pos_based_dir = f'{core_dir}/data/discanvis_base_files/positional_data_process'
    alphamissense_df = extract_pos_based_df(pd.read_csv(f"{pos_based_dir}/alphamissense_pos.tsv", sep='\t'))

    mobidb = extract_pos_based_df(pd.read_csv(f"{pos_based_dir}/mobidb_pos.tsv", sep='\t'))
    mobidb_am = mobidb.merge(alphamissense_df, on=['Protein_ID', 'Position'])
    mobidb_am['fname'] = "Exp.Dis"

    dibs_mfib_phasepro_binding = extract_pos_based_df(
        pd.read_csv(f"{pos_based_dir}/binding_mfib_phasepro_dibs_pos.tsv", sep='\t'))
    dibs_mfib_phasepro_binding_am = dibs_mfib_phasepro_binding.merge(alphamissense_df, on=['Protein_ID', 'Position'])

    mfib_df = dibs_mfib_phasepro_binding_am[dibs_mfib_phasepro_binding_am['mfib_info'].notna()]
    mfib_df['fname'] = "MFIB"
    #
    dibs_df = dibs_mfib_phasepro_binding_am[dibs_mfib_phasepro_binding_am['dibs_info'].notna()]
    dibs_df['fname'] = "DIBS"
    #
    phasepro_df = dibs_mfib_phasepro_binding_am[dibs_mfib_phasepro_binding_am['phasepro_info'].notna()]
    phasepro_df['fname'] = "PhasePro"
    #
    elm_df = extract_pos_based_df(pd.read_csv(f"{pos_based_dir}/elm_pos.tsv", sep='\t'))
    elm_am = elm_df.merge(alphamissense_df, on=['Protein_ID', 'Position'])
    elm_am['fname'] = "SLiM"

    df_list = [mobidb_am, mfib_df, dibs_df, phasepro_df]

    # FIG 6)
    # plot_functional_site_distribution(df_list, "AlphaMissense",
    #                                   f'Distribution of AlphaMissense score accross Disordered and Functional Sites',
    #                                   figsize=(4, 3)
    #                                   )

    # exit()

    """
    ------------------------
    """

    elm_regions = pd.read_csv(
        f"{core_dir}/data/discanvis_base_files/elm/elm_for_am_mapped.tsv",
        sep='\t')

    pfam_regions = pd.read_csv(
        f"{core_dir}/data/discanvis_base_files/pfam/pfamtable.tsv",
        sep='\t')
    only_domains_regions = pfam_regions.loc[pfam_regions['type'] == 'Domain']
    only_domains_regions = only_domains_regions.rename(columns={'envelope_start': 'Start', 'envelope_end': 'End'})

    pfam_df = extract_pos_based_df(pd.read_csv(f"{pos_based_dir}/pfam_pos.tsv", sep='\t'))
    only_domains_df = pfam_df.loc[pfam_df['Pfam_Info'].str.contains('Domain')]
    pfam_am = only_domains_df.merge(alphamissense_df, on=['Protein_ID', 'Position'])
    pfam_am['fname'] = "Domain"

    elm_sequential = create_sequential(elm_regions, shifting=0)
    elm_sequential_am = elm_sequential.merge(alphamissense_df, on=['Protein_ID', 'Position'])
    elm_sequential_am['fname'] = f"SLiM Flanking"

    pfam_sequential = create_sequential(only_domains_regions, shifting=0)
    pfam_sequential_am = pfam_sequential.merge(alphamissense_df, on=['Protein_ID', 'Position'])
    pfam_sequential_am['fname'] = f"Domain Flanking"

    sequential_df_list = [elm_am, elm_sequential_am, pfam_am, pfam_sequential_am]

    # FIG 4C)
    main_plots(df_list,sequential_df_list,figsize=(10,3),fig_dir=fig4)

    # FIG 7A)
    # sequential_processing_final(elm_regions,only_domains_regions,alphamissense_df,pfam_am,elm_am,figsize=(4,3))