import pandas as pd
import matplotlib.pyplot as plt
import os

def extract_pos_based_df(df):
    df[['Protein_ID', 'Position']] = df['AccessionPosition'].str.split('|', expand=True)
    df = df.drop(columns=['AccessionPosition'])
    df['Position'] = df['Position'].astype(int)
    return df

def disorder_filter(df):
    structure_path = '/dlab/home/norbi/PycharmProjects/DisCanVis_Data_Process/Processed_Data/gencode_process/positional_data_process'
    combined_disorder_df = extract_pos_based_df(pd.read_csv(f"{structure_path}/combined_dis_pos.tsv", sep='\t'))
    disorder = combined_disorder_df[combined_disorder_df['CombinedDisorder'] == 1]

    df_filtered = df.merge(disorder, on=['Protein_ID', 'Position'])
    print(df_filtered)
    print(df)
    return df_filtered

def plot_predictor_percentages_stacked_horizontal(results_df, plot_dir, figsize=(8, 6), width=0.6):
    fig, ax = plt.subplots(figsize=figsize)
    predictors = results_df['Predictor'].unique()
    y_labels = []
    y_positions = []

    for i, predictor in enumerate(predictors):
        total = results_df[results_df['Predictor'] == predictor]
        disorder_data = results_df[(results_df['Predictor'] == predictor) & (results_df['Region'] == 'disorder')]
        order_data = results_df[(results_df['Predictor'] == predictor) & (results_df['Region'] == 'order')]

        total_count = total['Pathogenic'].sum() + total['Benign'].sum()

        if predictor == "ClinVar":
            continue

        y_center = i  # Position each predictor on the y-axis
        y_labels.append(predictor)
        y_positions.append(y_center)

        # Initialize starting position for stacking
        start_position = 0

        # Disorder data
        if not disorder_data.empty:
            if predictor == "Mutation":
                pathogenic_total = disorder_data['Pathogenic'].sum()
                benign_total = disorder_data['Benign'].sum()
            else:
                pathogenic_total = disorder_data['Pathogenic'].values[0]
                benign_total = disorder_data['Benign'].values[0]

            total_disorder = benign_total + pathogenic_total

            ax.barh(y_center, pathogenic_total, height=width, left=start_position, color='red', alpha=0.7,
                    label='Pathogenic (Disorder)' if i == 0 else "")
            if predictor == "AlphaMissense":
                ax.text(start_position + pathogenic_total / 2, y_center + 0.3,
                        f'{pathogenic_total / 1_000_000:.1f}M \n ({pathogenic_total / total_disorder * 100:.0f}%)',
                        va='bottom', ha='center', fontsize=9)
            start_position += pathogenic_total

            ax.barh(y_center, benign_total, height=width, left=start_position, color='lightcoral', alpha=0.7,
                    label='Benign (Disorder)' if i == 0 else "")
            if predictor == "AlphaMissense":
                ax.text(start_position + benign_total / 2, y_center + 0.3 ,
                        f'{benign_total / 1_000_000:.1f}M \n ({benign_total / total_disorder * 100:.0f}%)',
                        va='bottom', ha='center', fontsize=9)
            start_position += benign_total

        # Order data
        if not order_data.empty:
            if predictor == "Mutation":
                pathogenic_total = order_data['Pathogenic'].sum()
                benign_total = order_data['Benign'].sum()
            else:
                pathogenic_total = order_data['Pathogenic'].values[0]
                benign_total = order_data['Benign'].values[0]

            total_order = benign_total + pathogenic_total

            ax.barh(y_center, pathogenic_total, height=width, left=start_position, color='blue', alpha=0.7,
                    label='Pathogenic (Order)' if i == 0 else "")
            if predictor == "AlphaMissense":
                ax.text(start_position + pathogenic_total / 2, y_center + 0.3 ,
                        f'{pathogenic_total / 1_000_000:.1f}M \n ({pathogenic_total / total_order * 100:.0f}%)',
                        va='bottom', ha='center', fontsize=9)
            start_position += pathogenic_total

            ax.barh(y_center, benign_total, height=width, left=start_position, color='lightblue', alpha=0.7,
                    label='Benign (Order)' if i == 0 else "")
            if predictor == "AlphaMissense":
                ax.text(start_position + benign_total / 2, y_center + 0.3  ,
                        f'{benign_total / 1_000_000:.1f}M \n ({benign_total / total_order * 100:.0f}%)',
                        va='bottom', ha='center', fontsize=9)
            start_position += benign_total

        # Display the total count on the right side of each bar
        ax.text(start_position + 1, y_center, f'{total_count / 1_000_000:.3f}M', va='center', ha='left', fontsize=10,
                color='black')

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Number Of Positions')
    ax.set_title('Clinical Significance Classification for AlphaMissense')

    # Get current x-axis limit
    xmin, xmax = ax.get_xlim()

    # Set new x-axis limit with a 10% increase on the maximum limit
    ax.set_xlim(xmin, xmax * 1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'alphamissense_predictor.png'), bbox_inches='tight')
    plt.show()

def plot_predictor_percentages_stacked_horizontal_mut_based(disorder_data,order_data, plot_dir, figsize=(8, 6), width=0.6):
    fig, ax = plt.subplots(figsize=figsize)
    # predictors = results_df['Predictor'].unique()
    y_labels = []
    y_positions = []
    predictor = "AlphaMissense"

    # for i, predictor in enumerate(predictors):
    #     total = results_df[results_df['Predictor'] == predictor]
    #     disorder_data = results_df[(results_df['Predictor'] == predictor) & (results_df['Region'] == 'disorder')]
    #     order_data = results_df[(results_df['Predictor'] == predictor) & (results_df['Region'] == 'order')]
    #
    #     total_count = total['Pathogenic'].sum() + total['Benign'].sum()
    #
    #     if predictor == "ClinVar":
    #         continue

    i = 0
    y_center = i
    y_labels.append(predictor)
    y_positions.append(y_center)

    # Initialize starting position for stacking
    start_position = 0
    total = pd.concat([disorder_data,order_data])
    total_count = total['pathogenic'].sum() + total['benign'].sum()

    # Disorder data
    if not disorder_data.empty:
        pathogenic_total = disorder_data['pathogenic'].sum()
        benign_total = disorder_data['benign'].sum()
        total_disorder = benign_total + pathogenic_total

        ax.barh(y_center, pathogenic_total, height=width, left=start_position, color='red', alpha=0.7,
                label='Pathogenic (Disorder)' if i == 0 else "")
        if predictor == "AlphaMissense":
            ax.text(start_position + pathogenic_total / 2, y_center + 0.3,
                    f'{pathogenic_total / 1_000_000:.1f}M \n ({pathogenic_total / total_disorder * 100:.0f}%)',
                    va='bottom', ha='center', fontsize=9)
        start_position += pathogenic_total

        ax.barh(y_center, benign_total, height=width, left=start_position, color='lightcoral', alpha=0.7,
                label='Benign (Disorder)' if i == 0 else "")
        if predictor == "AlphaMissense":
            ax.text(start_position + benign_total / 2, y_center + 0.3 ,
                    f'{benign_total / 1_000_000:.1f}M \n ({benign_total / total_disorder * 100:.0f}%)',
                    va='bottom', ha='center', fontsize=9)
        start_position += benign_total

    # Order data
    if not order_data.empty:
        pathogenic_total = order_data['pathogenic'].sum()
        benign_total = order_data['benign'].sum()
        total_order = benign_total + pathogenic_total

        ax.barh(y_center, pathogenic_total, height=width, left=start_position, color='blue', alpha=0.7,
                label='Pathogenic (Order)' if i == 0 else "")
        if predictor == "AlphaMissense":
            ax.text(start_position + pathogenic_total / 2, y_center + 0.3 ,
                    f'{pathogenic_total / 1_000_000:.1f}M \n ({pathogenic_total / total_order * 100:.0f}%)',
                    va='bottom', ha='center', fontsize=9)
        start_position += pathogenic_total

        ax.barh(y_center, benign_total, height=width, left=start_position, color='lightblue', alpha=0.7,
                label='Benign (Order)' if i == 0 else "")
        if predictor == "AlphaMissense":
            ax.text(start_position + benign_total / 2, y_center + 0.3  ,
                    f'{benign_total / 1_000_000:.1f}M \n ({benign_total / total_order * 100:.0f}%)',
                    va='bottom', ha='center', fontsize=9)
        start_position += benign_total

    # Display the total count on the right side of each bar
    ax.text(start_position + 1, y_center, f'{total_count / 1_000_000:.3f}M', va='center', ha='left', fontsize=10,
            color='black')

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    # ax.set_xlabel('Number Of Positions')
    # ax.set_title('Clinical Significance Classification for AlphaMissense')

    # Get current x-axis limit
    xmin, xmax = ax.get_xlim()

    # Set new x-axis limit with a 10% increase on the maximum limit
    ax.set_xlim(xmin, xmax * 1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'alphamissense_predictor.png'), bbox_inches='tight')
    plt.show()

def classify_mutations(df, region, rules,columns):
    results = {
        'Predictor': [],
        'Region': [],
        'Total': [],
        'Pathogenic': [],
        'Benign': [],
        'Predictor_Total': [],
        'NAN_Values': [],
    }
    total = df.shape[0]
    for predictor in columns:
        rule = rules[predictor]
        if predictor in df.columns:
            # Drop NaN values for the predictor column
            number_of_na_values = df[predictor].isna().sum()
            predictor_values = df[predictor].dropna()
            total_count = len(predictor_values)

            if total_count > 0:
                # Classify based on the threshold
                if rule['above']:
                    pathogenic_count = (predictor_values >= rule['threshold']).sum()
                else:
                    pathogenic_count = (predictor_values < rule['threshold']).sum()

                benign_count = total_count - pathogenic_count

                # Append the results
                results['Predictor'].append(predictor)
                results['Region'].append(region)
                results['Predictor_Total'].append(total_count)
                results['Total'].append(total)
                results['Pathogenic'].append(pathogenic_count)
                results['Benign'].append(benign_count)
                results['NAN_Values'].append(number_of_na_values)

    return  pd.DataFrame(results)

def make_groups_based_on_stucture_and_interpretation(df):
    structure = df['structure'].unique()

    results = {
        'Predictor': [],
        'Region': [],
        'Total': [],
        'Pathogenic': [],
        'Benign': [],
        'Uncertain': [],
    }


    for j in structure:
        current_df = df[df['structure'] == j]
        pathogenic = current_df[current_df['Interpretation'] == 'Pathogenic'].shape[0]
        benign = current_df[current_df['Interpretation'] == 'Benign'].shape[0]
        uncertain = current_df[current_df['Interpretation'] == 'Uncertain'].shape[0]

        # Append the results
        results['Predictor'].append("ClinVar")
        results['Region'].append(j)
        results['Total'].append(current_df.shape[0])
        results['Pathogenic'].append(pathogenic)
        results['Benign'].append(benign)
        results['Uncertain'].append(uncertain)

        # Append the results
        results['Predictor'].append("ClinVar Pathogenic")
        results['Region'].append(j)
        results['Total'].append(pathogenic)
        results['Pathogenic'].append(pathogenic)
        results['Benign'].append(0)
        results['Uncertain'].append(0)

    return pd.DataFrame(results)


def alphamissense_pos_based_distribution():
    clinvar_df = pd.read_csv(os.path.join(core_dir, "clinvar", 'clinvar_mutations_with_annotations_merged.tsv'),
                             sep='\t')
    print(clinvar_df)
    position_with_mutation_df = clinvar_df[['Protein_ID', "Position", "Interpretation", "structure"]].drop_duplicates()
    print(position_with_mutation_df)

    clinvar_classified = make_groups_based_on_stucture_and_interpretation(position_with_mutation_df)

    am_disorder = pd.read_csv(f"{core_dir}/alphamissense/am_disorder.tsv", sep='\t')
    am_order = pd.read_csv(f"{core_dir}/alphamissense/am_order.tsv", sep='\t')

    columns = [
        "AlphaMissense",
    ]

    rules = {
        'AlphaMissense': {'threshold': 0.5, 'above': True},
    }

    disorder_results_df = classify_mutations(am_disorder, 'disorder', rules, columns)
    order_results_df = classify_mutations(am_order, 'order', rules, columns)

    results_df = pd.concat([disorder_results_df, order_results_df, clinvar_classified])
    print(results_df)

    # results_df.to_csv( f'{core_dir}/alphamissense/alphamissense_prediction_results.tsv', index=False,sep='\t')

    plot_predictor_percentages_stacked_horizontal(results_df, plot_dir, figsize=(12, 3), width=0.4)


def alphamissense_mut_based_distribution():
    # Define the directory and file paths
    am_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/alphamissense'

    disorder_class_count_path = f'{am_dir}/am_disorder_class_count.tsv'
    order_class_count_path = f'{am_dir}/am_order_class_count.tsv'

    disorder_class_count_df = pd.read_csv(disorder_class_count_path, sep='\t')
    order_class_count_df = pd.read_csv(order_class_count_path, sep='\t')

    plot_predictor_percentages_stacked_horizontal_mut_based(disorder_class_count_df,order_class_count_df, plot_dir, figsize=(12, 2), width=0.4)


def plot_predictor_percentages_stacked_horizontal_percentage(results_df, plot_dir, figsize=(8, 6), width=0.6,fontsize=10):
    fig, ax = plt.subplots(figsize=figsize)
    predictors = results_df['Predictor'].unique()
    y_labels = []
    y_positions = []

    for i, predictor in enumerate(predictors):
        if predictor == "ClinVar":
            continue

        disorder_data = results_df[(results_df['Predictor'] == predictor) & (results_df['Region'] == 'disorder')]
        order_data = results_df[(results_df['Predictor'] == predictor) & (results_df['Region'] == 'order')]

        y_center = i  # Position each predictor on the y-axis
        y_labels.append(predictor)
        y_positions.append(y_center)

        start_position = 0  # Initialize starting position for stacking

        if predictor == "Mutation":
            total = disorder_data['Benign'].sum() + disorder_data['Pathogenic'].sum() + order_data['Pathogenic'].sum() + order_data['Benign'].sum()
        else:
            total = disorder_data['Benign'].values[0] + disorder_data['Pathogenic'].values[0] + order_data['Pathogenic'].values[0] + order_data['Benign'].values[0]

        for data, pathogenic_color, benign_color, region_label in [
            (disorder_data, 'red', 'lightcoral', 'Disorder'),
            (order_data, 'blue', 'lightblue', 'Order')
        ]:
            if not data.empty:
                if predictor == "Mutation":
                    pathogenic_total = data['Pathogenic'].sum()
                    benign_total = data['Benign'].sum()
                else:
                    pathogenic_total = data['Pathogenic'].values[0]
                    benign_total = data['Benign'].values[0]


                # total = pathogenic_total + benign_total
                if total == 0:
                    continue  # Avoid division by zero

                total_current = benign_total + pathogenic_total
                pathogenic_pct = (pathogenic_total / total) * 100
                benign_pct = (benign_total / total) * 100


                ax.barh(y_center, pathogenic_pct, height=width, left=start_position, color=pathogenic_color, alpha=0.7,
                        label=f'Pathogenic ({region_label})' if i == 0 else "")

                ax.text(start_position + pathogenic_pct / 2, y_center + 0.4 ,
                        f'{pathogenic_total / 1_000_000:.1f}M\n({pathogenic_total / total_current * 100:.0f}%)',
                        va='center', ha='left', fontsize=fontsize, color='black')

                ax.barh(y_center, benign_pct, height=width, left=start_position + pathogenic_pct, color=benign_color,
                        alpha=0.7,
                        label=f'Benign ({region_label})' if i == 0 else "")

                ax.text(start_position + pathogenic_pct + benign_pct / 2, y_center + 0.4 ,
                        f'{benign_total / 1_000_000:.1f}M\n({benign_total / total_current * 100:.0f}%)',
                        va='center', ha='left', fontsize=fontsize, color='black')

                start_position += pathogenic_pct + benign_pct  # Normalize to 100% per predictor

        # Display the total count on the right side of each bar
        ax.text(start_position + 1, y_center, f'{total / 1_000_000:.3f}M', va='center', ha='left', fontsize=fontsize,
                color='black')


    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels,fontsize=fontsize)
    ax.set_xlabel('Percentage',fontsize=fontsize)
    # fig.suptitle('Clinical Significance Classification for AlphaMissense')
    ax.set_xlim(0, 100)  # Ensure all bars span 100%

    plt.tight_layout()
    plt.subplots_adjust(top=0.8)
    plt.savefig(os.path.join(plot_dir, 'alphamissense_predictor_percentage.png'), bbox_inches='tight')
    plt.show()


def plot_pathogenic_piechart(results_df,figsize=(8, 6), fontsize=14,mut_type="Mutation"):
    # Calculate pathogenic totals for order and disorder regions
    disorder_data = results_df[(results_df['Region'] == 'disorder') & (results_df['Predictor'] == mut_type)]
    order_data = results_df[(results_df['Region'] == 'order') & (results_df['Predictor'] == mut_type)]

    # Total Pathogenic counts for Mutation and Position
    disorder_pathogenic_total = disorder_data['Pathogenic'].sum()
    order_pathogenic_total = order_data['Pathogenic'].sum()

    print(disorder_pathogenic_total)
    print(order_pathogenic_total)

    # Create the pie chart data
    labels = ['Disorder', 'Order']
    sizes = [disorder_pathogenic_total, order_pathogenic_total]
    explode = (0.1, 0)  # Explode the "Disorder" slice for emphasis

    # Plotting the pie chart
    fig, ax = plt.subplots(figsize=figsize)
    ax.pie(sizes, autopct='%1.1f%%', startangle=90,pctdistance=1.25,
           colors=['red', 'blue'])

    # Add total numbers
    # ax.text(-1.5, 0.8, f'Total Pathogenic: {total_pathogenic:,}', fontsize=fontsize, color='black')

    # Title and display
    ax.set_title(f'Pathogenic {mut_type} Distribution', fontsize=fontsize)

    plt.tight_layout()
    plt.show()

def alphamissense_both_distribution():
    # Define the directory and file paths
    am_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/alphamissense'

    disorder_class_count_path = f'{am_dir}/am_disorder_class_count.tsv'
    order_class_count_path = f'{am_dir}/am_order_class_count.tsv'

    disorder_class_count_df = pd.read_csv(disorder_class_count_path, sep='\t').rename(columns={"pathogenic":"Pathogenic","benign":"Benign"})
    order_class_count_df = pd.read_csv(order_class_count_path, sep='\t').rename(columns={"pathogenic":"Pathogenic","benign":"Benign"})

    am_disorder = pd.read_csv(f"{core_dir}/alphamissense/am_disorder.tsv", sep='\t')
    am_order = pd.read_csv(f"{core_dir}/alphamissense/am_order.tsv", sep='\t')

    columns = [
        "AlphaMissense",
    ]

    rules = {
        'AlphaMissense': {'threshold': 0.5, 'above': True},
    }

    disorder_results_df = classify_mutations(am_disorder, 'disorder', rules, columns)
    order_results_df = classify_mutations(am_order, 'order', rules, columns)

    disorder_results_df['Predictor'] = 'Position'
    order_results_df['Predictor'] = 'Position'


    disorder_class_count_df['Region'] = 'disorder'
    disorder_class_count_df['Predictor'] = 'Mutation'
    order_class_count_df['Region'] = 'order'
    order_class_count_df['Predictor'] = 'Mutation'

    results_df = pd.concat([disorder_results_df, order_results_df,disorder_class_count_df,order_class_count_df])

    # plot_predictor_percentages_stacked_horizontal_percentage(results_df, plot_dir, figsize=(10, 3), width=0.4,fontsize=11)

    # plot_pathogenic_piechart(results_df, figsize=(3, 3),fontsize=11)
    plot_pathogenic_piechart(results_df, figsize=(3, 3), fontsize=12, mut_type="Position")




if __name__ == "__main__":

    core_dir = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files"
    plot_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/plots/fig4'


    # alphamissense_mut_based_distribution()
    alphamissense_both_distribution()