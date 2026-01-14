import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def create_gene_distribution(disorder_data, order_data,max_count=20,file=None):
    disorder_gene_counts = disorder_data['Protein_ID'].value_counts()
    order_gene_counts = order_data['Protein_ID'].value_counts()

    # Create sets of all unique genes
    disorder_genes = set(disorder_gene_counts.index)
    order_genes = set(order_gene_counts.index)

    above_label = f">{max_count}"

    # Initialize data structures for the stacked bars
    mutation_distribution = {k: {'Disorder Only': [], 'Order Only': [], 'Both': [], 'Disorder Mostly': [], 'Order Mostly': [],'Equal':[]} for k in range(1, max_count + 1)}
    mutation_distribution[above_label] = {'Disorder Only': [], 'Order Only': [], 'Both': [], 'Disorder Mostly': [], 'Order Mostly': [],'Equal':[]}

    # Categorize and aggregate gene counts
    for gene, count in disorder_gene_counts.items():
        category = "Disorder Only" if gene not in order_genes else "Both"
        if category == "Both":
            ordered_count = order_gene_counts[gene]
            if ordered_count > count:
                inner_category = "Order Mostly"
            elif ordered_count < count:
                inner_category = "Disorder Mostly"
            else:
                inner_category = 'Equal'

            count = count + ordered_count

            if inner_category:
                if count > max_count:
                    mutation_distribution[above_label][inner_category].append(gene)
                else:
                    mutation_distribution[count][inner_category].append(gene)


        if count > max_count:
            mutation_distribution[above_label][category].append(gene)
        else:
            mutation_distribution[count][category].append(gene)

    for gene, count in order_gene_counts.items():
        if gene not in disorder_genes:
            category = "Order Only"
            if count > max_count:
                mutation_distribution[above_label][category].append(gene)
            else:
                mutation_distribution[count][category].append(gene)

    if file:
        with open(file,'w') as f:
            f.write("{}\t{}\t{}\n".format("Category", "Gene","Count"))
            for count, dct in mutation_distribution.items():
                for category, genes in dct.items():
                    for gene in genes:
                        f.write("{}\t{}\t{}\n".format(category, gene,count))


def create_gene_distribution_with_disease(disorder_data, order_data, max_count=20, file=None, extra_columns=None):
    if extra_columns is None:
        extra_columns = ['nDisease']

    disorder_gene_counts = disorder_data['Protein_ID'].value_counts()
    order_gene_counts = order_data['Protein_ID'].value_counts()

    # Create sets of all unique genes
    disorder_genes = set(disorder_gene_counts.index)
    order_genes = set(order_gene_counts.index)

    above_label = f">{max_count}"

    # Initialize data structures for the stacked bars
    mutation_distribution = {k: {'Disorder Only': [], 'Order Only': [], 'Both': [], 'Disorder Mostly': [], 'Order Mostly': [], 'Equal': []} for k in range(1, max_count + 1)}
    mutation_distribution[above_label] = {'Disorder Only': [], 'Order Only': [], 'Both': [], 'Disorder Mostly': [], 'Order Mostly': [], 'Equal': []}

    # Categorize and aggregate gene counts
    for gene, count in disorder_gene_counts.items():
        category = "Disorder Only" if gene not in order_genes else "Both"
        if category == "Both":
            ordered_count = order_gene_counts[gene]
            if ordered_count > count:
                inner_category = "Order Mostly"
            elif ordered_count < count:
                inner_category = 'Disorder Mostly'
            else:
                inner_category = 'Equal'

            count = count + ordered_count

            if inner_category:
                if count > max_count:
                    mutation_distribution[above_label][inner_category].append(gene)
                else:
                    mutation_distribution[count][inner_category].append(gene)

        if count > max_count:
            mutation_distribution[above_label][category].append(gene)
        else:
            mutation_distribution[count][category].append(gene)

    for gene, count in order_gene_counts.items():
        if gene not in disorder_genes:
            category = "Order Only"
            if count > max_count:
                mutation_distribution[above_label][category].append(gene)
            else:
                mutation_distribution[count][category].append(gene)

    if file:
        with open(file, 'w') as f:
            header = ["Category", "Gene", "Count"] + extra_columns
            f.write("{}\n".format("\t".join(header)))
            for count, dct in mutation_distribution.items():
                for category, genes in dct.items():
                    for gene in genes:
                        row = [category, gene, count] + [
                            disorder_data[disorder_data['Protein_ID'] == gene][col].iloc[0] if col in disorder_data.columns and not disorder_data[disorder_data['Protein_ID'] == gene].empty else ""
                            for col in extra_columns
                        ]
                        f.write("{}\n".format("\t".join(map(str, row))))

def plot_monogenic(disorder_data, order_data, monogenic_df, file=None,title="Distribution of Genes/Disease based on Mutation occurence for Monogenic Diseases",
        xlabel="Structural Classification", ylabel="Number of Genes",
                   ):

    monogenic_df = monogenic_df.dropna(subset=['nDisease'])

    disorder_monogenic = disorder_data[disorder_data['nDisease'].isin(monogenic_df['nDisease'])]
    order_monogenic = order_data[order_data['nDisease'].isin(monogenic_df['nDisease'])]

    # Group by 'nDisease' and 'Protein_ID' to get counts
    disorder_gene_counts = disorder_monogenic.groupby(['nDisease', 'Protein_ID']).size().reset_index(name='Count')
    order_gene_counts = order_monogenic.groupby(['nDisease', 'Protein_ID']).size().reset_index(name='Count')

    # Create sets of all unique genes
    disorder_genes = set(disorder_gene_counts['Protein_ID'])
    order_genes = set(order_gene_counts['Protein_ID'])

    # Initialize data structures for exclusive and common mutations
    mutation_distribution = {
        'Disorder Only': 0,
        'Order Only': 0,
        'Order Mostly': 0,
        'Disorder Mostly': 0,
        'Equal': 0,
        'Both': 0,
    }

    # Categorize and aggregate gene counts by disease
    for gene in disorder_genes:
        if gene not in order_genes:
            mutation_distribution['Disorder Only'] += 1
        else:
            current_disorder_df = disorder_gene_counts[disorder_gene_counts['Protein_ID'] == gene]
            for index, row in current_disorder_df.iterrows():
                disease = row['nDisease']
                count = row['Count']
                order_gene_subset = order_gene_counts[
                    (order_gene_counts['nDisease'] == disease) & (order_gene_counts['Protein_ID'] == gene)]
                if not order_gene_subset.empty:
                    order_count = order_gene_subset['Count'].values[0]
                    if order_count > count:
                        inner_category = "Order Mostly"
                    elif order_count < count:
                        inner_category = 'Disorder Mostly'
                    else:
                        inner_category = 'Equal'

                    mutation_distribution['Both'] += 1
                    mutation_distribution[inner_category] += 1

    for gene in order_genes:
        if gene not in disorder_genes:
            mutation_distribution["Order Only"] += 1

    # Prepare data for plotting
    mutation_index = ["Disorder Only", "Order Only", "Both", 'Disorder Mostly', 'Order Mostly', "Equal"]
    mutation_values = [mutation_distribution["Disorder Only"], mutation_distribution["Order Only"],
                       mutation_distribution["Both"],
                       mutation_distribution["Disorder Mostly"], mutation_distribution["Order Mostly"],
                       mutation_distribution["Equal"]
                       ]

    # Create a bar plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(mutation_index, mutation_values, color=['orange', 'lightblue', 'green', 'yellow', 'blue', 'lightgreen'])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.bar_label(ax.containers[0], fmt='{:,.0f}')
    plt.tight_layout()
    if file:
        plt.savefig(file)
    plt.show()


def calculate_distributions(clinvar_df):


    big_clinvar = pd.DataFrame()



    for i in clinvar_df['Interpretation'].unique():

        current_clinvar = clinvar_df[clinvar_df['Interpretation'] == i]

        disordered_df = current_clinvar[current_clinvar['structure'] == 'disorder']
        ordered_df = current_clinvar[current_clinvar['structure'] == 'order']

        # Group by 'nDisease' and 'Protein_ID' to get counts
        disorder_gene_counts = disordered_df.groupby(['nDisease', 'Protein_ID']).size().reset_index(name='Count')
        order_gene_counts = ordered_df.groupby(['nDisease', 'Protein_ID']).size().reset_index(name='Count')

        # Create sets of all unique genes
        disorder_genes = disorder_gene_counts['Protein_ID'].unique().tolist()
        order_genes = order_gene_counts['Protein_ID'].unique().tolist()

        mutation_distribution_lst = {
            'Disorder Only': {'Gene': set(), 'Mutation': 0, 'Disease': set()},
            'Order Only': {'Gene': set(), 'Mutation': 0, 'Disease': set()},
            'Both': {'Gene': set(), 'Mutation': 0, 'Disease': set()},
            'Disorder Mostly': {'Gene': set(), 'Mutation': 0, 'Disease': set()},
            'Order Mostly': {'Gene': set(), 'Mutation': 0, 'Disease': set()},
            'Equal': {'Gene': set(), 'Mutation': 0, 'Disease': set()},
            '-': {'Gene': set(), 'Mutation': 0, 'Disease': set()},
        }

        # Categorize and aggregate gene counts by disease
        for gene in tqdm(disorder_genes):
            current_disorder_df = disorder_gene_counts[disorder_gene_counts['Protein_ID'] == gene]
            for index, row in current_disorder_df.iterrows():
                disease = row['nDisease']
                disorder_count = int(row['Count'])

                order_gene_subset = order_gene_counts[
                    (order_gene_counts['nDisease'] == disease) & (order_gene_counts['Protein_ID'] == gene)]

                # Check if disease is '-'
                if disease == "-":
                    mutation_distribution_lst['-']['Mutation'] += disorder_count
                    mutation_distribution_lst['-']['Disease'].add(disease)
                    mutation_distribution_lst['-']['Gene'].add(gene)
                    disordered_df.loc[
                        (disordered_df['nDisease'] == disease) & (
                                disordered_df['Protein_ID'] == gene), 'Category'] = '-'
                    continue

                if not order_gene_subset.empty:
                    order_count = int(order_gene_subset['Count'].values[0])
                    count = order_count + disorder_count
                    proportion = disorder_count / count
                    if proportion <= 0.4:
                        inner_category = "Order Mostly"
                    elif proportion >= 0.6:
                        inner_category = "Disorder Mostly"
                    else:
                        inner_category = 'Equal'

                    mutation_distribution_lst['Both']['Mutation'] += count
                    mutation_distribution_lst['Both']['Disease'].add(disease)
                    mutation_distribution_lst['Both']['Gene'].add(gene)

                    mutation_distribution_lst[inner_category]['Mutation'] += count
                    mutation_distribution_lst[inner_category]['Disease'].add(disease)
                    mutation_distribution_lst[inner_category]['Gene'].add(gene)

                    disordered_df.loc[
                        (disordered_df['nDisease'] == disease) & (
                                    disordered_df['Protein_ID'] == gene), 'Category'] = inner_category
                    ordered_df.loc[
                        (ordered_df['nDisease'] == disease) & (
                                    ordered_df['Protein_ID'] == gene), 'Category'] = inner_category
                else:
                    mutation_distribution_lst['Disorder Only']['Mutation'] += disorder_count
                    mutation_distribution_lst['Disorder Only']['Disease'].add(disease)
                    mutation_distribution_lst['Disorder Only']['Gene'].add(gene)
                    disordered_df.loc[
                        (disordered_df['nDisease'] == disease) & (
                                    disordered_df['Protein_ID'] == gene), 'Category'] = 'Only Disorder'

        for gene in tqdm(order_genes):
            current_order_df = order_gene_counts[order_gene_counts['Protein_ID'] == gene]

            if gene not in disorder_genes:
                for index, row in current_order_df.iterrows():
                    disease = row['nDisease']
                    ordered_count = row['Count']

                    # Check if disease is '-'
                    if disease == "-":
                        mutation_distribution_lst['-']['Gene'].add(gene)
                        mutation_distribution_lst['-']['Disease'].add(disease)
                        mutation_distribution_lst['-']['Mutation'] += ordered_count
                        ordered_df.loc[
                            (ordered_df['Protein_ID'] == gene) & (ordered_df['nDisease'] == disease), 'Category'] = '-'
                    else:
                        mutation_distribution_lst["Order Only"]["Gene"].add(gene)
                        mutation_distribution_lst["Order Only"]['Disease'].add(disease)
                        mutation_distribution_lst["Order Only"]['Mutation'] += ordered_count
                        ordered_df.loc[
                            (ordered_df['Protein_ID'] == gene) & (
                                        ordered_df['nDisease'] == disease), 'Category'] = 'Only Order'
            else:
                for index, row in current_order_df.iterrows():
                    disease = row['nDisease']
                    ordered_count = row['Count']

                    # Check if disease is '-'
                    if disease == "-":
                        mutation_distribution_lst['-']['Gene'].add(gene)
                        mutation_distribution_lst['-']['Disease'].add(disease)
                        mutation_distribution_lst['-']['Mutation'] += ordered_count
                        ordered_df.loc[
                            (ordered_df['Protein_ID'] == gene) & (ordered_df['nDisease'] == disease), 'Category'] = '-'
                        continue

                    # If gene is in both disorder and order, check if it's missing in disorder for this disease
                    disordered_gene_subset = disorder_gene_counts[
                        (disorder_gene_counts['nDisease'] == disease) & (disorder_gene_counts['Protein_ID'] == gene)]
                    if disordered_gene_subset.empty:
                        mutation_distribution_lst["Order Only"]["Gene"].add(gene)
                        mutation_distribution_lst["Order Only"]['Disease'].add(disease)
                        mutation_distribution_lst["Order Only"]['Mutation'] += ordered_count
                        ordered_df.loc[
                            (ordered_df['nDisease'] == disease) & (
                                        ordered_df['Protein_ID'] == gene), 'Category'] = 'Only Order'

        # Initialize data structures for exclusive and common mutations
        mutation_distribution = {
            'Disorder Only': {'Gene': 0, 'Mutation': 0, 'Disease': 0},
            'Order Only': {'Gene': 0, 'Mutation': 0, 'Disease': 0},
            'Both': {'Gene': 0, 'Mutation': 0, 'Disease': 0},
            'Disorder Mostly': {'Gene': 0, 'Mutation': 0, 'Disease': 0},
            'Order Mostly': {'Gene': 0, 'Mutation': 0, 'Disease': 0},
            'Equal': {'Gene': 0, 'Mutation': 0, 'Disease': 0},
            '-': {'Gene': set(), 'Mutation': 0, 'Disease': set()},
        }

        for key, info_dct in mutation_distribution_lst.items():
            gene_set, mutation, disease_set = info_dct.values()
            count_unique_genes = len(gene_set)
            count_unique_diseases = len(disease_set)
            mutation_distribution[key]['Gene'] = count_unique_genes
            mutation_distribution[key]['Mutation'] = mutation
            mutation_distribution[key]['Disease'] = count_unique_diseases


        clinvar_final_df = pd.concat([disordered_df,ordered_df])

        big_clinvar = pd.concat([big_clinvar,clinvar_final_df])


    big_clinvar.loc[big_clinvar['Category'].isna(),'Category'] = "-"

    return big_clinvar


def calculate_distributions_optimized(clinvar_df):
    big_clinvar = pd.DataFrame()

    # Loop through unique interpretations
    for i in clinvar_df['Interpretation'].unique():

        current_clinvar = clinvar_df[clinvar_df['Interpretation'] == i]

        disordered_df = current_clinvar[current_clinvar['structure'] == 'disorder']
        ordered_df = current_clinvar[current_clinvar['structure'] == 'order']

        # Group by 'nDisease' and 'Protein_ID' to get counts
        disorder_gene_counts = disordered_df.groupby(['nDisease', 'Protein_ID']).size().reset_index(name='Count')
        order_gene_counts = ordered_df.groupby(['nDisease', 'Protein_ID']).size().reset_index(name='Count')

        disorder_genes = set(disorder_gene_counts['Protein_ID'].unique())
        order_genes = set(order_gene_counts['Protein_ID'].unique())

        # Initialize mutation distribution
        mutation_distribution_lst = {k: {'Gene': set(), 'Mutation': 0, 'Disease': set()} for k in
                                     ['Disorder Only', 'Order Only', 'Both',
                                      'Disorder Mostly', 'Order Mostly', 'Equal', '-']}

        # Helper function to handle mutation distribution updates
        def update_distribution(category, gene, disease, count):
            mutation_distribution_lst[category]['Gene'].add(gene)
            mutation_distribution_lst[category]['Disease'].add(disease)
            mutation_distribution_lst[category]['Mutation'] += count

        # Process disorder genes
        for gene in tqdm(disorder_genes):
            disorder_gene_df = disorder_gene_counts[disorder_gene_counts['Protein_ID'] == gene]
            for _, row in disorder_gene_df.iterrows():
                disease = row['nDisease']
                disorder_count = int(row['Count'])

                order_gene_subset = order_gene_counts[
                    (order_gene_counts['nDisease'] == disease) & (order_gene_counts['Protein_ID'] == gene)]

                if disease == "-":
                    update_distribution('-', gene, disease, disorder_count)
                    disordered_df.loc[(disordered_df['nDisease'] == disease) & (disordered_df['Protein_ID'] == gene),
                    'Category'] = '-'
                elif not order_gene_subset.empty:
                    order_count = int(order_gene_subset['Count'].values[0])
                    total_count = disorder_count + order_count
                    proportion = disorder_count / total_count

                    if proportion >= 0.6:
                        category = 'Disorder Mostly'
                    elif proportion <= 0.4:
                        category = 'Order Mostly'
                    else:
                        category = 'Equal'

                    update_distribution('Both', gene, disease, total_count)
                    update_distribution(category, gene, disease, total_count)

                    disordered_df.loc[(disordered_df['nDisease'] == disease) & (disordered_df['Protein_ID'] == gene),
                    'Category'] = category
                    ordered_df.loc[(ordered_df['nDisease'] == disease) & (ordered_df['Protein_ID'] == gene),
                    'Category'] = category
                else:
                    update_distribution('Disorder Only', gene, disease, disorder_count)
                    disordered_df.loc[(disordered_df['nDisease'] == disease) & (disordered_df['Protein_ID'] == gene),
                    'Category'] = 'Only Disorder'

        # Process order genes
        for gene in tqdm(order_genes - disorder_genes):
            order_gene_df = order_gene_counts[order_gene_counts['Protein_ID'] == gene]
            for _, row in order_gene_df.iterrows():
                disease = row['nDisease']
                order_count = row['Count']

                if disease == "-":
                    update_distribution('-', gene, disease, order_count)
                    ordered_df.loc[
                        (ordered_df['nDisease'] == disease) & (ordered_df['Protein_ID'] == gene), 'Category'] = '-'
                else:
                    update_distribution('Order Only', gene, disease, order_count)
                    ordered_df.loc[(ordered_df['nDisease'] == disease) & (ordered_df['Protein_ID'] == gene),
                    'Category'] = 'Only Order'

        # Concatenate final data
        clinvar_final_df = pd.concat([disordered_df, ordered_df])
        big_clinvar = pd.concat([big_clinvar, clinvar_final_df])

    big_clinvar.loc[big_clinvar['Category'].isna(), 'Category'] = "-"

    return big_clinvar


def process_gene_disorder(args):
    gene, disorder_gene_counts, order_gene_counts = args
    mutation_distribution = {
        'Disorder Only': {'Gene': set(), 'Mutation': 0, 'Disease': set()},
        'Order Only': {'Gene': set(), 'Mutation': 0, 'Disease': set()},
        'Both': {'Gene': set(), 'Mutation': 0, 'Disease': set()},
        'Disorder Mostly': {'Gene': set(), 'Mutation': 0, 'Disease': set()},
        'Order Mostly': {'Gene': set(), 'Mutation': 0, 'Disease': set()},
        'Equal': {'Gene': set(), 'Mutation': 0, 'Disease': set()},
    }

    category_assignments = []

    current_disorder_df = disorder_gene_counts[disorder_gene_counts['Protein_ID'] == gene]

    for _, row in current_disorder_df.iterrows():
        disease = row['nDisease']
        disorder_count = int(row['Count'])

        order_gene_subset = order_gene_counts[
            (order_gene_counts['nDisease'] == disease) & (order_gene_counts['Protein_ID'] == gene)]

        if not order_gene_subset.empty:
            order_count = int(order_gene_subset['Count'].values[0])
            count = order_count + disorder_count
            proportion = disorder_count / count
            if proportion <= 0.4:
                category = "Order Mostly"
            elif proportion >= 0.6:
                category = "Disorder Mostly"
            else:
                category = 'Equal'

            mutation_distribution['Both']['Mutation'] += count
            mutation_distribution['Both']['Disease'].add(disease)
            mutation_distribution['Both']['Gene'].add(gene)

            mutation_distribution[category]['Mutation'] += count
            mutation_distribution[category]['Disease'].add(disease)
            mutation_distribution[category]['Gene'].add(gene)

            category_assignments.append((disease, gene, category))
        else:
            mutation_distribution['Disorder Only']['Mutation'] += disorder_count
            mutation_distribution['Disorder Only']['Disease'].add(disease)
            mutation_distribution['Disorder Only']['Gene'].add(gene)
            category_assignments.append((disease, gene, 'Only Disorder'))

    return mutation_distribution, category_assignments


def process_gene_order(args):
    gene, disorder_gene_counts,disorder_genes, order_gene_counts = args
    mutation_distribution = {
        'Disorder Only': {'Gene': set(), 'Mutation': 0, 'Disease': set()},
        'Order Only': {'Gene': set(), 'Mutation': 0, 'Disease': set()},
        'Both': {'Gene': set(), 'Mutation': 0, 'Disease': set()},
        'Disorder Mostly': {'Gene': set(), 'Mutation': 0, 'Disease': set()},
        'Order Mostly': {'Gene': set(), 'Mutation': 0, 'Disease': set()},
        'Equal': {'Gene': set(), 'Mutation': 0, 'Disease': set()},
    }

    category_assignments = []


    current_order_df = order_gene_counts[order_gene_counts['Protein_ID'] == gene]

    for _, row in current_order_df.iterrows():
        disease = row['nDisease']
        ordered_count = row['Count']
        passed = False

        if gene not in disorder_genes:
            passed = True
        else:
            disordered_disease = disorder_gene_counts[(disorder_gene_counts['Protein_ID'] == gene) & (disorder_gene_counts['nDisease'] == disease) ]
            if disordered_disease.empty:
                passed = True

        if passed:
            mutation_distribution["Order Only"]["Gene"].add(gene)
            mutation_distribution["Order Only"]['Disease'].add(disease)
            mutation_distribution["Order Only"]['Mutation'] += ordered_count
            category_assignments.append((disease, gene, 'Only Order'))

    return mutation_distribution, category_assignments


def calculate_distributions_multiprocess(clinvar_df):
    from multiprocessing import Pool
    big_clinvar_list = []


    for i in clinvar_df['Interpretation'].unique():
        current_clinvar = clinvar_df[clinvar_df['Interpretation'] == i]
        disordered_df = current_clinvar[current_clinvar['structure'] == 'disorder']
        ordered_df = current_clinvar[current_clinvar['structure'] == 'order']

        disorder_gene_counts = disordered_df.groupby(['nDisease', 'Protein_ID']).size().reset_index(name='Count')
        order_gene_counts = ordered_df.groupby(['nDisease', 'Protein_ID']).size().reset_index(name='Count')

        disorder_genes = disorder_gene_counts['Protein_ID'].unique().tolist()
        order_genes = order_gene_counts['Protein_ID'].unique().tolist()

        # Process disorder genes
        with Pool() as pool:
            results_disorder = list(tqdm(pool.imap(process_gene_disorder,
                                                   [(gene, disorder_gene_counts, order_gene_counts) for gene in
                                                    disorder_genes]),
                                         total=len(disorder_genes)))

        # Process order genes
        with Pool() as pool:
            results_order = list(tqdm(pool.imap(process_gene_order,
                                                [(gene, disorder_gene_counts,disorder_genes, order_gene_counts) for gene in
                                                 order_genes]),
                                      total=len(order_genes)))

        # Combine results for disorder genes
        mutation_distribution_lst = {
            'Disorder Only': {'Gene': set(), 'Mutation': 0, 'Disease': set()},
            'Order Only': {'Gene': set(), 'Mutation': 0, 'Disease': set()},
            'Both': {'Gene': set(), 'Mutation': 0, 'Disease': set()},
            'Disorder Mostly': {'Gene': set(), 'Mutation': 0, 'Disease': set()},
            'Order Mostly': {'Gene': set(), 'Mutation': 0, 'Disease': set()},
            'Equal': {'Gene': set(), 'Mutation': 0, 'Disease': set()},
        }

        category_assignments_list = []

        for result in results_disorder:
            mutation_distribution, assignments = result
            category_assignments_list.extend(assignments)
            for key, value in mutation_distribution.items():
                mutation_distribution_lst[key]['Gene'].update(value['Gene'])
                mutation_distribution_lst[key]['Mutation'] += value['Mutation']
                mutation_distribution_lst[key]['Disease'].update(value['Disease'])

        # Combine results for order genes
        for result in results_order:
            mutation_distribution, assignments = result
            category_assignments_list.extend(assignments)
            for key, value in mutation_distribution.items():
                mutation_distribution_lst[key]['Gene'].update(value['Gene'])
                mutation_distribution_lst[key]['Mutation'] += value['Mutation']
                mutation_distribution_lst[key]['Disease'].update(value['Disease'])

        # Create DataFrame for category assignments
        category_df = pd.DataFrame(category_assignments_list, columns=['nDisease', 'Protein_ID', 'Category'])

        # Merge category assignments with original dataframes
        disordered_df = disordered_df.merge(category_df, on=['nDisease', 'Protein_ID'], how='left')
        ordered_df = ordered_df.merge(category_df, on=['nDisease', 'Protein_ID'], how='left')

        # Create final DataFrame for the current interpretation
        final_df = pd.concat([disordered_df, ordered_df])
        big_clinvar_list.append(final_df)

    # Concatenate all results at once
    big_clinvar = pd.concat(big_clinvar_list, ignore_index=True)
    big_clinvar.loc[big_clinvar['Category'].isna(), 'Category'] = "-"

    return big_clinvar


def plot_monogenic_threeplot(mutation_distribution,file=None, title="Distribution of Genes/Disease based on Mutation occurrence for Monogenic Diseases",
                   xlabel="Structural Classification", ylabel="Number of Genes"):

    # Prepare data for plotting
    categories = ["Disorder Only", "Order Only", "Both", 'Disorder Mostly', 'Order Mostly', "Equal"]
    data_types = ['Gene', 'Disease', 'Mutation', ]

    bar_width = 0.2
    x = range(len(categories))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, data_type in enumerate(data_types):
        values = [mutation_distribution[category][data_type] for category in categories]
        ax.bar([p + bar_width * i for p in x], values, width=bar_width, label=data_type)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks([p + 1.5 * bar_width for p in x])
    ax.set_xticklabels(categories)
    ax.bar_label(ax.containers[0], fmt='{:,.0f}')
    ax.bar_label(ax.containers[1], fmt='{:,.0f}')
    ax.bar_label(ax.containers[2], fmt='{:,.0f}')
    ax.legend()
    plt.tight_layout()

    if file:
        plt.savefig(file)
    plt.show()

def create_gene_distribution_with_disease_with_all_counts(disorder_data, order_data, monogenic_df=pd.DataFrame(), file=None):

    if monogenic_df.empty:
        disorder_monogenic = disorder_data
        order_monogenic = order_data
    else:
        monogenic_df = monogenic_df.dropna(subset=['nDisease'])

        disorder_monogenic = disorder_data[disorder_data['nDisease'].isin(monogenic_df['nDisease'])]
        order_monogenic = order_data[order_data['nDisease'].isin(monogenic_df['nDisease'])]

    # Group by 'nDisease' and 'Protein_ID' to get counts
    disorder_gene_counts = disorder_monogenic.groupby(['nDisease', 'Protein_ID']).size().reset_index(name='Count')
    order_gene_counts = order_monogenic.groupby(['nDisease', 'Protein_ID']).size().reset_index(name='Count')

    print(disorder_gene_counts)
    print(order_gene_counts)

    # Initialize data structures for the stacked bars
    mutation_distribution = {'Disorder Only': [], 'Order Only': [], 'Both': [], 'Disorder Mostly': [], 'Order Mostly': [], 'Equal': []}

    # Categorize and aggregate gene counts by disease
    for index, row in disorder_gene_counts.iterrows():
        gene = row['Protein_ID']
        disease = row['nDisease']
        disorder_count = int(row['Count'])
        order_gene_subset = order_gene_counts[(order_gene_counts['nDisease'] == disease) & (order_gene_counts['Protein_ID'] == gene)]
        if not order_gene_subset.empty:
            order_count = order_gene_subset['Count'].values[0]
            total_count = disorder_count + order_count
            proportion = disorder_count / total_count
            if proportion <= 0.4:
                inner_category = "Order Mostly"
            elif proportion >= 0.6:
                inner_category = "Disorder Mostly"
            else:
                inner_category = 'Equal'

            mutation_distribution[inner_category].append((disease, gene, total_count))
            mutation_distribution["Both"].append((disease, gene, total_count))
        else:
            mutation_distribution["Disorder Only"].append((disease, gene, disorder_count))

    for index, row in order_gene_counts.iterrows():
        gene = row['Protein_ID']
        disease = row['nDisease']
        order_count = int(row['Count'])
        disorder_gene_subset = disorder_gene_counts[
            (disorder_gene_counts['nDisease'] == disease) & (disorder_gene_counts['Protein_ID'] == gene)]
        if disorder_gene_subset.empty:
            mutation_distribution["Order Only"].append((disease, gene, order_count))

    # Optional: save the mutation_distribution to a file
    if file:
        with open(file, 'w') as f:
            header = ["Category", "Disease", "Gene", "Count"]
            f.write("{}\n".format("\t".join(header)))
            for category, tpls in mutation_distribution.items():
                for tpl in tpls:
                    disease, gene, count = tpl
                    row = [category, disease, gene, count]
                    f.write("{}\n".format("\t".join(map(str, row))))

def create_gene_distribution_with_disease_monogenic(disorder_data, order_data, monogenic_df=pd.DataFrame(), max_count=20, file=None,):

    if monogenic_df.empty:
        disorder_monogenic = disorder_data
        order_monogenic = order_data
    else:
        monogenic_df = monogenic_df.dropna(subset=['nDisease'])

        disorder_monogenic = disorder_data[disorder_data['nDisease'].isin(monogenic_df['nDisease'])]
        order_monogenic = order_data[order_data['nDisease'].isin(monogenic_df['nDisease'])]

    # Group by 'nDisease' and 'Protein_ID' to get counts
    disorder_gene_counts = disorder_monogenic.groupby(['nDisease', 'Protein_ID']).size().reset_index(name='Count')
    order_gene_counts = order_monogenic.groupby(['nDisease', 'Protein_ID']).size().reset_index(name='Count')

    print(disorder_gene_counts)
    print(order_gene_counts)

    above_label = f">{max_count}"

    # Initialize data structures for the stacked bars
    mutation_distribution = {k: {'Disorder Only': [], 'Order Only': [], 'Both': [], 'Disorder Mostly': [], 'Order Mostly': [], 'Equal': []} for k in range(1, max_count + 1)}
    mutation_distribution[above_label] = {'Disorder Only': [], 'Order Only': [], 'Both': [], 'Disorder Mostly': [], 'Order Mostly': [], 'Equal': []}

    # Categorize and aggregate gene counts by disease
    for (disease, gene), count in disorder_gene_counts.groupby(['nDisease', 'Protein_ID'])['Count'].sum().items():
        category = "Disorder Only"
        order_gene_subset = order_gene_counts[(order_gene_counts['nDisease'] == disease) & (order_gene_counts['Protein_ID'] == gene)]
        if not order_gene_subset.empty:
            order_count = order_gene_subset['Count'].values[0]
            category = "Both"
            total_count = count + order_count
            proportion = count / total_count
            if proportion <= 0.4:
                inner_category = "Order Mostly"
            elif proportion >= 0.6:
                inner_category = "Disorder Mostly"
            else:
                inner_category = 'Equal'

            count += order_count

            if inner_category:
                if count > max_count:
                    mutation_distribution[above_label][inner_category].append((disease, gene))
                else:
                    mutation_distribution[count][inner_category].append((disease, gene))

        if count > max_count:
            mutation_distribution[above_label][category].append((disease, gene))
        else:
            mutation_distribution[count][category].append((disease, gene))

    for (disease, gene), count in order_gene_counts.groupby(['nDisease', 'Protein_ID'])['Count'].sum().items():
        if not disorder_gene_counts[(disorder_gene_counts['nDisease'] == disease) & (disorder_gene_counts['Protein_ID'] == gene)].empty:
            continue
        category = "Order Only"
        if count > max_count:
            mutation_distribution[above_label][category].append((disease, gene))
        else:
            mutation_distribution[count][category].append((disease, gene))

    # Optional: save the mutation_distribution to a file
    if file:
        with open(file, 'w') as f:
            header = ["Category", "Disease", "Gene", "Count"]
            f.write("{}\n".format("\t".join(header)))
            for count, dct in mutation_distribution.items():
                for category, genes in dct.items():
                    for disease, gene in genes:
                        row = [category, disease, gene, count]
                        f.write("{}\n".format("\t".join(map(str, row))))



def create_monogenic_with_disease(disorder_data, order_data, file=None, col_to_check="nDisease",ismonogenic=True,iscomplex=False):
    all_df = pd.concat([disorder_data, order_data])
    check_df = all_df.groupby(col_to_check)['Protein_ID'].nunique().reset_index()
    check_df.columns = [col_to_check, 'Unique_Protein_ID_Count']

    if ismonogenic:
        monogenic_df = check_df[check_df['Unique_Protein_ID_Count'] == 1]
    else:
        if iscomplex:
            monogenic_df = check_df[check_df['Unique_Protein_ID_Count'] >= 5]
        else:
            monogenic_df = check_df[(check_df['Unique_Protein_ID_Count'] > 1) & (check_df['Unique_Protein_ID_Count'] < 5)]
    if file:
        monogenic_df.to_csv(file, index=False, sep='\t')

    return monogenic_df

def plot_monogenic_with_disease(disorder_data, order_data,max_count=20, col_to_check="nDisease"):
    all_df = pd.concat([disorder_data,order_data])
    check_df = all_df.groupby(col_to_check)['Protein_ID'].nunique().reset_index()
    check_df.columns = [col_to_check, 'Unique_Protein_ID_Count']

    # Count the number of counts
    count_of_counts = check_df['Unique_Protein_ID_Count'].value_counts().reset_index()
    count_of_counts.columns = ['Unique_Protein_ID_Count', 'Count']
    count_of_counts = count_of_counts.sort_values(by='Unique_Protein_ID_Count', ascending=False)

    # Aggregate values above max_count
    above_max_count = count_of_counts[count_of_counts['Unique_Protein_ID_Count'] > max_count]
    above_max_sum = above_max_count['Count'].sum()
    below_or_equal_max_count = count_of_counts[count_of_counts['Unique_Protein_ID_Count'] <= max_count]

    # Ensure order from 1 to 20 and then the aggregated value
    below_or_equal_max_count = below_or_equal_max_count.sort_values(by='Unique_Protein_ID_Count')
    aggregated_data = pd.concat(
        [below_or_equal_max_count,
         pd.DataFrame({'Unique_Protein_ID_Count': [f'>{max_count}'], 'Count': [above_max_sum]})],
        ignore_index=True
    )

    # Plot the distribution
    plt.figure(figsize=(12, 8))
    bars = plt.bar(aggregated_data['Unique_Protein_ID_Count'].astype(str), aggregated_data['Count'], color='skyblue')
    plt.xlabel('Unique Gene Counts')
    plt.ylabel('Number of Diseases')
    plt.title(f'Distribution Disease count in Genes for {col_to_check}')
    plt.xticks(rotation=45)
    plt.grid(axis='y')

    # Add count values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.show()

    print(check_df)


def categorize_mutations(disorder_pathogenic, order_pathogenic,categorized_type,filter_df=pd.DataFrame()):
    gene_distribution_file = os.path.join(files, 'distributions',f'clinvar_gene_distribution_file_{categorized_type}.tsv')
    clinvar_disordered_categorize_file = os.path.join(files, 'distributions',f'clinvar_disordered_categorized_{categorized_type}.tsv')
    clinvar_ordered_categorize_file = os.path.join(files, 'distributions',f'clinvar_ordered_categorized_{categorized_type}.tsv')
    create_gene_distribution_with_disease_with_all_counts(disorder_pathogenic, order_pathogenic, filter_df,file=gene_distribution_file)

    print(disorder_pathogenic[pd.isna(disorder_pathogenic['nDisease'])])
    exit()

    mutation_distribution_lst, mutation_distribution, order_categorized, disorder_categorized = calculate_distributions(disorder_pathogenic, order_pathogenic, filter_df)

    order_categorized.to_csv(clinvar_ordered_categorize_file,sep='\t',index=False)
    disorder_categorized.to_csv(clinvar_disordered_categorize_file,sep='\t',index=False)
    plot_monogenic_threeplot(mutation_distribution, title=f"Distribution of Genes/Disease/Mutations based on Mutation occurence for {categorized_type} Diseases")

    # create_gene_distribution_with_disease_monogenic(disorder_pathogenic, order_pathogenic,monogenic_df, max_count=10, file=gene_distribution_file_monogenic)
    # plot_monogenic(disorder_pathogenic, order_pathogenic,monogenic_df )
    # plot_monogenic_threeplot(disorder_pathogenic, order_pathogenic,monogenic_df,title="Distribution of Genes/Disease/Mutations based on Mutation occurence for Monogenic Diseases" )

def filter_pathogenic_df(clinvar_df):
    clinvar_df = clinvar_df[clinvar_df['ReviewStar'] > 0]

    return clinvar_df

# def create_genic_with_disease(all_df, col_to_check="nDisease"):
#
#     # check_df = all_df.groupby(col_to_check)['Protein_ID'].nunique().reset_index()
#     print(all_df.columns)
#     check_df = all_df.groupby(col_to_check).agg(
#         Unique_Proteins=('Protein_ID', lambda x: ','.join(x.unique())),
#         Unique_Protein_ID_Count=('Protein_ID', lambda x: x.nunique()),
#         Interpretations=('Interpretation', lambda x:','.join(x.unique())),
#     ).reset_index()
#
#     # check_df.columns = [col_to_check, 'Unique_Protein_ID_Count']
#
#     # Initialize the genic_category column with None
#     check_df['disease_genic'] = None
#
#
#     check_df.loc[check_df['Unique_Protein_ID_Count'] == 1, 'genic_category'] = 'Monogenic'
#
#
#     check_df.loc[check_df['Unique_Protein_ID_Count'] >= 5, 'genic_category'] = 'Complex'
#
#     check_df.loc[(check_df['Unique_Protein_ID_Count'] > 1) & (
#                     check_df['Unique_Protein_ID_Count'] < 5), 'genic_category'] = 'Multigenic'
#
#     # There are some not provided
#     check_df.loc[check_df['nDisease'] == "-", 'genic_category'] = "-"
#     print(check_df)
#     check_df.to_csv("/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/test/disease.tsv",sep='\t',index=False)
#     exit()
#
#     # Merge back the genic category information into the original DataFrame
#     all_df = all_df.merge(check_df[[col_to_check, 'genic_category']], on=col_to_check, how='left')
#
#     all_df.loc[pd.isna(all_df['genic_category']),'genic_category'] = "-"
#
#     return all_df


def create_genic_with_disease(all_df, col_to_check="nDisease"):
    final_df = pd.DataFrame()

    for interpretation in all_df["Interpretation"].unique():
        sub_df = all_df[all_df["Interpretation"] == interpretation]

        check_df = sub_df.groupby(col_to_check).agg(
            Unique_Proteins=('Protein_ID', lambda x: ','.join(x.unique())),
            Unique_Protein_ID_Count=('Protein_ID', lambda x: x.nunique()),
            # Interpretations=('Interpretation', lambda x: interpretation),
        ).reset_index()

        check_df["Interpretation"] = interpretation

        # Initialize the genic_category column with None
        check_df['genic_category'] = None

        check_df.loc[check_df['Unique_Protein_ID_Count'] == 1, 'genic_category'] = 'Monogenic'
        check_df.loc[check_df['Unique_Protein_ID_Count'] >= 5, 'genic_category'] = 'Complex'
        check_df.loc[(check_df['Unique_Protein_ID_Count'] > 1) &
                     (check_df['Unique_Protein_ID_Count'] < 5), 'genic_category'] = 'Multigenic'

        # Handle missing disease values
        check_df.loc[check_df[col_to_check] == "-", 'genic_category'] = "-"

        final_df = pd.concat([final_df, check_df], ignore_index=True)


    print(all_df.columns)
    print(final_df.columns)

    # Merge back the genic category information into the original DataFrame
    all_df = all_df.merge(final_df[[col_to_check, 'genic_category','Interpretation']], on=[col_to_check,'Interpretation'], how='left')
    all_df.loc[pd.isna(all_df['genic_category']), 'genic_category'] = "-"

    # Save output file
    final_df.to_csv("/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/test/disease.tsv",
                    sep='\t', index=False)

    return all_df


if __name__ == '__main__':
    core_dir = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations"
    files = os.path.join(core_dir, "processed_data", "files")

    to_clinvar_dir = os.path.join(files,'clinvar')

    clinvar_with_do_categories = f"clinvar_with_do_categories.tsv"

    clinvar_df = pd.read_csv(f'{to_clinvar_dir}/{clinvar_with_do_categories}', sep='\t')

    # Exclude the First Position
    clinvar_df = clinvar_df[clinvar_df['Protein_position'] != 1]

    clinvar_df.loc[pd.isna(clinvar_df['nDisease']), 'nDisease'] = clinvar_df.loc[pd.isna(clinvar_df['nDisease']), 'PhenotypeList']

    clinvar_df_genic_categorized = create_genic_with_disease(clinvar_df)

    print(clinvar_df_genic_categorized)


    # Calculate the mutation distribution
    df_categorized = calculate_distributions_multiprocess(clinvar_df_genic_categorized)

    print(df_categorized)

    df_categorized.to_csv(os.path.join(to_clinvar_dir,'clinvar_final.tsv'),sep='\t',index=False)

    exit()


    # Monogenic
    categorize_mutations(disorder_pathogenic, order_pathogenic,"monogenic",filter_df=monogenic_df)
    categorize_mutations(disorder_pathogenic, order_pathogenic,"polygenic",filter_df=polygenic_df)
    categorize_mutations(disorder_pathogenic, order_pathogenic,"complex",filter_df=complex_df)
    categorize_mutations(disorder_pathogenic, order_pathogenic,"all")

    # exit()

    gene_distribution_with_disease_file = os.path.join(files, 'distributions', 'clinvar_gene_distribution_with_disease.tsv')
    create_gene_distribution_with_disease(disorder_pathogenic, order_pathogenic,
                                          max_count=10,
                                          file=gene_distribution_with_disease_file,
                                          extra_columns = ['nDisease','Disease']
                                          )

    # likely_gene_distribution_with_disease_file = os.path.join(files, 'distributions',
    #                                                    'clinvar_gene_distribution_predicted_with_disease.tsv')
    # create_gene_distribution_with_disease(likely_pathogenic_disorder, likely_pathogenic_order,
    #                                       max_count=10,
    #                                       file=likely_gene_distribution_with_disease_file,
    #                                       extra_columns=['nDisease', 'Disease']
    #                                       )

    exit()

    # Number of mutations
    clinvar_mutation_structural_distribution = os.path.join(am_prediction_dir, "mutation_distribution")
    predicted_mutation_structural_distribution = os.path.join(am_prediction_dir, "mutation_distribution_predicted")


