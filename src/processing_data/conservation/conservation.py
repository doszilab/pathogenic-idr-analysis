import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def classify_alphamissense(score):
    if score <= 0.34:
        return 'likely_benign'
    elif score <= 0.564:
        return 'ambiguous'
    else:
        return 'pathogenic'

def classify_alphamissense_binary(score):
    if score >= 0.5:
        return 'pathogenic'
    else:
        return 'likely_benign'


def check_predictions(row):
    if row['AlphaMissense_Prediction'] == 'likely_benign' and row['label'] == 0:
        return True
    elif row['AlphaMissense_Prediction'] == 'pathogenic' and row['label'] == 1:
        return True
    else:
        return False


def plot_correlation(df, x_col, y_col, hue_col, title, ax):
    """
    Plot the correlation between two columns with hue indicating categories, adding lines for classification thresholds.

    :param df: DataFrame containing the data.
    :param x_col: Column name for the x-axis.
    :param y_col: Column name for the y-axis.
    :param hue_col: Column name for the hue (color coding).
    :param title: Title for the plot.
    :param ax: Matplotlib Axes object to plot on.
    """
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)
    ax.axhline(y=0.34, color='blue', linestyle='--', label='likely_benign threshold')  # likely_benign threshold
    ax.axhline(y=0.5, color='grey', linestyle='--', label='AlphaMissense threshold')  # likely_benign threshold
    ax.axhline(y=0.564, color='green', linestyle='--', label='pathogenic threshold')  # pathogenic threshold
    ax.set_title(title)
    ax.legend(title=hue_col, loc='best')


def plot_by_params(cols):

    for col in cols:

        current_df = clinvar_with_anotation_df[clinvar_with_anotation_df[col] != '-']
        current_disorder_df = clinvar_disorder[clinvar_disorder[col] != '-']
        current_ordered_df = clinvar_ordered[clinvar_ordered[col] != '-']

        # Prepare the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        # Call the function to plot
        plot_correlation(
            df=current_df,
            x_col=col,
            y_col='AlphaMissense',
            hue_col='Prediction_Correct',
            title=f'Correlation between AlphaMissense Prediction and {col} Score',
            ax=ax
        )
        plt.tight_layout()
        plt.show()

        for name,df_struct in [("Disordered",current_disorder_df), ("Ordered",current_ordered_df)]:

            # Prepare the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            # Call the function to plot
            plot_correlation(
                df=df_struct,
                x_col=col,
                y_col='AlphaMissense',
                hue_col='Prediction_Correct',
                title=f'Correlation between AlphaMissense Prediction and {col} Score in {name} Regions',
                ax=ax
            )
            plt.tight_layout()
            plt.show()

            # Prepare the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            # Call the function to plot
            plot_correlation(
                df=df_struct[df_struct['AlphaMissense_Prediction'] == 'pathogenic'],
                x_col=col,
                y_col='AlphaMissense',
                hue_col='Prediction_Correct',
                title=f'Correlation between AlphaMissense Prediction and {col} Score in {name} Regions for pathogenic',
                ax=ax
            )
            plt.tight_layout()
            plt.show()

            # Prepare the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            # Call the function to plot
            plot_correlation(
                df=df_struct[df_struct['AlphaMissense_Prediction'] != 'pathogenic'],
                x_col=col,
                y_col='AlphaMissense',
                hue_col='Prediction_Correct',
                title=f'Correlation between AlphaMissense Prediction and {col} Score in {name} Regions for benign',
                ax=ax
            )
            plt.tight_layout()
            plt.show()


def perform_pca_and_color(df, n_components=2, label_column='label', prediction_column='AlphaMissense',tilte='PCA of Dataset'):
    """
    Perform PCA on the dataset with normalization. Filter for true positives and false negatives for benign
    and pathogenic. Also, assign color labels for plotting.

    :param df: DataFrame containing the data.
    :param n_components: Number of components for PCA.
    :param label_column: Column name of the true labels.
    :param prediction_column: Column name of the predictions.
    :return: PCA components, explained variance ratio, and color labels for plotting.
    """

    # Define a new column for color labels
    def get_color(row):
        if row[prediction_column] == 'pathogenic' and row[label_column] == 1:
            return 'True Positive Pathogenic'
        elif row[prediction_column] == 'likely_benign' and row[label_column] == 1:
            return 'False Negative Pathogenic'
        elif row[prediction_column] == 'likely_benign' and row[label_column] == 0:
            return 'True Positive Benign'
        elif row[prediction_column] == 'pathogenic' and row[label_column] == 0:
            return 'False Negative Benign'
        else:
            return 'Other'

    df['Color_Label'] = df.apply(get_color, axis=1)

    # Select only numerical features for PCA
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    incorrect_cols = ['Position', 'label', 'AlphaMissense']
    numerical_cols = [col for col in numerical_cols if col not in incorrect_cols]

    numerical_data = df[numerical_cols].dropna(axis=0)

    # Normalize the data
    scaler = StandardScaler()
    numerical_data_scaled = scaler.fit_transform(numerical_data)

    # Perform PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(numerical_data_scaled)

    # Get color labels for the points
    color_labels = df.loc[numerical_data.index, 'Color_Label']  # Match index after dropna

    # Now you can plot the PCA components with color
    plt.figure(figsize=(10, 6))
    for label in np.unique(color_labels):
        condition = color_labels == label
        plt.scatter(components[condition, 0], components[condition, 1], label=label)

    plt.title(tilte)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()

def plot_roc_curve(df, score_column, label_column,tilte='Receiver Operating Characteristic Curve'):
    scores = df[score_column]
    labels = df[label_column]
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc_score = roc_auc_score(labels, scores)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(tilte)
    plt.legend(loc="lower right")
    plt.show()


def calculate_correctness_percentage(df):
    correctness_count = df['Prediction_Correct'].sum()
    total_count = len(df)
    return (correctness_count / total_count) * 100

def calculate_and_plot_metrics(df, label_column='label', prediction_column='AlphaMissense_Prediction',tilte='Combined'):
    # Convert predictions to binary
    df['Binary_Prediction'] = df[prediction_column].apply(lambda x: 1 if x == 'pathogenic' else 0)

    # Calculate Balanced Accuracy
    balanced_accuracy = balanced_accuracy_score(df[label_column], df['Binary_Prediction'])

    # Create Confusion Matrix
    confusion_matrix_result = confusion_matrix(df[label_column], df['Binary_Prediction'])

    # Plotting Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_result, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Benign', 'Predicted Pathogenic'],
                yticklabels=['True Benign', 'True Pathogenic'])
    plt.title(f'Confusion Matrix for {tilte}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return balanced_accuracy, confusion_matrix_result


def plot_distributions(df, column, title='Distribution by Category'):
    # Create a new column in the dataframe to identify TP, FP, TN, FN
    conditions = [
        (df['AlphaMissense_Prediction'] == 'pathogenic') & (df['label'] == 1),  # TP
        (df['AlphaMissense_Prediction'] == 'pathogenic') & (df['label'] == 0),  # FP
        (df['AlphaMissense_Prediction'] == 'likely_benign') & (df['label'] == 0),  # TN
        (df['AlphaMissense_Prediction'] == 'likely_benign') & (df['label'] == 1)  # FN
    ]
    choices = ['True Positive', 'False Positive', 'True Negative', 'False Negative']
    df['Prediction_Status'] = np.select(conditions, choices, default='Other')

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))
    plt.suptitle(title + column)

    # Extracting the individual categories
    categories = ['True Positive', 'False Positive', 'True Negative', 'False Negative']

    for i, category in enumerate(categories):
        plt.subplot(4, 1, i + 1)
        category_data = df[df['Prediction_Status'] == category]
        sns.histplot(category_data[column], kde=True, linewidth=0)
        plt.title(f'{category}')
        plt.xlabel(column if i == len(categories) - 1 else '')  # Only label the bottom subplot
        plt.ylabel('Number of Variants')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to make room for the suptitle
    plt.show()

def extract_pos_based_df(df):
    df[['Protein_ID', 'Position']] = df['AccessionPosition'].str.split('|', expand=True)
    df = df.drop(columns=['AccessionPosition'])
    df['Position'] = df['Position'].astype(int)
    return df

def create_conservation_files(mutations_in_disorder,mutations_in_ordered,mutations_in_ordered_predicted,mutations_in_disorder_predicted,pos_based_dir,to_dir):

    files ={
        "clinvar_disorder_conservation":mutations_in_disorder,
        "clinvar_order_conservation":mutations_in_ordered,
        "clinvar_order_conservation_predicted":mutations_in_ordered_predicted,
        "clinvar_disorder_conservation_predicted":mutations_in_disorder_predicted,
    }
    phastcons_df = extract_pos_based_df(pd.read_csv(f"{pos_based_dir}/phastcons_pos.tsv", sep='\t'))
    conservation_df = extract_pos_based_df(pd.read_csv(f"{pos_based_dir}/conservation_all_pos.tsv", sep='\t'))
    am_df = extract_pos_based_df(pd.read_csv(f"{pos_based_dir}/alphamissense_pos.tsv", sep='\t'))

    for file_name, df in files.items():
        mutations_with_conservation = df.merge(phastcons_df, on=['Protein_ID', 'Position'])
        mutations_with_conservation = mutations_with_conservation.merge(conservation_df,on=['Protein_ID', 'Position'])
        mutations_with_conservation = mutations_with_conservation.merge(am_df,on=['Protein_ID', 'Position'])
        mutations_with_conservation.to_csv(os.path.join(to_dir,f"{file_name}.tsv"),sep='\t',index=False)


def plot_functional_site_distribution(df_list, column_name,tilte,filename=None,figsize=(6, 3)):
    # Combine all dataframes into one
    combined_df = pd.concat(df_list)

    combined_df = combined_df.dropna(subset=[column_name])

    custom_palette = ['lightcoral','red','lightblue','blue']

    colors = {
        'Disorder - Pathogenic': 'red',
        'Disorder - Benign': 'lightcoral',
        'Order - Pathogenic': 'blue',
        'Order - Benign': 'lightblue'
    }

    # Create the plot
    plt.figure(figsize=figsize)
    sns.violinplot(data=combined_df, x='fname', y=column_name, linewidth=1.5,palette=custom_palette)

    # Set labels and title
    plt.xlabel('Category')
    plt.xlabel(None)
    plt.ylabel('Conservation Score')
    plt.title(tilte)

    plt.tight_layout()

    if filename:
        plt.savefig(filename)

    # Show the plot
    plt.show()

def plot_disorder_ordered_pathogenic_benign_conservation(clinvar_disorder_conservation,clinvar_order_conservation,column='global'):
    clinvar_disorder_conservation_benign = clinvar_disorder_conservation[clinvar_disorder_conservation['Interpretation'] == 'Benign']
    clinvar_disorder_conservation_pathogenic = clinvar_disorder_conservation[clinvar_disorder_conservation['Interpretation'] == 'Pathogenic']
    clinvar_order_conservation_benign = clinvar_order_conservation[
        clinvar_order_conservation['Interpretation'] == 'Benign']
    clinvar_order_conservation_pathogenic = clinvar_order_conservation[
        clinvar_order_conservation['Interpretation'] == 'Pathogenic']

    clinvar_disorder_conservation_benign['fname'] = f"Disorder Benign"
    clinvar_disorder_conservation_pathogenic['fname'] = f"Disorder Pathogenic"
    clinvar_order_conservation_benign['fname'] = f"Order Benign"
    clinvar_order_conservation_pathogenic['fname'] = f"Order Pathogenic"

    df_list = [clinvar_disorder_conservation_benign, clinvar_disorder_conservation_pathogenic,
               clinvar_order_conservation_benign,clinvar_order_conservation_pathogenic
               ]
    filename = os.path.join(plot_dir,f'conservation_category_distribution_clinvar_{column}.png')
    plot_functional_site_distribution(df_list, column,f'Conservation Scores - ClinVar',filename=filename)

def plot_disorder_ordered_pathogenic_benign_conservation_proteome(am_disorder,am_order,pos_based_dir,column='global',am_cutoff=0.5):

    conservation_df = extract_pos_based_df(pd.read_csv(f"{pos_based_dir}/conservation_all_pos.tsv", sep='\t'))

    am_disorder = am_disorder.merge(conservation_df,on=['Protein_ID', 'Position'])
    am_order = am_order.merge(conservation_df,on=['Protein_ID', 'Position'])

    am_disorder['Interpretation'] = np.where(am_disorder['AlphaMissense'] >= am_cutoff,'Pathogenic','Benign')
    am_order['Interpretation'] = np.where(am_order['AlphaMissense'] >= am_cutoff,'Pathogenic','Benign')

    clinvar_disorder_conservation_benign = am_disorder[am_disorder['Interpretation'] == 'Benign']
    clinvar_disorder_conservation_pathogenic = am_disorder[am_disorder['Interpretation'] == 'Pathogenic']
    clinvar_order_conservation_benign = am_order[am_order['Interpretation'] == 'Benign']
    clinvar_order_conservation_pathogenic = am_order[am_order['Interpretation'] == 'Pathogenic']

    clinvar_disorder_conservation_benign['fname'] = f"Disorder Benign"
    clinvar_disorder_conservation_pathogenic['fname'] = f"Disorder Pathogenic"
    clinvar_order_conservation_benign['fname'] = f"Order Benign"
    clinvar_order_conservation_pathogenic['fname'] = f"Order Pathogenic"

    df_list = [clinvar_disorder_conservation_benign, clinvar_disorder_conservation_pathogenic,
               clinvar_order_conservation_benign,clinvar_order_conservation_pathogenic
               ]
    filename = os.path.join(plot_dir,f'conservation_category_distribution_proteome_{column}.png')
    plot_functional_site_distribution(df_list, column,f'Conservation Scores - Human Proteome',filename=filename)

def plot_proteome_conservation(am_disorder,am_order,pos_based_dir,column='global'):

    conservation_df = extract_pos_based_df(pd.read_csv(f"{pos_based_dir}/conservation_all_pos.tsv", sep='\t'))

    am_disorder = am_disorder.merge(conservation_df,on=['Protein_ID', 'Position'])
    am_order = am_order.merge(conservation_df,on=['Protein_ID', 'Position'])

    am_disorder['fname'] = f"Disorder"
    am_order['fname'] = f"Order"

    df_list = [am_disorder, am_order]
    filename = os.path.join(plot_dir,f'conservation_distribution_proteome_{column}.png')
    plot_functional_site_distribution(df_list, column,f'Conservation Score Distribution Proteome',filename=filename,figsize=(3, 3))


def plot_disorder_ordered_pathogenic_conservation(clinvar_conservation,structtype="Disorder"):
    clinvar_conservation_pathogenic = clinvar_conservation[clinvar_conservation['Interpretation'] == 'Pathogenic']

    global_conservation = clinvar_conservation_pathogenic[['global']].rename(columns={"global":"Score"})
    global_conservation['fname'] = f"Global"
    mammalia_conservation = clinvar_conservation_pathogenic[['Mammalia']].rename(columns={"Mammalia":"Score"})
    mammalia_conservation['fname'] = f"Mammalia"
    vertebrata_conservation = clinvar_conservation_pathogenic[['Vertebrata']].rename(columns={"Vertebrata":"Score"})
    vertebrata_conservation['fname'] = f"Vertebrata"
    phastcons_conservation = clinvar_conservation_pathogenic[['phastConsScore']].rename(columns={"phastConsScore":"Score"})
    phastcons_conservation['fname'] = f"phastConsScore"

    df_list = [global_conservation, mammalia_conservation,
               vertebrata_conservation, phastcons_conservation]

    filename = os.path.join(plot_dir, f'conservation_pathogenic_distribution_{structtype}.png')

    plot_functional_site_distribution(df_list, "Score",f'Distribution of Conservation Scores for Pathogenic Variants ({structtype})',filename=filename)

def plot_correlation_matrices(df,columns,struct_type='Disorder'):
    correlation_matrix = df[columns].corr()
    print(correlation_matrix)

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=0, vmax=1)
    plt.title(f'Correlation Matrix for Conservation and AlphaMissense ({struct_type})')
    plt.tight_layout()
    filename = os.path.join(plot_dir, f'correlation_matrix_{struct_type}.png')
    plt.savefig(filename)
    plt.show()


def plot_pca(df, columns, struct_type='Disorder'):
    # Drop rows with NaN values only in the specified columns
    cleaned_df = df.dropna(subset=columns)

    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(cleaned_df[columns])

    # Create a DataFrame with the principal components
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Interpretation'] = cleaned_df['Interpretation'].reset_index(drop=True)

    # Ensure the correct order of labels
    # interpretation_order = ['Uncertain', 'Benign', 'Pathogenic']
    interpretation_order = ['Benign', 'Pathogenic']
    pca_df['Interpretation'] = pd.Categorical(pca_df['Interpretation'], categories=interpretation_order, ordered=True)

    # Sort the DataFrame by Interpretation
    pca_df = pca_df.sort_values(by='Interpretation')

    pca_df = pca_df[pca_df['Interpretation'] != "Uncertain"]

    # Define custom colors
    custom_palette = {'Uncertain': 'grey', 'Benign': 'lightblue', 'Pathogenic': 'red'}

    # Plot the PCA result
    # plt.figure(figsize=(10, 8))
    plt.figure()
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Interpretation', palette=custom_palette, hue_order=interpretation_order)
    plt.title(f'PCA of Scores with Label Coloring ({struct_type})')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Interpretation', loc='best')
    plt.show()

if __name__ == "__main__":
    # conservations to check: phastCons, Global, Mammalia, Vertebrata
    # 1 check the distribution of conservation for pathogenic vs benign variants in Clinvar
    # 2 Calculate a correlation coeff for AM vs conservation scores
    # 3 PCA for AM, Conservation scores labeling disorder and ordered for all? or only clinvar?

    origin = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations"
    to_clinvar_dir = os.path.join(origin, 'processed_data/files/clinvar')
    to_am_dir = os.path.join(origin, 'processed_data/files/alphamissense/clinvar')
    plot_dir = os.path.join(origin, 'plots/sm/conservation')

    # Clinvar
    mutations_in_disorder = pd.read_csv(f"{to_clinvar_dir}/clinvar_disorder.tsv", sep='\t').rename(
        columns={"Protein_position": "Position"})
    mutations_in_ordered = pd.read_csv(f"{to_clinvar_dir}/clinvar_order.tsv", sep='\t').rename(
        columns={"Protein_position": "Position"})
    mutations_in_ordered_predicted = pd.read_csv(f"{to_am_dir}/likely_pathogenic_order.tsv", sep='\t')
    mutations_in_disorder_predicted = pd.read_csv(f"{to_am_dir}/likely_pathogenic_disorder.tsv", sep='\t')

    pos_based_dir = os.path.join(origin, 'data/discanvis_base_files/positional_data_process')
    conservation_path = os.path.join(origin, 'processed_data/files/conservation')

    # create_conservation_files(mutations_in_disorder, mutations_in_ordered, mutations_in_ordered_predicted,
    #                           mutations_in_disorder_predicted, pos_based_dir,conservation_path)
    #
    # exit()



    # 1 check the distribution of conservation for pathogenic vs benign variants in Clinvar

    # conservation_path = '/dlab/home/norbi/PycharmProjects/AlphaMissense_Stat/processed_data/files/conservation/'
    am_path = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/alphamissense'
    clinvar_disorder_conservation = pd.read_csv(f'{conservation_path}/clinvar_disorder_conservation.tsv',sep='\t')
    clinvar_order_conservation = pd.read_csv(f'{conservation_path}/clinvar_order_conservation.tsv',sep='\t')
    clinvar_order_predicted_conservation = pd.read_csv(
        f'{conservation_path}/clinvar_order_conservation_predicted.tsv', sep='\t')
    clinvar_disorder_predicted_conservation = pd.read_csv(
        f'{conservation_path}/clinvar_disorder_conservation_predicted.tsv', sep='\t')

    plot_disorder_ordered_pathogenic_benign_conservation(clinvar_disorder_conservation,clinvar_order_conservation)
    # plot_disorder_ordered_pathogenic_benign_conservation(clinvar_disorder_conservation,clinvar_order_conservation,column='Vertebrata')

    am_disorder = pd.read_csv(f'{am_path}/am_disorder.tsv', sep='\t')
    am_order = pd.read_csv(f'{am_path}/am_order.tsv', sep='\t')
    #
    plot_disorder_ordered_pathogenic_benign_conservation_proteome(am_disorder,am_order,pos_based_dir)
    # plot_disorder_ordered_pathogenic_benign_conservation_proteome(am_disorder,am_order,pos_based_dir,column='Vertebrata')

    exit()
    # plot_proteome_conservation(am_disorder, am_order, pos_based_dir)
    # plot_proteome_conservation(am_disorder, am_order, pos_based_dir, column='Vertebrata')
    # plot_disorder_ordered_pathogenic_benign_conservation(clinvar_disorder_conservation,clinvar_order_conservation,column='phastConsScore')
    # plot_disorder_ordered_pathogenic_benign_conservation(clinvar_disorder_conservation,clinvar_order_conservation,column='Vertebrata')
    # plot_disorder_ordered_pathogenic_benign_conservation(clinvar_disorder_conservation,clinvar_order_conservation,column='Mammalia')
    # plot_disorder_ordered_pathogenic_conservation(clinvar_disorder_conservation,'Disorder')
    # plot_disorder_ordered_pathogenic_conservation(clinvar_order_conservation,'Ordered')

    # exit()

    # 2 Calculate a correlation coeff for AM vs conservation scores
    column_lst = [
        'phastConsScore',
        # 'Eukaryota','Eumetazoa',
        'Mammalia',
        # "Opisthokonta",
        "Vertebrata",
        # "Viridiplantae",
        'global',
        'AlphaMissense'
                  ]

    # For only Clinvar
    # plot_correlation_matrices(clinvar_disorder_conservation,column_lst,struct_type='Clinvar Disorder')
    # plot_correlation_matrices(clinvar_order_conservation,column_lst,struct_type='Clinvar Order')
    #
    # plot_correlation_matrices(pd.concat([clinvar_order_conservation,clinvar_disorder_conservation]),column_lst,struct_type='Clinvar All')

    # # For only All
    # all_pos_data = pd.read_csv("/dlab/home/norbi/PycharmProjects/DisCanVis_Data_Process/Processed_Data/gencode_process/positional_data_process/MotifPredPositionBasedAnnotations.gzip",compression='gzip',
    #                            # nrows=100,
    #                            sep='\t'
    #                            )
    # print(all_pos_data)
    # plot_correlation_matrices(all_pos_data, column_lst, struct_type='All')


    # exit()

    # 3 PCA for AM, Conservation scores labeling disorder and ordered for clinvar
    column_lst = [
        'phastConsScore',
        # 'Eukaryota','Eumetazoa',
        'Mammalia',
        # "Opisthokonta",
        "Vertebrata",
        # "Viridiplantae",
        'global',
        'AlphaMissense'
    ]
    plot_pca(clinvar_disorder_conservation,column_lst,struct_type='Disorder')
    plot_pca(clinvar_order_conservation,column_lst,struct_type='Order')

    exit()

    clinvar_with_anotation_df = pd.read_csv("../processed_data/clinvar_with_pos_data_rsa.tsv", sep="\t", header=0)

    # Cutoff for Evolutonary conservation checking
    evolutanry_conservation_percentile = 0.95

    # Cutoff for Evolutonary conservation checking
    evolutanry_conservation_percentile_ordered = 0.95
    evolutonary_level = "Vertebrata"

    clinvar_with_anotation_df = clinvar_with_anotation_df[clinvar_with_anotation_df[evolutonary_level] != '-']
    clinvar_with_anotation_df[evolutonary_level] = clinvar_with_anotation_df[evolutonary_level].astype(float)

    evolutanry_conservation_threshold = clinvar_with_anotation_df[evolutonary_level].quantile(
        evolutanry_conservation_percentile)
    clinvar_with_anotation_df = clinvar_with_anotation_df[
        clinvar_with_anotation_df[evolutonary_level] >= evolutanry_conservation_threshold]

    # https://www.sciencedirect.com/science/article/pii/S2001037023002143
    # Cutoffs

    plldt_disorder = 70
    iupred_disorder = 0.4
    rsa_cutoff = 0.849

    # alphamissense_likely_begin = 0 - 0.34
    # alphamissense_amibguous = 0.34 - 0.564
    # alphamissense_pathogen = 0.564 - 1


    # Apply the classification
    clinvar_with_anotation_df['AlphaMissense_Prediction'] = clinvar_with_anotation_df['AlphaMissense'].apply(
        classify_alphamissense_binary)

    clinvar_with_anotation_df['Binary_Prediction'] = clinvar_with_anotation_df['AlphaMissense_Prediction'].apply(
        lambda x: 1 if x == 'pathogenic' else 0)

    # Drop ambiguous values
    # clinvar_with_anotation_df = clinvar_with_anotation_df[clinvar_with_anotation_df['AlphaMissense_Prediction'] != 'ambiguous']

    # Determine the correctness of the predictions
    clinvar_with_anotation_df['Prediction_Correct'] = clinvar_with_anotation_df.apply(check_predictions, axis=1)

    # Polymorphism
    clinvar_with_anotation_df['Common_Polymorphism'] = clinvar_with_anotation_df['Polymorphism'].apply(lambda x: 1 if "Common" in x else 0 if "All" in x else -1)

    # # Select relevant columns for analysis
    # analysis_columns = ['phastConsscore', 'Polymorphism', 'AlphaMissense_Category']
    # analysis_df = clinvar_with_anotation_df[analysis_columns]

    clinvar_with_anotation_df = clinvar_with_anotation_df[clinvar_with_anotation_df['Plldtscores'] != '-' ]
    clinvar_with_anotation_df['Plldtscores'] = clinvar_with_anotation_df['Plldtscores'].astype(float)
    clinvar_with_anotation_df = clinvar_with_anotation_df[clinvar_with_anotation_df['RSA'] != '-']
    clinvar_with_anotation_df['RSA'] = clinvar_with_anotation_df['RSA'].astype(float)

    # Fill na values with 0
    for i in ["Mammalia", 'Vertebrata', 'Eumetazoa', 'Opisthokonta', 'Eukaryota', 'Viridiplantae']:
        clinvar_with_anotation_df[i] = clinvar_with_anotation_df[i].fillna(0)
        clinvar_with_anotation_df[i] = clinvar_with_anotation_df[i].replace('-', 0)
        clinvar_with_anotation_df[i] = clinvar_with_anotation_df[i].astype(float)

    for i in ["binding_info", "dibs_info", "phasepro_info", "mfib_info",'Elm_Info', 'Elm_Switch_Info','ptmdb','Binding','Roi','MobiDB']:
        clinvar_with_anotation_df[i] = clinvar_with_anotation_df[i].apply(lambda x: 1 if x != '-' else 0)
        clinvar_with_anotation_df[i] = clinvar_with_anotation_df[i].astype(float)

    clinvar_disorder = clinvar_with_anotation_df[((clinvar_with_anotation_df['Plldtscores'] < plldt_disorder) & (clinvar_with_anotation_df['RSA'] > rsa_cutoff))]
    clinvar_ordered = clinvar_with_anotation_df[~((clinvar_with_anotation_df['Plldtscores'] < plldt_disorder) & (clinvar_with_anotation_df['RSA'] > rsa_cutoff))]

    print(clinvar_disorder['Plldtscores'].describe())
    print(clinvar_disorder['AlphaMissense_Prediction'].value_counts())
    print(clinvar_disorder['Prediction_Correct'].value_counts())

    # For the combined dataset
    balanced_acc_combined, conf_matrix_combined = calculate_and_plot_metrics(clinvar_with_anotation_df,tilte="Combined")

    # For the disordered dataset
    balanced_acc_disordered, conf_matrix_disordered = calculate_and_plot_metrics(clinvar_disorder,tilte="Disordered")

    # For the ordered dataset
    balanced_acc_ordered, conf_matrix_ordered = calculate_and_plot_metrics(clinvar_ordered,tilte="Ordered")

    # Print Balanced Accuracy for each set
    print(f"Balanced Accuracy (Combined): {balanced_acc_combined}")
    print(f"Balanced Accuracy (Disordered): {balanced_acc_disordered}")
    print(f"Balanced Accuracy (Ordered): {balanced_acc_ordered}")

    print(clinvar_disorder.columns)
    print(clinvar_disorder['Gene'].unique().tolist())

    clinvar_disorder.to_csv("../processed_data/high_conserved_disordered.tsv",sep='\t',header=True,index=False)

    # exit()

    # Calculate correctness percentages
    overall_correctness = calculate_correctness_percentage(clinvar_with_anotation_df)
    disordered_correctness = calculate_correctness_percentage(clinvar_disorder)
    ordered_correctness = calculate_correctness_percentage(clinvar_ordered)


    # Create a DataFrame for plotting
    correctness_df = pd.DataFrame({
        'Region': ['All', 'Disordered', 'Ordered'],
        'Correctness (%)': [overall_correctness, disordered_correctness, ordered_correctness]
    })

    print(correctness_df)

    # Plotting
    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(x='Region', y='Correctness (%)', data=correctness_df)
    barplot.set_ylim(0, 100)  # Set the limit to be from 0 to 100 for percentages

    # Add the percentage text on each bar
    for index, row in correctness_df.iterrows():
        barplot.text(index, row['Correctness (%)'], f'{row["Correctness (%)"]:.2f}%', color='black', ha="center")

    plt.title('Prediction Correctness by Region')
    plt.show()

    print(clinvar_disorder)
    print(clinvar_ordered)

    plot_cols = [
            # 'phastConsscore',
            'global',
            'Mammalia',
            'Vertebrata',
            # "Common_Polymorphism",
            # 'Eumetazoa', 'Opisthokonta', 'Eukaryota', 'Viridiplantae'
         ]

    for i in plot_cols:

        plot_distributions(clinvar_with_anotation_df, i,title='Mean Values of All regions for ')
        plot_distributions(clinvar_disorder, i,title='Mean Values of Disordered regions for ')
        plot_distributions(clinvar_ordered, i,title='Mean Values of Ordered regions for ')



    # perform_pca_and_color(clinvar_with_anotation_df, label_column='label',prediction_column='AlphaMissense_Prediction')
    # perform_pca_and_color(clinvar_disorder, label_column='label',prediction_column='AlphaMissense_Prediction',tilte='PCA of Disordered regions')
    # perform_pca_and_color(clinvar_ordered, label_column='label',prediction_column='AlphaMissense_Prediction',tilte='PCA of Ordered regions')


    # # Plot ROC Curve for the whole dataset
    # plot_roc_curve(clinvar_with_anotation_df, 'AlphaMissense', 'label')
    #
    # # Separate ROC Curves for disordered and ordered
    # plot_roc_curve(clinvar_disorder, 'AlphaMissense', 'label','Plot curve for Disordered regions')
    # plot_roc_curve(clinvar_ordered, 'AlphaMissense', 'label','Plot curve for Ordered regions')

    plot_by_params(
        [
            # 'phastConsscore',
            'global',
            'Mammalia',
            # 'Vertebrata',
            # "Common_Polymorphism",
            # 'Eumetazoa', 'Opisthokonta', 'Eukaryota', 'Viridiplantae'
         ]
    )


