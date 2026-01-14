import pandas as pd
import os


def determine_known_for_predicted(predicted_df, known_df, columns=['Protein_ID', 'ELMIdentifier', "Start"]):
    """
    Merges predicted_df with known_df on the specified columns and adds a Found_Known column.
    """
    merged_df = predicted_df.merge(known_df[columns].drop_duplicates(), on=columns, how='left', indicator=True)
    merged_df['Found_Known'] = merged_df['_merge'] == 'both'
    return merged_df.drop(columns=['_merge'])


def class_based_counts(known_df,predicted_df,predicted_df_total):
    elm_types_count_predicted = predicted_df['ELMIdentifier'].value_counts().reset_index().rename(
        columns={"count": "Class_Count_Predicted"})

    elm_types_count_predicted_total = predicted_df_total['ELMIdentifier'].value_counts().reset_index().rename(
        columns={"count": "Class_Count_Predicted_Total"})

    bins = [0, 10, 25, 50, 100, 200, 500, 1000, float('inf')]
    labels = ['1-10', '10-25', '25-50', '50-100', '100-200', '200-500', '500-1000', "1000+"]

    elm_types_count_predicted_total['Count_Category'] = pd.cut(
        elm_types_count_predicted_total['Class_Count_Predicted_Total'],
        bins=bins,
        labels=labels,
        right=True  # inclusive of the upper bin value
    )

    match_keys = ['Protein_ID', 'ELMIdentifier', 'Start']
    deduped_matches = predicted_df[predicted_df['Found_Known'] == True][match_keys].drop_duplicates()
    elm_types_count_predicted_known_found = deduped_matches['ELMIdentifier'].value_counts().reset_index().rename(
        columns={"count": "Class_Count_Found_Known"})

    elm_types_count_known = known_df['ELMIdentifier'].value_counts().reset_index().rename(
        columns={"count": "Class_Count_Known"})

    merged_df = elm_types_count_known.merge(
        elm_types_count_predicted_known_found,
        on='ELMIdentifier',
        how='left'
    )

    merged_df = merged_df.merge(elm_types_count_predicted,on='ELMIdentifier',
        how='left')

    merged_df = merged_df.merge(elm_types_count_predicted_total, on='ELMIdentifier',
                                how='left')

    merged_df['Class_Count_Predicted'] = merged_df['Class_Count_Predicted'].fillna(0)
    merged_df['Class_Count_Found_Known'] = merged_df['Class_Count_Found_Known'].fillna(0)
    merged_df['Class_Count_Known'] = merged_df['Class_Count_Known'].fillna(0)

    merged_df['More_Predicted_than_Known'] = merged_df['Class_Count_Predicted'] > merged_df['Class_Count_Found_Known']
    merged_df['All_Known_Found'] = merged_df['Class_Count_Found_Known'] == merged_df['Class_Count_Known']

    merged_df['Class_Count_Predicted'] = merged_df['Class_Count_Predicted'].astype(int)
    merged_df['Class_Count_Found_Known'] = merged_df['Class_Count_Found_Known'].astype(int)
    merged_df['Class_Count_Known'] = merged_df['Class_Count_Known'].astype(int)

    merged_df['Known_Found_Percentage'] = merged_df['Class_Count_Found_Known'] / merged_df['Class_Count_Known'] * 100
    merged_df['Percentage_of_all_Instances'] = merged_df['Class_Count_Predicted'] / merged_df['Class_Count_Predicted'].sum() * 100


    merged_df['Filtering_Rate'] = merged_df['Class_Count_Predicted'] / merged_df['Class_Count_Predicted_Total']

    merged_df = merged_df.merge(
        predicted_df[['ELMIdentifier', 'ELMType']].drop_duplicates(),
        on='ELMIdentifier',
        how='inner'
    )

    return merged_df

if __name__ == "__main__":

    file_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/elm'
    error_dir = '/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/test'

    predicted_elm_classes = os.path.join(file_dir,'elm_predicted_disorder_with_confidence.tsv')
    # predicted_elm_classes = os.path.join(file_dir,'decision_tree','elm_predicted_with_am_info_all_disorder_class_filtered.tsv')
    predicted_elm_classes_df = pd.read_csv(predicted_elm_classes,sep='\t')
    predicted_elm_classes_total = os.path.join(file_dir,'decision_tree','elm_predicted_with_am_info_all_disorder_class_filtered.tsv')
    predicted_elm_classes_total_df = pd.read_csv(predicted_elm_classes_total, sep='\t')

    known_elm_classes = os.path.join(file_dir,'elm_known_with_am_info_all_disorder_class_filtered.tsv')
    known_elm_classes_df = pd.read_csv(known_elm_classes,sep='\t')

    predicted_df = determine_known_for_predicted(predicted_elm_classes_df,known_elm_classes_df)

    stat_df = class_based_counts(known_elm_classes_df, predicted_df,predicted_elm_classes_total_df)
    stat_df.to_csv(os.path.join(file_dir, 'class_based_approach', 'elm_pem_disorder_corrected_class_infos.tsv'),
                     index=False)

    merged_df = predicted_df.merge(
        stat_df,
        on=['ELMIdentifier','ELMType'],
        how='left'
    )

    merged_df = merged_df.sort_values(by=["Class_Count_Predicted",'ELMType','ELMIdentifier','Protein_ID'], ascending=True)
    merged_df.to_csv(os.path.join(file_dir,'class_based_approach','elm_pem_disorder_corrected_with_class_count.tsv'),index=False)

    lst_for_classes_df = merged_df[ ['Protein_ID' ,'Start', 'End','Found_Known',*stat_df.columns]]
    lst_for_classes_df.to_csv(os.path.join(file_dir,'class_based_approach','elm_pem_count_category.tsv'),index=False)
