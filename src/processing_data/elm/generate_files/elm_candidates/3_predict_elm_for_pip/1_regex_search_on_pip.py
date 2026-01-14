import os

import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import sys

sys.path.insert(0, "/dlab/home/norbi/PycharmProjects/Giga_Project/scripts")

# Now import the function from elm.py
from elm import count_key_residues_with_regex_search_and_extraction

def main(sequences,elm_classes_with_info,position_col="Position",mut_col="HGVSp_Short"):
    results = []

    wrong_ones = 0

    for index,row in tqdm(sequences.iterrows(),total=sequences.shape[0]):
        protein = row["Protein_ID"]
        sequence = row["Sequence"]
        length = row["Seq_Length"]
        pip_start = int(row["Start"])
        pip_end = int(row["End"])

        for _, elm_row in elm_classes_with_info.iterrows():
            elm_identifier = elm_row["ELMIdentifier"]
            elm_type = elm_row["ELMType"]
            regular_expression = elm_row["Regex"]
            pattern = re.compile(regular_expression)

            seq_start = 0
            while True:
                match = pattern.search(sequence, seq_start)
                if not match:
                    break

                start, end = match.span()
                # print(start, end, match.group())

                seq_start = start + 1

            # for match in pattern.finditer(sequence):
            #     start, end = match.span()
                matched_sequence = match.group()
                protein_start = pip_start + start
                protein_end = protein_start + len(matched_sequence) - 1

                # Terminal Rule Motifs should be in end of the Whole Protein
                if regular_expression.endswith("$"):
                    if protein_end != length:
                        continue
                elif regular_expression.startswith("^"):
                    if protein_start != 1:
                        continue

                try:
                    result = count_key_residues_with_regex_search_and_extraction(regular_expression, matched_sequence, protein_start)

                except Exception as e:
                    print(e,protein,protein_start,protein_end,matched_sequence,elm_type,regular_expression)
                    wrong_ones += 1

                data = {
                    "Protein_ID":protein,
                    "PIP_Start":pip_start + 1,
                    "PIP_End":pip_end,
                    "PIP_Sequence":sequence,
                    "Start":protein_start,
                    "End":protein_end,
                    "Sequence":matched_sequence,
                    "ELMType":elm_type,
                    "Regex":regular_expression,
                    "ELMIdentifier":elm_identifier,
                    **result
                }
                results.append(data)

    print('wrong_ones',wrong_ones)

    df = pd.DataFrame(results)
    return df


if __name__ == "__main__":
    base_dir = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations"
    file_dir = os.path.join(base_dir, "processed_data","files")

    # 2. Load Sequences
    main_file_path = os.path.join(base_dir, "data", "discanvis_base_files", "sequences","loc_chrom_with_names_main_isoforms_with_seq.tsv")
    main_df = pd.read_csv(main_file_path, sep="\t")

    # 2. Load Known ELMs with Regex
    known_elms_path = os.path.join(base_dir, "processed_data", "files", "elm",
                                   "elm_known_with_am_info_all_disorder_class_filtered.tsv")
    known_elms_df = pd.read_csv(known_elms_path, sep="\t")
    elm_classes_with_info = known_elms_df[["ELMType", "ELMIdentifier", "Regex"]].drop_duplicates()


    pips = [
        ['predicted_motif_region_by_kadane.tsv','motif_pred_on_pathogenic_region.tsv',False],
        ['predicted_motif_region_by_kadane_0_15.tsv','motif_pred_on_pip.tsv',False],
        ['predicted_motif_region_by_kadane_0_15.tsv','motif_pred_on_pip_extension.tsv',True],
    ]

    for file_name,output_file,extension in pips:
        # 3. Load PIP Region
        # pip_path = os.path.join(file_dir, "elm", "predicted_motif_region_by_am_sequential_rule.tsv")
        pip_path = os.path.join(file_dir, "pip", file_name)
        pip_df = pd.read_csv(pip_path, sep="\t")

        # Filter For only Kadane
        pip_df = pip_df[pip_df['Found_by'] == "Kadane"]

        proteins_with_mutation = pip_df["Protein_ID"].unique().tolist()

        sequences_df = pip_df.merge(main_df[['Protein_ID','Sequence']],how="left",on="Protein_ID")
        sequences_df['Seq_Length'] = sequences_df['Sequence'].str.len()

        if extension:
            e = 3
            sequences_df['Start'] = sequences_df.apply(lambda row:max(row['Start'] -e,1),axis=1)
            sequences_df['End'] = sequences_df.apply(lambda row:min(row['End'] +e ,row['Seq_Length']),axis=1)

        sequences_df['Sequence'] = sequences_df.apply(
            lambda row: row['Sequence'][row['Start'] - 1: row['End']], axis=1
        )

        # 4. Run regex on Sequences -> Modify Sequences based on mutations
        predicted_motifs_with_mut = main(sequences_df,elm_classes_with_info)
        to_dir  = os.path.join(file_dir, "pip")
        predicted_motifs_with_mut.to_csv(os.path.join(to_dir, output_file),sep="\t",index=False)