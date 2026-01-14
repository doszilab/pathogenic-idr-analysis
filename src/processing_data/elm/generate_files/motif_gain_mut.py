import os
import pandas as pd
import re
from tqdm import tqdm

def main(sequences,elm_classes_with_info,clinvar_df,position_col="Position",mut_col="HGVSp_Short"):
    results = []

    for index,row in tqdm(clinvar_df.iterrows(),total=clinvar_df.shape[0]):
        mutation = row[mut_col][2:]
        ref_aa = mutation[0]
        mut_aa = mutation[-1]
        position = row[position_col]
        protein = row["Protein_ID"]
        original_sequence =sequences[protein]
        sequence = list(original_sequence)
        sequence[position -1] = mut_aa
        modified_sequence = "".join(sequence)

        for _, elm_row in elm_classes_with_info.iterrows():
            elm_identifier = elm_row["ELMIdentifier"]
            elm_type = elm_row["ELMType"]
            regular_expression = elm_row["Regex"]
            pattern = re.compile(regular_expression)


            # Get original motif regions
            original_regions = {(m.start(), m.end()) for m in pattern.finditer(original_sequence)}

            for match in pattern.finditer(modified_sequence):
                start, end = match.span()
                matched_sequence = match.group()
                if (start, end) not in original_regions:
                    info = row.tolist() + elm_row.tolist() + [matched_sequence]
                    if start <= position <= end:
                        results.append(
                            info
                        )

    df = pd.DataFrame(results,columns=list(clinvar_df.columns) + list(elm_classes_with_info.columns) + ["Matched_Sequence"])
    return df

if __name__ == "__main__":
    base_dir = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations"
    file_dir = os.path.join(base_dir, "processed_data","files")
    # 1. Load mutations
    clinvar_path = os.path.join(file_dir,"clinvar","Pathogenic","disorder","clinvar_functional_categorized_final.tsv")
    clinvar_df = pd.read_csv(clinvar_path,sep="\t",
                             # nrows=100
                             )
    proteins_with_mutation = clinvar_df["Protein_ID"].unique().tolist()
    # 2. Load Known ELMs with Regex
    known_elms_path = os.path.join(base_dir,"data","discanvis_base_files","elm","proteome_scann","known_instances_with_match.tsv")
    known_elms_df = pd.read_csv(known_elms_path, sep="\t")
    elm_classes_with_info = known_elms_df[["ELMType","ELMIdentifier","Regex"]].drop_duplicates()
    # 3. Load Sequences
    main_file_path = os.path.join(base_dir, "data", "discanvis_base_files", "sequences", "loc_chrom_with_names_main_isoforms_with_seq.tsv")
    main_df = pd.read_csv(main_file_path, sep="\t")
    sequences = main_df[main_df["Protein_ID"].isin(proteins_with_mutation)][["Protein_ID","Sequence"]].set_index("Protein_ID")["Sequence"].to_dict()

    # 4. Run regex on Sequences -> Modify Sequences based on mutations
    predicted_motifs_with_mut = main(sequences,elm_classes_with_info,clinvar_df)
    print(predicted_motifs_with_mut)
    to_dir  = os.path.join(file_dir, "elm","motif_gain")
    predicted_motifs_with_mut.to_csv(os.path.join(to_dir, "motif_gain_mutations.tsv"),sep="\t",index=False)

    # Interesting Example NRL-206
    # 5. Filter For Disorder
    # 6. Filter Out those Motifs that match with the original motifs ?
    pass