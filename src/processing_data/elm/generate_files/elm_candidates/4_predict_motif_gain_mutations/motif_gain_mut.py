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
                    if start <= position <= end:
                        info = row.tolist() + elm_row.tolist() + [matched_sequence, original_sequence[start:end]]
                        results.append(info)

    df = pd.DataFrame(results,columns=list(clinvar_df.columns) + list(elm_classes_with_info.columns) + ["Matched_Sequence", "Original_Sequence"])
    return df


def precompute_original_motifs(sequences, elm_classes_with_info):
    """
    Precompute all motif matches for each (protein, ELM) on the *original* sequences.
    Returns a dict of the form:
        original_matches[protein_id][elm_identifier] = set of (start, end)
    and a dict of precompiled patterns:
        patterns[elm_identifier] = (compiled_regex, elm_type)
    """
    patterns = {}
    # Pre-compile all ELM regexes and store them along with elm_type
    for _, elm_row in elm_classes_with_info.drop_duplicates(subset=["ELMIdentifier"]).iterrows():
        elm_identifier = elm_row["ELMIdentifier"]
        elm_type = elm_row["ELMType"]
        regex_str = elm_row["Regex"]
        compiled = re.compile(regex_str)
        patterns[elm_identifier] = (compiled, elm_type)

    # Build a nested dictionary of original motif spans
    original_matches = {}
    for protein_id, seq in tqdm(sequences.items(), desc="Precomputing original motifs"):
        original_matches[protein_id] = {}
        for elm_identifier, (compiled_pattern, elm_type) in patterns.items():
            spans = set((m.start(), m.end()) for m in compiled_pattern.finditer(seq))
            original_matches[protein_id][elm_identifier] = spans

    return original_matches, patterns

def find_motif_gains(
    sequences,elm_classes_with_info,clinvar_df,
    position_col="Position",
    mut_col="HGVSp_Short",
    window_size=50
):
    """
    For each mutation in `clinvar_df`, mutates the protein sequence locally and checks
    if that mutation introduces any new motifs that include the mutated position.
    Uses a local window around the mutation to cut down on regex searching time.
    """

    # 1. Precompute original motifs + Precompile patterns
    original_matches, pattern_dict = precompute_original_motifs(sequences, elm_classes_with_info)

    results = []

    # 2. Iterate through each mutation
    for index, row in tqdm(clinvar_df.iterrows(), total=clinvar_df.shape[0], desc="Checking mutations"):
        mutation = row[mut_col][2:]  # e.g. 'p.A42T' => 'A42T' after [2:]
        ref_aa = mutation[0]
        mut_aa = mutation[-1]
        position = row[position_col]  # 1-based position
        protein_id = row["Protein_ID"]

        # Safety checks
        if protein_id not in sequences:
            # If for some reason the protein sequence is not available, skip.
            continue

        original_sequence = sequences[protein_id]
        seq_length = len(original_sequence)
        # Make sure position is valid in the sequence:
        # (If your data can have out-of-bounds positions, you might want to handle them carefully)
        if not (1 <= position <= seq_length):
            continue

        # 3. Mutate the sequence locally
        seq_list = list(original_sequence)
        seq_list[position - 1] = mut_aa
        modified_sequence = "".join(seq_list)

        # 4. Instead of searching the entire protein, define a local window around the mutation
        start_idx = max(0, position - 1 - window_size)
        end_idx = min(seq_length, position - 1 + window_size)
        local_original = original_sequence[start_idx:end_idx]
        local_modified = modified_sequence[start_idx:end_idx]

        # 5. For each ELM pattern, find new matches in the local modified region
        for elm_identifier, (compiled_pattern, elm_type) in pattern_dict.items():
            # All original motif spans for this protein & ELM
            # (We already computed them across the full sequence.)
            original_spans = original_matches[protein_id][elm_identifier]

            # Search only within the local modified region
            for match in compiled_pattern.finditer(local_modified):
                local_start, local_end = match.span()  # positions relative to local_modified
                # Convert local window coords to global coords
                global_start = local_start + start_idx
                global_end = local_end + start_idx

                # Check if mutated position is within this motif
                # (Remember that in Python indexing, position-1 is the 0-based index.)
                if global_start <= (position - 1) < global_end:
                    # Check if it's truly a new motif gain (not present in the original)
                    if (global_start, global_end) not in original_spans:
                        matched_seq_mod = modified_sequence[global_start:global_end]
                        matched_seq_orig = original_sequence[global_start:global_end]

                        # Build one output record
                        info = (
                            list(row.values)  # all columns from the clinvar_df row
                            + [elm_type, elm_identifier]  # from ELM info
                            + [matched_seq_mod, matched_seq_orig]  # new vs. original substring
                            + [f"{global_start +1}-{global_end}",row["mutation_count"]]  # new vs. original substring
                        )
                        results.append(info)

    # 6. Build an output DataFrame
    # We want columns from clinvar_df + ELMType + ELMIdentifier + Matched_Sequence + Original_Sequence
    out_columns = (
        list(clinvar_df.columns)
        + ["ELMType", "ELMIdentifier", "Matched_Sequence", "Original_Sequence", 'Motif_Region',"mutation_count"]
    )
    df = pd.DataFrame(results, columns=out_columns)
    return df

def get_unique_mutations(clinvar_df):

    group_lst = [
        'Protein_ID',
        'Position',
        'HGVSp_Short',
        'ID',
        'nDisease',
        "disease_category", "Rare", "Developmental",
        "category_names", 'genic_category', 'structure',
        'Interpretation', 'Category', 'info', 'info_cols',
    ]

    unique_mutational_df = clinvar_df.groupby(group_lst).size().reset_index(name='mutation_count')

    return unique_mutational_df

if __name__ == "__main__":
    base_dir = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations"
    file_dir = os.path.join(base_dir, "processed_data","files")

    mutation_types = [
        "Pathogenic",
        "Uncertain",
    ]

    for mut in mutation_types:

        # 1. Load mutations
        clinvar_path = os.path.join(file_dir,"clinvar",mut,"disorder","clinvar_functional_categorized_final.tsv")
        clinvar_df = pd.read_csv(clinvar_path,sep="\t",)
        clinvar_df = get_unique_mutations(clinvar_df)

        proteins_with_mutation = clinvar_df["Protein_ID"].unique().tolist()
        # 2. Load Known ELMs with Regex
        known_elms_path = os.path.join(base_dir, "processed_data", "files", "elm",
                                       f"elm_known_with_am_info_all_disorder_class_filtered.tsv")
        known_elms_df = pd.read_csv(known_elms_path, sep="\t")
        elm_classes_with_info = known_elms_df[["ELMType","ELMIdentifier","Regex"]].drop_duplicates()

        # 3. Load Sequences
        main_file_path = os.path.join(base_dir, "data", "discanvis_base_files", "sequences", "loc_chrom_with_names_main_isoforms_with_seq.tsv")
        main_df = pd.read_csv(main_file_path, sep="\t")
        sequences = main_df[main_df["Protein_ID"].isin(proteins_with_mutation)][["Protein_ID","Sequence"]].set_index("Protein_ID")["Sequence"].to_dict()

        # 4. Run regex on Sequences -> Modify Sequences based on mutations
        predicted_motifs_with_mut = find_motif_gains(sequences,elm_classes_with_info,clinvar_df)
        # predicted_motifs_with_mut = main(sequences,elm_classes_with_info,clinvar_df)
        to_dir  = os.path.join(file_dir, "elm",'clinvar',"motif_gain",mut)
        if not os.path.exists(to_dir):
            os.makedirs(to_dir)
        predicted_motifs_with_mut.to_csv(os.path.join(to_dir, f"motif_gain_mutations.tsv"),sep="\t",index=False)