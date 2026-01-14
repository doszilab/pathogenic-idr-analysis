import pandas as pd
from scipy.stats import fisher_exact
from collections import defaultdict
import pandas as pd
from scipy.stats import fisher_exact
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import random
from scipy.stats import entropy
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch
import os

def plot_enrichment_test_results(result_uncertain, result_pathogenic, result_predicted_pathogenic, plot_dir=None,motif="PEM"):
    labels = ["Uncertain", "Pathogenic", "Predicted Pathogenic"]
    enrichment_folds = [result_uncertain["Enrichment Fold"],result_pathogenic["Enrichment Fold"], result_predicted_pathogenic["Enrichment Fold"]]
    # log2_enrichment = [np.log2(x) for x in enrichment_folds]

    # Plot
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, enrichment_folds, color=["gray", "firebrick", "orange"])
    plt.axhline(0, color='black', linestyle='--')

    # Add values
    for bar, val in zip(bars, enrichment_folds):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.1 if val > 0 else val - 0.3,
                 f"{val:.2f}", ha='center', va='bottom' if val > 0 else 'top', fontsize=10)

    plt.ylabel("Enrichment Fold")
    plt.ylim(min(enrichment_folds), max(enrichment_folds) * 1.1)
    plt.title(f"Enrichment of Variants in Predicted Motifs ({motif})")
    plt.tight_layout()
    if plot_dir:
        plt.savefig(f"{plot_dir}/enrichment_test_results_{motif.lower()}.png")
    # plt.show()

def run_enrichment_test(disorder_df, mutation_df, pem_df, main_proteins_df, mutation_type="Uncertain"):
    # Filter to main isoforms and disordered positions only
    disorder_df = disorder_df[disorder_df["Protein_ID"].isin(main_proteins_df["Protein_ID"])]
    # disorder_df = disorder_df[disorder_df["CombinedDisorder"] == 1.0]
    disorder_df["in_IDR"] = True

    # Expand PEM positions
    pem_expanded = pem_df[["Protein_ID", "Start", "End"]].copy()
    pem_expanded = pem_expanded.assign(
        Position=pem_expanded.apply(lambda row: list(range(row["Start"], row["End"] + 1)), axis=1)
    ).explode("Position")[["Protein_ID", "Position"]]
    pem_expanded["in_PEM"] = True

    # Mark mutation positions
    mutation_df = mutation_df.copy()
    mutation_df["has_mut"] = True
    mutation_positions_df = mutation_df[["Protein_ID", "Position", "has_mut"]]

    # Merge
    merged = disorder_df.merge(pem_expanded, on=["Protein_ID", "Position"], how="left")
    merged = merged.merge(mutation_positions_df, on=["Protein_ID", "Position"], how="left")
    merged["in_PEM"] = merged["in_PEM"].fillna(False)
    merged["has_mut"] = merged["has_mut"].fillna(False)

    # Contingency table
    A = merged[(merged["in_PEM"]) & (merged["has_mut"])][["Protein_ID", "Position"]].drop_duplicates().shape[0]
    B = merged[(~merged["in_PEM"]) & (merged["has_mut"])][["Protein_ID", "Position"]].drop_duplicates().shape[0]
    C = merged[(merged["in_PEM"]) & (~merged["has_mut"])][["Protein_ID", "Position"]].drop_duplicates().shape[0]
    D = merged[(~merged["in_PEM"]) & (~merged["has_mut"])][["Protein_ID", "Position"]].drop_duplicates().shape[0]

    contingency_table = [[A, B], [C, D]]
    odds_ratio, p_value = fisher_exact(contingency_table)
    enrichment_fold = (A / (A + C)) / (B / (B + D)) if (A + C) > 0 and (B + D) > 0 else float("nan")

    return {
        "Mutation Type": mutation_type,
        "Contingency Table": contingency_table,
        "Odds Ratio": odds_ratio,
        "P-Value": p_value,
        "Enrichment Fold": enrichment_fold
    }


def calculate_expected_vs_observed_zscore(disorder_df, uncertain_df, pem_df, main_proteins_df):
    # Filter to main isoforms and IDR positions
    disorder_df = disorder_df[disorder_df["Protein_ID"].isin(main_proteins_df["Protein_ID"])]
    disorder_df = disorder_df[disorder_df["CombinedDisorder"] == 1.0]

    # Expand PEM positions
    pem_expanded = pem_df[["Protein_ID", "Start", "End"]].copy()
    pem_expanded = pem_expanded.assign(
        Position=pem_expanded.apply(lambda row: list(range(row["Start"], row["End"] + 1)), axis=1)
    ).explode("Position")[["Protein_ID", "Position"]]
    pem_expanded["in_PEM"] = True

    # Merge PEM info to disorder positions
    disorder_with_pem = disorder_df.merge(pem_expanded, on=["Protein_ID", "Position"], how="left")
    disorder_with_pem["in_PEM"] = disorder_with_pem["in_PEM"].fillna(False)

    # Add uncertain mutation marker
    uncertain_df = uncertain_df.copy()
    uncertain_df["has_uncertain_mut"] = True
    merged = disorder_with_pem.merge(
        uncertain_df[["Protein_ID", "Position", "has_uncertain_mut"]],
        on=["Protein_ID", "Position"], how="left"
    )
    merged["has_uncertain_mut"] = merged["has_uncertain_mut"].fillna(False)

    # Total positions
    total_idr_positions = disorder_with_pem[["Protein_ID", "Position"]].drop_duplicates().shape[0]
    total_pem_positions = disorder_with_pem[disorder_with_pem["in_PEM"]][["Protein_ID", "Position"]].drop_duplicates().shape[0]

    # Observed and expected uncertain in PEM
    observed_uncertain_in_pem = merged[(merged["in_PEM"]) & (merged["has_uncertain_mut"])] \
        [["Protein_ID", "Position"]].drop_duplicates().shape[0]
    total_uncertain_in_idr = merged[merged["has_uncertain_mut"]][["Protein_ID", "Position"]].drop_duplicates().shape[0]
    expected_uncertain_in_pem = total_uncertain_in_idr * (total_pem_positions / total_idr_positions)

    # Calculate z-score
    proportion = total_pem_positions / total_idr_positions
    std_dev = (total_uncertain_in_idr * proportion * (1 - proportion)) ** 0.5
    z_score = (observed_uncertain_in_pem - expected_uncertain_in_pem) / std_dev if std_dev > 0 else float("nan")

    return {
        "Observed uncertain mutations in PEM": observed_uncertain_in_pem,
        "Expected uncertain mutations in PEM": round(expected_uncertain_in_pem, 2),
        "Z-score": round(z_score, 2),
        "Total IDR positions": total_idr_positions,
        "Total PEM positions": total_pem_positions,
        "Total uncertain mutations in IDR": total_uncertain_in_idr
    }

def plot_observed_vs_expected(observed, expected, z_score, plot_dir=None):
    labels = ["Observed", "Expected"]
    values = [observed, expected]
    colors = ["gray", "steelblue"]

    plt.figure(figsize=(4.5, 4))
    bars = plt.bar(labels, values, color=colors)

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + max(values) * 0.01,
                 f"{val:.0f}", ha='center', va='bottom', fontsize=10)

    plt.ylabel("Number of Uncertain Mutations in PEMs")
    plt.title(f"Observed vs Expected\nZ-score = {z_score:.2f}")
    plt.tight_layout()

    if plot_dir:
        plt.savefig(f"{plot_dir}/uncertain_observed_vs_expected_in_pems.png")
    plt.show()

def generate_uncertain_mutation_density_table(pem_df, uncertain_df, disorder_df,main_proteins_df, window_size=20,region_type=False):
    tqdm.pandas()


    exclude_categories = ['Inborn Genetic Diseases', 'Unknown']
    uncertain_df = uncertain_df[~uncertain_df['category_names'].isin(exclude_categories)]

    disorder_df = disorder_df[disorder_df["Protein_ID"].isin(main_proteins_df["Protein_ID"])]
    # disorder_df = disorder_df[disorder_df["CombinedDisorder"] == 1.0]

    # Expand PEM positions
    pem_df = pem_df.copy()
    pem_df["Motif_ID"] = pem_df["Protein_ID"] + ":" + pem_df["Start"].astype(str) + "-" + pem_df["End"].astype(str)
    pem_expanded = pem_df.copy()

    uncertain_by_protein = uncertain_df.groupby("Protein_ID")["Position"].progress_apply(set).to_dict()
    disorder_by_protein = disorder_df.groupby("Protein_ID")["Position"].progress_apply(set).to_dict()

    results = []
    for _, row in tqdm(pem_expanded.iterrows(), total=len(pem_expanded)):
        pid = row.Protein_ID
        motif_start = row.Start
        motif_end = row.End
        motif_len = motif_end - motif_start + 1
        motif_id = row.Motif_ID

        # Define motif, flank, and idr regions
        motif_positions = set(range(motif_start, motif_end + 1))

        flanking_size = max(window_size, motif_len)
        flank_start = max(1, motif_start - flanking_size)
        flank_end = motif_end + flanking_size
        flank_positions = set(range(flank_start, flank_end + 1)) - motif_positions

        # Preloaded sets
        uncertain_positions = uncertain_by_protein.get(pid, set())
        disorder_positions = disorder_by_protein.get(pid, set())

        # Count uncertain mutations using set intersection
        motif_mut_count = len(motif_positions & uncertain_positions)
        motif_density = motif_mut_count / len(motif_positions)

        # IDR-based enrichment (exclude motif from IDR background)
        idr_background = disorder_positions - motif_positions
        idr_mut_count = len(idr_background & uncertain_positions) + motif_mut_count
        idr_density = idr_mut_count / len(idr_background) if len(idr_background) > 0 else 0
        enrichment_vs_idr = motif_density / idr_density if idr_density > 0 else float("inf")

        results.append({
            **row.to_dict(),
            "Motif_Mut_Count": motif_mut_count,
            "Motif_Mut_Density": motif_density,
            "Protein_Mut_Count": idr_mut_count,
            "Protein_Mut_Density": idr_density,
            "Enrichment_Protein": enrichment_vs_idr
        })

    df = pd.DataFrame(results)

    if region_type:
        return df

    # Define the columns that identify a motif (excluding 'known' and 'ELM_Accession')
    id_columns = ['known', "Protein_ID", "ELM_Accession", "ELMIdentifier", "Start", "End"]

    columns_to_compare = [col for col in df.columns if col not in id_columns]

    # Split into known and unknown
    known_df = df[df['known'] == True]
    unknown_df = df[df['known'] == False]

    # Find duplicates: rows in unknown that are already present in known
    # We use merge with an inner join on the motif-identifying columns
    duplicates = unknown_df.merge(known_df, on=columns_to_compare, how='inner')

    # Drop these duplicates from the unknown_df
    clean_unknown = pd.merge(unknown_df, duplicates, how='outer', indicator=True).query('_merge == "left_only"').drop(
        columns=['_merge'])

    # Combine the known and cleaned unknown
    final_df = pd.concat([known_df, clean_unknown], ignore_index=True)
    final_df = final_df[df.columns]

    return final_df


def plot_enriched_motifs(enriched_pems, plot_dir=None):
    # Count motif types with enriched uncertain mutations
    motif_counts = enriched_pems["ELMType"].value_counts().reset_index()
    motif_counts.columns = ["Motif Class", "Count"]

    # Take top 20 motif types
    top_motifs = motif_counts.head(20)

    # Plot
    plt.figure(figsize=(8, 5))
    bars = plt.barh(top_motifs["Motif Class"], top_motifs["Count"], color="mediumpurple")
    plt.xlabel("Number of Motifs with Enriched Uncertain Mutations")
    plt.title("Top Motif Classes with Enriched Uncertain Mutations")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # Save plot
    if plot_dir:
        plot_path = f"{plot_dir}/top_motif_classes_enriched_uncertain.png"
        plt.savefig(plot_path)
    plt.show()

def calculate_mutation_entropy(disorder_df, uncertain_df):
    entropies = {}

    disorder_by_gene = disorder_df.groupby("Protein_ID")["Position"].apply(list).to_dict()
    uncertain_by_gene = uncertain_df.groupby("Protein_ID")["Position"].apply(set).to_dict()

    for gene, disorder_pos in tqdm(disorder_by_gene.items()):
        uncertain_pos = uncertain_by_gene.get(gene, set())
        mutation_flags = np.array([1 if pos in uncertain_pos else 0 for pos in disorder_pos])
        if mutation_flags.sum() == 0:
            continue
        prob_dist = mutation_flags / mutation_flags.sum()
        entropies[gene] = entropy(prob_dist)
    return entropies

def get_non_uniform_proteins(disorder_df, uncertain_df, pem_df, entropy_threshold=3.5):
    disorder_subset = disorder_df[disorder_df["Protein_ID"].isin(pem_df["Protein_ID"])]
    uncertain_subset = uncertain_df[uncertain_df["Protein_ID"].isin(pem_df["Protein_ID"])]

    disorder_by_protein = disorder_subset.groupby("Protein_ID")["Position"].apply(list).to_dict()
    uncertain_by_protein = uncertain_subset.groupby("Protein_ID")["Position"].apply(set).to_dict()

    non_uniform_proteins = []

    for pid, positions in disorder_by_protein.items():
        mutation_flags = np.array([1 if pos in uncertain_by_protein.get(pid, set()) else 0 for pos in positions])
        if mutation_flags.sum() == 0:
            continue  # No mutations
        prob_dist = mutation_flags / mutation_flags.sum()
        ent = entropy(prob_dist)
        if ent < entropy_threshold:
            non_uniform_proteins.append(pid)

    return set(non_uniform_proteins)

def split_into_continuous_regions(position_list):
    if not position_list:
        return []
    sorted_pos = sorted(position_list)
    regions = [[sorted_pos[0]]]
    for pos in sorted_pos[1:]:
        if pos == regions[-1][-1] + 1:
            regions[-1].append(pos)
        else:
            regions.append([pos])
    return regions

def compare_uncertain_mutation_enrichment_vs_random(pem_df, disorder_df, uncertain_df, main_proteins_df,
                                                    n_random=1000):

    # Step 1: Pre-filter
    disorder_df = disorder_df[disorder_df["Protein_ID"].isin(main_proteins_df["Protein_ID"])]
    disorder_df = disorder_df[disorder_df["CombinedDisorder"] == 1.0]
    # pem_df = pem_df[pem_df["N_Motif_Predicted"] == "Low_Amount"]
    proteins_with_uncertain = set(uncertain_df["Protein_ID"].unique())
    disorder_df = disorder_df[disorder_df["Protein_ID"].isin(proteins_with_uncertain)]

    # Step 2: Get non-uniformly mutated proteins
    non_uniform_proteins = get_non_uniform_proteins(disorder_df, uncertain_df, pem_df, entropy_threshold=2)

    # Step 3: Filter all datasets
    pem_df = pem_df[pem_df["Protein_ID"].isin(non_uniform_proteins)]
    uncertain_df = uncertain_df[uncertain_df["Protein_ID"].isin(non_uniform_proteins)]
    disorder_df = disorder_df[disorder_df["Protein_ID"].isin(non_uniform_proteins)]

    print("Total PEMs considered:", len(pem_df))
    print("Proteins with non-uniform uncertain mutations:", len(non_uniform_proteins))

    # Step 4: Expand PEMs
    pem_expanded = pem_df.copy()
    pem_expanded["Motif_ID"] = pem_expanded["Protein_ID"] + ":" + pem_expanded["Start"].astype(str) + "-" + pem_expanded["End"].astype(str)
    pem_expanded = pem_expanded.assign(
        Position=pem_expanded.apply(lambda row: list(range(row["Start"], row["End"] + 1)), axis=1)
    ).explode("Position")[["Protein_ID", "Position", "Motif_ID"]]

    # Step 5: Keep only PEMs with actual uncertain mutations
    pem_with_mut = pem_expanded.merge(uncertain_df[["Protein_ID", "Position"]], on=["Protein_ID", "Position"], how="inner")
    valid_motifs = pem_with_mut["Motif_ID"].unique()
    pem_expanded = pem_expanded[pem_expanded["Motif_ID"].isin(valid_motifs)]
    pem_df = pem_df[pem_df["Protein_ID"].isin(pem_expanded["Protein_ID"].unique())]

    print("PEMs with uncertain mutations:", len(valid_motifs))

    # Step 6: Mutation density in PEMs
    uncertain_df = uncertain_df.copy()
    uncertain_df["has_mut"] = True
    mut_pos = uncertain_df[["Protein_ID", "Position", "has_mut"]]
    pem_mut_df = pem_expanded.merge(mut_pos, on=["Protein_ID", "Position"], how="left").fillna(False)
    pem_densities = pem_mut_df.groupby("Motif_ID")["has_mut"].mean().values
    pem_mut_rate = np.mean(pem_densities)

    # Step 7: Random region sampling
    disordered_by_protein = defaultdict(list)
    uncertain_by_protein = uncertain_df.groupby("Protein_ID")["Position"].apply(set).to_dict()

    for pid, pos_list in disorder_df.groupby("Protein_ID")["Position"]:
        continuous_regions = split_into_continuous_regions(pos_list.tolist())
        disordered_by_protein[pid] = continuous_regions

    random_densities = []
    motif_lengths = (pem_df["End"] - pem_df["Start"] + 1).unique()

    for length in tqdm(motif_lengths):
        for _ in range(n_random):
            eligible_proteins = list(non_uniform_proteins & disordered_by_protein.keys())
            if not eligible_proteins:
                continue
            protein = random.choice(eligible_proteins)
            regions = [r for r in disordered_by_protein[protein] if len(r) >= length]
            if not regions:
                continue
            region = random.choice(regions)
            start_idx = random.randint(0, len(region) - length)
            segment = region[start_idx:start_idx + length]
            mut_count = len(set(segment) & uncertain_by_protein.get(protein, set()))
            if mut_count == 0:
                continue  # skip random regions with no mutations (to match PEM filtering)
            density = mut_count / length
            random_densities.append(density)

    random_density_mean = np.mean(random_densities)
    enrichment = pem_mut_rate / random_density_mean if random_density_mean > 0 else float("inf")

    # Step 8: Plot
    plt.figure(figsize=(6, 4))

    # Histogram: Random (background)
    plt.hist(random_densities, bins=40, alpha=0.4, label='Random Regions (hist)', color='gray', density=True)

    # Histogram: PEMs (foreground)
    plt.hist(pem_densities, bins=30, alpha=0.5, label='PEMs (hist)', color='orangered', edgecolor='black',
             linewidth=1.2, density=True)

    # Smoothed KDE line: Random
    kde_random = gaussian_kde(random_densities)
    x_vals = np.linspace(min(random_densities + list(pem_densities)), max(random_densities + list(pem_densities)), 500)
    plt.plot(x_vals, kde_random(x_vals), color='black', linestyle='-', label='Random Regions (KDE)')

    # Smoothed KDE line: PEMs
    kde_pem = gaussian_kde(pem_densities)
    plt.plot(x_vals, kde_pem(x_vals), color='red', linestyle='-', label='PEMs (KDE)')

    # Mean lines
    plt.axvline(random_density_mean, color='black', linestyle='--', label=f'Random Mean = {random_density_mean:.2f}')
    plt.axvline(pem_mut_rate, color='red', linestyle='--', label=f'PEM Mean = {pem_mut_rate:.2f}')

    # Labels and styling
    plt.xlabel("Mutation Density")
    plt.ylabel("Normalized Frequency / Density")
    plt.title("Uncertain Mutation Density in PEMs vs Random IDRs\n(Filtered for Non-Uniformly Mutated Proteins)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"PEM mutation density (mean): {pem_mut_rate:.4f}")
    print(f"Random region mutation density (mean): {random_density_mean:.4f}")
    print(f"Enrichment fold: {enrichment:.2f}")

    return enrichment

def plot_gene_mutation_uniformity(uncertain_df, disorder_df, main_proteins_df, min_len=20):
    # Filter
    disorder_df = disorder_df[disorder_df["Protein_ID"].isin(main_proteins_df["Protein_ID"])]
    disorder_df = disorder_df[disorder_df["CombinedDisorder"] == 1.0]
    uncertain_df = uncertain_df.copy()

    # Group positions
    disorder_by_gene = disorder_df.groupby("Protein_ID")["Position"].apply(set).to_dict()
    uncertain_by_gene = uncertain_df.groupby("Protein_ID")["Position"].apply(set).to_dict()

    densities = []
    entropies = []

    for gene, disorder_pos in disorder_by_gene.items():
        if len(disorder_pos) < min_len:
            continue
        uncertain_pos = uncertain_by_gene.get(gene, set())
        mutation_counts = [1 if pos in uncertain_pos else 0 for pos in disorder_pos]
        if sum(mutation_counts) == 0:
            continue
        density = sum(mutation_counts) / len(disorder_pos)
        dens_vector = np.array(mutation_counts)
        prob_dist = dens_vector / dens_vector.sum() if dens_vector.sum() > 0 else np.ones_like(dens_vector) / len(dens_vector)
        ent = entropy(prob_dist)
        densities.append(density)
        entropies.append(ent)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.hist(densities, bins=50, alpha=0.8, color="lightcoral")
    plt.xlabel("Mutation Density in IDRs")
    plt.ylabel("Number of Genes")
    plt.title("Distribution of Uncertain Mutation Density Across Genes")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.hist(entropies, bins=50, alpha=0.8, color="cadetblue")
    plt.xlabel("Entropy of Mutation Distribution")
    plt.ylabel("Number of Genes")
    plt.title("Uniformity of Mutation Distribution (Entropy)")
    plt.tight_layout()
    plt.show()

    return pd.DataFrame({
        "Gene": list(disorder_by_gene.keys())[:len(densities)],
        "Mutation Density": densities,
        "Entropy": entropies
    })


def analyze_benign_constraint_in_pems(
    disorder_df,
    benign_df,
    pem_df,
    main_proteins_df,
    verbose=True,
    filepath=None,
    ylabel="Benign Mutation Frequency",
    title="Benign Mutation Rate\nKey Residues vs Other IDR"
):
    # Step 1: Filter PEMs to predicted (Low_Amount) motifs only
    pem_df = pem_df[pem_df["N_Motif_Predicted"] == "Low_Amount"]

    # Step 2: Collect key positions from PEMs
    key_positions = set()
    for _, row in pem_df.iterrows():
        if pd.isna(row["key_positions_protein"]):
            continue
        try:
            positions = [
                int(pos.strip())
                for pos in str(row["key_positions_protein"]).split(",")
                if pos.strip().isdigit()
            ]
            for pos in positions:
                key_positions.add((row["Protein_ID"], pos))
        except Exception as e:
            print(f"Error parsing row: {row['key_positions_protein']}, error: {e}")

    # Step 3: Filter disorder to main isoforms and disordered positions


    # Step 4: Keep only proteins that have both PEMs and benign mutations
    pem_proteins = pem_df["Protein_ID"].unique()
    benign_df = benign_df[benign_df["Protein_ID"].isin(main_proteins_df["Protein_ID"])]
    benign_df = benign_df[benign_df["Protein_ID"].isin(pem_proteins)]

    # Step 5: Filter benign mutations to those in disordered regions and overlapping with PEM proteins
    benign_df = benign_df.merge(disorder_df, on=["Protein_ID", "Position"], how="inner")
    benign_df["has_benign"] = True

    # Step 6: Keep only disorder positions from proteins with PEMs and benign mutations
    relevant_proteins = benign_df["Protein_ID"].unique()
    disorder_df = disorder_df[disorder_df["Protein_ID"].isin(relevant_proteins)]

    # Step 7: Mark key positions
    disorder_df["is_key"] = disorder_df.apply(
        lambda row: (row["Protein_ID"], row["Position"]) in key_positions, axis=1
    )

    # Step 8: Merge benign mutation info into disorder data
    merged = disorder_df.merge(
        benign_df[["Protein_ID", "Position", "has_benign"]],
        on=["Protein_ID", "Position"],
        how="left"
    )
    merged["has_benign"] = merged["has_benign"].fillna(False)

    # Step 9: Count mutations and background positions
    A = merged[(merged["is_key"]) & (merged["has_benign"])][["Protein_ID", "Position"]].drop_duplicates().shape[0]
    B = merged[(~merged["is_key"]) & (merged["has_benign"])][["Protein_ID", "Position"]].drop_duplicates().shape[0]
    C = merged[(merged["is_key"]) & (~merged["has_benign"])][["Protein_ID", "Position"]].drop_duplicates().shape[0]
    D = merged[(~merged["is_key"]) & (~merged["has_benign"])][["Protein_ID", "Position"]].drop_duplicates().shape[0]

    # Step 10: Calculate mutation frequencies and enrichment
    key_prop = A / (A + C) if (A + C) > 0 else 0
    idr_prop = B / (B + D) if (B + D) > 0 else 0
    contingency = [[A, B], [C, D]]
    odds_ratio, p_value = fisher_exact(contingency)
    enrichment = (key_prop / idr_prop) if idr_prop > 0 else float("nan")

    # Step 11: Plot
    plt.figure(figsize=(4, 3))
    bars = plt.bar(["Key residues", "Other IDR"], [key_prop, idr_prop], color=[COLORS["Benign"], COLORS["Benign"]])

    for bar, val in zip(bars, [key_prop, idr_prop]):
        plt.text(bar.get_x() + bar.get_width() / 2, val * 1.01, f"{val*100:.2f}%",
                 ha='center', va='bottom', fontsize=11)

    plt.ylim(0, max(key_prop, idr_prop) * 1.15)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    if filepath:
        plt.savefig(filepath, dpi=300)
    plt.show()

    # Step 12: Print results if verbose
    if verbose:
        print(f"Benign Mutation in key position       (A): {A}")
        print(f"Benign Mutation in other IDR position (B): {B}")
        print(f"NO benign in key                      (C): {C}")
        print(f"NO benign in other IDR                (D): {D}")
        print(f"Fisher exact p-value: {p_value:.3e}")
        print(f"Odds ratio: {odds_ratio:.2f}")
        print(f"Enrichment (Key / IDR): {enrichment:.2f}")

    return {
        "Key benign count": A,
        "IDR-nonKey benign count": B,
        "Odds ratio": odds_ratio,
        "P-value": p_value,
        "Enrichment Fold": enrichment
    }

def analyze_benign_constraint_multipanel_plot(
    disorder_df,
    benign_df,
    pem_df,
    all_pem_df,
    main_proteins_df,
    filepath=None,
    verbose=True
):
    pem_df = pem_df[pem_df["N_Motif_Predicted"] == "Low_Amount"]
    benign_df = benign_df[benign_df["Protein_ID"].isin(main_proteins_df["Protein_ID"])]
    benign_df = benign_df.merge(disorder_df, on=["Protein_ID", "Position"], how="inner")
    benign_df["has_benign"] = True

    proteins_with_function = pem_df["Protein_ID"].unique()
    benign_df = benign_df[benign_df["Protein_ID"].isin(proteins_with_function)]
    all_pem_df = all_pem_df[all_pem_df["Protein_ID"].isin(proteins_with_function)]

    # Refined PEM keys
    refined_keys = set()
    for _, row in pem_df.iterrows():
        if pd.isna(row["key_positions_protein"]):
            continue
        try:
            positions = [int(pos.strip()) for pos in str(row["key_positions_protein"]).split(",") if pos.strip().isdigit()]
            for pos in positions:
                refined_keys.add((row["Protein_ID"], pos))
        except Exception:
            continue

    # All predicted PEM keys
    all_keys = set()
    for _, row in all_pem_df.iterrows():
        if pd.isna(row.get("key_positions_protein")):
            continue
        try:
            positions = [int(pos.strip()) for pos in str(row["key_positions_protein"]).split(",") if pos.strip().isdigit()]
            for pos in positions:
                all_keys.add((row["Protein_ID"], pos))
        except Exception:
            continue

    non_refined_keys = all_keys - refined_keys

    # Prepare a disorder region including both refined and non-refined key positions
    all_key_positions = list(refined_keys | non_refined_keys)
    key_df = pd.DataFrame(all_key_positions, columns=["Protein_ID", "Position"])
    key_df["key_type"] = key_df.apply(lambda row: "refined" if (row["Protein_ID"], row["Position"]) in refined_keys else "non_refined", axis=1)

    merged = key_df.merge(benign_df[["Protein_ID", "Position", "has_benign"]],
                          on=["Protein_ID", "Position"], how="left")
    merged["has_benign"] = merged["has_benign"].fillna(False)

    A = merged[(merged["key_type"] == "refined") & (merged["has_benign"])].shape[0]
    B = merged[(merged["key_type"] == "non_refined") & (merged["has_benign"])].shape[0]
    C = merged[(merged["key_type"] == "refined") & (~merged["has_benign"])].shape[0]
    D = merged[(merged["key_type"] == "non_refined") & (~merged["has_benign"])].shape[0]

    refined_prop = A / (A + C) if (A + C) > 0 else 0
    non_refined_prop = B / (B + D) if (B + D) > 0 else 0
    odds_ratio, p_value = fisher_exact([[A, B], [C, D]])
    enrichment = (refined_prop / non_refined_prop) if non_refined_prop > 0 else float("nan")

    # Plot
    plt.figure(figsize=(4, 3))
    bars = plt.bar(["Refined PEM Key", "All Motif Key"],
                   [refined_prop, non_refined_prop],
                   color=["salmon", "gray"])

    for bar, val in zip(bars, [refined_prop, non_refined_prop]):
        plt.text(bar.get_x() + bar.get_width() / 2, val * 1.01,
                 f"{val*100:.2f}%",
                 ha='center', va='bottom', fontsize=11)

    plt.ylabel("Benign Mutation Frequency")
    plt.title("Benign Mutation Rate\nRefined vs All Motif Key Residues")
    plt.ylim(0, max(refined_prop, non_refined_prop) * 1.15)
    plt.tight_layout()

    if filepath:
        plt.savefig(filepath, dpi=300)
    # plt.show()

    if verbose:
        print(f"\nBenign Mutation in refined key       (A): {A}")
        print(f"Benign Mutation in other PEM key     (B): {B}")
        print(f"NO benign in refined key             (C): {C}")
        print(f"NO benign in other PEM key           (D): {D}")
        print(f"Fisher exact p-value: {p_value:.3e}")
        print(f"Odds ratio: {odds_ratio:.2f}")
        print(f"Enrichment (Refined / Other): {enrichment:.2f}")

    return {
        "A (Refined + Benign)": A,
        "B (Other Key + Benign)": B,
        "C (Refined + No Benign)": C,
        "D (Other Key + No Benign)": D,
        "Refined Rate": refined_prop,
        "Other Rate": non_refined_prop,
        "Odds Ratio": odds_ratio,
        "P-Value": p_value,
        "Enrichment Fold": enrichment
    }


def plot_enrichment(n_top=20):
    # Reload the required files independently for this corrected plot
    base_dir = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations"
    uncertain_file = f"{base_dir}/processed_data/files/elm/clinvar/enriched/enriched_pems_with_uncertain_mutations.tsv"
    pathogenic_file = f"{base_dir}/processed_data/files/elm/clinvar/enriched/enriched_pems_with_pathogenic_mutations.tsv"

    to_dir = f"{base_dir}/plots/fig6"

    known_color = '#B4F8C8'
    predicted_color = '#FFB68A'

    # Load data
    uncertain_df = pd.read_csv(uncertain_file, sep="\t")
    pathogenic_df = pd.read_csv(pathogenic_file, sep="\t")

    # pathogenic_df["known"] = pathogenic_df['Found_Known']
    # uncertain_df["known"] = uncertain_df['Found_Known']

    # uncertain_df['Enrichment_Protein'] = np.log(uncertain_df['Enrichment_Protein'])
    # pathogenic_df['Enrichment_Protein'] = np.log(pathogenic_df['Enrichment_Protein'])

    base_table = pd.read_csv(os.path.join(base_dir, 'data', 'discanvis_base_files', 'sequences',
                                          'loc_chrom_with_names_isoforms_with_seq.tsv'), sep='\t')

    # Clean and filter
    def prepare_enriched_df(df, n_top):
        df = df.merge(
            base_table[['Protein_ID', 'Gene_Uniprot']],
            on='Protein_ID',
            how='left'
        )
        df = df.replace([float("inf"), float("Inf")], pd.NA)
        df = df.dropna(subset=["Enrichment_Protein"])
        df = df[df["Motif_Mut_Count"] > 0]
        df = df.sort_values(by="Enrichment_Protein", ascending=False)[:n_top]
        return df

        # Preprocess and get top n ELMIdentifiers by frequency

    uncertain_df_clean = prepare_enriched_df(uncertain_df, n_top=n_top)
    pathogenic_df_clean = prepare_enriched_df(pathogenic_df, n_top=n_top)

    # Plotting function
    def plot_enrichment_by_gene(df, title, with_legend=False,save_path=None):
        plt.figure(figsize=(6, 3))
        x_labels = df["Gene_Uniprot"].tolist()
        y_vals = df["Enrichment_Protein"].tolist()

        # Color by known
        known_vals = df['known']
        color_map = {
            True: known_color,   # Known motif
            False: predicted_color             # Predicted (PEM)
        }
        colors = [color_map.get(k, "gray") for k in known_vals]

        bars = plt.bar(x_labels, y_vals, color=colors)

        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Enrichment Score")
        plt.title(title)
        plt.tight_layout()
        # plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.ylim(0, max(y_vals)*1.2)

        # Optional legend
        if with_legend:
            legend_elements = [
                Patch(facecolor=known_color, label='Known Motif'),
                Patch(facecolor=predicted_color, label='Predicted (PEM)')
            ]
            plt.legend(handles=legend_elements, loc='upper right', frameon=False)

        if save_path:
            plt.savefig(save_path,dpi=300)
        plt.show()

    def prepare_frequency_df(df, n_top):
        df = df[df["Motif_Mut_Count"] > 0]

        # Count by ELMIdentifier and known status separately
        freq_df = (
            df.groupby(["ELMIdentifier", "known"])
            .size()
            .reset_index(name="count")
            .sort_values(by="count", ascending=False)
        )

        # Optionally keep only top N most frequent ELMIdentifiers (regardless of known/predicted)
        top_ids = (
            freq_df.groupby("ELMIdentifier")["count"]
            .sum()
            .sort_values(ascending=False)
            .head(n_top)
            .index
        )
        freq_df = freq_df[freq_df["ELMIdentifier"].isin(top_ids)]

        return freq_df

    uncertain_freq_df = prepare_frequency_df(uncertain_df, n_top=n_top)
    pathogenic_freq_df = prepare_frequency_df(pathogenic_df, n_top=n_top)

    # Plotting function
    def plot_elm_frequency(df, title, with_legend=False,save_path=None):

        # Convert to format expected by stacked bar plot
        pivot_df = df.pivot_table(index="ELMIdentifier", columns="known", values="count", fill_value=0)
        # Sort by total (known + predicted)
        pivot_df["total"] = pivot_df.sum(axis=1)
        pivot_df = pivot_df.sort_values(by="total", ascending=False).drop(columns="total")

        # Extract values
        x_labels = pivot_df.index.tolist()
        known_vals = pivot_df.get(True, pd.Series([0] * len(x_labels)))
        unknown_vals = pivot_df.get(False, pd.Series([0] * len(x_labels)))

        # Plot
        plt.figure(figsize=(6, 3))
        plt.bar(x_labels, known_vals, label="Known", color=known_color)
        plt.bar(x_labels, unknown_vals, bottom=known_vals, label="Predicted (PEM)", color=predicted_color)

        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Motif Instance Count")
        plt.title(title)
        plt.tight_layout()
        # plt.grid(axis='y', linestyle='--', alpha=0.5)

        # Legend
        if with_legend:
            legend_elements = [
                Patch(facecolor='mediumseagreen', label='Known Motif'),
                Patch(facecolor='gray', label='Predicted (PEM)')
            ]
            plt.legend(handles=legend_elements, loc='upper right', frameon=False)

        if save_path:
            plt.savefig(save_path,dpi=300)
        plt.show()

    # Plot each separately
    plot_elm_frequency(pathogenic_freq_df, f"Top {n_top} Frequent ELM Classes (Pathogenic - HotspotPEM)", with_legend=False,save_path=f"{to_dir}/C2.png")
    plot_elm_frequency(uncertain_freq_df, f"Top {n_top} Frequent ELM Classes (Uncertain - HotspotPEM)", with_legend=False)

    # Plot each separately
    plot_enrichment_by_gene(pathogenic_df_clean, f"Top {n_top} Mutated Genes (Pathogenic - HotspotPEM)", with_legend=True,save_path=f"{to_dir}/C.png")
    plot_enrichment_by_gene(uncertain_df_clean, f"Top {n_top} Mutated Genes (Uncertain - HotspotPEM)", with_legend=False)

def extract_pos_based_df(df):
    df[['Protein_ID', 'Position']] = df['AccessionPosition'].str.split('|', expand=True)
    df = df.drop(columns=['AccessionPosition'])
    df['Position'] = df['Position'].astype(int)
    return df

def enrichment_in_disordered_functional_regions(
    mutation_df,
    disorder_df,
    functional_df,
    main_proteins_df,
    mutation_label="Benign",
    color="#f4a6a6",
    plot_path=None,
    verbose=True,
    ylabel="Benign Mutation Frequency",
    title="Benign Mutation Rate per Functional Category"
):

    # Step 1: Filter disorder to main isoforms and disordered positions
    disorder_df = disorder_df[
        (disorder_df["Protein_ID"].isin(main_proteins_df["Protein_ID"])) &
        (disorder_df["CombinedDisorder"] == 1.0)
    ][["Protein_ID", "Position"]].drop_duplicates()

    # Step 2: Expand functional regions to per-position format
    expanded_func = []
    for _, row in functional_df.iterrows():
        for pos in range(row["Start"], row["End"] + 1):
            expanded_func.append((row["Protein_ID"], pos, row["Data"]))
    expanded_func_df = pd.DataFrame(expanded_func, columns=["Protein_ID", "Position", "Category"])

    # Step 3: Keep only functional annotations within disordered regions
    expanded_func_df = expanded_func_df.merge(disorder_df, on=["Protein_ID", "Position"], how="inner")

    # Step 4: Keep only proteins that have at least one functional annotation in disorder
    proteins_with_function = expanded_func_df["Protein_ID"].unique()
    disorder_df = disorder_df[disorder_df["Protein_ID"].isin(proteins_with_function)]
    mutation_df = mutation_df[mutation_df["Protein_ID"].isin(proteins_with_function)]

    # Step 5: Filter mutations to disordered positions of relevant proteins
    mutation_df = mutation_df.merge(disorder_df, on=["Protein_ID", "Position"], how="inner")

    # Step 6: Merge functional categories into disorder data
    disorder_df = disorder_df.merge(expanded_func_df, on=["Protein_ID", "Position"], how="left")
    disorder_df["Category"] = disorder_df["Category"].fillna("No Annotation")

    # Step 7: Mark mutation presence
    mutation_df = mutation_df[["Protein_ID", "Position"]].drop_duplicates()
    mutation_df["has_mut"] = True

    # Step 8: Merge mutation info into the disordered+functional table
    merged = disorder_df.merge(mutation_df, on=["Protein_ID", "Position"], how="left")
    merged["has_mut"] = merged["has_mut"].fillna(False)

    # Step 9: Remove duplicates (e.g., position annotated to multiple categories)
    merged = merged.drop_duplicates(subset=["Protein_ID", "Position", "Category"])

    # Step 10: Group by functional category and compute mutation frequency
    category_stats = []
    for category, group in merged.groupby("Category"):
        total = group[["Protein_ID", "Position"]].drop_duplicates().shape[0]
        mutated = group[group["has_mut"]][["Protein_ID", "Position"]].drop_duplicates().shape[0]
        rate = mutated / total if total > 0 else 0
        category_stats.append((category, rate, mutated, total))

    # Step 11: Sort categories by mutation rate (descending)
    category_stats = sorted(category_stats, key=lambda x: -x[1])

    # Step 12: Plot results
    labels, rates, mutated_counts, total_counts = zip(*category_stats)
    plt.figure(figsize=(5, 3))
    bars = plt.bar(labels, rates, color=color)

    for bar, rate in zip(bars, rates):
        plt.text(bar.get_x() + bar.get_width() / 2, rate * 1.05, f"{rate * 100:.2f}%",
                 ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, max(rates) * 1.2)
    plt.tight_layout()

    if plot_path:
        plt.savefig(plot_path, dpi=300)
    plt.show()

    # Step 13: Print category mutation statistics
    if verbose:
        print(f"\n--- Mutation Rates for {mutation_label} (IDRs + functional proteins only) ---")
        for cat, rate, mut, total in category_stats:
            print(f"{cat:<20s}: {mut}/{total} → {rate:.4f}")

    # Step 14: Return the stats
    return {
        "label": mutation_label,
        "stats": [
            {"Category": cat, "Mutation rate": rate, "Mutated": mut, "Total": total}
            for cat, rate, mut, total in category_stats
        ]
    }




def generate_enriched_motifs(enricment_score = 3):
    main_df = pd.read_csv(
        f"{base_dir}/data/discanvis_base_files/sequences/loc_chrom_with_names_main_isoforms_with_seq.tsv", sep="\t")

    # disorder_df = extract_pos_based_df(pd.read_csv(f"{base_dir}/data/discanvis_base_files/positional_data_process/combined_dis_pos.tsv", sep="\t"))
    disorder_df = pd.read_csv(f"{base_dir}/data/discanvis_base_files/positional_data_process/CombinedDisorderNew_Pos.tsv", sep="\t")

    # disorder_df = disorder_df[
    #     (disorder_df["Protein_ID"].isin(main_df["Protein_ID"])) &
    #     (disorder_df["CombinedDisorder"] == 1)
    #     ][["Protein_ID", "Position"]].drop_duplicates()


    uncertain_df = pd.read_csv(
        f"{base_dir}/processed_data/files/clinvar/Uncertain/positional_clinvar_functional_categorized_final.tsv",
        sep="\t")

    pathogenic_df = pd.read_csv(
        f"{base_dir}/processed_data/files/clinvar/Pathogenic/positional_clinvar_functional_categorized_final.tsv",
        sep="\t")

    # pip
    pip_df = pd.read_csv("/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/pip/pip_regions_with_overlapping_motifs.tsv",sep='\t')
    print(pip_df)

    pip_enrhiced_pathogenic = generate_uncertain_mutation_density_table(pip_df, pathogenic_df,
                                                                                     disorder_df,
                                                                                     main_df, window_size=20, region_type=True).sort_values(by="Enrichment_Protein", ascending=False)

    enriched_pems = pip_enrhiced_pathogenic[
        (pip_enrhiced_pathogenic["Enrichment_Protein"] > enricment_score) |
        ((pip_enrhiced_pathogenic["Motif_Mut_Count"] > 0) &
         (pip_enrhiced_pathogenic["Protein_Mut_Count"] == 0))
        ]
    enriched_pems = enriched_pems[enriched_pems["Motif_Mut_Count"] > 0]

    enriched_pems.to_csv(
        f"{base_dir}/processed_data/files/pip/enriched_pip_pathogenic_mutations.tsv", sep="\t",
        index=False)

    pip_enrhiced_uncertain = generate_uncertain_mutation_density_table(pip_df, uncertain_df,
                                                                        disorder_df,
                                                                        main_df, window_size=20, region_type=True).sort_values(by="Enrichment_Protein", ascending=False)

    enriched_pems = pip_enrhiced_uncertain[
        (pip_enrhiced_uncertain["Enrichment_Protein"] > enricment_score) |
        ((pip_enrhiced_uncertain["Motif_Mut_Count"] > 0) &
         (pip_enrhiced_uncertain["Protein_Mut_Count"] == 0))
        ]
    enriched_pems = enriched_pems[enriched_pems["Motif_Mut_Count"] > 0]

    enriched_pems.to_csv(
        f"{base_dir}/processed_data/files/pip/enriched_pip_uncertain_mutations.tsv", sep="\t",
        index=False)

    exit()

    # Calculate mutation density for Uncertain
    pem_df = pd.read_csv(f"{base_dir}/processed_data/files/elm/class_based_approach/elm_pem_disorder_corrected_with_class_count.tsv")
    # pem_df = pd.read_csv(f"{base_dir}/processed_data/files/elm/elm_predicted_disorder_low_predicted.tsv",sep='\t')

    # polk = pem_df[pem_df['Protein_ID'] == "POLK-201"]
    # print(polk)
    # print(polk[['ELMIdentifier', 'Start', 'End', 'known',"prediction_decision_tree"]])
    # print(polk.columns)
    #
    # exit()

    # print(pem_df)
    # exit()

    high_amount = pem_df[(pem_df["N_Motif_Predicted"] == "High_Amount") & (pem_df["known"] == False)]
    # low_or_known_pems = pem_df
    pems_high_amount_enrhiced_pathogenic = generate_uncertain_mutation_density_table(high_amount, pathogenic_df,
                                                                                 disorder_df,
                                                                                 main_df, window_size=20).sort_values(by="Enrichment_Protein", ascending=False)

    enriched_pems = pems_high_amount_enrhiced_pathogenic[
        (pems_high_amount_enrhiced_pathogenic["Enrichment_Protein"] > enricment_score) |
        ((pems_high_amount_enrhiced_pathogenic["Motif_Mut_Count"] > 0) &
         (pems_high_amount_enrhiced_pathogenic["Protein_Mut_Count"] == 0))
        ]
    enriched_pems = enriched_pems[enriched_pems["Motif_Mut_Count"] > 0]

    enriched_pems.to_csv(
        f"{base_dir}/processed_data/files/elm/clinvar/enriched/enriched_pems_high_amount_with_pathogenic_mutations.tsv", sep="\t",
        index=False)

    exit()


    low_or_known_pems = pem_df[(pem_df["N_Motif_Predicted"] == "Low_Amount") | (pem_df["known"] == True)]
    # low_or_known_pems = pem_df
    pems_with_enriched_uncertain_mut = generate_uncertain_mutation_density_table(low_or_known_pems, uncertain_df,
                                                                                 disorder_df, main_df,
                                                                                 window_size=20).sort_values(by="Enrichment_Protein", ascending=False)



    print(pems_with_enriched_uncertain_mut)
    enriched_pems = pems_with_enriched_uncertain_mut[
        (pems_with_enriched_uncertain_mut["Enrichment_Protein"] > enricment_score) |
        ((pems_with_enriched_uncertain_mut["Motif_Mut_Count"] > 0) &
         (pems_with_enriched_uncertain_mut["Protein_Mut_Count"] == 0))
        ]
    enriched_pems = enriched_pems[enriched_pems["Motif_Mut_Count"] > 0]
    print(enriched_pems)
    enriched_pems.to_csv(
        f"{base_dir}/processed_data/files/elm/clinvar/enriched/enriched_pems_with_uncertain_mutations.tsv", sep="\t",
        index=False)

    # plot_enriched_motifs(enriched_pems, plot_dir)

    # Calculate mutation density for Pathogen
    pems_with_enriched_pathogenic = generate_uncertain_mutation_density_table(low_or_known_pems, pathogenic_df,
                                                                              disorder_df,
                                                                              main_df, window_size=20).sort_values(by="Enrichment_Protein", ascending=False)
    print(pems_with_enriched_pathogenic)
    enriched_pems = pems_with_enriched_pathogenic[
        (pems_with_enriched_pathogenic["Enrichment_Protein"] > enricment_score) |
        ((pems_with_enriched_pathogenic["Motif_Mut_Count"] > 0) &
         (pems_with_enriched_pathogenic["Protein_Mut_Count"] == 0))
        ]
    enriched_pems = enriched_pems[enriched_pems["Motif_Mut_Count"] > 0]
    print(enriched_pems)
    enriched_pems.to_csv(
        f"{base_dir}/processed_data/files/elm/clinvar/enriched/enriched_pems_with_pathogenic_mutations.tsv", sep="\t",
        index=False)

    plot_enrichment()
    exit()



if __name__ == "__main__":
    base_dir = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations"
    plot_dir = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/plots/sm/clinvar"

    COLORS = {
        "disorder": '#ffadad',
        "Disorder": '#ffadad',
        "order": '#a0c4ff',
        "Order": '#a0c4ff',
        "both": '#ffc6ff',
        "Pathogenic": '#ff686b',
        "Benign": "#f4a6a6",
        "Uncertain": "#f8edeb"
    }



    # generate_enriched_motifs()
    # exit()

    plot_enrichment()
    exit()

    # Input files
    main_df = pd.read_csv(f"{base_dir}/data/discanvis_base_files/sequences/loc_chrom_with_names_main_isoforms_with_seq.tsv", sep="\t")
    disorder_df = pd.read_csv(f"{base_dir}/data/discanvis_base_files/positional_data_process/CombinedDisorderNew_Pos.tsv", sep="\t")

    disorder_df = disorder_df[
        (disorder_df["Protein_ID"].isin(main_df["Protein_ID"])) &
        (disorder_df["CombinedDisorder"] == 1.0)
    ][["Protein_ID", "Position"]].drop_duplicates()

    pem_df = pd.read_csv(f"{base_dir}/processed_data/files/elm/class_based_approach/elm_pem_disorder_corrected_with_class_count.tsv")
    all_pem_df = pd.read_csv(f"{base_dir}/processed_data/files/elm/decision_tree/elm_predicted_with_am_info_all_disorder_class_filtered.tsv",sep="\t")
    # print(all_pem_df.shape[0])
    # exit()

    # Ylva Ivarsson total predicted
    # ylva_ivarsson_df = pd.read_csv(f"{base_dir}/processed_data/files/elm/decision_tree/publication_predicted_with_am_info_all_disorder_class_filtered.tsv", sep="\t")
    # ylva_ivarsson_df = ylva_ivarsson_df[ylva_ivarsson_df['prediction_decision_tree'] == False]
    # Norman Davey total predicted
    # norman_davey_df = pd.read_csv(f"{base_dir}/processed_data/files/elm/decision_tree/publication_predicted_effected_norman_davey_with_am_info_all_disorder_class_filtered.tsv", sep="\t")
    # norman_davey_df = norman_davey_df[norman_davey_df['prediction_decision_tree'] == False]

    # pem_df = pem_df[pem_df["N_Motif_Predicted"] == "Low_Amount"]

    # Mutations
    uncertain_df = pd.read_csv(f"{base_dir}/processed_data/files/clinvar/Uncertain/positional_clinvar_functional_categorized_final.tsv", sep="\t")
    pathogenic_df = pd.read_csv(f"{base_dir}/processed_data/files/clinvar/Pathogenic/positional_clinvar_functional_categorized_final.tsv", sep="\t")
    # predicted_pathogenic_df = pd.read_csv(f"{base_dir}/processed_data/files/alphamissense/clinvar/likely_pathogenic_disorder_pos_based.tsv", sep="\t")

    benign_disorder_df = pd.read_csv(f"{base_dir}/processed_data/files/clinvar/Benign/positional_clinvar_functional_categorized_final.tsv", sep="\t")

    functional_df = pd.read_csv(f"{base_dir}/processed_data/files/pip/functional_region_with_score.tsv",sep='\t')
    functional_df = functional_df[functional_df["Data"] != "Pfam"]

    benign_mutation_functional_rate_path = f"{plot_dir}/benign_mutation_rate_functional.png"
    # result = enrichment_in_disordered_functional_regions(
    #     mutation_df=benign_disorder_df,
    #     disorder_df=disorder_df,
    #     functional_df=functional_df,
    #     main_proteins_df=main_df,
    #     mutation_label="Benign",
    #     plot_path=benign_mutation_functional_rate_path,
    #     color=COLORS["Benign"]
    # )
    # exit()

    benign_mutation_rate_path = f"{plot_dir}/benign_mutation_rate.png"

    # result = analyze_benign_constraint_in_pems(
    #     disorder_df=disorder_df,
    #     benign_df=benign_disorder_df,
    #     pem_df=pem_df,
    #     main_proteins_df=main_df,
    #     filepath=benign_mutation_rate_path,
    # )

    result = analyze_benign_constraint_multipanel_plot(
        disorder_df=disorder_df,
        benign_df=benign_disorder_df,
        pem_df=pem_df,
        all_pem_df=all_pem_df,
        main_proteins_df=main_df,
        filepath=benign_mutation_rate_path,
    )

    exit()

    uncertain_disorder_df = pd.read_csv(f"{base_dir}/processed_data/files/clinvar/Uncertain/disorder/positional_clinvar_functional_categorized_final.tsv", sep="\t")
    # Goal 1: PEM vs Random
    # compare_uncertain_mutation_enrichment_vs_random(pem_df, disorder_df, uncertain_disorder_df, main_df)
    # exit()

    # Goal 2: Mutation Uniformity
    # uniformity_df = plot_gene_mutation_uniformity(uncertain_disorder_df, disorder_df, main_df)
    #
    # print(uniformity_df)

    # exit()
    # motif_set= [
    #     ["PEM",pem_df],
    #     ["SlimPrint",norman_davey_df],
    # ]
    #
    # for motif, df in motif_set:
    #
    #     # Run tests
    #     result_uncertain = run_enrichment_test(disorder_df, uncertain_df, df, main_df, mutation_type="Uncertain")
    #     result_pathogenic = run_enrichment_test(disorder_df, pathogenic_df, df, main_df, mutation_type="Pathogenic")
    #     result_predicted_pathogenic = run_enrichment_test(disorder_df, predicted_pathogenic_df, df, main_df, mutation_type="Predicted Pathogenic")
    #
    #
    #     print("Uncertain",result_uncertain)
    #     print("Pathogenic",result_pathogenic)
    #     print("Predicted Pathogenic",result_predicted_pathogenic)
    #
    #     # Plot
    #     plot_enrichment_test_results(result_uncertain, result_pathogenic, result_predicted_pathogenic, plot_dir,motif)
    #     # exit()

    # zscore_result = calculate_expected_vs_observed_zscore(disorder_df, uncertain_df, pem_df, main_df)
    #
    # plot_observed_vs_expected(
    #     observed=zscore_result["Observed uncertain mutations in PEM"],
    #     expected=zscore_result["Expected uncertain mutations in PEM"],
    #     z_score=zscore_result["Z-score"],
    #     plot_dir=plot_dir
    # )





