import os
import re
import pandas as pd

PDB_ID_COL = "pdb_id"  # az oligomer fájlban ez a PDB azonosító oszlop neve

def _normalize_pdb_token(tok: str) -> tuple[str, str]:
    """
    '2BYDA' -> ('2byd', 'A')
    '2EOG'  -> ('2eog', '')
    Bemenet lehet szóközzel, vesszővel szennyezett is.
    """
    if not isinstance(tok, str):
        tok = str(tok)
    s = re.sub(r"[^A-Za-z0-9]", "", tok).strip()
    if len(s) < 4:
        return "", ""
    pdb4 = s[:4].lower()
    chain = s[4:]  # maradék (opcionális)
    return pdb4, chain

if __name__ == "__main__":
    pdb_base_path = r"/dlab/home/norbi/PycharmProjects/DisCanVis_Data_Process/Processed_Data/gencode_process/annotations/structures"
    to_dir = r"/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/pdb"
    pdb_oligomer_summary = os.path.join(pdb_base_path, "pdb_oligomer_summary.tsv")
    pdb_my_proteins_lst = os.path.join(pdb_base_path, "pdb_ids_disorder.tsv")
    out_fp = os.path.join(to_dir, "protein_pdb_oligomer_join.tsv")

    # --- load
    df_prot = pd.read_csv(pdb_my_proteins_lst, sep="\t")        # elvár: Protein_ID, pdb_ids (vesszővel elválasztott)
    df_olig = pd.read_csv(pdb_oligomer_summary, sep="\t")        # elvár: pdb_id (+ protein_* oszlopok)

    # --- oligomer PDB id normalizálás (lower)
    df_olig[PDB_ID_COL] = df_olig[PDB_ID_COL].astype(str).str.strip().str.lower()

    # --- proteins: explode PDB list
    # a kód feltételezi, hogy az oszlop neve 'pdb_ids' – ha más, itt cseréld
    pdb_list_col = "pdb_ids"
    if pdb_list_col not in df_prot.columns:
        raise ValueError(f"Hiányzik a '{pdb_list_col}' oszlop a {pdb_my_proteins_lst} fájlban.")

    tmp = (
        df_prot[["Protein_ID", pdb_list_col]]
        .dropna(subset=[pdb_list_col])
        .assign(_pdb_list=lambda d: d[pdb_list_col].astype(str).str.split(","))
        .explode("_pdb_list")
        .assign(_pdb_list=lambda d: d["_pdb_list"].str.strip())
        .loc[lambda d: d["_pdb_list"] != ""]
        .copy()
    )

    # --- token -> (pdb4, chain) + normalizált megjelenítés
    norm = tmp["_pdb_list"].apply(_normalize_pdb_token)
    tmp["pdb_id_norm"] = norm.apply(lambda x: x[0])
    tmp["chain_token"] = norm.apply(lambda x: x[1])
    # emberbarát PDB_ID (4-karakter nagybetűs)
    tmp["PDB_ID"] = tmp["pdb_id_norm"].str.upper()

    # --- szűrés: csak ahol sikerült a 4-karaktert kinyerni
    tmp = tmp.loc[tmp["pdb_id_norm"].astype(bool)].copy()

    # --- join az oligomer infóra
    keep_cols = [
        "protein_oligomeric_state",
        "protein_complex_type",
        "protein_n_chains",
        "protein_n_unique_uniprot",
    ]
    missing = [c for c in keep_cols if c not in df_olig.columns]
    if missing:
        raise ValueError(f"Az oligomer fájlból hiányzik(anak) az oszlop(ok): {missing}")

    merged = (
        tmp.merge(
            df_olig[[PDB_ID_COL] + keep_cols],
            left_on="pdb_id_norm",
            right_on=PDB_ID_COL,
            how="left",
        )
        .drop(columns=[pdb_list_col, "_pdb_list", "pdb_id_norm", PDB_ID_COL, "chain_token"])
        .drop_duplicates(subset=["Protein_ID", "PDB_ID", "protein_oligomeric_state", "protein_complex_type"])
        .reset_index(drop=True)
    )

    # --- oszlopok sorrendje és mentés
    merged = merged[[
        "Protein_ID",
        "PDB_ID",
        "protein_oligomeric_state",
        "protein_complex_type",
        "protein_n_chains",
        "protein_n_unique_uniprot",
    ]]

    merged.to_csv(out_fp, sep="\t", index=False)
    print(f"[OK] Write: {out_fp}  (n={len(merged)})")

    # opcionális: gyors diagnosztika
    n_all_pairs = tmp[["Protein_ID", "PDB_ID"]].drop_duplicates().shape[0]
    n_joined = merged.shape[0]
    n_unmatched = n_all_pairs - n_joined
    if n_unmatched > 0:
        print(f"[INFO] {n_unmatched} (Protein_ID, PDB_ID) párhoz nem találtam oligomer rekordot (valszeg nincs benne a summary TSV-ben).")
