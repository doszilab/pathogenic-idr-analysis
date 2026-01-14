import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

_COMPLEX_COLOR = {
    "monomer":   "#8da0cb",
    "homomer":   "#66c2a5",
    "heteromer": "#fc8d62",
}
_COMPLEX_ORDER = ["monomer", "homomer", "heteromer"]  # megjelenítési sorrend

def _normalize_pdb4(tok: str) -> str:
    s = re.sub(r'[^A-Za-z0-9]', '', str(tok)).strip()
    return s[:4].upper() if len(s) >= 4 else ""

def _row_pdb_iter(info: str, info_cols: str):
    """Visszaadja az adott sorból a PDB tokenek (akár láncbetűvel) iterátorát."""
    parts = [x.strip() for x in str(info).split(";")]
    if "PDB" not in parts:
        return []
    idx = parts.index("PDB")
    cols = str(info_cols).split(";")
    if idx >= len(cols):
        return []
    # vessző+whitespace szeletelés
    return re.split(r'[,\s]+', cols[idx].strip())

def _norm_type(x):
    s = "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)
    s = s.strip().lower()
    return s if s in {"monomer","homomer","heteromer"} else ""

def collect_unique_pdb_types(df: pd.DataFrame, pdb_dict: dict[str, list[str]]) -> list[str]:
    unique_pdb4 = set()
    for _, row in df.iterrows():
        for tok in _row_pdb_iter(row["info"], row["info_cols"]):
            pdb4 = _normalize_pdb4(tok)
            if pdb4:
                unique_pdb4.add(pdb4)

    types = []
    for pdb4 in unique_pdb4:
        vals = pdb_dict.get(pdb4)
        if not vals:
            continue
        ctype = _norm_type(vals[1])
        if ctype:
            types.append(ctype)
    return types



def summarize_pdb_props(pdb_list_str: str, pdb_dict: dict[str, list[str]]) -> tuple[str, str]:
    """
    Visszaad két oszlop-értéket:
      - egyedi protein_oligomeric_state értékek ", " szeparátorral
      - egyedi protein_complex_type értékek ", " szeparátorral
    """
    if not isinstance(pdb_list_str, str) or not pdb_list_str.strip():
        return "", ""

    states, types = set(), set()
    # tokenizálás vessző + whitespace alapján
    for tok in re.split(r'[,\s]+', pdb_list_str):
        pdb4 = _normalize_pdb4(tok)
        if not pdb4:
            continue
        vals = pdb_dict.get(pdb4)
        if not vals:
            continue
        state, ctype = vals[0], vals[1]
        if isinstance(state, str) and state.strip():
            states.add(state.strip())
        if isinstance(ctype, str) and ctype.strip():
            types.add(ctype.strip())

    states_str = ", ".join(sorted(states)) if states else ""
    types_str  = ", ".join(sorted(types))  if types else ""
    return states_str, types_str



def _split_complex_types_cell(s: str) -> list[str]:
    """Vesszővel elválasztott cellából kinyeri a normalizált komplex-típusokat."""
    valid = set(_COMPLETEX_ORDER) if (_COMPLETEX_ORDER := _COMPLEX_ORDER) else set()
    if not isinstance(s, str) or not s.strip():
        return []
    toks = re.split(r"[,\s]+", s.strip())
    out = []
    for t in toks:
        t2 = t.strip().lower()
        if t2 in valid:
            out.append(t2)
    return out

def _counts_from_df(df: pd.DataFrame, col: str) -> pd.Series:
    """A megadott oszlop többértékű celláiból összesít (monomer / homomer / heteromer)."""
    all_vals = []
    for v in df.get(col, pd.Series([], dtype=object)).fillna(""):
        all_vals.extend(_split_complex_types_cell(v))
    counts = pd.Series(all_vals).value_counts().reindex(_COMPLEX_ORDER).fillna(0).astype(int)
    return counts

def plot_complex_type_two_pies(df_disordered: pd.DataFrame,
                               df_ordered: pd.DataFrame,
                               col: str = "pdb_protein_complex_types",
                               titles=("Disordered mutations", "Ordered mutations"),
                               out_png: str = "pdb_complex_type_distribution_dis_vs_ord.png"):
    counts_dis = _counts_from_df(df_disordered, col)
    counts_ord = _counts_from_df(df_ordered,   col)

    # ha egyikben sincs adat, ne rajzoljunk
    if counts_dis.sum() == 0 and counts_ord.sum() == 0:
        print("[WARN] Nincs komplex-típus adat egyik dataframe-ben sem – nem készül plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    for ax, counts, title in zip(axes, (counts_dis, counts_ord), titles):
        total = counts.sum()
        if total == 0:
            # üres panel jelzés
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)
            ax.axis("off")
            continue

        def _autopct(p):
            return f"{p:.1f}%" if p > 0 else ""

        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=None,
            startangle=90,
            autopct=_autopct,
            counterclock=False,
            colors=[_COMPLEX_COLOR[k] for k in _COMPLEX_ORDER]
        )
        ax.set_title(title)
        ax.axis("equal")

    # közös legenda
    handles = [plt.Line2D([0],[0], marker='o', linestyle='', color=_COMPLEX_COLOR[k], label=k)
               for k in _COMPLEX_ORDER]
    fig.legend(handles=handles, labels=[k.capitalize() for k in _COMPLEX_ORDER],
               loc="lower center", ncol=3, frameon=True)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.suptitle("Pathogenic Mutations Protein Complex-type Composition (PDB)")
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(out_png, dpi=170)
    plt.close()
    print(f"[OK] Saved: {out_png}")

def _counts_from_types(types: list[str]) -> pd.Series:
    order = ["monomer","homomer","heteromer"]
    s = pd.Series(types, dtype=str)
    return s.value_counts().reindex(order).fillna(0).astype(int)

def plot_complex_type_two_pies_unique_pdb(types_dis: list[str],
                                          types_ord: list[str],
                                          titles=("Disordered mutations", "Ordered mutations"),
                                          out_png: str = "pdb_complex_type_distribution_uniquePDB.png"):
    counts_dis = _counts_from_types(types_dis)
    counts_ord = _counts_from_types(types_ord)

    if counts_dis.sum() == 0 and counts_ord.sum() == 0:
        print("[WARN] Nincs komplex-típus a unique PDB-kre – nem készül plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    for ax, counts, title in zip(axes, (counts_dis, counts_ord), titles):
        tot = counts.sum()
        if tot == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title); ax.axis("off"); continue
        def _autopct(p): return f"{p:.1f}%" if p > 0 else ""
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=None,
            startangle=90,
            autopct=_autopct,
            counterclock=False,
            colors=[_COMPLEX_COLOR[k] for k in _COMPLEX_ORDER]
        )
        ax.set_title(title); ax.axis("equal")

    handles = [plt.Line2D([0],[0], marker='o', linestyle='', color=_COMPLEX_COLOR[k], label=k)
               for k in _COMPLEX_ORDER]
    fig.legend(handles=handles, labels=[k.capitalize() for k in _COMPLEX_ORDER],
               loc="lower center", ncol=3, frameon=True)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.suptitle("Pathogenic mutations — PDB complex types (unique PDBs)")
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    plt.savefig(out_png, dpi=170)
    plt.close()
    print(f"[OK] Saved: {out_png}")


def preprocess_clinvar(df,pdb_dict):
    print(df)
    pdb_clinvar = df[df["info"].str.contains("PDB")]
    print(pdb_clinvar)

    pdb_clinvar["pdb_list"] = pdb_clinvar.apply(lambda x: get_pdb_lst(x), axis=1)
    print(pdb_clinvar)

    pdb_clinvar[["pdb_protein_oligomeric_states", "pdb_protein_complex_types"]] = (
        pdb_clinvar["pdb_list"].apply(lambda s: pd.Series(summarize_pdb_props(s, pdb_dict)))
    )
    return pdb_clinvar

def subset_with_pdb(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["info"].astype(str).str.contains(r"\bPDB\b", na=False)
    return df.loc[mask].copy()

if __name__ == "__main__":

    clinvar_disorder_pos = pd.read_csv("/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/clinvar/Pathogenic/disorder/positional_clinvar_functional_categorized_final.tsv",sep="\t")
    clinvar_order_pos = pd.read_csv("/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/clinvar/Pathogenic/order/positional_clinvar_functional_categorized_final.tsv",sep="\t")
    pdb_oligomer = pd.read_csv("/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/pdb/protein_pdb_oligomer_join.tsv",sep="\t")
    out_png = "/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/plots/pdb/pdb_complex_type_distribution.png"

    pdb_dict = {
        row["PDB_ID"]: [row["protein_oligomeric_state"], row["protein_complex_type"], row["protein_n_chains"]]
        for _, row in tqdm(pdb_oligomer.iterrows(),total=len(pdb_oligomer))
    }

    pdb_clinvar_disorder = subset_with_pdb(clinvar_disorder_pos)
    pdb_clinvar_order = subset_with_pdb(clinvar_order_pos)


    pdb_clinvar_disorder.to_csv("/dlab/home/norbi/paper_projects/IDP_Pathogenic_Mutations/processed_data/files/pdb/pdb_disorder_clinvar_pathogenic.tsv",sep="\t",index=False)

    types_dis = collect_unique_pdb_types(pdb_clinvar_disorder, pdb_dict)
    types_ord = collect_unique_pdb_types(pdb_clinvar_order, pdb_dict)

    # plot
    plot_complex_type_two_pies_unique_pdb(
        types_dis=types_dis,
        types_ord=types_ord,
        titles=("Disordered mutations", "Ordered mutations"),
        out_png=out_png
    )