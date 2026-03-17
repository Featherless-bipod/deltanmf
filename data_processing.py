from pathlib import Path
import numpy as np
import sys
import argparse
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deltanmf.api import run_twostage_deltanmf
from deltanmf.io import h5ad_to_npy
from deltanmf.ddp_utils import setup_ddp

def _normalize_ensgid(value):
    s = str(value).strip()
    if not s:
        return None
    return s.split(".")[0]


def _map_to_ensembl_and_collapse_duplicates(X_ntc, X_spec, gene_names, gene_matrix_path):
    df = pd.read_csv(gene_matrix_path, sep="\t")
    required = {"ensgid", "gene_name", "gene_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in geneMatrix.tsv: {sorted(missing)}")

    symbol_to_ensg = {}
    geneid_to_ensg = {}
    ensg_set = set()

    for _, row in df.iterrows():
        ensg = _normalize_ensgid(row["ensgid"])
        if ensg is None or not ensg.startswith("ENSG"):
            continue
        ensg_set.add(ensg)

        gene_name = str(row["gene_name"]).strip()
        if gene_name and gene_name not in symbol_to_ensg:
            symbol_to_ensg[gene_name] = ensg

        gene_id = str(row["gene_id"]).strip()
        if gene_id:
            geneid_to_ensg.setdefault(gene_id, ensg)
            geneid_to_ensg.setdefault(_normalize_ensgid(gene_id), ensg)

    mapped = []
    for g in np.asarray(gene_names, dtype=object):
        gs = str(g).strip()
        g_base = _normalize_ensgid(gs)
        ensg = None
        if gs in ensg_set:
            ensg = gs
        elif g_base in ensg_set:
            ensg = g_base
        elif gs in symbol_to_ensg:
            ensg = symbol_to_ensg[gs]
        elif g_base in symbol_to_ensg:
            ensg = symbol_to_ensg[g_base]
        elif gs in geneid_to_ensg:
            ensg = geneid_to_ensg[gs]
        elif g_base in geneid_to_ensg:
            ensg = geneid_to_ensg[g_base]
        mapped.append(ensg)

    mapped = np.asarray(mapped, dtype=object)
    keep = mapped != None  # noqa: E711
    if keep.sum() == 0:
        raise ValueError("No genes could be mapped to Ensembl IDs using geneMatrix.tsv")

    X_ntc_keep = X_ntc[keep, :]
    X_spec_keep = X_spec[keep, :]
    mapped_keep = mapped[keep]

    codes, uniques = pd.factorize(mapped_keep, sort=False)
    n_unique = len(uniques)

    X_ntc_collapsed = np.zeros((n_unique, X_ntc_keep.shape[1]), dtype=np.float32)
    X_spec_collapsed = np.zeros((n_unique, X_spec_keep.shape[1]), dtype=np.float32)
    np.add.at(X_ntc_collapsed, codes, X_ntc_keep.astype(np.float32, copy=False))
    np.add.at(X_spec_collapsed, codes, X_spec_keep.astype(np.float32, copy=False))

    print(
        f"Gene mapping: input={len(gene_names)}, mapped={keep.sum()}, unique_ensembl={n_unique}",
        flush=True,
    )
    return X_ntc_collapsed, X_spec_collapsed, np.asarray(uniques, dtype=object)


def main():
    local_rank, world_size, is_ddp = setup_ddp()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-rel-gamma", type=float, default=0.0)
    args = parser.parse_args()
    gamma = float(args.stage2_rel_gamma)

    h5ad_path = "/hpc/group/gersbachlab/agk21/Schizophrenia_PROVEIT_Perturbseq/output/adata_gex_with_qc_fixeddoubletremoved_HQ_geneexpressionfeatures_filtergenes_deltanmfupdate.h5ad"
    gamma_tag = f"{gamma:.6g}".replace(".", "p")
    out_root = Path("/hpc/group/gersbachlab/agk21/Schizophrenia_PROVEIT_Perturbseq/deltanmf")
    out = out_root / f"noFM_stage2_rel_gamma_{gamma_tag}"
    
    if local_rank == 0:
        out.mkdir(parents=True, exist_ok=True)

    # Implement Memory-Efficient Data Loading for DDP
    # Rank 0 loads h5ad, collapses it, and saves it to a persistent temp dir.
    # Other ranks wait, and then everyone mmaps the numpy arrays to share RAM.
    tmp_data_dir = Path("/work/zy231/deltanmf/tmp_data")
    
    if local_rank == 0:
        tmp_data_dir.mkdir(parents=True, exist_ok=True)
        print("Rank 0 parsing data...")
        X_ntc, X_spec, gene_names = h5ad_to_npy(
            h5ad_path,
            ntc_key="negative_control",
            condition_key="guide_type",
            case_key=None,
            layer=None,
        )
        gene_matrix_path = Path(
            "/hpc/group/gersbachlab/agk21/Schizophrenia_PROVEIT_Perturbseq/clustering/geneMatrix.tsv"
        )
        if not gene_matrix_path.exists():
            raise FileNotFoundError(f"Missing: {gene_matrix_path}")
        X_ntc, X_spec, gene_names = _map_to_ensembl_and_collapse_duplicates(
            X_ntc, X_spec, gene_names, gene_matrix_path
        )
        np.save(tmp_data_dir / "X_ntc.npy", X_ntc)
        np.save(tmp_data_dir / "X_spec.npy", X_spec)
        np.save(tmp_data_dir / "gene_names.npy", gene_names)
        print("Rank 0 saved cached data to disk.")

    if is_ddp:
        import torch.distributed as dist
        dist.barrier()
        
    print(f"Rank {local_rank} loading mmaps...")
    X_ntc = np.load(tmp_data_dir / "X_ntc.npy", mmap_mode="r")
    X_spec = np.load(tmp_data_dir / "X_spec.npy", mmap_mode="r")
    gene_names = np.load(tmp_data_dir / "gene_names.npy", allow_pickle=True)

    S_E_PATH = Path("/hpc/group/singhlab/user/agk21/projects/NMF/src/scGPT/scgpt_similarity_human_updated20251027_S_E_relu.npy")
    S_E_GENES_PATH = Path("/hpc/group/singhlab/user/agk21/projects/NMF/src/scGPT/scgpt_similarity_human_updated20251027_genes_order.json")

    if not S_E_PATH.exists():
        raise FileNotFoundError(f"Missing: {S_E_PATH}")
    if not S_E_GENES_PATH.exists():
        raise FileNotFoundError(f"Missing: {S_E_GENES_PATH}")

    res = run_twostage_deltanmf(
        X_ntc, X_spec, gene_names, S_E_PATH, S_E_GENES_PATH,
        K_stage1=30, K_stage2=60,
        MIN_CELLS=10,
        stage1_rel_alpha=0.0,
        stage2_rel_alpha=0.0,
        stage2_rel_gamma=gamma,
        stage1_max_iter=10000,
        stage2_max_iter=10000,
        FM_NONNEG="softplus",
        FM_SOFTPLUS_BETA=5.0,
        lr=0.01,
        stage1_warmup_n_runs=20,
        stage2_warmup_n_runs=20,
        stage2_use_hybrid_memory=True,
        stage1_use_minibatch_ntc = False,
        stage1_batchsize=40000,
    )

    if local_rank == 0:
        np.save(out / "W_stage1.npy", res["W_stage1"])
        np.save(out / "H_stage1.npy", res["H_stage1"])
        np.save(out / "W_stage2.npy", res["W_stage2"])
        np.save(out / "H_stage2.npy", res["H_stage2"])
        np.save(out / "gene_names_aligned.npy", res["gene_names_aligned"])
        np.savetxt(out / "gene_names_aligned.tsv", res["gene_names_aligned"], fmt="%s")
        np.save(out / "ntc_cell_ids.npy", res["ntc_cell_ids"])
        np.savetxt(out / "ntc_cell_ids.tsv", res["ntc_cell_ids"], fmt="%s")
        np.save(out / "specific_cell_ids.npy", res["specific_cell_ids"])
        np.savetxt(out / "specific_cell_ids.tsv", res["specific_cell_ids"], fmt="%s")


if __name__ == "__main__":
    main()