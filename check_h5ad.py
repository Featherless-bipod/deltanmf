import anndata as ad
import sys

h5ad_path = '/hpc/group/gersbachlab/zy231/schizo_analysis/data/schizo_patient_unique.h5ad'
try:
    adata = ad.read_h5ad(h5ad_path, backed='r')
    print("Shape:", adata.shape)
    print("Obs columns:", adata.obs.columns)
    
    # Let's count NTC and SPEC
    cond = adata.obs['Disease'].astype(str)
    ntc_count = (cond == 'CTRL').sum()
    spec_count = (cond == 'SCZ').sum()
    print(f"NTC count: {ntc_count}")
    print(f"SPEC count: {spec_count}")
    print(f"Total cells: {adata.shape[0]}")
    print(f"Total genes: {adata.shape[1]}")
    
except Exception as e:
    print("Error:", e)
