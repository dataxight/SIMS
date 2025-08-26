import sys
sys.path.insert(0, ".")

import logging

import scsims
from scsims import SIMS
from pytorch_lightning.loggers import WandbLogger
import scanpy as sc
import anndata

if __name__ == "__main__":
    logger = WandbLogger(offline=True)

    # for this example, the file modality 1 of the
    # dataset CITE-seq profiles of 8k Cord Blood Mononuclear Cells
    # this dataset is accessible at
    # https://openproblems.bio/datasets/openproblems_v1_multimodal/citeseq_cbmc
    filepath = 'dataset_mod1.h5ad'
    
    
    adata = anndata.read_h5ad(filepath)
    if adata is None:
        logging.error("adata is None")
        sys.exit(1)
    logging.error(f"adata: {adata.__class__}")
    
    #Perform some light filtering
    counts = adata.layers['counts']
    sc.pp.filter_cells(counts, min_genes=100)
    sc.pp.filter_genes(counts, min_cells=3)
    #Transform the data for model ingestion
    sc.pp.normalize_total(adata,layer="counts")#Normalize counts per cell
    sc.pp.log1p(adata,layer="counts") ### Logarithmizing the data
    sc.pp.scale(adata,layer="counts") #Scale mean to zero and variance to 1

    # disabling as there's an error to be resolved here
    # at this moment, our need is to be able to initialize the classifer
    #sims = SIMS(data=adata, class_label='size_factors')
    #sims.setup_trainer(accelerator="gpu", devices=1, logger=logger)
    #sims.train()

    classifier = scsims.SIMSClassifier(10,4)
