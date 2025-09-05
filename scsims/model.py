from functools import partial
from typing import Any, Callable, Dict, Union

import os
import anndata as an
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_tabnet.tab_network import TabNet
from pytorch_tabnet.utils import create_explain_matrix
from scipy.sparse import csc_matrix
#from torchmetrics.functional.classification.stat_scores import _stat_scores_update
from torchmetrics.classification import MulticlassStatScores
from tqdm import tqdm
import torch.utils.data
from scipy.sparse import csr_matrix
from scsims.data import CollateLoader
from scsims.inference import DatasetForInference
from scsims.temperature_scaling import _ECELoss
from torchmetrics import Accuracy, F1Score, Precision, Recall, Specificity
from sklearn.preprocessing import LabelEncoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, Specificity

class SIMSClassifier(L.LightningModule):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        cat_idxs=[],
        cat_dims=[],
        cat_emb_dim=1,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
        lambda_sparse=1e-3,
        optim_params: dict = None,
        scheduler_params: dict = None,
        weights: torch.Tensor = None,
        loss: callable = None,
        pretrained: dict = None,
        no_explain: bool = False,
        genes: list[str] = None,
        label_encoder=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["loss", "weights", "label_encoder"])

        self.genes = genes
        self.label_encoder = label_encoder
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lambda_sparse = lambda_sparse
        self.weights = weights
        self.loss = loss if loss is not None else F.cross_entropy

        if pretrained is not None:
            self._from_pretrained(**pretrained.get_params())

        # Optimizer defaults
        self.optim_params = optim_params or {
            "optimizer": torch.optim.Adam,
            "lr": 1e-2,
            "weight_decay": 1e-2,
        }

        self.scheduler_params = scheduler_params or {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
            "factor": 0.75,
            "patience": 5,
        }

        # Network
        self.network = TabNet(
            input_dim=input_dim,
            output_dim=output_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=cat_emb_dim,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            mask_type=mask_type,
            group_attention_matrix=torch.ones(1, input_dim)  # FIXED shape
        )

        # Explainer
        if not no_explain:
            self.reducing_matrix = create_explain_matrix(
                self.network.input_dim,
                self.network.cat_emb_dim,
                self.network.cat_idxs,
                self.network.post_embed_dim,
            )

        # Metrics: use MetricCollection for auto reset
        task = "binary" if output_dim == 2 else "multiclass"
        num_classes = None if output_dim == 2 else output_dim

        metrics = MetricCollection({
            "micro_acc": Accuracy(task=task, num_classes=num_classes, average="micro"),
            "macro_acc": Accuracy(task=task, num_classes=num_classes, average="macro"),
            "weighted_acc": Accuracy(task=task, num_classes=num_classes, average="weighted"),
            "precision": Precision(task=task, num_classes=num_classes, average="macro"),
            "recall": Recall(task=task, num_classes=num_classes, average="macro"),
            "f1": F1Score(task=task, num_classes=num_classes, average="macro"),
            "specificity": Specificity(task=task, num_classes=num_classes, average="macro"),
        })

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")

        # Temperature scaling
        self.temperature = torch.nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits, M_loss = self.network(x)
        return self.temperature_scale(logits), M_loss

    def _common_step(self, batch):
        x, y = batch
        logits, M_loss = self.network(x)
        loss = self.loss(logits, y, weight=self.weights)
        loss = loss - self.lambda_sparse * M_loss
        return logits, y, loss

    def training_step(self, batch, batch_idx):
        logits, y, loss = self._common_step(batch)
        probs = logits.softmax(dim=-1)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.train_metrics.update(probs, y)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, y, loss = self._common_step(batch)
        probs = logits.softmax(dim=-1)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.val_metrics.update(probs, y)

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), prog_bar=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True)
        self.val_metrics.reset()

    def configure_optimizers(self):
        opt_class = self.optim_params.pop("optimizer", torch.optim.Adam)
        optimizer = opt_class(self.parameters(), **self.optim_params)
        if self.scheduler_params is not None:
            sched_class = self.scheduler_params.pop("scheduler")
            scheduler = sched_class(optimizer, **self.scheduler_params)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
            }
        return optimizer

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature
