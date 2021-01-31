from pytorch_lightning.metrics import Metric
from sklearn.metrics import ndcg_score
import numpy as np
import torch

class nDCG(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("ndcg", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, y_true: torch.Tensor, y_score: torch.Tensor):

        self.total += 1
        self.ndcg += ndcg_score(np.asarray([y_true]), np.asarray(y_score.unsqueeze(0)))

    def compute(self):
        return self.ndcg.float() / self.total
