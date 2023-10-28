import torch
import torch.nn as nn
import numpy as np
import copy
import time
import os
import sys
import wandb
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseServer import BaseServer
from algorithms.fedlmd_tf.ClientTrainer import ClientTrainer
from algorithms.fedlmd_tf.criterion import *
from algorithms.measures import *

__all__ = ["Server"]


class Server(BaseServer):
    def __init__(
        self, algo_params, model, data_distributed, optimizer, scheduler=None, **kwargs
    ):
        super(Server, self).__init__(
            algo_params, model, data_distributed, optimizer, scheduler, **kwargs
        )
        local_criterion = self._get_local_criterion(self.algo_params, self.num_classes)

        self.client = ClientTrainer(
            local_criterion,
            algo_params=algo_params,
            model=copy.deepcopy(model),
            local_epochs=self.local_epochs,
            device=self.device,
            num_classes=self.num_classes,
        )

        # Count the Major labels
        self.client_y_lst = {}
        for client_idx, d in self.data_distributed["local"].items():
            label_distribute_d = defaultdict(lambda: 0)
            for _, y_lst in d["train"]:
                for y in y_lst:
                    label_distribute_d[y.item()] += 1

            mean_v = np.mean(self.data_distributed["data_map"][client_idx])
            y_lst = np.where(np.array(self.data_distributed["data_map"][client_idx]) > mean_v)[0]
            self.client_y_lst[client_idx] = copy.deepcopy(torch.tensor(y_lst, device=self.device))
        print("\n>>> fedlmd-tf Server initialized...\n")

    def _get_local_criterion(self, algo_params, num_classes):
        tau = algo_params.tau
        beta = algo_params.beta

        criterion = LMD_Tf_Loss(num_classes, tau, beta)

        return criterion

    def _set_client_data(self, client_idx):
        super()._set_client_data(client_idx)
        self.client.major_labels = self.client_y_lst[client_idx]