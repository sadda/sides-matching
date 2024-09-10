import copy
import os
import numpy as np
import pandas as pd
from wildlife_datasets import datasets
from wildlife_tools.data import WildlifeDataset

class WD(WildlifeDataset):
    def plot_grid(self, transform=None, **kwargs):
        self_copy = copy.deepcopy(self)
        self_copy.transform = transform
        self_copy.load_label = False
        loader = lambda k: self_copy.__getitem__(k)
        rotate = kwargs.pop('rotate', False)
        return datasets.DatasetFactory(self.root, self.metadata).plot_grid(rotate=rotate, loader=loader, **kwargs)

    def plot_predictions(self, y_true, y_pred, **kwargs):
        from collections.abc import Iterable

        if not isinstance(y_true, Iterable) and np.array(y_pred).ndim == 1:
            y_true = [y_true]
            y_pred = [y_pred]
        if len(y_true) > 1:
            header_cols = ["Query", ""] + [f"Match {i+1}" for i in range(len(y_pred[0]))]
        else:
            identity = self.metadata['identity'].to_numpy()
            header_cols = [identity[y_true[0]], ""] + [identity[y_p] for y_p in y_pred[0]]
        n_cols = len(header_cols)
        idx = []
        for y_t, y_p in zip(y_true, y_pred):
            idx.append([y_t, -1] + list(y_p))
        n_rows = kwargs.pop('n_rows', min(len(y_true), 5))
        return self.plot_grid(idx=idx, n_rows=n_rows, n_cols=n_cols, header_cols=header_cols, **kwargs)
