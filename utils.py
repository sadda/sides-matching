import os
import copy
import numpy as np
import timm
from wildlife_datasets import datasets
from wildlife_tools.data import WildlifeDataset
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.features import DeepFeatures

def get_normalized_features(file_name, dataset=None, extractor=None, normalize=True):
    if os.path.exists(file_name):
        features = np.load(file_name)
    else:
        features = extractor(dataset)
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        np.save(file_name, features)
    if normalize:
        for i in range(len(features)):
            features[i] /= np.linalg.norm(features[i])
    return features

def get_extractor(model_name='hf-hub:BVRA/MegaDescriptor-T-224', **kwargs):
    model = timm.create_model(model_name, num_classes=0, pretrained=True)
    return DeepFeatures(model, **kwargs)

def compute_predictions_disjoint(features, k=4, batch_size=1000):
    n_query = len(features)
    n_chunks = int(np.ceil(n_query / batch_size))
    chunks = np.array_split(range(n_query), n_chunks)

    matcher = CosineSimilarity()
    idx_true = np.array(range(n_query))    
    idx_pred = np.zeros((n_query, k), dtype=np.int32)
    for chunk in chunks:
        similarity = matcher(query=features[chunk], database=features)['cosine']
        idx_x = np.arange(len(chunk))
        idx_y = np.arange(chunk[0], chunk[0]+len(chunk))
        similarity[idx_x, idx_y] = -1        
        idx_pred[chunk,:] = (-similarity).argsort(axis=-1)[:, :k]
    return idx_true, idx_pred

def compute_predictions_closed(features_query, features_database, k=4, batch_size=1000):
    n_query = len(features_query)
    n_chunks = int(np.ceil(n_query / batch_size))
    chunks = np.array_split(range(n_query), n_chunks)

    matcher = CosineSimilarity()
    idx_true = np.array(range(n_query))    
    idx_pred = np.zeros((n_query, k), dtype=np.int32)
    for chunk in chunks:
        similarity = matcher(query=features_query[chunk], database=features_database)['cosine']
        idx_pred[chunk,:] = (-similarity).argsort(axis=-1)[:, :k]
    return idx_true, idx_pred

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
