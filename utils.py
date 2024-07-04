import os
import copy
import numpy as np
import pandas as pd
import timm
from sklearn.metrics.pairwise import cosine_similarity
from wildlife_datasets import datasets
from wildlife_tools.data import WildlifeDataset
from wildlife_tools.features import DeepFeatures
from typing import Optional, List, Tuple, Callable

class AmvrakikosTurtles(datasets.DatasetFactory):
    def create_catalogue(self) -> pd.DataFrame:
        data = pd.read_csv(os.path.join(self.root, 'annotations.csv'))

        # Get the bounding box
        columns_bbox = ['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height']
        bbox = data[columns_bbox].to_numpy()
        bbox = pd.Series(list(bbox))

        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': range(len(data)),
            'path': 'images' + os.path.sep + data['image_name'],
            'identity': data['image_name'].apply(lambda x: x.split('_')[0]).astype(int),
            'date': data['image_name'].apply(lambda x: x.split('_')[1]).astype(int),
            'orientation': data['image_name'].apply(lambda x: x.split('_')[2]),
            'bbox': bbox,
        })
        df = df[df['orientation'] != 'top']
        df['image_id'] = range(len(df))
        return self.finalize_catalogue(df)

class ReunionTurtles(datasets.DatasetFactory):
    def create_catalogue(self) -> pd.DataFrame:
        data = pd.read_csv(os.path.join(self.root, 'Data.csv'))

        date = pd.to_datetime(data['Date'])
        year = date.apply(lambda x: x.year)
        path = data['Species'] + os.path.sep + data['Turtle_ID'] + os.path.sep + year.astype(str) + os.path.sep + data['Photo_name']
        orientation = data['Photo_name'].apply(lambda x: os.path.splitext(x)[0].split('_')[2])
        orientation = orientation.replace({'L': 'left', 'R': 'right'})

        # Extract and convert ID codes
        id_code = list(data['ID_Code'].apply(lambda x: x.split(';')))
        max0 = 0
        max1 = 0
        for x in id_code:
            for y in x:
                max0 = max(max0, int(y[0]))
                max1 = max(max1, int(y[1]))
        code = np.zeros((len(id_code), max0, max1), dtype=int)
        for i, x in enumerate(id_code):
            for y in x:
                code[i, int(y[0])-1, int(y[1])-1] = int(y[2])
        code = code.reshape(len(id_code), -1)

        # Finalize the dataframe
        df = pd.DataFrame({
            'image_id': range(len(data)),
            'path': path,
            'identity': data['Turtle_ID'],
            'date': date,
            'orientation': orientation,
            'species': data['Species'],
            'id_code': list(code)
        })
        return self.finalize_catalogue(df)

def get_normalized_features(
        file_name: str,
        dataset: Optional[WildlifeDataset] = None,
        extractor: Optional[DeepFeatures] = None,
        normalize: bool = True,
        force_compute: bool = False,
        ) -> np.ndarray:
    """Loads already computed features from `file_name` or computes and saves them.

    Args:
        file_name (str): Filename of the saved features.
        dataset (Optional[WildlifeDataset], optional): Dataset for which compute the features.
        extractor (Optional[DeepFeatures], optional): Extractor to extract the features.
        normalize (bool, optional): Whether the features should be normalized to l2-norm one.
        force_compute (bool, optional): Whether the file should be overwritten if it exists.

    Returns:
        Computed features of size n_dataset*n_features.
    """

    if os.path.exists(file_name) and not force_compute:
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

def get_extractor(
        model_name: str = 'hf-hub:BVRA/MegaDescriptor-T-224',
        **kwargs
        ) -> DeepFeatures:
    """Loads an extractor via `timm.create_model`.

    Args:
        model_name (str, optional): Name of the model.

    Returns:
        Loaded extractor.
    """

    model = timm.create_model(model_name, num_classes=0, pretrained=True)
    return DeepFeatures(model, **kwargs)

def compute_predictions(
        features_query: np.ndarray,
        features_database: np.ndarray,
        ignore: Optional[List[List[int]]] = None,
        matcher: Callable = cosine_similarity,
        k: int = 4,
        batch_size: int = 1000
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Computes a closest match in the database for each vector in the query set.

    Args:
        features_query (np.ndarray): Query features of size n_query*n_feature. 
        features_database (np.ndarray): Database features of size n_database*n_feature
        ignore (Optional[List[List[int]]], optional): `ignore[i]` is a list of indices
            in the database ignores for i-th query.
        matcher (Callable, optional): function computing similarity.
        k (int, optional): Returned number of predictions.
        batch_size (int, optional): Size of the computaiton batch.

    Returns:
        Vector of size (n_query,) and array of size (n_query,k). The latter are indices
            in the database for the closest matches (with ignored `ignore` indices)
    """

    # Create batch chunks
    n_query = len(features_query)
    n_chunks = int(np.ceil(n_query / batch_size))
    chunks = np.array_split(range(n_query), n_chunks)
    # If ignore is not provided, initialize as empty
    if ignore is None:
        ignore = [[] for _ in range(n_query)]
    
    idx_true = np.array(range(n_query))
    idx_pred = np.zeros((n_query, k), dtype=np.int32)
    for chunk in chunks:
        # Compute the cosine similarity between the query chunk and the database
        similarity = matcher(features_query[chunk], features_database)
        # Set -infinity for ignored indices
        for i in range(len(chunk)):
            similarity[i, ignore[chunk[i]]] = -np.inf
        # Find the closest matches (k highest values)
        idx_pred[chunk,:] = (-similarity).argsort(axis=-1)[:, :k]
    return idx_true, idx_pred

def compute_predictions_disjoint(
        features: np.ndarray,
        ignore: Optional[List[List[int]]] = None,
        **kwargs
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the closest matches for the disjoint-set setting.

    Args:
        features (np.ndarray): List of database=query features of size n*n_feature. 
        ignore (Optional[List[List[int]]], optional): `ignore[i]` is a list of indices
            in the database ignores for i-th query.

    Returns:
        Vector of size (n_query,) and array of size (n_query,k). The latter are indices
            in the database for the closest matches (with ignored `ignore` indices)
    """

    # If ignore is not provided, initialize as empty
    if ignore is None:
        ignore = [[] for _ in range(len(features))]
    # Add diagonal elements to ignore
    for i in range(len(ignore)):
        ignore[i].append(i)
    return compute_predictions(features, features, ignore=ignore, **kwargs)

def compute_predictions_closed(
        features_query: np.ndarray,
        features_database: np.ndarray,
        ignore: Optional[List[List[int]]] = None,
        **kwargs
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the closest matches for the closed-set setting.

    Args:
        features_query (np.ndarray): Query features of size n_query*n_feature. 
        features_database (np.ndarray): Database features of size n_database*n_feature
        ignore (Optional[List[List[int]]], optional): `ignore[i]` is a list of indices
            in the database ignores for i-th query.

    Returns:
        Vector of size (n_query,) and array of size (n_query,k). The latter are indices
            in the database for the closest matches (with ignored `ignore` indices)
    """

    return compute_predictions(features_query, features_database, ignore=ignore, **kwargs)

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
