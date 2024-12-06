import os
import numpy as np
import pandas as pd
import pickle
import timm
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as T
from wildlife_datasets.datasets import WildlifeDataset
from wildlife_tools.features import DeepFeatures
from typing import Optional, List, Tuple, Callable

def get_features(
        file_name: str,
        dataset: Optional[WildlifeDataset] = None,
        extractor: Optional[DeepFeatures] = None,
        force_compute: bool = False,
        ) -> object:
    """Loads already computed features from `file_name` or computes and saves them.

    Args:
        file_name (str): Filename of the saved features.
        dataset (Optional[WildlifeDataset], optional): Dataset for which compute the features.
        extractor (Optional[DeepFeatures], optional): Extractor to extract the features.
        force_compute (bool, optional): Whether the file should be overwritten if it exists.

    Returns:
        Computed features of size n_dataset*n_features.
    """

    if os.path.exists(file_name) and not force_compute:
        with open(file_name, 'rb') as file: 
            features = pickle.load(file)
    else:
        features = extractor(dataset)
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        with open(file_name, 'wb') as file: 
            pickle.dump(features, file) 
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
        return_score: bool = False
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Computes a closest match in the database for each vector in the query set.

    Args:
        features_query (np.ndarray): Query features of size n_query*n_feature. 
        features_database (np.ndarray): Database features of size n_database*n_feature
        ignore (Optional[List[List[int]]], optional): `ignore[i]` is a list of indices
            in the database ignores for i-th query.
        matcher (Callable, optional): function computing similarity.
        k (int, optional): Returned number of predictions.
        return_score (bool, optional): Whether the similalarity is returned.

    Returns:
        Vector of size (n_query,) and array of size (n_query,k). The latter are indices
            in the database for the closest matches (with ignored `ignore` indices).
            If `return_score`, it also returns an array of size (n_query,k) of scores.
    """

    # Create batch chunks
    n_query = len(features_query)
    # If ignore is not provided, initialize as empty
    if ignore is None:
        ignore = [[] for _ in range(n_query)]
    
    idx_true = np.array(range(n_query))
    idx_pred = np.zeros((n_query, k), dtype=np.int32)
    scores = np.zeros((n_query, k))
    # Compute the cosine similarity between the query and the database
    similarity = matcher(features_query, features_database)
    # Set -infinity for ignored indices
    for i in range(len(ignore)):
        similarity[i, ignore[i]] = -np.inf
    # Find the closest matches (k highest values)
    idx_pred = (-similarity).argsort(axis=-1)[:, :k]
    if return_score:
        scores = np.take_along_axis(similarity, idx_pred, axis=-1)
        return idx_true, idx_pred, scores
    else:
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

def get_dataset(dataset_class, root_dataset):
    d = dataset_class(root_dataset)
    if 'date' in d.df.columns:
        date = d.df['date']
        idx = d.df.index[~date.isnull()]
        if len(idx) > 0 and len(str(date.iloc[idx[0]])) != 4:
            d.df['year'] = np.nan
            d.df.loc[idx, 'year'] = pd.to_datetime(d.df.loc[idx, 'date']).apply(lambda x: x.year)
        else:
            d.df['year'] = d.df['date']
    else:
        d.df['date'] = np.nan
        if 'year' not in d.df.columns:
            d.df['year'] = np.nan
    return d

def get_df_split(dataset_class, root_dataset, analysis, **kwargs):
    d = get_dataset(dataset_class, root_dataset)
    idx_unknown_side = d.df['orientation'].apply(lambda x: x not in analysis.sides.keys())
    idx_unknown_identity = d.df['identity'] == 'unknown'
    idx_ignore = (idx_unknown_side + idx_unknown_identity).to_numpy()
    idx_database, idx_query = analysis.get_split(d.df, idx_ignore=idx_ignore, **kwargs)

    return d.df, idx_database, idx_query

def unique_no_sort(array):
    return pd.Series(array).unique()

def get_transform(flip=False, grayscale=False, img_size=None, normalize=False):
    transform = T.Compose([])
    if flip:
        transform = T.Compose([*transform.transforms, T.RandomHorizontalFlip(1)])
    if grayscale:
        transform = T.Compose([*transform.transforms, T.Grayscale(3)])        
    if img_size is not None:
        transform = T.Compose([*transform.transforms, T.Resize([img_size, img_size])])
    if normalize:
        transform = T.Compose([*transform.transforms, T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    return transform

def get_box_plot_data(boxplot):
    rows_list = []
    for i in range(len(boxplot['boxes'])):
        dict1 = {}
        dict1['lower_whisker'] = boxplot['whiskers'][i*2].get_ydata()[1]
        dict1['lower_quartile'] = boxplot['boxes'][i].get_ydata()[1]
        dict1['median'] = boxplot['medians'][i].get_ydata()[1]
        dict1['upper_quartile'] = boxplot['boxes'][i].get_ydata()[2]
        dict1['upper_whisker'] = boxplot['whiskers'][(i*2)+1].get_ydata()[1]
        rows_list.append(dict1)
    return pd.DataFrame(rows_list)