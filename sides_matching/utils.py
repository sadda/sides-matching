import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import pickle
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

def unpack(x):
    return f'{x[0]}_{x[1]}'

def convert_img_keypoints(img, keypoints, flip_img=False):
    if flip_img:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        keypoints = [cv2.KeyPoint(img.size[0] - int(x[0]), int(x[1]), 1) for x in keypoints]
    else:
        keypoints = [cv2.KeyPoint(int(x[0]), int(x[1]), 1) for x in keypoints]
    return np.array(img), keypoints

def visualise_matches(
        img0: Image,
        keypoints0: np.ndarray,
        img1: Image,
        keypoints1: list,
        ax = None,
        flip_img0 = False,
        flip_img1 = False
        ):
    """
    Visualise matches between two images.

    Args:
        img0 (np.array or PIL Image): First image.
        keypoints0 (np.array): Keypoints in the first image.
        img1 (np.array): Second image.
        keypoints1 (np.array): Keypoints in the second image.
        ax (matplotlib.axes.Axes, optional): Matplotlib axis to draw on. If None, a new axis is created.
    """

    # Convert images and keypoints to desired format
    img0, keypoints0 = convert_img_keypoints(img0, keypoints0, flip_img=flip_img0)
    img1, keypoints1 = convert_img_keypoints(img1, keypoints1, flip_img=flip_img1)

    # Create dummy matches (DMatch objects)
    matches = [cv2.DMatch(i, i, 0) for i in range(len(keypoints0))]

    # Draw matches
    img_matches = cv2.drawMatches(
        img0,
        keypoints0,
        img1,
        keypoints1,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # Plotting
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(img_matches)
    ax.axis("off")