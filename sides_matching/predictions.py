import os
import numpy as np
from scipy.spatial.distance import cdist
from wildlife_tools.similarity import CosineSimilarity, MatchLightGlue
from wildlife_tools.features import AlikedExtractor, DiskExtractor, SiftExtractor, SuperPointExtractor
from .utils import get_features, compute_predictions, unique_no_sort

class Data():
    def __init__(self, path_features_query, path_features_database):
        self.path_features_query = path_features_query
        self.path_features_database = path_features_database

    def get_features(self):
        return get_features(self.path_features_query), get_features(self.path_features_database)

    def compute_scores(self, ignore_diagonal=False):
        features_query, features_database = self.get_features()
        if ignore_diagonal and len(features_query) == len(features_database):
            idx_ignore = [[i] for i in range(len(features_query))]
            k = len(features_database) - 1
        else:
            idx_ignore = [[] for i in range(len(features_query))]
            k = len(features_database)
        return compute_predictions(features_query, features_database, ignore=idx_ignore, k=k, matcher=self.matcher, return_score=True)

class MegaDescriptor(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.matcher = CosineSimilarity()

class Aliked(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.matcher = MatchLightGlue('aliked')

class Sift(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.matcher = MatchLightGlue('sift')

class TORSOOI(Data):
    def __init__(self, df):
        self.df = df
        self.matcher = lambda x, y: cdist(x, y, lambda a, b: sum(a==b))

    def get_features(self):
        features = np.array([list(x) for x in self.df['id_code'].to_numpy()])
        return features, features

class Prediction():
    def __init__(self, df, true, pred, scores):
        self.df = df
        self.identity = df['identity'].to_numpy()
        self.orientation = df['orientation'].to_numpy()
        self.year = df['year'].to_numpy()
        self.true = true
        self.pred = pred
        self.scores = scores
        self.n_individuals = df['identity'].nunique()
    
    def compute_accuracy(self, mods):
        metrics = [f'top {i}' for i in range(1, 1+self.n_individuals)]
        accuracy = {mod: {metric: 0 for metric in metrics} for mod in mods}
        
        # Loop over individual query images
        for i, (i_pred, i_true) in enumerate(zip(self.pred, self.true)):
            # Extract identity, orientation and year        
            identity_pred_full = self.identity[i_pred]
            orientation_pred = self.orientation[i_pred]
            year_pred = self.year[i_pred]
            identity_true = self.identity[i_true]
            orientation_true = self.orientation[i_true]
            year_true = self.year[i_true]
            same_identity = identity_pred_full == identity_true
            # Save metrics for individual mods
            for mod in mods:            
                # Select indices to ignore for individual mods
                if mod == 'full':
                    ignore = np.zeros(len(identity_pred_full), dtype='bool')
                elif mod == 'different orientation':
                    ignore = orientation_true == orientation_pred
                elif mod == 'same orientation':
                    ignore = orientation_true != orientation_pred
                elif mod == 'different year':
                    ignore = year_true == year_pred
                elif mod == 'same year':
                    ignore = year_true != year_pred
                elif mod == 'different both':
                    ignore = (orientation_true == orientation_pred) + (year_true == year_pred)
                else:
                    raise Exception('Unknown mod')            
                # Ignore selected indices but only of the same individual
                identity_pred = identity_pred_full[~(same_identity * ignore)]
                # Get the unique predictions
                identity_pred_unique = unique_no_sort(identity_pred)            
                # Compute the metrics
                for i in range(1, 1+self.n_individuals):
                    accuracy[mod][f'top {i}'] += (identity_true in identity_pred_unique[:i]) / len(self.true)
        self.accuracy = accuracy

    def split_scores(self, save_idx=False):
        scores_split = {x: {y: {z: [] for z in {True, False}} for y in {True, False}} for x in {True, False}}
        for i_score, (i_pred, i) in enumerate(zip(self.pred, self.true)):
            for j_score, j in enumerate(i_pred):
                equal_identity = self.identity[i] == self.identity[j]
                equal_orientation = self.orientation[i] == self.orientation[j]
                equal_year = self.year[i] == self.year[j]
                score = self.scores[i_score, j_score]
                if save_idx:
                    score_add = (score, i, j)
                else:
                    score_add = score
                scores_split[equal_identity][equal_orientation][equal_year].append(score_add)
        return scores_split