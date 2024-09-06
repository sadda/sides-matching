import os
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity

from sift_matching import Reader, Loader, SIFT, L1Matcher
from .utils import get_normalized_features, compute_predictions, unique_no_sort

class Data():
    def compute_scores(self, idx_true):
        features_database, features_query = self.get_features()
        idx_ignore = [[i] for i in range(len(idx_true))]
        k = len(idx_true) - 1

        # Get the prediction metric
        _, idx_pred, scores = compute_predictions(features_query[idx_true], features_database[idx_true], ignore=idx_ignore, k=k, matcher=self.matcher, return_score=True)
        idx_pred = idx_true[idx_pred]
        return idx_true, idx_pred, scores

class Data_MegaDescriptor(Data):
    def __init__(self, path_features_query, path_features_database):
        self.path_features_query = path_features_query
        self.path_features_database = path_features_database
        self.matcher = cosine_similarity

    def get_features(self):
        features_database = get_normalized_features(self.path_features_database)
        features_query = get_normalized_features(self.path_features_query)
        return features_database, features_query

class Data_TORSOOI(Data):
    def __init__(self, df):
        self.df = df
        self.matcher = lambda x, y: cdist(x, y, lambda a, b: sum(a==b))

    def get_features(self):
        features_database = np.array([list(x) for x in self.df['id_code'].to_numpy()])
        features_query = np.array([list(x) for x in self.df['id_code'].to_numpy()])
        return features_database, features_query    

class Data_SIFT():
    def __init__(
            self,
            root_images,
            root_results,
            df,
            image_loader=None,
            keypoint_extractor = None, 
            keypoint_matcher = None, 
            reader=None
            ):
                
        df = df.copy()
        df['path'] = df['path'].apply(lambda x: os.path.join(root_images, x))
        if image_loader is None:
            if 'bbox' in df.columns:
                image_loader = Loader(img_load='bbox', img_size=90)
            else:
                image_loader = Loader(img_load='full', img_size=90)
        if keypoint_extractor is None:
            keypoint_extractor = SIFT()
        if keypoint_matcher is None:
            keypoint_matcher = L1Matcher()
        if reader is None:
            reader = Reader(df, image_loader, keypoint_extractor, keypoint_matcher, None, root_results)

        self.df = df
        self.root_results = root_results
        self.image_loader = image_loader
        self.keypoint_extractor = keypoint_extractor
        self.keypoint_matcher = keypoint_matcher
        self.reader = reader        

    def compute_scores(self, idx_true, n_matches=15):
        if self.reader.n_batches != 1:
            raise Exception('Works only for readers with one batch.')
        
        self.reader.create_database()
        self.reader.create_matches()
        matches = self.reader.load_matches(0, 0)

        n = len(idx_true)
        scores = np.full((n, n), -np.inf)
        for i0, i in enumerate(idx_true):
            for j0, j in enumerate(idx_true):
                if i < j and len(matches[i][j]) >= n_matches:
                    value = -np.sum([np.sqrt(match.distance) for match in matches[i][j][:n_matches]])
                    scores[i0,j0] = value
                    scores[j0,i0] = value
        idx_pred = np.argsort(scores, axis=1)[:,::-1][:,:-1]
        scores = np.array([s[i] for s, i in zip(scores, idx_pred)])
        idx_pred = idx_true[idx_pred]
        return idx_true, idx_pred, scores

class Prediction():
    def __init__(self, df, true, pred, scores, n_individuals):
        self.df = df
        self.identity = df['identity'].to_numpy()
        self.orientation = df['orientation'].to_numpy()
        self.year = df['year'].to_numpy()
        self.true = true
        self.pred = pred
        self.scores = scores
        self.n_individuals = n_individuals
    
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