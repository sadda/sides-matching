import os
import numpy as np
from scipy.spatial.distance import cdist
from wildlife_tools.similarity import CosineSimilarity, MatchLightGlue
from wildlife_tools.features import AlikedExtractor, DiskExtractor, SiftExtractor, SuperPointExtractor

from sift_matching import Reader, Loader, SIFT, L1Matcher
from .utils import get_features, compute_predictions, unique_no_sort

class Data():
    def compute_scores(self, ignore_diagonal=False):
        features_query, features_database = self.get_features()
        if ignore_diagonal and len(features_query) == len(features_database):
            idx_ignore = [[i] for i in range(len(features_query))]
            k = len(features_database) - 1
        else:
            idx_ignore = [[] for i in range(len(features_query))]
            k = len(features_database)
        return compute_predictions(features_query, features_database, ignore=idx_ignore, k=k, matcher=self.matcher, return_score=True)

class Data_WildlifeTools(Data):
    def __init__(self, path_features_query, path_features_database):
        self.path_features_query = path_features_query
        self.path_features_database = path_features_database

    def get_features(self):
        return get_features(self.path_features_query), get_features(self.path_features_database)

class MegaDescriptor(Data_WildlifeTools):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.matcher = CosineSimilarity()

class Aliked(Data_WildlifeTools):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.matcher = MatchLightGlue('aliked')

class Sift(Data_WildlifeTools):
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

    def compute_scores(self, n_matches=15, ignore_diagonal=False):
        # TODO: missing ignore_diagonal
        if self.reader.n_batches != 1:
            raise Exception('Works only for readers with one batch.')
        
        self.reader.create_database()
        self.reader.create_matches()
        matches = self.reader.load_matches(0, 0)

        n = len(matches)
        scores = np.full((n, n), -np.inf)
        for i in range(n):
            for j in range(n):
                if i < j and len(matches[i][j]) >= n_matches:
                    value = -np.sum([np.sqrt(match.distance) for match in matches[i][j][:n_matches]])
                    scores[i,j] = value
                    scores[j,i] = value
        idx_pred = np.argsort(scores, axis=1)[:,::-1][:,:-1]
        scores = np.array([s[i] for s, i in zip(scores, idx_pred)])
        return np.array(range(n)), idx_pred, scores

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