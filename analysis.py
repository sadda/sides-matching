import os
import numpy as np
import pandas as pd

class Analysis():
    def __init__(self):
        # TODO: change names to something normal
        self.n_sides = len(self.sides)
        self.names_split = ['database-database', 'query-query', 'database-query']
        # TODO: add check for sides

    def get_split(self, *args, **kwargs):
        raise NotImplementedError('Must be implemented by subclasses')

    def get_matches(
            self,
            y_true,
            y_pred,
            orientation_true,
            orientation_pred
            ):
        
        matches = {}
        for k in range(orientation_pred.shape[1]):
            matches[k] = {}
            for key in self.diff_to_matches.values():
                matches[k][key] = 0
            matches[k]['other'] = 0
            matches[k]['wrong match'] = 0
            for i in range(len(orientation_true)):
                if y_true[i] == y_pred[i][k]:
                    if orientation_true[i] not in self.sides.keys() or orientation_pred[i][k] not in self.sides.keys():
                        matches[k]['other'] += 1
                    else:
                        diff = abs(self.sides[orientation_true[i]] - self.sides[orientation_pred[i][k]])
                        matches[k][self.diff_to_matches[diff]] += 1
                else:
                    matches[k]['wrong match'] += 1
        matches = pd.DataFrame(matches).T
        for column in matches.columns:
            matches[column] = (matches[column] / len(orientation_pred) * 100).round(2).astype(str) + '%'
        matches.index = 'match k = ' + (matches.index + 1).astype(str)
        return matches

    def initialite_results_split_similarity(self):
        results = {}
        for name1 in self.names_split:
            results1 = {}
            for diff in range(self.n_sides):
                results2 = {}
                for name2 in self.names_categories:
                    results2[name2] = []
                results1[diff] = results2
            results[name1] = results1
        return results

    def compute_data_split_similarity(self, df):
        identity = df['identity'].to_numpy()
        orientation = df['orientation'].apply(lambda x: self.sides[x] if x in self.sides else np.nan).to_numpy()
        return {'identity': identity, 'orientation': orientation}
    
    def compute_index_split_similarity(self, data, i, j):
        if data['identity'][i] == data['identity'][j]:
            return 'same ind'
        else:
            return 'diff ind'

    def split_similarity_matrix(self, df, similarity, idx_database, idx_query):
        if len(set(idx_database).intersection(set(idx_query))):
            raise Exception('idx must be disjoint')
        idx_database = np.sort(np.array(idx_database))
        idx_query = np.sort(np.array(idx_query))
        
        data = self.compute_data_split_similarity(df)
        results = self.initialite_results_split_similarity()
        for idx1, idx2, name in zip((idx_database, idx_query, idx_database), (idx_database, idx_query, idx_query), self.names_split):
            array_equal = np.array_equal(idx1, idx2)
            for i_index, i in enumerate(idx1):
                idx2_range = idx2[i_index+1:] if array_equal else idx2
                for j in idx2_range:
                    diff = abs(data['orientation'][i] - data['orientation'][j])
                    index = self.compute_index_split_similarity(data, i, j)
                    results[name][diff][index].append(similarity[i,j])
        return results

class Analysis_SarahZelvy(Analysis):
    def __init__(self):
        self.sides = {'left': 0, 'right': 1}
        self.diff_to_matches = {0: 'match same side', 1: 'match diff side'}
        self.names_categories = ['same ind', 'diff ind']
        super().__init__()

    def get_split(self, df, daytime='all'):
        idx_database = []
        if daytime in ['day', 'night']:
            idx_query = np.where(df['daytime'] == daytime)[0]
        else:
            idx_query = np.arange(len(df))
        return idx_database, idx_query

class Analysis_WildlifeDataset(Analysis):
    def get_split_general(
            self,
            df,
            dataset_name,
            empty_database=False,
            idx_ignore=None,
            ):
        
        if idx_ignore is None:
            idx_ignore = np.zeros(len(df), dtype=bool)
        assert len(df) == len(idx_ignore)

        # Load the split on which MegaDescriptor was trained
        train_df = pd.read_csv(os.path.join('csv', 'combined_all.csv'))
        # Select data for the dataset in question
        train_df = train_df[train_df['path'].str.startswith(dataset_name + '/')]
        # Extract the names of the training examples
        train_names = []
        for _, row in train_df.iterrows():
            image_id = str(row['image_id'])
            file_name = os.path.basename(row['path'])
            file_name, ext = os.path.splitext(file_name)
            if not file_name.endswith(image_id):
                raise Exception('Something went wrong')
            file_name = file_name[:len(file_name)-len(image_id)-1]
            train_names.append(file_name + ext)
        # Extract names from  and check for uniqueness
        df_names = [os.path.basename(path) for path in df['path']]
        if len(df_names) != len(np.unique(df_names)):
            raise Exception('File names must be unique.')
        # Extract query and database indices
        idx_train = np.array([x in train_names for x in df_names])
        idx_query = np.where((~idx_ignore) * (~idx_train))[0]
        if empty_database:
            idx_database = []
        else:
            idx_database = np.where((~idx_ignore) * idx_train)[0]
        return idx_database, idx_query

    def get_split(self, df, idx_ignore=None, **kwargs):
        if idx_ignore is None:
            idx_ignore = np.array(df['identity'] == 'unknown')
        name = self.__class__.__name__.split('_')[-1]
        return self.get_split_general(df, name, idx_ignore=idx_ignore, **kwargs)

class Analysis_SeaTurtleIDHeads(Analysis_WildlifeDataset):
    def __init__(self):
        self.sides = {'left': 0, 'topleft': 1, 'top': 2, 'topright': 3, 'right': 4}
        self.diff_to_matches = {0: 'match diff = 0', 1: 'match diff = 1', 2: 'match diff = 2', 3: 'match diff = 3', 4: 'match diff = 4'}
        self.names_categories = ['same ind - same day', 'same ind - same setup', 'same ind - diff setup', 'diff ind - same setup', 'diff ind - diff setup']
        super().__init__()
        
    def get_setup_id(self, date):
        if date.year <= 2013:
            return 1
        elif date.year <= 2017:
            return 2
        else:
            return 3

    def compute_data_split_similarity(self, df):
        data = super().compute_data_split_similarity(df)
        data['date'] = pd.to_datetime(df['date']).to_numpy()
        data['camera_setup'] = pd.to_datetime(df['date']).apply(lambda x: self.get_setup_id(x)).to_numpy()
        return data
    
    def compute_index_split_similarity(self, data, i, j):
        if data['identity'][i] == data['identity'][j]:
            if data['date'][i] == data['date'][j]:
                return 'same ind - same day'
            elif data['camera_setup'][i] == data['camera_setup'][j]:
                return 'same ind - same setup'
            else:
                return 'same ind - diff setup'
        else:
            if data['camera_setup'][i] == data['camera_setup'][j]:
                return 'diff ind - same setup'
            else:
                return 'diff ind - diff setup'

class Analysis_ZindiTurtleRecall(Analysis_WildlifeDataset):
    def __init__(self):
        self.sides = {'left': 0, 'top': 1, 'right': 2}
        self.diff_to_matches = {0: 'match diff = 0', 1: 'match diff = 1', 2: 'match diff = 2'}
        self.names_categories = ['same ind', 'diff ind']
        super().__init__()
