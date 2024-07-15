import os
import numpy as np
import pandas as pd

class Analysis():
    def __init__(self):
        self.n_sides = len(self.sides)
        self.names_split = ['database-database', 'query-query', 'database-query']
        if self.sides_cycle:
            self.max_diff = int(np.floor(self.n_sides / 2))
        else:
            self.max_diff = self.n_sides - 1
        if self.max_diff == 1:
            diff_to_matches = {0: 'same side', 1: 'diff side'}        
        else:
            diff_to_matches = {i: f'diff = {i}' for i in range(self.max_diff+1)}
        self.diff_to_matches = diff_to_matches
        self.check_data()

    def check_data(self):
        array1a = range(self.n_sides)
        array1b = np.sort(list(self.sides.values()))
        array2a = range(self.max_diff+1)
        array2b = np.sort(list(self.diff_to_matches.keys()))
        if not np.array_equal(array1a, array1b):
            raise Exception('Values in self.sides must be 0..n')
        if not np.array_equal(array2a, array2b):
            raise Exception('Values in self.diffs_to_matches are wrong')

    def get_split(self, *args, **kwargs):
        raise NotImplementedError('Must be implemented by subclasses')

    def difference(self, i_side, j_side):
        # TODO: write test function
        if self.sides_cycle:
            return min(np.mod(int(i_side)-int(j_side), self.n_sides), np.mod(int(j_side)-int(i_side), self.n_sides))
        else:
            return np.abs(int(i_side)-int(j_side))
    
    def get_split_initialize_idx_ignore(self, df, idx_ignore=None):        
        if idx_ignore is None:
            idx_ignore = np.zeros(len(df), dtype=bool)
        assert len(df) == len(idx_ignore)
        return idx_ignore

    def initialite_results_split_similarity(self):
        return {i: {j: {k: [] for k in self.names_categories} for j in range(self.max_diff+1)} for i in self.names_split}
    
    def compute_data_split_similarity(self, df):
        identity = df['identity'].to_numpy()
        orientation = df['orientation'].apply(lambda x: self.sides[x] if x in self.sides else np.nan).to_numpy()
        # Test if there are only integers
        orientation_test1 = pd.Series(orientation)
        orientation_test1 = orientation_test1[~orientation_test1.isnull()]
        orientation_test2 = orientation_test1.astype(int)
        if (orientation_test2 - orientation_test1).abs().sum() != 0:
            raise Exception('df column orientation must contain integers only')
        return {'identity': identity, 'orientation': orientation}
    
    def compute_index_split_similarity(self, data, i, j):
        if data['identity'][i] == data['identity'][j]:
            return 'same ind'
        else:
            return 'diff ind'

    def split_similarity_matrix(self, df, similarity, idx_database, idx_query):
        if len(set(idx_database).intersection(set(idx_query))) > 0:
            raise Exception('idx must be disjoint')
        idx_database = np.sort(np.array(idx_database))
        idx_query = np.sort(np.array(idx_query))
        
        data = self.compute_data_split_similarity(df)
        results = self.initialite_results_split_similarity()
        for idx, jdx, name in zip((idx_database, idx_query, idx_database), (idx_database, idx_query, idx_query), self.names_split):
            array_equal = np.array_equal(idx, jdx)
            for i_index, i in enumerate(idx):
                jdx_range = jdx[i_index+1:] if array_equal else jdx
                for j in jdx_range:
                    diff = self.difference(data['orientation'][i], data['orientation'][j])
                    index = self.compute_index_split_similarity(data, i, j)
                    results[name][diff][index].append(similarity[i,j])
        return results

class Analysis_SarahZelvy(Analysis):
    def __init__(self, **kwargs):
        self.sides = {'left': 0, 'right': 1}
        self.sides_cycle = False
        self.names_categories = ['same ind', 'diff ind']
        super().__init__(**kwargs)

    def get_split(self, df, idx_ignore=None):
        idx_ignore = self.get_split_initialize_idx_ignore(df, idx_ignore=idx_ignore)
        idx_database = []
        idx_query = np.where(~idx_ignore)[0]
        return idx_database, idx_query

class Analysis_WildlifeDataset(Analysis):
    def get_split_general(
            self,
            df,
            dataset_name,
            idx_ignore=None,
            ):
        
        idx_ignore = self.get_split_initialize_idx_ignore(df, idx_ignore=idx_ignore)
        # Load the split on which MegaDescriptor was trained
        train_df = pd.read_csv(os.path.join('csv', 'combined_all.csv'))
        # Select data for the dataset in question
        train_df = train_df[train_df['path'].str.startswith(dataset_name + '/')]
        df_names = []
        for _, row in df.iterrows():
            image_id = str(row['image_id'])
            file_name = os.path.basename(row['path'])
            file_name, ext = os.path.splitext(file_name)
            df_names.append(f'{file_name}_{image_id}{ext}')
        train_names = [os.path.basename(path) for path in train_df['path']]
        if len(df_names) != len(np.unique(df_names)):
            raise Exception('File names must be unique.')
        # Extract query and database indices
        idx_train = np.array([x in train_names for x in df_names])
        idx_query = np.where((~idx_ignore) * (~idx_train))[0]
        idx_database = np.where((~idx_ignore) * idx_train)[0]
        return idx_database, idx_query

    def get_split(self, df, **kwargs):
        name = self.__class__.__name__.split('_')[-1]
        return self.get_split_general(df, name, **kwargs)

class Analysis_HyenaID2022(Analysis_WildlifeDataset):
    def __init__(self, **kwargs):
        self.sides = {'left': 0, 'frontleft': 1, 'front': 2, 'frontright': 3, 'right': 4, 'backright': 5, 'back': 6, 'backleft': 7}
        self.sides_cycle = True
        self.names_categories = ['same ind', 'diff ind']
        super().__init__(**kwargs)

class Analysis_AmvrakikosTurtles(Analysis_WildlifeDataset):
    def __init__(self, **kwargs):
        self.sides = {'left': 0, 'right': 1}
        self.sides_cycle = False
        self.names_categories = ['same ind - same year', 'same ind - diff year', 'diff ind - same year', 'diff ind - diff year']
        super().__init__(**kwargs)

    def compute_data_split_similarity(self, df):
        data = super().compute_data_split_similarity(df)
        data['year'] = df['year'].to_numpy()
        return data
    
    def compute_index_split_similarity(self, data, i, j):
        if data['identity'][i] == data['identity'][j]:
            if data['year'][i] == data['year'][j]:
                return 'same ind - same year'
            else:
                return 'same ind - diff year'
        else:
            if data['year'][i] == data['year'][j]:
                return 'diff ind - same year'
            else:
                return 'diff ind - diff year'

class Analysis_ReunionTurtles(Analysis_AmvrakikosTurtles):
    pass

class Analysis_LeopardID2022(Analysis_WildlifeDataset):
    def __init__(self, **kwargs):
        self.sides = {'left': 0, 'front': 1, 'right': 2, 'back': 3}
        self.sides_cycle = True
        self.names_categories = ['same ind', 'diff ind']
        super().__init__(**kwargs)

class Analysis_NyalaData(Analysis_WildlifeDataset):
    def __init__(self, **kwargs):
        self.sides = {'left': 0, 'right': 1}
        self.sides_cycle = False
        self.names_categories = ['same ind', 'diff ind']
        super().__init__(**kwargs)

class Analysis_SeaTurtleIDHeads(Analysis_WildlifeDataset):
    def __init__(self, **kwargs):
        self.sides = {'left': 0, 'topleft': 1, 'top': 2, 'topright': 3, 'right': 4}
        self.sides_cycle = False
        self.names_categories = ['same ind - same day', 'same ind - same setup', 'same ind - diff setup', 'diff ind - same setup', 'diff ind - diff setup']
        super().__init__(**kwargs)
        
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

class Analysis_SeaTurtleID2022(Analysis_SeaTurtleIDHeads):
    def get_split(self, df, df_old, idx_ignore=None):
        idx_ignore = self.get_split_initialize_idx_ignore(df, idx_ignore=idx_ignore)
        new_identities = set(df['identity']) - set(df_old['identity'])
        idx_query = []
        for i, (_, df_row) in enumerate(df.iterrows()):
            if df_row['identity'] in new_identities and not idx_ignore[i]:
                idx_query.append(i)
        idx_query = np.array(idx_query)
        idx_database = []
        return idx_database, idx_query

class Analysis_StripeSpotter(Analysis_WildlifeDataset):
    def __init__(self, **kwargs):
        self.sides = {'left': 0, 'right': 1}
        self.sides_cycle = False
        self.names_categories = ['same ind', 'diff ind']
        super().__init__(**kwargs)

class Analysis_WhaleSharkID(Analysis_WildlifeDataset):
    def __init__(self, **kwargs):
        self.sides = {'left': 0, 'back': 1, 'right': 2}
        self.sides_cycle = False
        self.names_categories = ['same ind', 'diff ind']
        super().__init__(**kwargs)

class Analysis_ZindiTurtleRecall(Analysis_WildlifeDataset):
    def __init__(self, **kwargs):
        self.sides = {'left': 0, 'top': 1, 'right': 2}
        self.sides_cycle = False
        self.names_categories = ['same ind', 'diff ind']
        super().__init__(**kwargs)
