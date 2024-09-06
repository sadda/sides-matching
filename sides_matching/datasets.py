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
        data = pd.read_csv(os.path.join(self.root, 'data.csv'))

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

class ZakynthosTurtles(datasets.DatasetFactory):
    def create_catalogue(self) -> pd.DataFrame:
        data = pd.read_csv(os.path.join(self.root, 'annotations.csv'))
        bbox = pd.read_csv(os.path.join(self.root, 'bbox.csv'))
        data = pd.merge(data, bbox, left_on='path', right_on='image_name')

        dates = data['date'].str.split('_')
        dates = dates.apply(lambda x: x[2] + '-' + x[1] + '-' + x[0])
        df = pd.DataFrame({
            'image_id': range(len(data)),
            'path': 'images/' + data['path'],
            'identity': data['identity'],
            'date': dates,
            'orientation': data['orientation'],
            'bbox': data[['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height']].values.tolist()
        })
        return self.finalize_catalogue(df)

