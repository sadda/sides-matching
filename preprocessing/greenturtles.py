import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

from wildlife_datasets import datasets
from utils import WD

root_full = '../data/SarahZelvy_Full'
root_cropped = '../data/SarahZelvy'
 
if not os.path.exists(root_cropped):
    d = datasets.SarahZelvy(root_full)
    dataset = WD(d.df, d.root, img_load='bbox', load_label=False)
    for i in tqdm(range(len(dataset))):
        image = dataset[i]
        path = os.path.join(root_cropped, dataset.metadata.iloc[i]["path"])
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        image.save(path)
    for file_name in ['annotations.csv', 'already_downloaded']:
        shutil.copy(os.path.join(root_full, file_name), os.path.join(root_cropped, file_name))

d = datasets.SarahZelvy(root_cropped)
dataset = WD(d.df, d.root, load_label=False)
metadata = pd.read_csv(os.path.join(root_cropped, 'annotations.csv'))
if not 'daytime' in dataset.metadata.columns:
    metadata['daytime'] = ''
    for i in range(len(dataset)):
        means = np.array(dataset[i]).mean(axis=(0,1))
        if means[1] + means[2] <= 100:
            metadata.loc[metadata.index[i], 'daytime'] = 'night'
        else:
            metadata.loc[metadata.index[i], 'daytime'] = 'day'
metadata.to_csv(os.path.join(root_full, 'annotations.csv'))
metadata.to_csv(os.path.join(root_cropped, 'annotations.csv'))

