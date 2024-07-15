import os
import shutil
import numpy as np
import pandas as pd

root_input = '/data/wildlife_datasets/data/SeaTurtleIDSubset_Orig'
root_output = '/data/wildlife_datasets/data/SeaTurtleIDSubset/images'
root_annotation = '/data/wildlife_datasets/data/SeaTurtleIDSubset'
extensions = ['jpg', 'jpeg']

file_names = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(root_input) for f in filenames] 
file_names = [x for x in file_names if x.split('.')[-1].lower() in extensions]
file_names_red = [x.split(os.path.sep)[-1] for x in file_names]
if len(np.unique(file_names_red)) != len(file_names_red):
    raise Exception('File names not unique')

if not os.path.exists(root_output):
    os.makedirs(root_output)
df = []
for (file_name, file_name_red) in zip(file_names, file_names_red):
    file_name_split = file_name.split(os.path.sep)
    image_name = file_name_split[-1]
    if image_name.startswith('left'):
        orientation = 'left'
    elif image_name.startswith('right'):
        orientation = 'right'
    else:
        raise Exception('File name must start with left or right')    
    df.append({
        'identity': file_name_split[-3],
        'path': image_name,
        'orientation': orientation,
        'date': file_name_split[-2]
    })
    file_name_output = os.path.join(root_output, image_name)
    shutil.copy(file_name, file_name_output)
df = pd.DataFrame(df)
df.to_csv(os.path.join(root_annotation, 'annotations.csv'), index=False)