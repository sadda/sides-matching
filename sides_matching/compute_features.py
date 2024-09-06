import os
from utils import AmvrakikosTurtles, ReunionTurtles, SeaTurtleIDSubset, WD, get_extractor, get_normalized_features, get_transform

model_name = 'MegaDescriptor-L-384'
root_datasets = '/data/wildlife_datasets/data'

dataset_classes = [
    (AmvrakikosTurtles, 'AmvrakikosTurtles'),
    (ReunionTurtles, 'ReunionTurtles'),
    (SeaTurtleIDSubset, 'SeaTurtleIDSubset'),
]

img_size = int(model_name.split('-')[-1])
for dataset_class, dataset_root in dataset_classes:
    for flip in [True, False]:
        for grayscale in [True, False]:
            transform = get_transform(flip=flip, grayscale=grayscale, img_size=img_size, normalize=True)
            root = os.path.join(root_datasets, dataset_root)
            file_name = os.path.join('features', f'features_{dataset_class.__name__}_flip={flip}_grayscale={grayscale}_{model_name}.npy')
            if not os.path.exists(file_name):
                d = dataset_class(root)
                if 'bbox' in d.df:
                    dataset = WD(d.df, d.root, transform=transform, img_load='bbox')
                else:
                    dataset = WD(d.df, d.root, transform=transform)
                extractor = get_extractor(model_name='hf-hub:BVRA/'+model_name, batch_size=32, device='cuda')
                features = get_normalized_features(file_name, dataset, extractor)