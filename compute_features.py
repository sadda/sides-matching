import os
import torchvision.transforms as T

from wildlife_datasets import datasets
from utils import KyparissiaTurtles, KyparissiaTurtles_nobbox, ReunionTurtles, WD, get_extractor, get_normalized_features

model_name = 'MegaDescriptor-L-384'
root_datasets = '/data/wildlife_datasets/data'

dataset_classes = [
    (datasets.HyenaID2022, 'HyenaID2022'),
    (KyparissiaTurtles, 'KyparissiaTurtles'),
    (KyparissiaTurtles_nobbox, 'KyparissiaTurtles'),
    (ReunionTurtles, 'ReunionTurtles'),
    (datasets.LeopardID2022, 'LeopardID2022'),
    (datasets.NyalaData, 'NyalaData'),    
    (datasets.SeaTurtleID2022, 'SeaTurtleID2022'),
    (datasets.SeaTurtleIDHeads, 'SeaTurtleIDHeads'),
    (datasets.StripeSpotter, 'StripeSpotter'),
    (datasets.WhaleSharkID, 'WhaleSharkID'),
    (datasets.ZindiTurtleRecall, 'ZindiTurtleRecall'),
]

img_size = int(model_name.split('-')[-1])
for dataset_class, dataset_root in dataset_classes:
    for flip in [True, False]:
        if flip:
            transform = T.Compose([T.RandomHorizontalFlip(1), T.Resize([img_size, img_size]), T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        else:
            transform = T.Compose([T.Resize([img_size, img_size]), T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

        root = os.path.join(root_datasets, dataset_root)
        file_name = os.path.join('features', f'features_{dataset_class.__name__}_flip={flip}_{model_name}.npy')
        if not os.path.exists(file_name):
            d = dataset_class(root)
            if 'bbox' in d.df:
                dataset = WD(d.df, d.root, transform=transform, img_load='bbox')
            else:
                dataset = WD(d.df, d.root, transform=transform)
            extractor = get_extractor(model_name='hf-hub:BVRA/'+model_name, batch_size=32, device='cuda')
            features = get_normalized_features(file_name, dataset, extractor)