import numpy as np
import pandas as pd
import os

import torch
import torchvision

from sklearn.model_selection import train_test_split

# Mean/stddev of R/G/B image channels for ImageNet
mean_inet_RGB_3 = [0.485, 0.456, 0.406]
stddev_inet_RGB_3 = [0.229, 0.224, 0.225]


DEFAULT_IM_PREPROCESSING = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),  
    torchvision.transforms.Normalize(
        mean=mean_inet_RGB_3, std=stddev_inet_RGB_3),
])

# For visualization to show humans, don't do the normalization
IM_PREPROCESSING_FOR_VIEW = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),  
])

class BirdsnapDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, 
            transform=None, 
            target_transform=None,
            n_samples_per_class=None):
        super().__init__(root,
            transform=transform, target_transform=target_transform)
        self.n_samples_per_class = n_samples_per_class

        if self.n_samples_per_class is not None:
            self._filter_samples()

    def transform_for_viz(self, x):
        return IM_PREPROCESSING_FOR_VIEW(x)

    def _filter_samples(self):
        class_counts = {}
        filtered_samples = []

        for sample, target in self.samples:
            if target not in class_counts:
                class_counts[target] = 0

            if class_counts[target] < self.n_samples_per_class:
                filtered_samples.append((sample, target))
                class_counts[target] += 1

        self.samples = filtered_samples
        self.targets = [target for _, target in filtered_samples]

def make_birdsnap_data_loaders(
        root=os.path.abspath('.'),
        transform=DEFAULT_IM_PREPROCESSING,
        target_transform=None,
        batch_size=64,
        n_samples_per_class_trainandvalid=50,
        frac_valid=0.2,
        random_state=1234,
        verbose=True):
    birdsnap_dev = BirdsnapDataset(
        os.path.join(root, 'train'), transform=transform,
        n_samples_per_class=n_samples_per_class_trainandvalid)
    birdsnap_test = BirdsnapDataset(
        os.path.join(root, 'test'), transform=transform)

    # Stratified sampling for train and val
    tr_idx, val_idx = train_test_split(np.arange(len(birdsnap_dev)),
                                             test_size=frac_valid,
                                             random_state=random_state,
                                             shuffle=True,
                                             stratify=birdsnap_dev.targets)

    # Create data subsets from indices
    Subset = torch.utils.data.Subset
    tr_set = Subset(birdsnap_dev, tr_idx)
    va_set = Subset(birdsnap_dev, val_idx)
    te_set = Subset(birdsnap_test, np.arange(len(birdsnap_test)))

    if verbose:
        # Print summary of dataset, in terms of counts by class for each split
        def get_y(subset):
            return [subset.dataset.targets[i]
                    for i in subset.indices]
        y_vals = np.unique(np.union1d(get_y(tr_set), get_y(te_set)))
        row_list = list()
        for splitname, dset in [('train', tr_set),
                                ('valid', va_set),
                                ('test', te_set)]:
            y_U, ct_U = np.unique(get_y(dset), return_counts=True)
            y2ct_dict = dict(zip(y_U, ct_U))
            row_dict = dict(splitname=splitname)
            for y in y_vals:
                row_dict[y] = y2ct_dict.get(y, 0)
            row_list.append(row_dict)
        df = pd.DataFrame(row_list)
        print(df.to_string(index=False))

    # Convert to DataLoaders
    DataLoader = torch.utils.data.DataLoader
    tr_loader = DataLoader(tr_set, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va_set, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te_set, batch_size=batch_size, shuffle=False)
    return tr_loader, va_loader, te_loader
