import torch
import numpy as np
import pandas as pd
import os

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class MyCSVDataset(Dataset):
    def __init__(self, filepath_x, filepath_y=None,
                 n_samples_per_class=None):
        xdf = pd.read_csv(filepath_x)
        x_N2 = torch.from_numpy(xdf.values.astype(np.float32))
        self.x_N2 = x_N2

        if filepath_y is None:
            self.samples = self.x_N2
        else:
            ydf = pd.read_csv(filepath_y)
            self.targets = torch.from_numpy(ydf.values[:,0].astype(np.int64))
            self.samples = zip(x_N2, self.targets)
            if n_samples_per_class is not None:
                self.n_samples_per_class = n_samples_per_class
                self._filter_samples()
        
    def __getitem__(self, idx):
        if hasattr(self, 'targets'):
            return self.x_N2[idx, :], self.targets[idx]
        else:
            return self.x_N2[idx, :]

    def __len__(self):
        return self.x_N2.shape[0]

    def _filter_samples(self):
        class_counts = {}
        filtered_samples = []

        for sample, target in self.samples:
            target_int = target.item()
            if target_int not in class_counts:
                class_counts[target_int] = 0
            if class_counts[target_int] < self.n_samples_per_class:
                filtered_samples.append((sample, target))
                class_counts[target_int] += 1
        self.samples = filtered_samples
        self.x_N2 = torch.vstack([s[0] for s in filtered_samples])
        self.targets = torch.hstack([target for _, target in filtered_samples])

def make_moons_data_loaders(
        data_dir=None,
        batch_size=128,
        transform=None,
        target_transform=None,
        n_samples_per_class_trainandvalid=None,
        frac_valid=0.4,
        data_creation_seed=1234,
        verbose=True):
    moons_dev = MyCSVDataset(
        os.path.join(data_dir, 'train_x.csv.gz'),
        os.path.join(data_dir, 'train_y.csv.gz'),
        n_samples_per_class=n_samples_per_class_trainandvalid)
    moons_test = MyCSVDataset(
        os.path.join(data_dir, 'test_x.csv.gz'),
        os.path.join(data_dir, 'test_y.csv.gz'))
    moons_unlab = MyCSVDataset(
        os.path.join(data_dir, 'unlab_x.csv.gz'),
        None)

    # Stratified sampling for train and val
    tr_idx, val_idx = train_test_split(np.arange(len(moons_dev)),
                                             test_size=frac_valid,
                                             random_state=data_creation_seed,
                                             shuffle=True,
                                             stratify=moons_dev.targets)

    # Create data subsets from indices
    Subset = torch.utils.data.Subset
    unlab_set = Subset(moons_unlab, np.arange(len(moons_unlab)))
    tr_set = Subset(moons_dev, tr_idx)
    va_set = Subset(moons_dev, val_idx)
    te_set = Subset(moons_test, np.arange(len(moons_test)))
    if verbose:
        # Print summary of dataset, in terms of counts by class for each split
        def get_y(subset):
            return [subset.dataset.targets[i].item()
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
    tr_loader = DataLoader(tr_set, batch_size=batch_size, shuffle=True)
    unlab_loader = DataLoader(unlab_set, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va_set, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te_set, batch_size=batch_size, shuffle=False)
    return tr_loader, va_loader, te_loader, unlab_loader