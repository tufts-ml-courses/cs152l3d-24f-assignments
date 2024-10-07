import torch
import torch.utils
import numpy as np

def make_expanded_data_loader(
        source_model, tr_loader, unlab_loader,
        threshold_quantile=0.95,
        trch_prng=None,
        verbose=False,
        ):
    # Extract full unlabeled data as tensor
    xu_N2 = unlab_loader.dataset.dataset.x_N2
    # Build pseudo-labeled subset
    xnew_X2, ynew_X = make_pseudolabels_for_most_confident_fraction(
        source_model, xu_N2,
        threshold_quantile=threshold_quantile,
        trch_prng=trch_prng,
        verbose=verbose)

    # Concatenate this new subset with existing train set
    xtr_N2 = tr_loader.dataset.dataset.x_N2[tr_loader.dataset.indices]
    ytr_N = tr_loader.dataset.dataset.targets[tr_loader.dataset.indices]
    expanded_dset = torch.utils.data.TensorDataset(
        torch.cat([xtr_N2, xnew_X2]),
        torch.cat([ytr_N, ynew_X]),
        )
    # Create and return DataLoader
    expanded_loader = torch.utils.data.DataLoader(
        expanded_dset,
        batch_size=tr_loader.batch_size,
        shuffle=True)
    return expanded_loader

def make_pseudolabels_for_most_confident_fraction(
        source_model, xu_N2,
        threshold_quantile=0.5,
        trch_prng=None,
        verbose=False):
    ''' Create pseudolabeled version of provided dataset.

    Obtains pseudolabels and associated confidences for each instance.
    Then, for each label, identifies label-specific threshold for desired
    quantile. Keeps only those instances with confidence above that threshold.

    Returns
    -------
    xnew_XF : torch tensor, shape (X, F)
        Equal to subset of provided xu tensor
    ynew_X : torch tensor, shape (X,)
        Corresponding pseudolabels for each row of xnew_XF
    '''
    N = xu_N2.shape[0]
    with torch.no_grad():
        # TODO FIXME replace this block with actual code for two steps
        # 1) get predicted probabilities from source_model for xu_N2
        # 2) get largest probabilities (=phat_N) and class indicators (=yhat_N)
        yhat_N = torch.rand(N, generator=trch_prng) > 0.5       # TODO FIXME
        phat_N = 0.5 + 0.5 + torch.rand(N, generator=trch_prng) # TODO FIXME

    phat_N = phat_N.detach()
    yhat_N = yhat_N.detach()

    # Report on how unique different predicted probas are across dataset
    if verbose:
        uvals, counts = np.unique(
            phat_N.numpy().round(3),
            return_counts=True)
        U = len(uvals)
        print("Unlabeled set of size %3d maps to %3d unique 3-digit probas" % (
            N, U))
        for ii, (u, c) in enumerate(zip(uvals, counts)):
            if ii < 4:
                print("%3d %s" % (c,u))
            elif ii == 4:
                print("%3d %s" % (c,u))
                print("...")
            elif ii >= U - 4:
                print("%3d %s" % (c,u))
    
    # Find class-specific thresholds
    pthresh0 = torch.quantile(phat_N[yhat_N==0], threshold_quantile)
    pthresh1 = torch.quantile(phat_N[yhat_N==1], threshold_quantile)

    # keepmask0_N is bool indicator of whether both are true
    # * yhat == 0 for example i
    # * predicted probability is above the threshold
    keepmask0_N = torch.logical_and(phat_N >= pthresh0, yhat_N == 0)
    keepmask1_N = torch.logical_and(phat_N >= pthresh1, yhat_N == 1)
    size0 = int((1 - threshold_quantile) * N / 2)
    size1 = int(size0)

    # Break ties, in case many examples are "tied"
    # We only want a subset of the target size, not more than expected.
    if torch.sum(keepmask0_N) > size0:
        ids0 = torch.nonzero(keepmask0_N, as_tuple=True)[0]
        perm = torch.randperm(ids0.size(0), generator=trch_prng)
        ids0 = ids0[perm[:size0]]
        keepmask0_N[:] = 0
        keepmask0_N[ids0] = 1
    if torch.sum(keepmask1_N) > size1:
        ids1 = torch.nonzero(keepmask1_N, as_tuple=True)[0]
        perm = torch.randperm(ids1.size(0), generator=trch_prng)
        ids1 = ids1[perm[:size1]]
        keepmask1_N[:] = 0
        keepmask1_N[ids1] = 1

    # Create new tensors to represent x and y for pseudolabeled subset
    xnew_X2 = xu_N2[torch.logical_or(keepmask0_N, keepmask1_N)]
    ynew_X = yhat_N[torch.logical_or(keepmask0_N, keepmask1_N)]
    return xnew_X2, ynew_X



