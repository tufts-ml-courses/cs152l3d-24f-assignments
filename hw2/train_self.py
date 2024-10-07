import torch
import sklearn.metrics
import tqdm

import numpy as np


def train_model(model, device, unlab_loader,
                loss_module, unlab_loss_kws={},
                n_epochs=10, lr=0.001, l2pen_mag=0.0, data_order_seed=42,
                do_early_stopping=True,
                n_epochs_without_va_improve_before_early_stop=15,
                ):
    ''' Train model via stochastic gradient descent.

    Assumes provided model's trainable params already set to initial values.

    Returns
    -------
    best_model : PyTorch model
        Model corresponding to epoch with best validation loss (xent)
        seen at any epoch throughout this training run
    info : dict
        Contains history of this training run, for diagnostics/plotting
    '''
    # Make sure data loader shuffling reproducible
    trch_prng = torch.Generator(device=device)
    trch_prng.manual_seed(data_order_seed)
    torch.manual_seed(data_order_seed)      
    torch.cuda.manual_seed(data_order_seed)
    model.to(device)
    
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr)

    # Allocate lists for tracking progress each epoch
    tr_info = {'loss':[]}
    epochs = []

    # Count size of datasets, for adjusting metric values to be per-example
    n_train_examples = float(len(unlab_loader.dataset))
    n_batches = float(len(unlab_loader))

    # Progress bar
    progressbar = tqdm.tqdm(range(n_epochs + 1))
    pbar_info = {}

    # Loop over epochs
    for epoch in progressbar:
        if epoch > 0:
            model.train() # In TRAIN mode
            tr_loss = 0.0  # aggregate total loss
            pbar_info['batch_done'] = 0
            for bb, (x) in enumerate(unlab_loader):
                optimizer.zero_grad()
                x_BF = x.to(device)

                loss = loss_module.calc_self_loss_for_batch(
                    model, x_BF, trch_prng=trch_prng, **unlab_loss_kws)
                loss.backward()
                optimizer.step()
    
                pbar_info['batch_done'] += 1        
                progressbar.set_postfix(pbar_info)
    
                # Increment loss metrics we track for debugging/diagnostics
                tr_loss += loss.item() / n_batches
        else:
            # First epoch (0) doesn't train
            tr_loss = np.nan

        # Update diagnostics and progress bar
        epochs.append(epoch)
        tr_info['loss'].append(tr_loss)
        pbar_info.update({
            "loss": tr_loss,
            })
        progressbar.set_postfix(pbar_info)
    info = {
        'tr':tr_info,
        'epochs': epochs}
    return model, info
