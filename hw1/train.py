import torch
import sklearn.metrics
import tqdm

import numpy as np

def train_model(model, device, tr_loader, va_loader,
                n_epochs=10, lr=0.001, l2pen_mag=0.0, data_order_seed=42,
                do_early_stopping=True,
                n_epochs_without_va_improve_before_early_stop=15,
                xent_loss_func=torch.nn.functional.cross_entropy,
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
    # Make sure tr_loader shuffling reproducible
    torch.manual_seed(data_order_seed)      
    torch.cuda.manual_seed(data_order_seed)
    model.to(device)
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr)

    # Allocate lists for tracking progress each epoch
    tr_info = {'xent':[], 'err':[], 'loss':[]}
    va_info = {'xent':[], 'err':[]}
    epochs = []

    # Init vars needed for early stopping
    best_va_loss = float('inf')
    curr_wait = 0 # track epochs we are waiting since last best_va_loss change

    # Count size of datasets, for adjusting metric values to be per-example
    n_train = float(len(tr_loader.dataset))
    n_batch_tr = float(len(tr_loader))
    n_valid = float(len(va_loader.dataset))

    # Progress bar
    progressbar = tqdm.tqdm(range(n_epochs + 1))
    pbar_info = {}

    # Loop over epochs
    for epoch in progressbar:
        if epoch > 0:
            model.train() # In TRAIN mode
            tr_loss = 0.0  # aggregate total loss
            tr_xent = 0.0  # aggregate cross-entropy
            tr_err = 0     # count mistakes on train set
            pbar_info['batch_done'] = 0
            for bb, (x, y_B) in enumerate(tr_loader):
                optimizer.zero_grad()
    
                logits_BC = model(x.to(device))
                loss_xent = torch.tensor((0.4567,), device=device, requires_grad=True) # TODO FIXME 
                # Hint 1: use provided xent_loss_func
                # HInt 2: compute average over examples in current batch

                loss_l2 = torch.tensor((0.0,),  device=device, requires_grad=True) # TODO FIXME 
                # Hint: access weights of last layer in model.trainable_params
                # No need to penalize bias params, those less likely to overfit

                loss = loss_xent + float(l2pen_mag) / n_train * loss_l2
                loss.backward()
                optimizer.step()
    
                pbar_info['batch_done'] += 1        
                progressbar.set_postfix(pbar_info)
    
                # Increment loss metrics we track for debugging/diagnostics
                tr_loss += loss.item() / n_batch_tr
                tr_xent += loss_xent.item() / n_batch_tr
                tr_err += sklearn.metrics.zero_one_loss(
                    logits_BC.argmax(axis=1).detach().cpu().numpy(),
                    y_B, normalize=False)
            tr_err_rate = tr_err / n_train
        else:
            # First epoch (0) doesn't train, just measures initial perf on val
            tr_loss = np.nan
            tr_xent = np.nan
            tr_err_rate = np.nan

        # Track performance on val set
        with torch.no_grad():
            model.eval() # In EVAL mode
            va_xent = 0.0
            va_err = 0
            for xva_B3HW, yva_B in va_loader:
                logits_BC = model(xva_B3HW.to(device))
                va_xent += 0.3 * (1.0 - epoch / n_epochs) # TODO FIXME
                # Hint: Make sure va_ent is per-example average over val set
                # That way, its numerical scale will be same as tr_xent

                va_err += sklearn.metrics.zero_one_loss(
                    logits_BC.argmax(axis=1).detach().cpu().numpy(),
                    yva_B, normalize=False)
            va_err_rate = va_err / n_valid

        # Update diagnostics and progress bar
        epochs.append(epoch)
        tr_info['loss'].append(tr_loss)
        tr_info['xent'].append(tr_xent)
        tr_info['err'].append(tr_err_rate)        
        va_info['xent'].append(va_xent)
        va_info['err'].append(va_err_rate)
        pbar_info.update({
            "tr_xent": tr_xent, "tr_err": tr_err_rate,
            "va_xent": va_xent, "va_err": va_err_rate,
            })
        progressbar.set_postfix(pbar_info)

        # Early stopping logic
        # If loss is dropping, mark current weights as best, save for later
        if va_xent < best_va_loss:
            best_va_loss = va_xent
            best_epoch = epoch
            best_tr_err_rate = tr_err_rate
            best_va_err_rate = va_err_rate
            curr_wait = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            curr_wait += 1
                
        wait_enough = False # TODO FIXME. When should early stopping happen?
        # Hint: Compare the current wait to the provided hyperparameter
        # n_epochs_without_va_improve_before_early_stop

        if do_early_stopping and wait_enough:
            print("Stopped early.")
            break
    print(f"Finished after epoch {epoch}, best epoch={best_epoch}")
    print("best va_xent %.3f" % best_va_loss)
    print("best tr_err %.3f" % best_tr_err_rate)
    print("best va_err %.3f" % best_va_err_rate)

    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    result = {
        'tr':tr_info,
        'va':va_info,
        'best_tr_err': best_tr_err_rate,
        'best_va_err': best_va_err_rate,
        'best_va_loss': best_va_loss,
        'best_epoch': best_epoch,
        'epochs': epochs}
    return model, result