import numpy as np
import matplotlib.pyplot as plt

import torch

def plot_embeddings_on_unit_circle(model, x_N2, y_N, jitter=0.02):
    z_N2 = model.encode(x_N2).detach().numpy()
    ax = plt.gca()
    ax.plot(z_N2[y_N==0,0], z_N2[y_N==0,1], 'b.')
    ax.plot(z_N2[y_N==1,0] + jitter, z_N2[y_N==1,1] + jitter, 'r.')
    ax.set_aspect('equal');
    ax.set_xlim([-1.05, 1.05]);
    ax.set_ylim([-1.05, 1.05]);

def plot_losses_and_learned_boundary(model, info, xtr_N2, ytr_N):
    _, axgrid = plt.subplots(
        nrows=1, ncols=2, figsize=(8, 4))
    plt.sca(axgrid[1])
    plot_probas_over_dense_grid(model, xtr_N2, ytr_N)
    
    t1_str = "seed={data_order_seed} l2pen_mag={l2pen_mag}".format(**info)
    t2_str = "lr={lr} best_epoch={best_epoch} n_epochs={n_epochs}".format(**info)
    axgrid[1].set_title(t1_str + "\n" + t2_str, fontsize=8)

    ax = axgrid[0]
    ax.plot(info['epochs'], info['tr']['xent'], 'b--', label='tr  xent')
    ax.plot(info['epochs'], info['va']['xent'], 'r--', label='val xent')
    ax.legend(loc='upper right')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    # Fix awk spacing between two panels
    plt.subplots_adjust(wspace=0.3);

    return axgrid

def plot_data_colored_by_labels(x_N2, y_N, msize=5, alpha=1.0):
    if y_N is None:
        plt.plot(x_N2[:,0], x_N2[:,1],
            color='#333333', marker='d', linestyle='', markersize=msize,
            mew=2, alpha=0.4);
    else:
        plt.plot(x_N2[y_N==0,0], x_N2[y_N==0,1],
            color='r', marker='x', linestyle='', markersize=msize, alpha=alpha,
            mew=2, label='y=0');
        plt.plot(x_N2[y_N==1,0], x_N2[y_N==1,1],
            color='b', marker='+', linestyle='', markersize=msize, alpha=alpha,
            mew=2, label='y=1');
    ax = plt.gca()
    ax.set_xticks([-2, -1, 0, 1, 2]);
    ax.set_yticks([-2, -1, 0, 1, 2]);
    ax.set_xlim([-2.05, 2.05]);
    ax.set_ylim([-2.05, 2.05]);
    ax.set_xlabel('x_1'); 
    yticklabels = ax.get_yticklabels()
    if len(yticklabels) > 0:
        ax.set_ylabel('x_2');
    ax.set_aspect('equal');



def plot_probas_over_dense_grid(
        model, x_N2, y_N,
        do_show_colorbar=True,
        x1_ticks=np.asarray([-2, -1, 0, 1, 2]),
        x2_ticks=np.asarray([-2, -1, 0, 1, 2]),
        c_levels=np.linspace(0, 1, 21),
        c_ticks=np.asarray([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        x1_grid=np.linspace(-3, 3, 100),
        x2_grid=np.linspace(-3, 3, 100),
        x1_lims=(-2.05, 2.05),
        x2_lims=(-2.05, 2.05)):
    cur_ax = plt.gca()
    G = x1_grid.size
    H = x2_grid.size
    
    # Get regular grid of G x H points, where each point is an (x1, x2) location
    x1_GH, x2_GH = np.meshgrid(x1_grid, x2_grid)
    
    # Combine the x1 and x2 values into one array
    # Flattened into M = G x H rows
    # Each row of x_M2 is a 2D vector [x_m1, x_m2]
    x_M2 = np.hstack([
        x1_GH.flatten()[:,np.newaxis],
        x2_GH.flatten()[:,np.newaxis]]).astype(np.float32)
        
    # Predict proba for each point in the flattened grid
    with torch.no_grad():
        yproba1_M__pt = model.predict_proba(torch.from_numpy(x_M2))[:,1]
        yproba1_M = yproba1_M__pt.detach().cpu().numpy()
    
    # Reshape the M probas into the GxH 2D field
    yproba1_GH = np.reshape(yproba1_M, x1_GH.shape)
    
    cmap = plt.cm.RdYlBu
    my_contourf_h = plt.contourf(
    	x1_GH, x2_GH, yproba1_GH, levels=c_levels, 
        vmin=0, vmax=1.0, cmap=cmap, alpha=0.5)
    plt.xticks(x1_ticks, x1_ticks);
    plt.yticks(x2_ticks, x2_ticks);
    
    if do_show_colorbar:
        left, bottom, width, height = plt.gca().get_position().bounds
        cax = plt.gcf().add_axes([left+1.03*width, bottom, 0.03, height])
        plt.colorbar(my_contourf_h, orientation='vertical',
            cax=cax, ticks=c_ticks);
        plt.sca(cur_ax);
    plot_data_colored_by_labels(x_N2, y_N);

    #plt.legend(bbox_to_anchor=(1.0, 0.5));
    plt.xlabel('x_1');
    plt.ylabel('x_2');

    plt.gca().set_aspect(1.0);
    plt.gca().set_xlim(x1_lims);
    plt.gca().set_ylim(x2_lims);