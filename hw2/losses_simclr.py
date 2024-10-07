import torch
import torch.nn

def calc_self_loss_for_batch(model, xu_BF, 
        temp=1.0, sigma=0.001, trch_prng=None, 
        method='forloop',
        **unlab_loss_kws):
    ''' Compute the SimCLR-style self-supervised loss for our toy dataset.

    Returns
    -------
    loss : torch float
        Loss value, lower is better.
    '''
    B, F = xu_BF.shape

    # Add Gaussian noise to create "left"/"right" variants of each unlabeled
    xleft_BF = xu_BF + sigma * torch.randn(B, F, generator=trch_prng)
    xright_BF = xu_BF + sigma * torch.randn(B, F, generator=trch_prng)

    # Encode the variants as projections on the unit circle (D=2 for toy data)
    zleft_BD = model.encode(xleft_BF, enforce_L2_norm_is_one=True)
    zright_BD = model.encode(xright_BF, enforce_L2_norm_is_one=True)

    if method == 'fast':
        return calc_simclr_loss__fast(zleft_BD, zright_BD, temp)
    else:
        return calc_simclr_loss__forloop(zleft_BD, zright_BD, temp)



def calc_simclr_loss__forloop(z_left_ND, z_right_ND, temp):
    """ Compute contrastive loss (naive version using for loops).

    Args
    ----
    z_left_ND : tensor, shape (N, D)
        Encoding of first augmentation of each instance in current batch.
    z_right_ND : tensor, shape (N, D)
        Encoding of second augmentation of each instance in current batch.
        z_left_ND[i] and z_right_ND[i] form a "positive pair" for all valid i
    temp : float
        Temperature that determins scale of the similarity function

    Returns
    -------
    loss : torch float
        Scalar loss, summed over all positive pairs.
    """
    N = z_left_ND.shape[0] # Num examples in the batch
    M = 2 * N
    # Concatenate left and right into tensor with M=2*N rows
    z_MD = torch.cat([z_left_ND, z_right_ND], dim=0)
    total_loss = 0.0
    for i in range(N):  # loop through each positive pair
        zL_i_1D, zR_i_1D = z_left_ND[i:i+1], z_right_ND[i:i+1]

        # TODO calculate loss(i, i+N)
        # - calculate numerator (ideally, its logarithm for stability)
        # - calculate denominator (again, ideally, its logarithm)
        lossL = torch.sum(z_left_ND) # TODO FIXME

        # TODO calculate loss(i+N, i), as above
        lossR = torch.sum(z_right_ND) # TODO FIXME

        total_loss = total_loss + lossL + lossR
    # Return AVERAGE loss (over all variants in the batch)
    total_loss = total_loss / (2*N)
    return total_loss


def calc_simclr_loss__fast(z_left_ND, z_right_ND, temp):
    """ Compute contrastive loss fast via vectorized code. No for loops.

    Args
    ----
    z_left_ND : tensor, shape (N, D)
        Encoding of first augmentation of each instance in current batch.
    z_right_ND : tensor, shape (N, D)
        Encoding of second augmentation of each instance in current batch.
        z_left_ND[i] and z_right_ND[i] form a "positive pair" for all valid i
    temp : float
        Temperature that determins scale of the similarity function

    Returns
    -------
    loss : torch float
        Scalar loss, summed over all positive pairs.
    """
    N = z_left_ND.shape[0] # Num examples in the batch
    M = 2 * N

    # Concatenate left and right into tensor with M=2*N rows
    z_MD = torch.cat([z_left_ND, z_right_ND], dim=0)

    # Bool mask version of identity matrix, shape (M, M)
    # This might be useful.
    identity_mask_MM = torch.eye(M, dtype=torch.bool, device=z_MD.device)
    
    # Bool mask with one indicator per row, indicating the partner of that row
    # This also might be useful.
    partner_mask_MM = torch.roll(identity_mask_MM, N, dims=0)
    # Should be true in general: partner_mask_MM[i, i+N] == 1 for all i
    assert partner_mask_MM[0, N] == 1
    assert partner_mask_MM[1, N+1] == 1
    assert partner_mask_MM[N, 0] == 1

    total_loss = 0.0 # TODO FIXME
    # TODO calculate loss's (log) numerator for each entry (row) in z_MD
    # TODO calculate loss's (log) denominator for each entry
    # Hint: can use torch.logsumexp, and other utils in torch or torch.nn
    return total_loss
