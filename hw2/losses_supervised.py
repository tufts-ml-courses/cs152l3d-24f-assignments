import torch.nn
import numpy as np

METHOD_TYPE = 'supervised'

def calc_xent_loss_base2(logits, y, reduction='sum'):
	xent_base_e = torch.nn.functional.cross_entropy(logits, y, reduction=reduction) 
	xent_base_2 = xent_base_e / np.log(2.0)
	return xent_base_2

def calc_labeled_loss_for_batch(logits_BC, y_B):
	xent_loss_per_example = calc_xent_loss_base2(
		logits_BC, y_B, reduction='mean')
	return xent_loss_per_example