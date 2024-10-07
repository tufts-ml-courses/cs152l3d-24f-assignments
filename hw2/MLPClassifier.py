import torch
import torch.nn
import numpy as np

from collections import OrderedDict

class MLPClassifier(torch.nn.Module):

    def __init__(self,
            hidden_layer_sizes=[32, 2, 2],
            activation_functions=['relu', 'relu', 'l2normalization'],
            n_input_features=2,
            n_output_classes=2,
            seed=42):
        super().__init__()
        torch.manual_seed(int(seed))        

        layer_list = []
        n_hidden_layers = len(hidden_layer_sizes)
        for layer_id in range(n_hidden_layers):
            if layer_id == 0:
                prev_dim = n_input_features
            else:
                prev_dim = hidden_layer_sizes[layer_id-1]
            layer_list.append((
                f'hidden{layer_id}',
                torch.nn.Linear(prev_dim, hidden_layer_sizes[layer_id])))
            if activation_functions[layer_id] == 'relu':
                act_layer = torch.nn.ReLU()
            elif activation_functions[layer_id] == 'l2normalization':
                act_layer = L2NormalizationLayer(dim=1)
            layer_list.append((f'hidden{layer_id}_act', act_layer))
        self.encoder = torch.nn.Sequential(OrderedDict(layer_list))
        self.output = torch.nn.Linear(hidden_layer_sizes[-1], n_output_classes)

        # Create list of layer names for easy access
        param_names = [n for n,p in self.named_parameters()]
        raw_layer_names = [
            n.replace(".weight","").replace(".bias","")
            for n in param_names]
        uniq_layer_names = OrderedDict(
            zip(raw_layer_names, raw_layer_names))
        self.layer_names = [k for k in uniq_layer_names]
        self.setup_trainable_parameters('all')


    def encode(self, x, enforce_L2_norm_is_one=True):
        return self.encoder(x)
        # Equivalent to forceable normalization
        # z / torch.sqrt(1e-12 + torch.sum(torch.square(z), 1, keepdim=True))

    def forward(self, x):
        return self.output(self.encode(x))

    def predict_proba(self, x):
        logit = self.output(self.encode(x))
        return torch.nn.functional.softmax(logit, dim=1)

    def setup_trainable_parameters(self, n_trainable_layers='all'):
        ''' Set last n layers as trainable, other layers as not.

        Post Condition
        --------------
        - Create self.n_trainable_layers : int
        - Create self.trainable_layer_names : list of strings
        - Create self.trainable_params : dict
        - Modify each param tensor so gradient tracking is enabled/disabled,
          based on its layer name.

        Returns
        -------
        None. Attributes of self modified in-place.
        '''
        if n_trainable_layers == 'all':
            n_trainable_layers = len(self.layer_names)
        self.n_trainable_layers = int(n_trainable_layers)
        # Define last n layers as the trainable ones
        self.trainable_layer_names = self.layer_names[-self.n_trainable_layers:]
        self.trainable_params = dict()
        n_params = 0
        # Iterate over parameters in our current model
        # name : string, like 'output.weight' or 'features.stage1.conv.bias'
        # param : Tensor containing the parameter values
        for name, param in self.named_parameters():
            # Determine if current param is trainable, by asking if its
            # name contains the name of a trainable layer
            is_trainable = sum([name.count(n) for n in self.trainable_layer_names])
            if is_trainable:
                param.requires_grad = True
                self.trainable_params[name] = param
                n_params += np.prod(param.shape)
            else:
                param.requires_grad = False
        msg = "Trainable parameter count=%d over %d tensors in layers: %s." % (
            n_params, len(self.trainable_params),
            ','.join(self.trainable_layer_names))
        print("Setup complete. " + msg)


class L2NormalizationLayer(torch.nn.Module):
    ''' Credit: https://discuss.pytorch.org/t/l2-normalization-layer/176772
    '''

    def __init__(self, dim=1, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=self.dim, eps=self.eps)
