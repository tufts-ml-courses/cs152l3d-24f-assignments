from typing import OrderedDict
import torch
import numpy as np
from pytorchcv.model_provider import get_model as ptcv_get_model

arch2modelname = {
        'ResNet10': 'resnet10',
        'ResNet26': 'resnet26',
        'MobileNet_x1':'mobilenet_w1',
        'MobileNet_x0.25': 'mobilenet_wd4',
}

class PretrainedResNetForBirdSnap10(torch.nn.Module):
    def __init__(self,
                 src_dataset='ImageNet1k', 
                 arch='ResNet10',
                 model_dir='.',
                 n_trainable_layers=1,
                 n_target_classes=10,
                 seed=42):
        super().__init__()

        assert n_trainable_layers >= 0, "Must provide int value >= 1 for n_trainable_layers"

        if src_dataset.lower().count('imagenet'):
            # Load model as pytorch object
            self.model = ptcv_get_model(arch2modelname[arch], pretrained=True, root=model_dir)
            self.srcdata_name = 'imagenet1000'
        elif src_dataset.lower().count('cub'):
            # Load model as pytorch object
            self.model = ptcv_get_model(arch2modelname[arch] + "_cub", pretrained=True, root=model_dir)
            self.srcdata_name = 'cub200'
        else:
            raise ValueError("Unknown src_dataset: %s" % src_dataset)
        # Replace the last layer with a new one
        torch.manual_seed(int(seed))
        self.model.output = torch.nn.Linear(self.model.output.in_features, n_target_classes)

        # Create list of layer names for easy access
        param_names = [n for n,p in self.model.named_parameters()]
        raw_layer_names = [
            n.replace(".weight","").replace(".bias","")
            for n in param_names]
        uniq_layer_names = OrderedDict(
            zip(raw_layer_names, raw_layer_names))
        self.layer_names = [k for k in uniq_layer_names]
        self.setup_trainable_parameters(n_trainable_layers)

    def setup_trainable_parameters(self, n_trainable_layers):
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
        self.n_trainable_layers = int(n_trainable_layers)
        # Define last n layers as the trainable ones
        self.trainable_layer_names = [n for n in self.layer_names] # TODO FIXME
        self.trainable_params = dict()
        n_params = 0
        # Iterate over parameters in our current model
        # name : string, like 'output.weight' or 'features.stage1.conv.bias'
        # param : Tensor containing the parameter values
        for name, param in self.model.named_parameters():
            # Determine if current param is trainable, by asking if its
            # name contains the name of a trainable layer
            is_trainable = sum([name.count(n) for n in self.trainable_layer_names])
            if is_trainable:
                # TODO FIXME: modify param so it is trainable
                self.trainable_params[name] = param
                n_params += np.prod(param.shape)
            else:
                # TODO FIXME: modify param so it is NOT trainable
                pass # does nothing, just placeholder
        msg = "Trainable parameter count=%d over %d tensors in layers: %s." % (
            n_params, len(self.trainable_params),
            ','.join(self.trainable_layer_names))
        print("Setup complete. " + msg)

    def forward(self,x):
        return self.model(x)
    
    def predict_proba(self, x):
        return torch.nn.functional.softmax(self.forward(x), dim=1)
    
    
