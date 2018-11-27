import os
import torch
import torch.nn as nn
import numpy as np
import scipy.misc
import time

# refer https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/14
def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)

def unfreeze_layer(to_train):
    for param in to_train.parameters(): 
        param.requires_grad = True

def _expensive_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if len(x.shape) > 1:
        ps = np.empty(x.shape)
        for i in range(x.shape[0]):
            ps[i,:]  = np.exp(x[i,:])
            ps[i,:] /= np.sum(ps[i,:])
        return ps
    else: 
        return np.exp(x) / np.sum(np.exp(x), axis=0)

def tensor_to_np(x):
    """x is the tensor"""
    y = x.data.cpu().numpy()
    if len(y.shape) > 2:
        if y.shape[1] == 1:
            y = y.reshape(y.shape[0], y.shape[1] * y.shape[2])
        else:
            y = y.reshape(y.shape[0] * y.shape[1], y.shape[2])
    return y
def _debug_layer_output(x, layer_name):

    # Do your print / debug stuff here
    if os.environ.get('debug_layer') == 'True':
        # print("==== input to layer %s of shape %s ===" % (layer_name, str(x.shape)))
        # print(x)
        log_dir = '/data/work/voicezen/wip/ai/deepspeech/pt-app/ds-pt/debug_log'
        fname = os.path.join(log_dir, "%s-%s" % (layer_name, str(int(time.time()))))
        y = tensor_to_np(x)
        y = _expensive_softmax(y)
        img = scipy.misc.toimage(y)
        img.save("%s.png" % fname)
        np.savetxt("%s.txt" % fname, y, fmt = "%2.5f")

def log_tensor(x, identifier):
    _debug_layer_output(x, identifier)

class PrintLayer(nn.Module):
    def __init__(self, layer_name):
        super(PrintLayer, self).__init__()
        self.layer_name = layer_name
    
    def forward(self, x):
        _debug_layer_output(x, self.layer_name)
        return x

def decorate_for_debug(norm_layer, layer_name  = "unknown"):
        original = norm_layer.forward
        def func(x): 
            _debug_layer_output(x, layer_name)
            return original(x)
        norm_layer.forward = func
        return norm_layer

