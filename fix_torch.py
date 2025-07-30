import torch

def fix_torch_classes():
    if hasattr(torch._classes, '__path__'):
        delattr(torch._classes, '__path__')

fix_torch_classes()
