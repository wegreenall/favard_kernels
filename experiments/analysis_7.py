import torch
import torch.distributions as D
import matplotlib.pyplot as plt

"""
This script will analyse the "findings" from the experiemnts in experiment_7.py. 
This calculates:
    -the KL-divergence between the Mercer predictive density under
     Gaussian inputs and the true distribution (i.e. mean + noise) at x = 0, 

    - the KL-divergence between the Mercer predictive density under NON
     Gaussian inputs and the true distribution (i.e. mean + noise) at x = 0.

There is little data at 0 in the non-Gaussian inputs version. 
However it is not clear that evaluating at 0 is the right number. 

Also it will be necessary to read the data carefully as it has been saved as 
text representations of torch.Tensors. This is probably not the best way to do
it.
"""
