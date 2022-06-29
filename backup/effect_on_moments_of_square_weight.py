import torch
import torch.distributions as D

"""
This file aims to investigate the effect on the maximal entropy density moments
after multiplication by e^(-lx^2). The resulting moments will be smaller than
the original as the new density is everywhere lower than the original.

For this to be truly correct, however, we need to find the way to fix the
maximal entropy moments to be correct... not sure why they are not...
"""
