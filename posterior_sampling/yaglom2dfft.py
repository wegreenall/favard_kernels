import numpy as np
import torch
import torch.distributions as D
from np.fft import fft2
from mercergp.kernels import MercerKernel
from ortho.basis_functions import (
    smooth_exponential_eigenvalues_fasshauer,
    Basis,
)
from ortho.builders import get_orthonormal_basis_from_sample

"""
This script will make use of a theorem of Yaglom that states that 
a kernel is positive definite iff it is the inverse fourier transform
of a 2-d density.

k(x, y) = \int F(v, u) cos(xv - yu)dp(v, u)


The function we will use is the numpy implementation of the ifft2. From docs:  
"This function computes the inverse of the 2-dimensional discrete Fourier transform
over any number of axes in an M-dimensional array by means of the FFT. By 
default, the inverse transform is calculated over the last two axes of the input array,
i.e. a 2-d IFFT-

Parameters: a: array_like
            s: sequence of ints; optional; the length of each transformed axis.

"""

# get the kernel
kernel_args = {
    "ard_parameter": torch.Tensor([1.0]),
    "precision_parameter": torch.Tensor([1.0]),
    "noise_parameter": torch.Tensor([1.0]),
}
sample = 
kernel = MercerKernel(order, basis, eigenvalues, kernel_args)
