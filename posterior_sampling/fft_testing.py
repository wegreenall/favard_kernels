import torch
from pynufft import NUFFT
import math

from scipy.fft import fft, ifft
import numpy as np
import matplotlib.pyplot as plt
from ortho.measure import MaximalEntropyDensity

# nufft stuff
nufft_obj = NUFFT()


def se_kernel(delta):
    return np.exp(-(delta ** 2))


def laplace_kernel(delta):
    return np.exp(-(np.abs(delta)))


order = 10
medensity = MaximalEntropyDensity(order, torch.zeros(2 * order), torch.ones(2 * order))

N = 1024
# x = torch.linspace(-5, 5, N)
x = torch.distributions.Uniform(-math.pi, math.pi).sample([N, 1])
y = se_kernel(x)
# y = medensity(x).numpy()
torchfft = torch.fft.fft2(y)
y_fft = nufft_obj.forward(y)
plt.plot(x.numpy(), y_fft.real, label="real")
plt.plot(x.numpy(), y_fft.imag, label="imag")
plt.plot(x.numpy(), torchfft.real, label="real")
plt.plot(x.numpy(), torchfft.imag, label="imag")
plt.legend()
# x_numps = x.numpy()
# nufft_obj.plan(x_numps, (256,), (512,), (6,))

plt.plot(x, y)
plt.plot(x, y_fft)
plt.show()
