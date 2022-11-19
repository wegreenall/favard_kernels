import torch
import torch.distributions as D


"""
This script will attempt to construct the spectral density of our "symmetrised"
Mercer kernel.

The Mercer representation of a given kernel allows for construction of a 
well-fit Gaussian posterior using finite basis functions. However, 
representation of the prior component to a posterior sample fails
due to the degenerate nature of the kernel - the basis functions must eventually
collapse to 0. We aim to remedy this by constructing a random Fourier feature 
map to represent the prior. However, in order to do so, the main tool is to use
Bochner's theorem, which requires that the kernel be stationary.
If the kernel k is stationary, and depends only on the distance between two inputs,
then Bochner's theorem gives:

        k(Δ) = int e^{-2iω}p(ω} dΔ

In general, however, the Mercer kernel is not stationary, and so direct appeal
to Bochner's theorem is not really feasible. However, the Mercer expansion
of the smooth exponential kernel (which is stationary), under a Gaussian input 
measure, can be written with a non-stationary Mercer representation for
a given basis (Zhu, 2007). Feasibly, then, Mercer kernels can represent 
stationary kernels in the limit. This may allow us to build an approximation
to the spectral density of a kernel that, in the limit, we assume would be 
stationary.

If a kernel k(.,.) is stationary, one can write:
         k(x - y, 0) = k(x, y) = k(0, y - x)


Given that the kernel is written:
        k(x, y) = Σ_i λ_i φ_i(x) φ_i(y)

We can calculate the inverse Fourier transform of the symmetrised kernel:
        k(x, y) = Σ_i λ_i φ_i(0) φ_i(Δ)

Note that our formulation means that the basis functions are written:
         φ_i(x)  = c_i P_i(x) exp{-lx^2 / 2}

Given that P_i is an orthogonal polynomial, we can write it as

         P_i(x)  = Σ_j α_ij Η_j(sqrt(l/2) Δ)

Taking the Fourier transform then gives:
    Σ_i λ_i φ_i(0) Σ_j α_ij  int H_i(sqrt(l/2) Δ) e^(-lx^2/2) e^{iωΔ}dΔ
       
adding appropriate constants and a change of variable gives the form in
terms of the Hermite functions, which are eigenfunctions of the Fourier
transform.

As a result it is possible to build the spectral density as long
as one knows the expression of the polynomial P in terms of the
orthogonal Hermite polynomials.

Given that the orthogonal polynomials we construct are monic, this is the
We can actually "convert" them to Hermite polynomials by shifting the three-term
recurrence coefficients.
"""
