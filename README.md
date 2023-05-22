## Code for paper: Favard Kernels

The data and experiment code can be found in `datasets/wine_quality` and `datasets/formula1`.

# Dependencies
The current library depends on `mercergp` our sparse Gaussian process library, 
and `ortho`, our orthogonal polynomials manipulation library. Links to these 
are available in the supplementary material that accompanies the paper that 
this library is connected to.
A superset of the dependencies can be found in dependencies.txt, which contains
a dump of pipdeptree on this project.

# Wine data
To run the wine data experiments, use "datasets/wine_quality/wine_quality_analysis_2.py". 
In the code, changed the pretrained and precompared flags to False, and run.

# Formula 1 data
To run the wine data experiments, use "datasets/formula1/analysis_2.py". 
In the code, changed the pretrained and precompared flags to False, and run.

# Eigenvalue consistency
Code for generation of the eigenvalue consistency diagram can be found in 
`./eigenvalue_consistency/`; run the file "eigenvalue_consistency_diagram.py".


# Posterior Sampling
Code for generation of the posterior sampling diagram can be found in 
`./posterior_sampling/`; run the file `paper_example.py`.

# Predictive Density
Code for generation of the posterior sampling diagram can be found in 
`./predictive_density/`; run the file `experiment.py` to generate the data;
then use `analysis.py` to generate the diagram.
