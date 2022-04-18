# master_thesis

This is a GitHub project containing the code used in master's thesis Semisupervised Learning of Heterogenous Structured Data.

## Structure

- Folder `src` constains necessary source files. All models and core functions are implemented there.
- Folder `scripts` contains scripts for running and testing. It is divided into subfolders, most importantly `MIProblems` (for running experiments on MIL datasets) and `MNIST` (MNIST point cloud experiments).

## Instalation

This code base is using the Julia Language and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> master_thesis

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box.
