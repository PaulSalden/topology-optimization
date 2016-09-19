# topology-optimization
Python code for MSc thesis

## Method

These scripts implement the Solid Isotropic Material with Penalization (SIMP) method for 2D minimum compliance problems. Isogeometric analysis is used throughout. The code accompanies my MSc thesis, which aims to provide a SIMP implementation more suitable to modern manufacturing techniques such as additive manufacturing (3D printing).

## Optimization scripts

Two optimization scripts are included. Script `opt-continuous.py` allows for optimization using a consistent, continuous density field. It has the option to determine displacements on a finer mesh than density values. Use it as follows:

    python opt-continuous.py <input file>

Script `opt-elementwise.py` implements the more classical piecewise constant densities. Apart from the two meshes approach, it allows for application of a density filter. Use:

    python opt-elementwise.py <input file>

An example of an input file is included.

## Utility scripts

The final compliance value, together with required iterations and run time may be printed for a batch of results using:

    python printcompliances.py <results directory>

A history of compliance values may be plotted using:

    python plotcompliances.py <history file>

Where history files are located in the "hist" directory of optimization results.

Reconstructing plots without doing optimization again is possible through `replot-continuous.py` and `replot-elementwise.py`. Their uses is analogous to the original optimization scripts.

## Python structure

Throughout the scripts, I have attempted to make maximum use of numpy's array capabilities. Various options for NURBS geometry are defined in `geometry.py`. Script `algorithms.py` contains algorithms that have largely been based on work by others. The main optimization scripts also depend on routines that I have constructed myself, found in `routines.py`. Additional functionality is included in `aid.py`, but is unnecessary for typical use.