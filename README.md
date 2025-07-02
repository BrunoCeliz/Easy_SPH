# Smoothed Particle Hydrodynamics (SPH)
by Bruno Celiz & Daniela Couriel - HPC FaMAF UNC 2025

This code was modified from its original version (author Matthias Varnholt: https://github.com/varnholt/smoothed_particle_hydrodynamics).

We aim to present a flexible and highly optimized SPH code, useful for fast and on-the-fly visual representation of fluids evolving in time. Several modules of the code can be modified to tackle, among multiple escenarios: i) viscous, protoplanetary discs; ii) collapse of a gas cloud with supernova feedback (kinetic) within a static potential; iii) Fluid under a constant gravity within a set boundary.

![animation](SPH_example.gif)

## Recommended Reading:

* "Smoothed Particle Hydrodynamics". Monaghan J.J., 1992, ARA&A, 30, 543

* "High Performance Computing and Numerical Modelling". Volker Springel within the Saas-Fee Advanced Course 43 of the Swiss Society for Astrophysics and Astronomy ("Star Formation in Galaxy Evolution: Connecting Numerical Models to Reality").

* ...and references therein.

## Instructions:

* Clone this repository `git clone https://github.com/BrunoCeliz/Easy_SPH.git`.

* On the folder, create a "build" directory `mkdir build`.

* Go to the recently created directory `cd build` and point to the CMake file `cmake ..`.

* Compile with Make `make`.

* Run and visualise the default example! `./sph`.

### Fine-tune your preferred conditions:

* On the file `src/sph.cpp` you will find the initial positions & velocities, viscosity, target density, amount of particles and neighbors to search, total time and time step, kinetic kick and density threshold, size, etc.

## Warnings:

The GUI part of the code highly relies on Qt5. Particularly, check for `apt install qtbase5-dev qtbase5-dev-tools`.

### WIP:

Ease the selection of options & parameters. Create a few initial conditions (not only a rotating gas cloud).