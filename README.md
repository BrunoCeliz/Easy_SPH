# Smoothed Particle Hydrodynamics (SPH)
by Bruno Celiz & Daniela Couriel - HPC FaMAF UNC 2025

This code was modified from its original version (author Matthias Varnholt: https://github.com/varnholt/smoothed_particle_hydrodynamics).

We aim to present a flexible and highly optimized SPH code, useful for fast and on-the-fly visual representation of fluids evolving in time. Several modules of the code can be modified to tackle, among multiple escenarios: i) viscous, protoplanetary discs; ii) collapse of a gas cloud with supernova feedback (kinetic) within a static potential; iii) Fluid under a constant gravity within a set boundary.

![animation](SPH_example.gif)

## Recommended Reading:

* "Smoothed Particle Hydrodynamics". Monaghan J.J., 1992, ARA&A, 30, 543

* "High Performance Computing and Numerical Modelling". Volker Springel within the Saas-Fee Advanced Course 43 of the Swiss Society for Astrophysics and Astronomy ("Star Formation in Galaxy Evolution: Connecting Numerical Models to Reality").

* ...and references therein.

## Warnings:

The GUI part of the code highly relies on Qt5. Particularly, check for `apt install qtbase5-dev qtbase5-dev-tools`.