# Smoothed Particle Hydrodynamics (SPH)
by Bruno Celiz & Daniela Couriel - HPC FaMAF UNC 2025

This code was modified from its original version (author Matthias Varnholt)

![animation](SPH_example.gif)

## Why SPH?

The Smooth Particle Hydrodynamics (SPH) is a robust method to tackle the
evolution of a fluid through a Lagrangian approach. Kernels of compact support
and a "nieghbor distance" allows highly adaptative resolution for particles
at very different densities.

### The 'equation of motion'

The equation of motion says that changes in velocity are described as
a function of

- Gravity ($g$).
- A pressure gradient $\Delta P$ (fluid flows from high to low pressure regions).
- Velocity $u ~ \nabla^2 v$
- Viscosity $u$ (the fluid "stickiness")

### The 'mass continuity equation'

$\rho ~ \nabla \cdot v = 0$

where $\rho$ is the density and $\nabla \cdot v$ is the divergence of velocity.

The equation of mass continuity expresses that mass is neither created
nor destroyed.

For the incompressible equation the density is assumed to be constant.
Therefore for our purpose it does not need to be considered. All the
mass we will start with, we will still have at the end.

Since the mass continuity equations just says that we must not create
nor destroy particles while we're computing (i.e. we are assuming a
constant number of particles), this term does not need to be solved.

So back to the equation of motion.

It looks like this:

$\rho \times \left[dv/dt + v \cdot \nabla v \right] = \rho \times g - \nabla p + u\nabla^2 v$

where $\rho$ and $p$ are scalars, but $g$ and $v$ are vectors. $v \cdot \nabla v$ is the
**convective acceleration term** (if fluids have to move through a region of limited space, their
velocity increases); $- \nabla p$ is the **pressure gradient** (negative since fluids go from
a region of high pressure to a region of low pressure); $u\nabla^2 v$ is the **viscosity term**
($\nabla^2$ is the Laplacian operator and acts to diffuse velocity/momentum, resulting in a system
with similar velocities).

The pressure $p$ is defined as follows:
$p = \kappa \times (\rho - \rho_0)$

where $\rho_0$ is a "target density" (at equilibrium), such that
$\rho > \rho_0 \Rightarrow$ positive pressure (and viceversa).


### The material derivative

Is the derivative along a path with a given velocity $v$:
$\rho \times Dv/Dt = \rho \times g - \nabla p + u\nabla^2 v$

$\therefore$ we have a simple equation for the motion of a single particle $i$:

$\frac{dv_i}{dt} = g - \frac{\nabla p}{\rho_i} + \frac{u}{\rho_i} \times \nabla^2 v$

These are the equations we will actually solve.

### From Navier Stokes to Smoothed Particle Hydrodynamics
We can represent any quantity by the summation of nearby points multiplied with a weighting function $W$. $W$ is also called 'smoothing kernel'. They've been introduced by Monaghan in '92.

$W$ will give more strength to points that are close to us
- Points that are further away from us have a weaker influence on us.
- For points away more than a certain distance, $W$ will return 0, i.e. they don't affect us at all. How far a quantity must be away before it stops interacting with us is called the 'interaction radius'

This is evaluating a quantity by sampling a neighborhood of space and weighting points by how close they are to our sampling points.

Monaghan also introduced approximations to the terms of the incompressible Navier-Stokes equation.

## Implementation in SPH
Here are the approximations for the individual terms of the incompressible Navier-Stokes equations:

$\rho_i = \sum_j m_j W(r - r_j, h)$

i.e. the density is approximately equal to the sum of the masses of nearby points, weighted
by the smoothing kernel $W$. We approximate the density at point $i$ by the summation of
neighboring points $j$ (weighted appropriately).

$\frac{\nabla p_i}{\rho_i} = \sum_j m_j \times (\frac{p_i}{\rho^2_i} + \frac{p_j}{\rho^2_j}) \times \nabla W(r - r_j, h)$

The pressure gradient divided by $\rho_i$ is approximately equal to the sums of the masses at various points $j$ multiplied by a scalar quantity of pressure over density. The term is multiplied by the **gradient of a smoothing kernel** ($\nabla W$, vector).

$\frac{u}{\rho_i} \nabla^2 v_i = \frac{u}{\rho_i} \sum_j m_j \times (\frac{v_j - v_i}{\rho_j} \times \nabla^2 W(r - r_j, h)$

$u$ is a scalar coefficient $\propto$ viscosity. The term is multiplied by the Laplacian of the smoothing kernel 4\nabla^2 W$.
If $v_j$ and $v_i$ are equal, there's no viscous interaction between them. Over time this term has the effect of encouraging particles to travel together, in the same direction.

## Smoothing kernels

Attributes of smoothing kernels:
- At a distance $h$, $W$ will drop to 0, i.e. particles that are too far away will not interact with the particle currently processed.
- Over a sphere of radius $h$, $W$ sums to 1.

A typically used 3D kernel with a compact support is given by:

$W = \frac{315}{64 \pi h^9} \times (h^2 - |r - r_0|^2)^3$

And its 1st and 2nd derivatives are:

$\nabla W = \frac{-45}{\pi h^6} \times (h - |r - r_0|)^2 \times \frac{r - r_0}{|r - r_0|}$

$\nabla^2 W = \frac{-45}{\pi h^6} \times (h - |r - r_0|)$

### Solving the equations

Based on the material derivative, we do the following:

1) Compute an approximation for $\rho$ for every particle.

2) Evaluate the pressure gradient. It depends on $\rho$, which needs to be computed first. It also depends on the pressure gradient which is calculated by computing the difference between $\rho$ and $\rho_0$.

3) Evaluate the viscous term. It depends on $\rho$ and on the current velocity of the particle.

4) After having computed all the approximations, we can put them together in the material derivative formula. Using a simple numerical scheme we can timestep the velocity first and then compute the new position of the particle.

## Caveats

Without optimization we'd have to test every particle against every particle ($\mathcal{O}(n^2)$). The result will still be correct because the smoothing kernel $W$ will give 0 for as result for all particles that are outside the interaction radius $h$. Then, we can use:

1) Voxels
   Device space into local regions of space (voxels). Those voxels should have the size $2h$ on a side (twice the interactivity radius; quantities outside this radius are not interacting with the particle).

   So a particle can only interact with particles in the same voxel and in adjacent voxels (2x2x2).

2) Amount of neighbors
   Choose a random subset of $n$ particles (for example 32) because in most cases it's not required to take all the particles into account.
