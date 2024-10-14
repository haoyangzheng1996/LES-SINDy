# Laplace-Enhanced SINDy (LES-SINDy)

This is the code repository for Laplace-Enhanced Sparse Identification of Nonlinear Dynamical Systems (LES-SINDy).

More details will be updated later...

![pde_plot](https://github.com/user-attachments/assets/ed64f36a-47de-40f5-a22b-bb3bc60805a5)

# 1. Convection-Diffusion Equation

The Convection-Diffusion equation is a partial differential equation (PDE) that describes the flow of a substance or quantity within a physical system, taking into account both advection (transport due to the flow) and diffusion (spreading due to random motion). The general form of the equation is:

$\frac{\partial C}{\partial t} + u \frac{\partial C}{\partial x} - D \frac{\partial^2 C}{\partial x^2} = 0$,

where:
- $C(x,t)$ is the concentration of the substance.
- $u=1.0$ is the constant velocity of the flow.
- $D=1.0$ is the diffusion coefficient.
- $x$ is the spatial coordinate.
- $t$ is time.


# 2. Burgers' Equation

Burgers' equation is a fundamental PDE that combines elements of nonlinear advection and diffusion. It is used to model various physical processes, such as gas dynamics and traffic flow. The equation is given by:

$\frac{\partial u}{\partial t} + \mu \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0$,

where:
- $\mu=1.0$ is the velocity field.
- $\nu=0.5$ is the kinematic viscosity.
- $x$ is the spatial coordinate.
- $t$ is time.

# 3. Kuramoto–Sivashinsky Equation

The Kuramoto–Sivashinsky equation is a nonlinear PDE used to describe the diffusive instabilities in a laminar flame front, but it also applies to a variety of other physical contexts involving pattern formation and spatiotemporal chaos. The equation is:

$\frac{\partial u}{\partial t} + \frac{\partial^2 u}{\partial x^2} + \frac{\partial^4 u}{\partial x^4} + u \frac{\partial u}{\partial x} = 0$,

where:
- $u(x,t)$ is a scalar field representing the system's state.
- $x$ is the spatial coordinate.
- $t$ is time.
