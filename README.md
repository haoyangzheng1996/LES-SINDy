# Laplace-Enhanced SINDy (LES-SINDy)

This is the code repository for Laplace-Enhanced SINDy.

# 1. Advection-Diffusion Equation

The Advection-Diffusion equation is a partial differential equation (PDE) that describes the flow of a substance or quantity within a physical system, taking into account both advection (transport due to the flow) and diffusion (spreading due to random motion). The general form of the equation is:

$\frac{\partial C}{\partial t} + u \frac{\partial C}{\partial x} - D \frac{\partial^2 C}{\partial x^2} = 0$,

where:
- $C(x,t)$ is the concentration of the substance.
- $u=1.0$ is the constant velocity of the flow.
- $D=1.0$ is the diffusion coefficient.
- $x$ is the spatial coordinate.
- $t$ is time.

The equation is widely used in fields such as fluid dynamics, environmental engineering, and chemical engineering to model phenomena like pollutant dispersion in the atmosphere, heat conduction, and solute transport in porous media.

# 2. Burgers' Equation

Burgers' equation is a fundamental PDE that combines elements of nonlinear advection and diffusion. It is used to model various physical processes, such as gas dynamics and traffic flow. The equation is given by:

$\frac{\partial u}{\partial t} + \mu \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0$,

where:
- $\mu=1.0$ is the velocity field.
- $\nu=0.5$ is the kinematic viscosity.
- $x$ is the spatial coordinate.
- $t$ is time.

In its inviscid form (when $\nu=0$), Burgers' equation reduces to a simple hyperbolic PDE, which can develop shock waves. With viscosity, it resembles the Navier-Stokes equation in one dimension, making it useful for studying turbulence and shock wave formation.

# 3. Kuramoto–Sivashinsky Equation

The Kuramoto–Sivashinsky equation is a nonlinear PDE used to describe the diffusive instabilities in a laminar flame front, but it also applies to a variety of other physical contexts involving pattern formation and spatiotemporal chaos. The equation is:

$\frac{\partial u}{\partial t} + \frac{\partial^2 u}{\partial x^2} + \frac{\partial^4 u}{\partial x^4} + u \frac{\partial u}{\partial x} = 0$,

where:
- $u(x,t)$ is a scalar field representing the system's state.
- $x$ is the spatial coordinate.
- $t$ is time.

This equation exhibits chaotic behavior and complex spatiotemporal patterns, making it a key model in the study of turbulence, chaotic dynamics, and pattern formation in dissipative systems.
