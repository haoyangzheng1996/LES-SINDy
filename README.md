
![les_sindy_framework](https://github.com/user-attachments/assets/de34c63b-3b47-4fe7-8b8c-2586e10e1001)

# Laplace-Enhanced SINDy (LES-SINDy)

This is the code repository for [Laplace-Enhanced Sparse Identification of Nonlinear Dynamical Systems (LES-SINDy)](https://arxiv.org/abs/2411.01719).



# Introduction
Sparse Identification of Nonlinear Dynamical Systems (SINDy) is a powerful tool for the data-driven discovery of governing equations. However, it encounters challenges when modeling complex dynamical systems involving high-order derivatives or discontinuities, particularly in the presence of noise. These limitations restrict its applicability across various fields in applied mathematics and physics. To mitigate these, we propose Laplace-Enhanced SparSe Identification of Nonlinear Dynamical Systems (LES-SINDy). By transforming time-series measurements from the time domain to the Laplace domain using the Laplace transform and integration by parts, LES-SINDy enables more accurate approximations of derivatives and discontinuous terms. It also effectively handles unbounded growth functions and accumulated numerical errors in the Laplace domain, thereby overcoming challenges in the identification process. The model evaluation process selects the most accurate and parsimonious dynamical systems from multiple candidates. Experimental results across diverse ordinary and partial differential equations show that LES-SINDy achieves superior robustness, accuracy, and parsimony compared to existing methods.


More details will be updated later...

# Prerequisites

Please refer to "environment.yml"

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

Please run:
```python
$ python demo_advection.py
```

# 2. Burgers' Equation

Burgers' equation is a fundamental PDE that combines elements of nonlinear advection and diffusion. It is used to model various physical processes, such as gas dynamics and traffic flow. The equation is given by:

$\frac{\partial u}{\partial t} + \mu \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0$,

where:
- $\mu=1.0$ is the velocity field.
- $\nu=0.5$ is the kinematic viscosity.
- $x$ is the spatial coordinate.
- $t$ is time.
- 
Please run:
```python
$ python demo_burgers.py
```

# 3. Kuramoto–Sivashinsky Equation

The Kuramoto–Sivashinsky equation is a nonlinear PDE used to describe the diffusive instabilities in a laminar flame front, but it also applies to a variety of other physical contexts involving pattern formation and spatiotemporal chaos. The equation is:

$\frac{\partial u}{\partial t} + \frac{\partial^2 u}{\partial x^2} + \frac{\partial^4 u}{\partial x^4} + u \frac{\partial u}{\partial x} = 0$,

where:
- $u(x,t)$ is a scalar field representing the system's state.
- $x$ is the spatial coordinate.
- $t$ is time.
- 
Please run:
```python
$ python demo_ks.py
```

## References:
This repo is built upon [PySINDy](https://github.com/dynamicslab/pysindy)

