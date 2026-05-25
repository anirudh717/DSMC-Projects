This is a High-fidelity, One-dimensional Three-velocity-component shock tube simulator using DSMC.
   
The code has been adapted from Dr. Garcia's DSMC MATLAB code:
Alejandro Garcia (2025). Numerical Methods for Physics, 2e 
(https://www.mathworks.com/matlabcentral/fileexchange/2270-numerical-methods-for-physics-2e), MATLAB Central File Exchange. Retrieved December 20, 2025.

A probability-based acceptance-rejection method has been employed to stochastically solve for the Boltzmann equation via an N-particle collision.
   
The solution is high-fidelity, which can be confirmed  by comparison with the Analytical Riemann solution of the same shock-tube (Code #2).



Comparison of Mean Velocity Profile - DSMC (top) vs Analytical Solution (bottom)


<img width="4500" height="1800" alt="5) DSMC-velocity profile" src="https://github.com/user-attachments/assets/52b7d069-a61b-4546-9ebe-bca311e67247" />


<img width="1498" height="747" alt="6) Analytical Riemann -Velocity profile" src="https://github.com/user-attachments/assets/98781e7a-f6fe-4678-bbde-6d5096be961b" />




