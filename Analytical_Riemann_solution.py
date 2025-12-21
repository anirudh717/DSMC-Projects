#========================================================================================
# THEORETICAL SHOCK SPEED CALCULATION FOR SHOCKTUBE PROBLEM - RIEMANN ANALYTICAL SOLUTION
#========================================================================================


from scipy.optimize import fsolve
import numpy as np

# initial base states in the upstream and far left regions
P1 = 1.846e5
rho1 = 1.78

P4 = 3.888e6
rho4 = 12.46

gamma = 5/3 # monoatomic

a1 = np.sqrt(gamma * P1 / rho1)
a4 = np.sqrt(gamma * P4 / rho4)

# Shock function 
def f_shock(P, P1, rho1, gamma):
    A = 2 / ((gamma + 1) * rho1)
    B = (gamma - 1) / (gamma + 1) * P1
    return (P - P1) * np.sqrt(A / (P + B))

# Rarefaction function 
def f_rarefaction(P, P4, rho4, gamma):
    term = (P / P4)**((gamma - 1) / (2 * gamma))
    return (2 * a4 / (gamma - 1)) * (term - 1)

# Solve for contact discontinuity f_L + f_R = 0
P_star = fsolve(lambda P: f_shock(P, P1, rho1, gamma) + f_rarefaction(P, P4, rho4, gamma),P4*0.5)[0]

# Calculate Shock Mach No.:
M_s = np.sqrt(((P_star/P1 - 1) * (gamma + 1) / (2 * gamma)) + 1)

print("Shock Mach =", M_s)
print("Shock speed =", M_s * a1)
