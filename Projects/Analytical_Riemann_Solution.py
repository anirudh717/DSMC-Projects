# ==============================================================================================================================
#                                    Analytical Riemann Solution for Sod Shock Tube Problem (1D Euler equations)
#-------------------------------------------------------------------------------------------------------------------------------
#                                                                                                 -By: Anirudh Renganathan, 2025
# ==============================================================================================================================

import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace
from scipy.optimize import fsolve

# Geometry and Grid
L=1e-6
x=np.linspace (0,L,5000)
x0=0.5*L

# =========================
# Initial Conditions
# =======================

# Gas properties
gamma = 5/3 # monoatomic
kB = 1.380649e-23    # J/K
m  = 6.63e-26        # kg
R  = kB / m

# Right state (1)
P1 = 1.846e5
rho1 = 1.78
T1 = 500
a1 = np.sqrt(gamma * P1 / rho1)
u1=0

#Left state (4)
P4 = 3.888e6
rho4 = 12.46
T4 = 1500
a4 = np.sqrt(gamma * P4 / rho4)
u4=0

t = 3.43763e-10
# =========================

# =====================================================================
#  INITIAL CONDITIONS
# =====================================================================
rho_init = np.where(x < x0, rho4, rho1)
u_init   = np.where(x < x0, u4,   u1)
P_init   = np.where(x < x0, P4,   P1)
T_init   = np.where(x < x0, T4,   T1)

# =====================================================================
# Riemann Problem - Solution
# =====================================================================
def u3(P):
 A=(2*a4/(gamma-1))*(1-(P/P4)**((gamma-1)/(2*gamma)))
 return A

def u2(P):
 B= (P-P1)*np.sqrt(2/(rho1*(P*(gamma+1))+P1*(gamma-1)))
 return B

def contact_pressure_eqn(Pc):
 return u3(Pc) - u2(Pc)

Pc_guess = 0.5 * (P1 + P4) # Pls note: Most crucial line - everything calculated further is dependent on this guess
# Pc_guess=np.sqrt(P1*P4)  #Alternate

Pc = fsolve(contact_pressure_eqn, Pc_guess)[0]
uc = u3(Pc)

# In region 3: b/w tail wave and contact discontinuity
rho3 = rho4 * (Pc/P4)**(1/gamma)
P3=Pc
T3 = T4*((Pc/P4)**((gamma-1)/gamma))
a3 = np.sqrt(gamma * P3 / rho3)

# In region 2: b/w contact discontinuity and shock
rho2 = rho1*((Pc +(((gamma-1)/(gamma+1))*P1))/(P1+(((gamma-1)/(gamma+1))*Pc)))
P2=Pc
T2 = P2/(rho2*R)
a2= np.sqrt(gamma * P2 / rho2)

#Shock speed:
S_shock = np.sqrt((Pc*(gamma+1)+ (gamma-1)*P1)/(2*rho1))

# Contact discontinuity speed:
S_CD = uc

# EF wave speeds:
S_head = u4-a4  # =-a4
S_tail = uc-a3

# Initialize macrostate variables
rho = np.zeros_like(x)
u   = np.zeros_like(x)
P   = np.zeros_like(x)
T   = np.zeros_like(x)
a   = np.zeros_like(x)

xi = (x - x0) / t

# Masks
mask_4 = xi < S_head
mask_f = (xi >= S_head) & (xi <= S_tail)
mask_3 = (xi > S_tail) & (xi < uc)
mask_2 = (xi >= uc) & (xi < S_shock)
mask_1 = xi >= S_shock

# ========================
# Region assignments
# ========================
rho[mask_4] = rho4
u[mask_4]   = u4
P[mask_4]   = P4
T[mask_4]   = T4
a[mask_4]   = a4

u[mask_f] = (2/(gamma+1)) * (a4 + xi[mask_f])
a[mask_f] = a4 - (gamma-1)/2 * u[mask_f]
rho[mask_f] = rho4 * (a[mask_f]/a4)**(2/(gamma-1))
P[mask_f]   = P4   * (a[mask_f]/a4)**(2*gamma/(gamma-1))
T[mask_f]   = P[mask_f] / (rho[mask_f] * R)

rho[mask_3] = rho3
u[mask_3]   = uc
P[mask_3]   = P3
T[mask_3]   = T3
a[mask_3]   = a3

rho[mask_2] = rho2
u[mask_2]   = uc
P[mask_2]   = P2
T[mask_2]   = T2
a[mask_2]   = a2

rho[mask_1] = rho1
u[mask_1]   = u1
P[mask_1]   = P1
T[mask_1]   = T1
a[mask_1]   = a1

assert np.all(mask_4 | mask_f | mask_3 | mask_2 | mask_1)

# ============================
# PLOTTING: RIEMANN SOLUTION
# ============================

# Non-dimensional x
x_nd = x / L

# ============================
# Styling function
# ============================
def style(ax, ylabel, title=None, ylim=None, labelpad=10):  
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    ax.set_xlabel("x / L", fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold', labelpad=labelpad)  

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.grid(True, linestyle=":", linewidth=1.0)

    ax.tick_params(axis='both', labelsize=11, width=1.5)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    for spine in ax.spines.values():
        spine.set_linewidth(1.8)

# =====================================================================
# PLOTTING INITIAL CONDITIONS
# =====================================================================
fig_ic, axs_ic = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

rho_max = np.max(np.abs(rho_init))
axs_ic[0].plot(x_nd, rho_init, 'b', linewidth=2)
style(axs_ic[0], ylabel="Density (kg/m³)",ylim=(-0.05*rho_max, 1.22*rho_max),title="Initial Analytical Riemann Solution",labelpad=34)

axs_ic[0].text(0.99, 0.92,"Elapsed Time: 0.000e+00 s\nCreated By: Anirudh Renganathan, 2025",transform=axs_ic[0].transAxes,fontsize=10, fontweight='bold',ha='right', va='top',bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))

u_max = np.max(np.abs(u_init)) + 1e-16
axs_ic[1].plot(x_nd, u_init, 'g', linewidth=2)
style(axs_ic[1], ylabel="Velocity u (m/s)",ylim=(-0.2*u_max, 1.22*u_max),labelpad=26)

P_max = np.max(np.abs(P_init))
axs_ic[2].plot(x_nd, P_init, 'k', linewidth=2)
style(axs_ic[2], ylabel="Pressure (Pa)",ylim=(-0.05*P_max, 1.22*P_max),labelpad=41)

T_max = np.max(np.abs(T_init))
axs_ic[3].plot(x_nd, T_init, 'r', linewidth=2)
style(axs_ic[3], ylabel="Temperature (K)",ylim=(0.2*T_max, 1.22*T_max),labelpad=18)

x_ticks = np.arange(0, 1.1, 0.1)
for ax in axs_ic:
    ax.axvline(x0/L, color='gray', linestyle='--', alpha=0.6)
    ax.set_xticks(x_ticks)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

plt.tight_layout()
plt.savefig("initial_conditions.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================
# FINAL RIEMANN SOLUTION PLOT
# ============================
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

rho_max = np.max(np.abs(rho))
axs[0].plot(x_nd, rho, 'b', linewidth=2)
style(axs[0],ylabel="Density (kg/m³)",ylim=(-0.05*rho_max, 1.22*rho_max),title="Analytical Riemann Solution".format(t),labelpad=34)

axs[0].text(0.99, 0.92,f"Elapsed Time: {t:.3e} s\nCreated By: Anirudh Renganathan, 2025",transform=axs[0].transAxes,fontsize=10, fontweight='bold',ha='right', va='top',bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))

u_max = np.max(np.abs(u))
axs[1].plot(x_nd, u, 'g', linewidth=2)
style(axs[1],ylabel="Velocity u (m/s)",ylim=(-0.2*u_max, 1.22*u_max),labelpad=26)

P_max = np.max(np.abs(P))
axs[2].plot(x_nd, P, 'k', linewidth=2)
style(axs[2],ylabel="Pressure (Pa)",ylim=(-0.05*P_max, 1.22*P_max),labelpad=41)

T_max = np.max(np.abs(T))
axs[3].plot(x_nd, T, 'r', linewidth=2)
style(axs[3],ylabel="Temperature (K)",ylim=(0.2*T_max, 1.22*T_max),labelpad=18)

x_head  = (x0 + S_head * t) / L
x_tail  = (x0 + (uc - a3) * t) / L
x_cd    = (x0 + uc * t) / L
x_shock = (x0 + S_shock * t) / L

for ax in axs:
    ax.axvline(x_head,  color='gray', linestyle='--', alpha=0.6)
    ax.axvline(x_tail,  color='gray', linestyle='--', alpha=0.6)
    ax.axvline(x_cd,    color='gray', linestyle='--', alpha=0.6)
    ax.axvline(x_shock, color='gray', linestyle='--', alpha=0.6)
    ax.set_xticks(x_ticks)

plt.tight_layout()
plt.savefig("riemann_solution.png", dpi=300, bbox_inches="tight")
plt.show()

print("Shock speed =", S_shock)
print("Shock Mach number = ", S_shock / a1)
print("Contact discontinuity speed =", S_CD)
print("Head wave speed =", S_head)
print("Tail wave speed =", S_tail)






