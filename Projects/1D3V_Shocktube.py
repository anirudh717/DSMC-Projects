# ============================================================================================================
#                                      ========================================
#                                      AAE-590 - MOLECULAR GAS DYNAMICS PROJECT
#                                      ========================================
#                 1D3V SHOCK-TUBE SIMULATION : A DSMC ANALOGUE TO THE SOD SHOCK-TUBE RIEMANN PROBLEM
#-------------------------------------------------------------------------------------------------------------
#                                                                                      By: Anirudh Renganathan
#                                                    School of Aeronautics and Astronautics, PURDUE University                                                                                            
#                                                                         Project Advisor: Dr. Alina Alexeenko
#                                                                                                   YEAR: 2025
# ============================================================================================================

# Note: Pls read the READ ME section below and make sure to uncomment lines in the code as needed for visualization
# or saving plot files
#------------------------------------------------------------------------------------------------------
#======================================================================================================
# READ ME :-)
#======================================================================================================
# This code simulates a 1-D 3-V shocktube problem using the Direct Simulation Monte Carlo (DSMC) method.
# Simulation is 1D in physical space and 3D in velocity space (1D3V)
# The Code has been adapted from Dr. Alejandro Garcia's DSMC-equilibrium MATLAB code (cited in README.md).
# MONTE CARLO METHODOLOGY : Probability-based ACCEPTANCE-REJECTION based on relative velocities
# SORTER function sorts particles into cells based on their positions in the shocktube domain
# COLIDER function redistributes kinetic energies and the particles have all 3 components of velocity
# 
#------------------------------------------------------------------------------------------------------
#
#======================================================================================================
# GAS DESCRIPTION: 
#======================================================================================================
# GAS: ARGON (MONO-ATOMIC GAS)
# ASSUMPTIONS: THE GAS IS IN THERMODYNAMIC EQUILIBRIUM INITIALLY IN BOTH REGIONS
#------------------------------------------------------------------------------------------------------
#
#======================================================================================================
# DESCRIPTION OF SIMULATION DOMAIN AND COLLISION MODEL
#======================================================================================================
# SIMULATION DOMAIN: 1D SHOCKTUBE WITH 2 REGIONS of P,T,rho, separated by a Contact Discontinuity 
# BOUNDARY CONDITIONS : HARD-REFLECTING WALL BCs at the left and right edges/ends of the shocktube
# COLLISION  MODEL : HARD SPHERES
#------------------------------------------------------------------------------------------------------
#
#======================================================================================================
# SIMULATION PARAMETERS AND OUTPUTS
#======================================================================================================
# Simulation parameters : No. of cells, No. of particles, No. of timesteps, f-number, collision coeff
# Outputs: Macrostates - (Density, Temperature, Pressure),  Mean Velocity, and Mean Molecular Speed
# TIME STEPPING : Tau - Adaptive every 20 time steps based on maximum particle velocity
#------------------------------------------------------------------------------------------------------
# =====================================================================================================
# ASCII ART FIGURE OF THE 1-D SHOCKTUBE SETUP
# =====================================================================================================

'''                INITIAL CONDITIONS IN THE SHOCKTUBE
                   ===================================

|___________________________________                                  | 
|                                   |          RIGHT REGION           |              
|           LEFT REGION             |          (Low P,T,rho)          |             
|          (High P,T,rho)           |_________________________________|
|                                  C.D.                               |
LW -->Left Wall                                                       RW ---> Right Wall
Hard-Reflection BCs at both the walls

'''
#-------------------------------------------------------------------------------------------------------
#=======================================================================================================

# LOAD ALL DEPENDENCIES
import numpy as np
import pylab as plt
from pdb import set_trace
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import matplotlib
#matplotlib.use("Agg") # Use 'Agg' backend for non-interactive plotting on Supercomputers

# ==============================================================
# SET PHYSICAL CONSTANTS AND GAS PARAMETERS
# ==============================================================

boltz = 1.3806e-23 # Boltzmann constant (m^2 kg s^-2 K^-1)
mass  = 6.63e-26 # Mass of Argon in Kg
diam  = 3.66e-10 # Diameter of Argon Atom/Molecule in meters

# ==============================================================
# SET BASE PRESSURE AND TEMPERATURE FOR LEFT AND RIGHT REGIONS
# ==============================================================

rho_L = 7 * (1.78)          # left density (kg/m^3)
T_L   = 1500                # left temperature (K)

rho_R = 1* (1.78)           # right density (kg/m^3)
T_R   = 500                 # right temperature (K)

#==============================================================
# DEFINE THE SIMULATION DOMAIN AND TOTAL NUMBER OF PARTICLES
#==============================================================
L = 1e-6       # domain length (m)
A = 1            # cross-sectional area (m^2)

# Enter TOTAL number of particles: 
npart = 2000000

# Particle weighing using average density and eff_num
rho_avg = (rho_L + rho_R) / 2 # average density
eff_num = ((rho_avg/mass) * ((L*A)/npart)) #eff_num - effective number of real molecules represented by each simulated particle

# PLOT-flags
initial_plot = True
shock_hist_initial = True
plot_solution_for_animation = True
final_plot=True
final_maxwellian_speed_hist=True
shock_hist=True

# =======================================================================
# INITIALIZE PARTICLE POSITIONS AND VELOCITIES - FIXED FOR THE SHOCK TUBE
# =======================================================================

rng = np.random.default_rng(0) # RNG with fixed seed for reproducibility

# Distribute particles in the 2 halves proportional to the density ratio
density_ratio = rho_L / rho_R 
n_left = int(npart * density_ratio / (density_ratio + 1))  # 7/8 of npart
n_right = npart - n_left  # 1/8 of npart

# Position the particles inside the shocktube according to their regions' density
x_left = (L/2) * rng.random(n_left)           # [0, L/2) - MORE particles
x_right = L/2 + (L/2) * rng.random(n_right)   # [L/2, L) - FEWER particles
x = np.concatenate([x_left, x_right])

# Initialize velocities with zeros
v = np.zeros((npart, 3)) #first, initialize particle velocity (in x,y,z) with zeros (m/s)

# Maxwellian RMS distribution function
def maxwell_v(T):
    return np.sqrt(3 * boltz * T / mass)

# Obtain Maxwellian Molecular RMS Velocities using the function previously defined
v_L = maxwell_v(T_L)  #RMS velocity of left region's particles ---> SAME VALUE FOR ALL LEFT PARTICLES (m/s)
v_R = maxwell_v(T_R)  #RMS velocity of right region's particles ---> SAME VALUE FOR ALL RIGHT PARTICLES (m/s)

# CALCULATE 1-D MAXWELLIAN THERMAL VELOCITY PER COMPONENT FOR LEFT REGION
v_thermal_L = np.sqrt(boltz * T_L / mass)   #sqrt K_B*T/m ---> variance of 1-D Maxwellian distribution

# Obtain Initial ABSOLUTE Velocity for LEFT region from RMS Velocities via Box-Muller Transform
for i in range(3):
    v[:n_left, i] = v_thermal_L * rng.standard_normal(n_left) ## BOX MULLER TRASNFORM FOR 1D MAXWELLIAN THERMAL VELOCITY
    
# CALCULATE 1-D MAXWELLIAN THERMAL VELOCITY PER COMPONENT FOR RIGHT REGION
v_thermal_R = np.sqrt(boltz * T_R / mass)

# Obtain Initial Velocity RIGHT region from RMS Velocities via Box-Muller Transform
for i in range(3):
    v[n_left:, i] = v_thermal_R * rng.standard_normal(n_right) ## BOX MULLER TRASNFORM FOR 1D MAXWELLIAN THERMAL VELOCITY

# Define left_ids and right_ids - aids in calculation and plotting
left_ids = np.arange(n_left)
right_ids = np.arange(n_left, npart)

#Calculate magnitude of absolute velocity for left and right regions
vmag_L = np.sqrt((v[left_ids,0]**2) + (v[left_ids,1]**2) + (v[left_ids,2]**2))
vmag_R = np.sqrt((v[right_ids,0]**2) + (v[right_ids,1]**2) + (v[right_ids,2]**2))

#==============================================================
# PLOT THE INITIAL SPEED DISTRIBUTION (LEFT AND RIGHT REGIONS)
#==============================================================


# Bold spine function
def bold_spines(ax, width=2):
    for spine in ax.spines.values():
     spine.set_linewidth(width)

if initial_plot:
    
    plt.figure(figsize=(12, 6))

    # Define histogram bin centers_hist and edges 
    centers_hist = np.arange(50, 3051, 150)
    step = centers_hist[1] - centers_hist[0]
    edges = np.r_[centers_hist - step/2, centers_hist[-1] + step/2]  #Patch the edges

    ###### FOR LEFT SIDE #####
    ax1 = plt.subplot(1, 2, 1)
    counts_L, edges_L, patches_L = ax1.hist(vmag_L, bins=edges)

    #Color settings
    cmap_L = plt.cm.get_cmap('tab20', len(patches_L))
    for i, p in enumerate(patches_L):
        p.set_facecolor(cmap_L(i))
        p.set_edgecolor('black')
        p.set_label(f"{edges_L[i]:.0f}–{edges_L[i+1]:.0f} m/s")
 
    #plotting
    ax1.set_title(
        f"LEFT Region — High P, $\\boldsymbol{{\\rho}}$, T — {n_left} particles",
        fontsize=12, fontweight='bold'
    )
    ax1.set_xlabel('Speed (m/s)', fontsize=10, fontweight = 'bold')
    ax1.set_ylabel('Particle Count', fontsize=10, fontweight = 'bold')
    ax1.tick_params(axis='both', labelsize=10)
    for label in ax1.get_xticklabels():
     label.set_fontweight('bold')

    for label in ax1.get_yticklabels():
     label.set_fontweight('bold')
    #ax1.legend(fontsize=9)

    bold_spines(ax1)   # <--- APPLY BOLD SPINES

    ####### FOR RIGHT REGION #######
    ax2 = plt.subplot(1, 2, 2)
    counts_R, edges_R, patches_R = ax2.hist(vmag_R, bins=edges)

    #Color settings
    cmap_R = plt.cm.get_cmap('tab20c', len(patches_R))
    for i, p in enumerate(patches_R):
        p.set_facecolor(cmap_R(i))
        p.set_edgecolor('black')
        p.set_label(f"{edges_R[i]:.0f}–{edges_R[i+1]:.0f} m/s")
    
    #plotting
    ax2.set_title(
        f"RIGHT Region — Low P, $\\boldsymbol{{\\rho}}$, T — {n_right} particles",
        fontsize=12, fontweight='bold'
    )
    ax2.set_xlabel('Speed (m/s)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Particle Count', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=10)
    for label in ax2.get_xticklabels():
     label.set_fontweight('bold')

    for label in ax2.get_yticklabels():
     label.set_fontweight('bold')
    #ax2.legend(fontsize=10)

    bold_spines(ax2)   # <--- APPLY BOLD SPINES

    plt.tight_layout()
    plt.savefig("initial_hist.png", dpi=300)
    plt.close()
    #plt.show()
   
#==============================================================
# DSMC Simulation parameters
#==============================================================

ncell = 700 # NUMBER OF CELLS

# Calculate magnitude of absolute velocity to determine maximum - used in time stepping
v_mag = np.sqrt(v[:,0]**2 + v[:,1]**2 + v[:,2]**2)
vmax = np.max(v_mag)
tau = 0.2 * (L/ncell) / vmax ##Time step is defined based on vmax and cell size

vrmax   = 3 * max(v_L, v_R) * np.ones(ncell) #Intialize max relative velocity per cell, calculated from RMS velocities V_L and V_R
selxtra = np.zeros(ncell) #DEFINE CARRY-OVER VARIABLE FOR FRACTIONAL COLLISIONS
coeff   = 0.5 * eff_num * np.pi*(diam**2)*(tau/((L*A)/ncell)) #collision coefficient
coltot  = 0 #total collisions counter

# ==============================================================
# CALCULATE AND PLOT THE INITIAL MACROSTATES 
# ==============================================================

cell_edges = np.linspace(0, L, ncell+1)
centers    = 0.5*(cell_edges[:-1] + cell_edges[1:])

rho_0 = np.zeros(ncell)  # initial density
T0   = np.zeros(ncell)   # initial temperature
Tx_0 = np.zeros(ncell)   # initial temperature in x-direction
Ty_0 = np.zeros(ncell)   # initial temperature in y-direction
Tz_0 = np.zeros(ncell)   # initial temperature in z-direction
P0   = np.zeros(ncell)   # initial pressure
num_density_0 = np.zeros(ncell)  # initial number density
mean_mol_speed_0 = np.zeros(ncell)    # initial mean molecular speed
u_mean_0 = np.zeros(ncell)  # initial mean velocity


for j in range(ncell):
    idx = np.where((x >= cell_edges[j]) & (x < cell_edges[j+1]))[0]
    if len(idx) > 0:
        # calculate initial number density per cell:
        num_density_0[j] = len(idx) * eff_num / (L*A/ncell)

        rho_0[j] = num_density_0[j] * mass

        # calculate initial mean molecular speed magnitude
        v_mag_0 = np.sqrt(v[idx,0]**2 + v[idx,1]**2 + v[idx,2]**2)
        mean_mol_speed_0[j] = np.mean(v_mag_0)  
        
        # calculate initialcomponents of mean velocity per cell- used to obtain thermal velocities for temperature calculation
        u_local_x_0 = np.mean(v[idx,0])
        u_local_y_0 = np.mean(v[idx,1])
        u_local_z_0 = np.mean(v[idx,2])
        
        # calculate initial temperature from all 3 components of thermal velocity
        T0[j] = mass * np.mean((v[idx,0] - u_local_x_0)**2 + (v[idx,1] - u_local_y_0)**2 + (v[idx,2] - u_local_z_0)**2) / (3 * boltz)
        
        Tx_0[j] = mass * np.mean((v[idx,0] - u_local_x_0)**2)/boltz
        Ty_0[j] = mass * np.mean((v[idx,1] - u_local_y_0)**2)/boltz
        Tz_0[j] = mass * np.mean((v[idx,2] - u_local_z_0)**2)/boltz

        # calculate initial pressure
        P0[j] = num_density_0[j] * boltz * T0[j]

        # calculate initial mean velocity
        u_mean_0[j] = np.mean(v[idx,0]) 


######### PLOT THE INITIAL MACROSTATES #########
plt.figure(figsize=(12,11))

xticks = np.arange(0, L + 1e-12, 0.1*L)

#GLOBAL STYLING FUNCTION FOR AXES STYLING
def style_axis(ax, title=None, ylabel=None, xlabel=None):
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold')

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, fontweight='bold', labelpad= 15) 

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, fontweight='bold')

    ax.set_xticks(xticks)

    ax.ticklabel_format(style='plain', axis='x')

    ax.set_xticklabels([f"{x/L:.1f}" for x in xticks]) # Non-dimensionalize x-axis

    ax.tick_params(axis='both', labelsize=8.5)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.grid(True, linestyle=':', linewidth=1.0)

    for spine in ax.spines.values():
        spine.set_linewidth(1.8)

# ================= Density =================
ax1 = plt.subplot(5,1,1)
ax1.plot(centers, rho_0, 'b-', linewidth=1.7)
style_axis(ax1, title="Initial Macrostates", ylabel="Density (kg/m³)", xlabel="x/L")

# ================= Temperature ==============
ax2 = plt.subplot(5,1,2)
ax2.plot(centers, T0, 'r-', linewidth=1.7)
style_axis(ax2, ylabel="Temperature (K)", xlabel="x/L")

# ================= Speed magnitude ==========
ax3 = plt.subplot(5,1,3)
ax3.plot(centers, mean_mol_speed_0, 'm-', linewidth=1.7)
style_axis(ax3, ylabel="Mean Molecular Speed (m/s)", xlabel="x/L")

# ================= Mean Velocity ============
ax4 = plt.subplot(5,1,4)
ax4.plot(centers, u_mean_0, 'g-', linewidth=1.7)
style_axis(ax4, ylabel="Mean Velocity (m/s)", xlabel="x/L")

# ================= Pressure =================
ax5 = plt.subplot(5,1,5)
ax5.plot(centers, P0, 'k-', linewidth=1.7)
style_axis(ax5, ylabel="Pressure (Pa)", xlabel="x/L")

plt.tight_layout()
plt.savefig("initial_macrostates.png", dpi=300)
plt.close()
#plt.show()

#================================================================
# PLOT INITIAL DIRECTIONAL TEMPERATURES Tx, Ty, Tz
#================================================================
plt.figure(figsize = (12,11))

ax1 = plt.subplot(3,1,1)
ax1.plot(centers,Tx_0,'r-', linewidth = 1.7)
style_axis(ax1, title = "Initial Temperatures in x,y,z directions", xlabel = "x/L", ylabel = "Tx (K)")

ax2 = plt.subplot(3,1,2)
ax2.plot(centers,Ty_0,'k-', linewidth = 1.7)
style_axis(ax2,  xlabel = "x/L", ylabel = "Ty (K)")

ax3 = plt.subplot(3,1,3)
ax3.plot(centers,Tz_0,'g-', linewidth = 1.7)
style_axis(ax3, xlabel = "x/L", ylabel = "Tz (K)")

plt.tight_layout()
plt.savefig("initial_temperatures_xyz.png", dpi=300)
plt.close()
#plt.show()


#==========================================================================================
# PLOT INITIAL X-THERMAL VELOCITY DISTRIBUTION IN THE SHOCK REGION - BEFORE SHOCK FORMATION
#==========================================================================================
if shock_hist_initial:

    # ID the shock - region
    shock_idx = np.where((x > 0.7*L) & (x < 0.95*L))[0]    # Note that the shock region is to be visually estimated since there is statistical noise

    if len(shock_idx) > 0:
        # compute thermal velocities v'_x = v_x - u_x
        u_local_x = np.mean(v[shock_idx, 0])
        v_th_x = v[shock_idx, 0] - u_local_x

        # set histogram bins
        vmin = np.min(v_th_x)
        vmax = np.max(v_th_x)
        bins = np.linspace(vmin, vmax, 3000)  

        plt.figure(figsize=(10,6))
        ax = plt.subplot(1,1,1)

        counts, edges, patches = ax.hist(v_th_x, bins=bins, color='red', edgecolor='black')

        ax.set_title("X-Thermal Velocity Distribution in the Shock Region - Before Shock Formation",
                     fontsize=16, fontweight='bold')
        ax.set_xlabel("X-Thermal Velocity (m/s)", fontsize=16, fontweight='bold')
        ax.set_ylabel("Particle Count", fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', labelsize=14)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')    

        bold_spines(ax)

        plt.tight_layout()
        plt.savefig("Initial_Shock_thermal_velocity_hist.png", dpi=121)
        plt.close()
        #plt.show()


# ======================================================================================
# SortData dictionary - intializwe keys andd values used in sorter and colider functions
# ======================================================================================

sortData = {
    "ncell":  ncell,                              #number of cells
    "npart":  npart,                              #number of particles
    "cell_n": np.zeros(ncell, dtype=int),         #number of particles per cell
    "index":  np.zeros(ncell, dtype=int),         #starting index of each cell in Xref
    "Xref":   np.zeros(npart, dtype=int),         #reference array for sorted particle indices
}

# ==============================================================
# sorter function 
# ==============================================================

def sorter(x, L, sD):
    ncell, npart = sD["ncell"], sD["npart"]

    # Compute cell indices for all particles at once
    jx = np.clip((x * ncell / L).astype(int), 0, ncell - 1)

    # Count particles per cell 
    sD["cell_n"][:] = 0
    for ip in range(npart):
        sD["cell_n"][jx[ip]] += 1

    # Compute prefix sum to get starting indices
    m = 0
    for j in range(ncell):
        sD["index"][j] = m
        m += sD["cell_n"][j]

    # Sort particle references by cell
    temp = np.zeros(ncell, dtype=int)
    for i in range(npart):
        j = jx[i]
        k = sD["index"][j] + temp[j]
        sD["Xref"][k] = i
        temp[j] += 1

    return sD

'''
#SIMPLER VERSION OF SORTER FUNCTION USING NUMPY BINCOUT AND ARGSORT

def sorter(x, L, sD):
    
    ncell, npart = sD["ncell"], sD["npart"]
    
    # Compute cell indices for all particles at once
    jx = np.clip((x * ncell / L).astype(int), 0, ncell - 1)
    
    # Count particles per cell 
    sD["cell_n"] = np.bincount(jx, minlength=ncell)
    
    # Compute prefix sum to get starting indices
    sD["index"] = np.concatenate([[0], np.cumsum(sD["cell_n"][:-1])])
    
    # Sort particle references by cell
    sort_indices = np.argsort(jx)
    sD["Xref"] = sort_indices

    return sD
'''

# ==============================================================
# colider function
# ==============================================================

def colider(v, crmax, tau, selxtra, coeff, sD, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    ncell = sD["ncell"]
    col = 0

    for jcell in range(ncell):
        number = int(sD["cell_n"][jcell])
        if number <= 1: continue

        select = coeff * (number**2) * crmax[jcell] + selxtra[jcell]
        nsel = int(np.floor(select))
        selxtra[jcell] = select - nsel

        crm = float(crmax[jcell])
        start = int(sD["index"][jcell])

        for _ in range(nsel):

            k  = int(rng.integers(0, number))
            kk = int((k + rng.integers(1, number)) % number)

            i = int(sD["Xref"][start+k])
            j = int(sD["Xref"][start+kk])

            dv = v[i] - v[j]
            cr = float(np.linalg.norm(dv))
            if cr > crm:
                crm = cr

            if cr / crmax[jcell] > rng.random():

                col += 1

                vcm = 0.5*(v[i] + v[j])
                cos_th = 1.0 - 2.0*rng.random()
                sin_th = np.sqrt(max(0.0, 1 - cos_th*cos_th))
                phi = 2*np.pi*rng.random()

                vrel = np.array([
                    cr*cos_th,
                    cr*sin_th*np.cos(phi),
                    cr*sin_th*np.sin(phi)
                ])

                v[i] = vcm + 0.5*vrel
                v[j] = vcm - 0.5*vrel

        crmax[jcell] = crm

    return v, crmax, selxtra, col

# ============================
# Set Simulation time-stepping
# ============================

nstep = 3800#number of time steps
tphys = 0 #  time counter

#re-seed the RNG for collision process
rng = np.random.default_rng(0)

#VARIABLES FOR ANIMATION
bulk_velocity = np.zeros(ncell)
bulk_velocity_history = []
time_history = []

num_density_cell = np.zeros(ncell)
v_th_cell = np.zeros(ncell)
lambda_cell = np.zeros(ncell)
check_history = []
check_history_2 = []

v_max_old = np.max(np.sqrt(v[:,0]**2 + v[:,1]**2 + v[:,2]**2))
#================================================================================================
# MAIN DSMC TIME-STEPPING LOOP
#================================================================================================
for istep in range(nstep+1):
    print("it: ", istep)

    if istep % 20 == 0:
     v_mag = np.sqrt(v[:,0]**2 + v[:,1]**2 + v[:,2]**2)
     vmax = np.max(v_mag)
     tau = 0.2 * (L/ncell) / vmax

    # Move particles
    x += v[:,0] * tau

###############################
# HARD-REFLECTING WALL BCs
###############################
    # hard-reflecting left wall
    left = np.where(x < 0)[0]
    if len(left) > 0:
        v[left,0] *= -1
        x[left] = -x[left]

    # hard-reflecting right wall
    right = np.where(x > L)[0]
    if len(right) > 0:
        v[right,0] *= -1
        x[right] = 2*L - x[right]

    # Sort 
    sortData = sorter(x, L, sortData)

    # Collide
    v, vrmax, selxtra, col = colider(v, vrmax, tau, selxtra, coeff, sortData, rng)
    
    # Calculate bulk velocity profile for animation
    for j in range(ncell):
     idx = np.where((x >= cell_edges[j]) & (x < cell_edges[j+1]))[0]
     if len(idx) > 0:
        bulk_velocity[j] = np.mean(v[idx, 0])
     else:
        bulk_velocity[j] = 0.0

    #Append bulk velocity profile and time to history lists
    bulk_velocity_history.append(bulk_velocity.copy())
    time_history.append(tphys)

    ###### Recalculate v_max_new every 20 steps and see how many times it exceeds v_max_old by updating check-sum counter #######
    if istep % 20==0:
     v_max_new = np.max(np.sqrt(v[:,0]**2 + v[:,1]**2 + v[:,2]**2))

     if v_max_new > v_max_old:
        check_history_2.append(1)
     else:
        check_history_2.append(0)    


    ########## For each step, calculate t_col and update check-sum counter if tau>t_col at any step ############    
    # for j in range(ncell):

    #  idx = np.where((x >= cell_edges[j]) & (x < cell_edges[j+1]))[0]

    #  if len(idx) > 0:
    #     # number density
    #     num_density_cell[j] = len(idx) * eff_num / (L*A/ncell)

    #     # local mean velocity
    #     ux = np.mean(v[idx,0])
    #     uy = np.mean(v[idx,1])
    #     uz = np.mean(v[idx,2])

    #     # thermal velocities
    #     vx_th = v[idx,0] - ux
    #     vy_th = v[idx,1] - uy
    #     vz_th = v[idx,2] - uz

    #     v_th = np.sqrt(vx_th**2 + vy_th**2 + vz_th**2)
    #     v_th_cell[j] = np.mean(v_th)

    #     # mean free path
    #     lambda_cell[j] = 1.0 / (np.sqrt(2)*np.pi*(diam**2)*num_density_cell[j])

    #  else:
    #   num_density_cell[j] = 0.0
    #   lambda_cell[j] = np.inf
    #   v_th_cell[j] = np.inf

    # # compute all stability conditions
    # tcol_cell = lambda_cell / v_th_cell
    # tcol_cell = np.nan_to_num(tcol_cell, nan=np.inf)   

    # #check if any cell violated tau>t_col condition
    # if np.any(tau > tcol_cell):
    #  check_history.append(1)
    # else:
    #  check_history.append(0)

    
    # ==================================================
    # CALCULATE FINAL MACROSTATES AT THE LAST TIME STEP
    # ==================================================
    if istep == nstep:

        rho_final   = np.zeros(ncell) # initialize final density
        T_final     = np.zeros(ncell) # initialize final temperature
        Tx_final = np.zeros(ncell) # initialize final temperature in x-direction
        Ty_final = np.zeros(ncell) # initialize final temperature in y-direction
        Tz_final = np.zeros(ncell) # initialize final temperature in z-direction
        P_final     = np.zeros(ncell) # initialize final pressure
        mean_mol_speed_final = np.zeros(ncell) # initialize final mean molecular speed
        num_density = np.zeros(ncell) # initialize final number density
        u_mean_final = np.zeros(ncell) # initialize final mean velocity

        for j in range(ncell):
            idx = np.where((x >= cell_edges[j]) & (x < cell_edges[j+1]))[0]
            if len(idx) > 0:
                # calculate number density per cell:
                num_density[j] = len(idx) * eff_num / (L*A/ncell)

                # calculate mean molecular speed per cell:
                v_mag_final = np.sqrt(v[idx,0]**2 + v[idx,1]**2 + v[idx,2]**2)
                mean_mol_speed_final[j] = np.mean(v_mag_final)

                # calculate density per cell:
                rho_final[j] = num_density[j] * mass

                # calculate final components of mean velocity per cell - used to obtain thermal velocities for temperature calculation
                u_local_x_final = np.mean(v[idx,0])
                u_local_y_final = np.mean(v[idx,1])
                u_local_z_final = np.mean(v[idx,2])

                # calculate temperature per cell:
                T_final[j] = mass * np.mean( (v[idx,0] - u_local_x_final)**2 + (v[idx,1] - u_local_y_final)**2 + (v[idx,2] - u_local_z_final)**2) / (3 * boltz)
                
                Tx_final[j] = mass * np.mean((v[idx,0] - u_local_x_final)**2)/boltz
                Ty_final[j] = mass * np.mean((v[idx,1] - u_local_y_final)**2)/boltz
                Tz_final[j] = mass * np.mean((v[idx,2] - u_local_z_final)**2)/boltz

                # calculate per cell:
                P_final[j] = num_density[j] * boltz * T_final[j]

                # calculate mean velocity per cell:
                u_mean_final[j] = np.mean(v[idx,0])

    # Update total collision counter and physical simulation time counter
    coltot += col
    tphys += tau          
#-------------------------------------------------------------------------------------------------------------          
# END OF MAIN TIME-STEPPING LOOP #
# ============================================================================================================
  

# ================================================================
# PLOTTING SETUP FOR ANIMATION OF MEAN VELOCITY PROFILE EVOLUTION#
# ================================================================
if plot_solution_for_animation:
    figwidth = 20
    figheight = 10
    lineWidth = 3
    textFontSize = 30
    gcafontSize = 20

    # Create figure for animation
    fig = plt.figure(0, figsize=(figwidth, figheight))
    ax = fig.add_subplot(1, 1, 1)

    u_line, = ax.plot(centers/L, bulk_velocity_history[0][:], '-g', linewidth=7, label="Mean Velocity u_x")
    ax.set_title(f"Evolution of Mean Velocity in X-Direction inside the Shock Tube", fontsize=16, fontweight='bold')
    ax.set_xlabel("x/L", fontsize=15, fontweight='bold')
    ax.set_ylabel("Mean Velocity in X-Direction (m/s) ", fontsize=15, fontweight='bold')
    ax.xaxis.get_offset_text().set_fontweight('bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(-1e2,7e2)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, linestyle=':', linewidth=1.5)

    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')    
    bold_spines(ax)

    text_overlay = ax.text(0.99, 0.98, "", transform=ax.transAxes,fontsize=10, fontweight='bold',ha='right', va='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))

    # define update function for animation
    def update(frame):
     u_line.set_ydata(bulk_velocity_history[frame][:])
     text_overlay.set_text(f"Elapsed Time: {time_history[frame]:.3e} s\n  Created By: Anirudh Renganathan, 2025")
     return u_line, text_overlay 

    # Create animation
    ani = animation.FuncAnimation(fig, update,frames=len(bulk_velocity_history),interval=0.05, blit=True)
    #writer = animation.PillowWriter(fps=30, metadata={'artist': 'Anirudh Renganathan'}, bitrate=8000)
    #ani.save("bulk_velocity_evolution.gif", writer=writer, dpi=300)
    
    writer = FFMpegWriter(
    fps=120,
    metadata={'artist': 'Anirudh Renganathan'},
    bitrate=12000,
    codec='libx264',
    extra_args=['-pix_fmt', 'yuv420p']
    )
    #ani.save("bulk_velocity_fast_evolution.mp4", writer=writer, dpi=300)

   
    print("saved")
    plt.show()
    #plt.close()
    
    
'''
#-----------------------------------------------------------------------------
#=============================================================================
# QUANTITATIVE ANALYSIS OF SHOCKWAVE, CONTACT DISCONTINUITY, AND EXPANSION FAN
#=============================================================================
#-----------------------------------------------------------------------------

# =================================================================
# CALCULATION OF SHOCKSPEED AND SHOCK THICKNESS:
# =================================================================

# Upstream region (unshocked, right side: 0.95L to 1.0L)
pre_indices = np.where((centers >= 0.95*L) & (centers <= 1.0*L))[0]  # Note that the upstream region is to be visually estimated since there is statistical noise


# Downstream region (shocked plateau: 0.6L to 0.7L)
post_indices = np.where((centers >= 0.6*L) & (centers <= 0.7*L))[0]  # Note that the downstream region is to be visually estimated since there is statistical noise


# Upstream properties (just average over the selected cells)
rho_pre = np.mean(rho_final[pre_indices])
P_pre = np.mean(P_final[pre_indices])
T_pre = np.mean(T_final[pre_indices])
u_mean_pre = np.mean(u_mean_final[pre_indices])  # NO abs()!
mean_mol_speed_pre = np.mean(mean_mol_speed_final[pre_indices])

# Downstream properties (just average over the selected cells)
rho_post = np.mean(rho_final[post_indices])
P_post = np.mean(P_final[post_indices])
T_post = np.mean(T_final[post_indices])
u_mean_post = np.mean(u_mean_final[post_indices])  # NO abs()!
mean_mol_speed_post = np.mean(mean_mol_speed_final[post_indices])

# Calculate shock speed using Rankine-Hugoniot relations
shock_speed  = ((rho_post * u_mean_post) - (rho_pre * u_mean_pre))/(rho_post - rho_pre)

# Calculate Mach number of shock wave
gamma = 5/3  
a_pre = np.sqrt(gamma * P_pre / rho_pre)  # speed of sound upstream 
M_shock = shock_speed / a_pre # Shock Mach number

#calculate speed of sound in the downstream CD region (shock-shot region)
a_post = np.sqrt(gamma * P_post / rho_post)

# Calculate upstream and downstream number densities
n_pre = rho_pre / mass
n_post = rho_post / mass

# Calculate upstream and downstream number mean free paths
lambda_pre = 1 / (np.sqrt(2) * np.pi * (diam**2) * n_pre)
lambda_post = 1 / (np.sqrt(2) * np.pi * (diam**2) * n_post)

# Calculate Shock thickness in m (estimated approximately from graphs)
shock_thickness = 0.25*L  

#Normalize shock thickness with upstream and downstream mean free paths
normalised_shock_thickness_1 =  shock_thickness/lambda_pre
normalised_shock_thickness_2 =  shock_thickness/lambda_post
# =================================================================
# CALCULATION OF CONTACT DISCONTINUITY SPEED 
# =================================================================

#Note: Speed of contact discontinuity is equal to the mean velocity in the post-shock C-D region
contact_discontinuity_speed = u_mean_post 

# =================================================================
# CALCULATION OF EXPANSION FAN SPEED AND THICKNESS:
# =================================================================

#Calculate the mean density and pressure in the head region of the Expansion Fan (0.1L to 0.15L)
head_region_indices = np.where((centers >= 0*L) & (centers <= 0.15*L))[0]

rho_head_region = np.mean(rho_final[head_region_indices])
P_head_region = np.mean(P_final[head_region_indices])

# Calculate speed of sound in head region of Expansion Fan
a_head_region = np.sqrt(gamma * P_head_region / rho_head_region) 
u_head_region = np.mean(u_mean_final[head_region_indices])  

#calculate head-wave speed and tail-wave speed for the Expansion Fan
head_wave_speed = u_head_region-a_head_region
tail_wave_speed = u_mean_post-a_post 

#calculate expansion fan thickness
expansion_fan_thickness = 0.45*L


#----------------------------------------------------------------------------------------------
#==============================================================================================
# PLOTS OF FINAL VARIABLES/PROPERTIES:
#==============================================================================================
#----------------------------------------------------------------------------------------------

# ==========================================================================
# TEXT PLOT - PROPERTIES OF SHOCKWAVE, EXPANSION FAN, CONTACT DISCONTINUITY
# ==========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12)) 
ax1.axis('off')
ax2.axis('off')

# ---------------------------------------------------------------------
# 2-column formatting function (LABEL | VALUE)
# ---------------------------------------------------------------------
def format_line(label, value, label_width=42, value_width=35):
    return f"{label:<{label_width}}{value:<{value_width}}"

# ---------------------------------------------------------------------
# Line list -1 and Line list -2 - to store TEXT contents
# ---------------------------------------------------------------------
line_1 = [
   
    "=" * 85,
    "SHOCK KINEMATICS:",
    "=" * 85,
    format_line("Shock Speed:",              f"= {shock_speed:.5f} m/s"),
    format_line("Shock Mach Number:",        f"= {M_shock:.5f}"),
    "",

    "=" * 85,
    "UPSTREAM STATE (Unshocked Region):",
    "=" * 85,
    format_line("Density:",                  f"= {rho_pre:.5f} kg/m³"),
    format_line("Velocity:",                 f"= {u_mean_pre:.5f} m/s"),
    format_line("Pressure:",                 f"= {P_pre:.5f} Pa"),
    format_line("Temperature:",              f"= {T_pre:.5f} K"),
    format_line("Mean Molecular Speed:",     f"= {mean_mol_speed_pre:.5f} m/s"),
    format_line("Number Density:",           f"= {n_pre:.5e} m⁻³"),
    format_line("Mean Free Path:",           f"= {lambda_pre:.5e} m"),
    format_line("Speed of Sound (upstream):",f"= {a_pre:.5f} m/s"),
    "",

    "=" * 85,
    "DOWNSTREAM STATE (Shock-shot region):",
    "=" * 85,
    format_line("Density:",                  f"= {rho_post:.5f} kg/m³"),
    format_line("Velocity:",                 f"= {u_mean_post:.5f} m/s"),
    format_line("Pressure:",                 f"= {P_post:.5f} Pa"),
    format_line("Temperature:",              f"= {T_post:.5f} K"),
    format_line("Mean Molecular Speed:",     f"= {mean_mol_speed_post:.5f} m/s"),
    format_line("Number Density:",           f"= {n_post:.5e} m⁻³"),
    format_line("Mean Free Path:",           f"= {lambda_post:.5e} m"),
    format_line("Speed of Sound (downstream):",f"= {a_post:.5f} m/s"),
    "",

    "=" * 85,
    "JUMP RATIOS ACROSS SHOCK:",
    "=" * 85,
    format_line("Density Ratio:",    f"= {rho_post/rho_pre:.5f}"),
    format_line("Pressure Ratio:",   f"= {P_post/P_pre:.5f}"),
    format_line("Temperature Ratio:",f"= {T_post/T_pre:.5f}"),
    format_line("Velocity Ratio:",   f"= {u_mean_post/u_mean_pre if u_mean_pre!=0 else 0:.5f}"),
    "",

    "=" * 85,
    "SHOCK STRUCTURE:",
    "=" * 85,
    format_line("Shock span:",          f"= {shock_thickness:.5e} m"),
    format_line("Normalized Thickness 1:",   f"= {normalised_shock_thickness_1:.5e}"),
    format_line("Normalized Thickness 2:",   f"= {normalised_shock_thickness_2:.5e}"),
    "",

    "=" * 85,
    "SIMULATION PARAMETERS:",
    "=" * 85,
    format_line("Simulation Time:",          f"= {tphys:.5e} s"),
    format_line("Total Collisions:",         f"= {coltot}"),
    format_line("Number of Cells:",          f"= {ncell}"),
    format_line("Cell Size:",                f"= {L/ncell:.5e} m"),
    format_line("Total Particles:",          f"= {npart}"),
    #format_line("Check sum (epsilon):",      f"= {np.sum(check_history)}"),
    format_line("Check sum - 2 (Zeta):",      f"= {np.sum(check_history_2)}"),
    "=" * 85,
]


line_2 = [
   
    "=" * 85,
    "EXPANSION FAN KINEMATICS:",
    "=" * 85,
    format_line("Head-Wave Speed:",              f"= {head_wave_speed:.5f} m/s"),
    format_line("Tail-Wave Speed:",              f"= {tail_wave_speed:.5f} m/s"),
    format_line("Expansion Fan Thickness:",      f"= {expansion_fan_thickness:.5e} m"),
    format_line("Speed of Sound (CD region):",   f"= {a_post:.5f} m/s"),
    "",

    "=" * 85,
    "CONTACT DISCONTINUITY KINEMATICS:",
    "=" * 85,
    format_line("Contact Discontinuity Speed:",      f"= {contact_discontinuity_speed:.5f} m/s"),
    "=" * 85,
]

# Convert the "line" list into a single block of monospaced text
text_content_1 = "\n".join(line_1)
text_content_2 = "\n".join(line_2)

# SUBPLOT-1 - line_content_1
ax1.text(0.05, 0.95, text_content_1,
        transform=ax1.transAxes,
        fontsize=9.5,
        verticalalignment='top',
        horizontalalignment='left',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

# SUBPLOT-2 - line_content_2
ax2.text(0.05, 0.95, text_content_2,
        transform=ax2.transAxes,
        fontsize=9.5,
        verticalalignment='top',
        horizontalalignment='left',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

plt.tight_layout()
plt.savefig("shock_properties_table.png", dpi=300, bbox_inches='tight')
#plt.show()
plt.close()

# =====================================================
# PLOT OF FINAL MACROSTATES (ALL IN ONE PLOT)
# =====================================================

plt.figure(figsize=(12,11))

# ================= Density =================
ax1 = plt.subplot(5,1,1)
ax1.plot(centers, rho_final, 'b-', linewidth=1.7)
style_axis(ax1, title=f"Final Macrostates at T={tphys:.3e}s",  ylabel="Density (kg/m³)", xlabel="x/L")

# ================= Temperature ==============
ax2 = plt.subplot(5,1,2)
ax2.plot(centers, T_final, 'r-', linewidth=1.7)
style_axis(ax2, ylabel="Temperature T (K)", xlabel="x/L")

# ================= Speed magnitude ==========
ax3 = plt.subplot(5,1,3)
ax3.plot(centers, mean_mol_speed_final, 'm-', linewidth=1.7)
style_axis(ax3, ylabel="Mean Molecular speed (m/s)", xlabel="x/L")

# ================= Mean Velocity ============
ax4 = plt.subplot(5,1,4)
ax4.plot(centers, u_mean_final, 'g-', linewidth=1.7)
style_axis(ax4, ylabel="Mean Velocity (m/s)", xlabel="x/L")

# ================= Pressure =================
ax5 = plt.subplot(5,1,5)
ax5.plot(centers, P_final, 'k-', linewidth=1.7)
style_axis(ax5, ylabel="Pressure (Pa)", xlabel="x/L")

plt.tight_layout()
plt.savefig("final_macrostates.png", dpi=300)
#plt.show()
plt.close()


# ==================================================================
# Plots of Final directional temperatures (Tx, Ty, Tz) in one figure
# ==================================================================

plt.figure(figsize = (12,11))

ax1 = plt.subplot(3,1,1)
ax1.plot(centers, Tx_final, 'r-', linewidth = 1.7)
style_axis(ax1, title = f"Final Directional Temperatures at T={tphys:.3e}s", xlabel = "x/L", ylabel = "Tx (K)")

ax2= plt.subplot(3,1,2)
ax2.plot(centers, Ty_final, 'k-', linewidth = 1.7)
style_axis(ax2, xlabel = "x/L", ylabel = "Ty (K)")

ax3= plt.subplot(3,1,3)
ax3.plot(centers, Tz_final, 'g-', linewidth = 1.7)
style_axis(ax3, xlabel = "x/L", ylabel = "Tz (K)")

plt.tight_layout()
plt.savefig("final_temperatures_xyz.png", dpi=300)
plt.close()
#plt.show()

# ===============================================================
#  PLOT OF FINAL MACROSTATES (1-PLOT per figure VERSION)
# ===============================================================
# ========== 1. Density ========================================
plt.figure(figsize=(15,6))
ax = plt.subplot(1,1,1)
ax.plot(centers, rho_final, 'b-', linewidth=1.7)
style_axis(ax, title="Density", ylabel="Density (kg/m³)", xlabel="x/L")
plt.tight_layout()
#plt.show()
plt.savefig("density.png", dpi=300)
plt.close()

# ========== 2. Temperature ====================================
plt.figure(figsize=(15,6))
ax = plt.subplot(1,1,1)
ax.plot(centers, T_final, 'r-', linewidth=1.7)
style_axis(ax, title="Temperature", ylabel="Temperature (K)", xlabel="x/L")
plt.tight_layout()
#plt.show()
plt.savefig("temperature.png", dpi=300)
plt.close()

# ========== 2A. Temperature-Tx ====================================
plt.figure(figsize=(15,6))
ax = plt.subplot(1,1,1)
ax.plot(centers, Tx_final, 'r-', linewidth=1.7)
style_axis(ax, title="Temperature - Tx", ylabel="Tx (K)", xlabel="x/L")
plt.tight_layout()
#plt.show()
plt.savefig("temperature_Tx.png", dpi=300)
plt.close()

# ========== 2B. Temperature-Ty ====================================
plt.figure(figsize=(15,6))
ax = plt.subplot(1,1,1)
ax.plot(centers, Ty_final, 'r-', linewidth=1.7)
style_axis(ax, title="Temperature - Ty", ylabel="Ty (K)", xlabel="x/L")
plt.tight_layout()
#plt.show()
plt.savefig("temperature_Ty.png", dpi=300)
plt.close()

# ========== 2C. Temperature-Tz ====================================
plt.figure(figsize=(15,6))
ax = plt.subplot(1,1,1)
ax.plot(centers, Tz_final, 'r-', linewidth=1.7)
style_axis(ax, title="Temperature -  Tz", ylabel="Tz (K)", xlabel="x/L")
plt.tight_layout()
#plt.show()
plt.savefig("temperature_Tz.png", dpi=300)
plt.close()

# ========== 3. Mean Molecular Speed ===========================
plt.figure(figsize=(15,6))
ax = plt.subplot(1,1,1)
ax.plot(centers, mean_mol_speed_final, 'm-', linewidth=1.7)
style_axis(ax, title="Mean Molecular Speed", ylabel="Mean Molecular Speed(m/s)", xlabel="x/L")
plt.tight_layout()
#plt.show()
plt.savefig("mean_molecular_speed.png", dpi=300)
plt.close()

# ========== 4. Mean Velocity (Bulk velocity in x) ==============
plt.figure(figsize=(15,6))
ax = plt.subplot(1,1,1)
ax.plot(centers, u_mean_final, 'g-', linewidth=1.7)
style_axis(ax, title="Mean Velocity", ylabel="Mean velocity u_x (m/s)", xlabel="x/L")
plt.tight_layout()
#plt.show()
plt.savefig("mean_velocity.png", dpi=300)
plt.close()

# ========== 5. Pressure ========================================
plt.figure(figsize=(15,6))
ax = plt.subplot(1,1,1)
ax.plot(centers, P_final, 'k-', linewidth=1.7)
style_axis(ax, title="Pressure", ylabel="Pressure (Pa)", xlabel="x/L")
plt.tight_layout()
#plt.show()
plt.savefig("pressure.png", dpi=300)
plt.close()

# ====================================================================
# PLOT THE FINAL SPEED DISTRIBUTION HISTOGRAM (LEFT AND RIGHT REGIONS)
# ====================================================================
if final_plot:

    # Calculate indices for particles in LEFT and RIGHT regions based on FINAL positions
    final_left_ids  = np.where(x < L/2)[0]
    final_right_ids = np.where(x >= L/2)[0]

    # re-calculate the magnitude of absolute velocity for final LEFT and RIGHT regions
    vmag_L_final = np.sqrt(v[final_left_ids,0]**2 + v[final_left_ids,1]**2 + v[final_left_ids,2]**2)
    vmag_R_final = np.sqrt(v[final_right_ids,0]**2 + v[final_right_ids,1]**2 + v[final_right_ids,2]**2)

    plt.figure(figsize=(12, 6))
    
    #calculate histogram edges, bin centers_hist
    centers_hist = np.arange(50, 3051, 150)
    step = centers_hist[1] - centers_hist[0]
    edges = np.r_[centers_hist - step/2, centers_hist[-1] + step/2]  #patch the edges

    ######## LEFT REGION ########
    ax1 = plt.subplot(1, 2, 1)
    counts_L, edges_L, patches_L = ax1.hist(vmag_L_final, bins=edges)
    
    #Color settings
    cmap_L = plt.cm.get_cmap('tab20', len(patches_L))
    for i, p in enumerate(patches_L):
        p.set_facecolor(cmap_L(i))
        p.set_edgecolor('black')

    #plotting
    ax1.set_title(f"FINAL LEFT Region — {len(final_left_ids)} particles",
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Speed (m/s)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Particle Count', fontsize=10, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=10)
    for label in ax1.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')
        #ax1.legend(fontsize=10)

    bold_spines(ax1)  # <--- Apply bold spines

    ######### RIGHT REGION ########
    ax2 = plt.subplot(1, 2, 2)
    counts_R, edges_R, patches_R = ax2.hist(vmag_R_final, bins=edges)
    
    #Color settings
    cmap_R = plt.cm.get_cmap('tab20c', len(patches_R))
    for i, p in enumerate(patches_R):
        p.set_facecolor(cmap_R(i))
        p.set_edgecolor('black')
    
    #plotting
    ax2.set_title(f"FINAL RIGHT Region — {len(final_right_ids)} particles",
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Speed (m/s)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Particle Count', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=10)
    for label in ax2.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax2.get_yticklabels():
        label.set_fontweight('bold')
        #ax2.legend(fontsize=10)

    bold_spines(ax2)  # <--- Apply bold spines

    plt.tight_layout()
    plt.savefig("final_hist.png", dpi=300)
    #plt.show()
    plt.close()


# ========================================================================================
# Same histogram of speed distribution but with fewer bins to view the Maxwellian shape
# ========================================================================================
if final_maxwellian_speed_hist:

    plt.figure(figsize=(12, 6))
    
    #calculate histogram edges, bin centers_hist
    centers_hist = np.arange(50, 3051, 5)
    step = centers_hist[1] - centers_hist[0]
    edges = np.r_[centers_hist - step/2, centers_hist[-1] + step/2]  #patch the edges

    ######## LEFT REGION ########
    ax1 = plt.subplot(1, 2, 1)
    counts_L, edges_L, patches_L = ax1.hist(vmag_L_final, bins=edges)
    
    #Color settings
    cmap_L = plt.cm.get_cmap('tab20', len(patches_L))
    for i, p in enumerate(patches_L):
        p.set_facecolor(cmap_L(i))
        p.set_edgecolor('black')

    #plotting
    ax1.set_title(f"FINAL LEFT Region — {len(final_left_ids)} particles",
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Speed (m/s)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Particle Count', fontsize=10, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=10)
    for label in ax1.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')
        #ax1.legend(fontsize=10)

    bold_spines(ax1)  # <--- Apply bold spines

    ######### RIGHT REGION ########
    ax2 = plt.subplot(1, 2, 2)
    counts_R, edges_R, patches_R = ax2.hist(vmag_R_final, bins=edges)
    
    #Color settings
    cmap_R = plt.cm.get_cmap('tab20c', len(patches_R))
    for i, p in enumerate(patches_R):
        p.set_facecolor(cmap_R(i))
        p.set_edgecolor('black')
    
    #plotting
    ax2.set_title(f"FINAL RIGHT Region — {len(final_right_ids)} particles",
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Speed (m/s)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Particle Count', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=10)
    for label in ax2.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax2.get_yticklabels():
        label.set_fontweight('bold')
        #ax2.legend(fontsize=10)

    bold_spines(ax2)  # <--- Apply bold spines

    plt.tight_layout()
    plt.savefig("final_maxwellian_speed_hist.png", dpi=300)
    #plt.show()
    plt.close()    

# ==================================================================
# Plot the histogram of final x-thermal velocity in the shock region
# ==================================================================
if shock_hist:

    # ID the shock - region
    shock_idx = np.where((x > 0.7*L) & (x < 0.95*L))[0]

    if len(shock_idx) > 0:
        # compute thermal velocities v'_x = v_x - u_x
        u_local_x = np.mean(v[shock_idx, 0])
        v_th_x = v[shock_idx, 0] - u_local_x

        # set histogram bins
        vmin = np.min(v_th_x)
        vmax = np.max(v_th_x)
        bins = np.linspace(vmin, vmax, 3000)  

        plt.figure(figsize=(10,6))
        ax = plt.subplot(1,1,1)

        counts, edges, patches = ax.hist(v_th_x, bins=bins, color='red', edgecolor='black')

        ax.set_title("X-Thermal Velocity Distribution in the Shock Region",
                     fontsize=14, fontweight='bold')
        ax.set_xlabel("X-Thermal Velocity (m/s)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Particle Count", fontsize=12, fontweight='bold')
        ax.tick_params(axis='both', labelsize=10)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')    

        bold_spines(ax)

        plt.tight_layout()
        plt.savefig("Shock_thermal_velocity_hist.png", dpi=300)
        plt.close()
        #plt.show()
        '''
