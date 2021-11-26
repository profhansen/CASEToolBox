from casetoolbox.casestab import casestab
from casetoolbox.casestab import math_functions as mf
import os
import numpy as np
from matplotlib import pyplot as plt
# Get current work directory
work_folder = os.getcwd()
# Set result folder
res_folder = 'BladeModes/results_H2_elements'
#========================================================================================================
# Setup rotor models for each operational point
rotor_models = casestab.rotor_models(os.path.join(work_folder,'DTU10MW_H2_elements_few_ops.json'))

#========================================================================================================
# Structural standstill blade modes (standstill = first operational point)
blade = rotor_models.models[0].substructures[1]
# Create mass and stiffness matrices
rotor_models.models[0].update_all_substructures(0.0)
# Compute modes
casestab_sol = blade.compute_modes()
casestab_sol['solname'] = 'CASEStab'
# Read and translate HAWCStab2 results to CASEStab's "subs_mode_solution" format
HS2_freq=np.loadtxt(os.path.join(work_folder,'HAWCStab2_results/blade.cmb'),skiprows=1)
HS2_mode=np.loadtxt(os.path.join(work_folder,'HAWCStab2_results/blade_standstill.amp'),skiprows=3)
HS2_name=['1F','1E','2F','2E','3F','3E','4F','1T','5F','4F']
nmodes = 10
HS2_sol={}
HS2_sol['zpos']=HS2_mode[:,1]
HS2_sol['freq']=HS2_freq[0,1:nmodes+1]
HS2_sol['modeamp']=np.zeros((HS2_mode.shape[0],3,nmodes))
HS2_sol['modepha']=np.zeros((HS2_mode.shape[0],3,nmodes))
for imode in range(nmodes):
    for idof in range(3):
        HS2_sol['modeamp'][:,idof,imode]=           HS2_mode[:,2+2*idof+6*imode]
        HS2_sol['modepha'][:,idof,imode]=np.radians(HS2_mode[:,2+2*idof+6*imode+1])
HS2_sol['name']=HS2_name
HS2_sol['solname']='HAWCStab2'
# Collect solutions
subs_mode_solutions = [casestab_sol,HS2_sol]
# Plot 10 modes
fname = os.path.join(work_folder, res_folder + '/standstill')
blade.plot_substructure_modes(subs_mode_solutions,nmodes,False,fname)

#========================================================================================================
# Structural blade modes at rated rpm (rated rpm = second operational point)
blade = rotor_models.models[1].substructures[1]
# Setup matrices and deflect blades under the centrifugal forces
rotor_models.models[1].compute_substructure_steady_state_deformation(1)
# Compute modes
casestab_sol = blade.compute_modes()
casestab_sol['solname'] = 'CASEStab'
# Read and translate HAWCStab2 results to CASEStab's "subs_mode_solution" format
HS2_freq=np.loadtxt(os.path.join(work_folder,'HAWCStab2_results/blade.cmb'),skiprows=1)
HS2_mode=np.loadtxt(os.path.join(work_folder,'HAWCStab2_results/blade_10_rpm.amp'),skiprows=3)
HS2_name=['1F','1E','2F','2E','3F','3E','4F','1T','5F','4F']
nmodes = 10
HS2_sol={}
HS2_sol['zpos']=HS2_mode[:,1]
HS2_sol['freq']=HS2_freq[-1,1:nmodes+1]
HS2_sol['modeamp']=np.zeros((HS2_mode.shape[0],3,nmodes))
HS2_sol['modepha']=np.zeros((HS2_mode.shape[0],3,nmodes))
for imode in range(nmodes):
    for idof in range(3):
        HS2_sol['modeamp'][:,idof,imode]=           HS2_mode[:,2+2*idof+6*imode]
        HS2_sol['modepha'][:,idof,imode]=np.radians(HS2_mode[:,2+2*idof+6*imode+1])
HS2_sol['name']=HS2_name
HS2_sol['solname']='HAWCStab2'
# Collect solutions
subs_mode_solutions = [casestab_sol,HS2_sol]
# Plot 10 modes
fname = os.path.join(work_folder, res_folder + '/10_rpm')
blade.plot_substructure_modes(subs_mode_solutions,nmodes,False,fname)

#========================================================================================================
# Structural modes of deflected blade operating at 11 m/s (third operational point)
blade = rotor_models.models[2].substructures[1]
# Setup matrices and deflect blades under the centrifugal forces
rotor_models.models[2].compute_rotor_stationary_steady_state(0)
# Compute modes
casestab_sol = blade.compute_modes()
casestab_sol['solname'] = 'CASEStab'
# Read and translate HAWCStab2 results to CASEStab's "subs_mode_solution" format
HS2_freq=np.loadtxt(os.path.join(work_folder,'HAWCStab2_results/blade_steady_state.cmb'),skiprows=1)
HS2_mode=np.loadtxt(os.path.join(work_folder,'HAWCStab2_results/blade_11ms.amp'),skiprows=3)
HS2_name=['1F','1E','2F','2E','3F','3E','4F','1T','5F','4F']
nmodes = 10
HS2_sol={}
HS2_sol['zpos']=HS2_mode[:,1]
HS2_sol['freq']=HS2_freq[6,1:nmodes+1]
HS2_sol['modeamp']=np.zeros((HS2_mode.shape[0],3,nmodes))
HS2_sol['modepha']=np.zeros((HS2_mode.shape[0],3,nmodes))
for imode in range(nmodes):
    for idof in range(3):
        HS2_sol['modeamp'][:,idof,imode]=           HS2_mode[:,2+2*idof+6*imode]
        HS2_sol['modepha'][:,idof,imode]=np.radians(HS2_mode[:,2+2*idof+6*imode+1])
HS2_sol['name']=HS2_name
HS2_sol['solname']='HAWCStab2'
# Collect solutions
subs_mode_solutions = [casestab_sol,HS2_sol]
# Plot 10 modes
fname = os.path.join(work_folder, res_folder + '/11ms')
blade.plot_substructure_modes(subs_mode_solutions,nmodes,False,fname)


