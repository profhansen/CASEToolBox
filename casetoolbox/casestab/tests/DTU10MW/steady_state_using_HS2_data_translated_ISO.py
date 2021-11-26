from casetoolbox.casestab import casestab
from casetoolbox.casestab import math_functions as mf
import os
import numpy as np
from matplotlib import pyplot as plt
# Get current work directory
work_folder = os.getcwd()
# Set result folder
res_folder = 'SteadyState/results_ISO'
# Setup rotor models for each operational point
rotor_models = casestab.rotor_models(os.path.join(work_folder,'DTU10MW_ISO.json'))
# Compute the steady for all operational points
rotor_models.steady_state_computation()
# Save the steady for all operational points
rotor_models.save_steady_state_results('/' + res_folder + '/ind/')
# Copy for shorter notation in plotting below
ops = rotor_models.ops
pwr = rotor_models.pwr
thr = rotor_models.thr
models = rotor_models.models
# Plot comparison to HS2 for each operational point
for iops in range(len(models)):
    # Generate output data
    defl_data = models[iops].substructures[1].create_data_for_deflection_state()
    bem_data = models[iops].rotors[0].create_data_for_BEM_results()
    # Wind speed for labels
    wsp = ops[iops,0]
    # Read HS2 results
    hs2_ind = np.loadtxt(os.path.join(work_folder,'HAWCStab2_results/dtu10mw_u{:d}.ind'.format(int(1000*wsp))),skiprows = 1)
    # Plot deflections and forces in the rotor plane
    fig1=plt.figure(figsize=[12,6])
    axs1=fig1.subplots(3,2)
    axs1_2 = []
    for i in range(3):
        axdummy = []
        for j in range(2):
            axdummy.append(axs1[i,j].twinx())
        axs1_2.append(axdummy)
    axs1[0,0].plot(defl_data[:,0],defl_data[:,4],'.-',hs2_ind[:,0],hs2_ind[:,9],'.-')
    axs1[1,0].plot(defl_data[:,0],defl_data[:,5],'.-',hs2_ind[:,0],hs2_ind[:,10],'.-')
    axs1[2,0].plot(defl_data[:,0],defl_data[:,6],'.-',hs2_ind[:,0],hs2_ind[:,11],'.-')
    axs1[0,1].plot(defl_data[:,0],defl_data[:,9],'.-',hs2_ind[:,0],np.degrees(hs2_ind[:,34]*hs2_ind[:,37]),'.-')
    axs1[1,1].plot(bem_data[:,0],bem_data[:,5]*1e-3,'.-',hs2_ind[:,0],1e-3*hs2_ind[:,6],'.-')
    axs1[2,1].plot(bem_data[:,0],bem_data[:,6]*1e-3,'.-',hs2_ind[:,0],1e-3*hs2_ind[:,7],'.-')
    axs1[0,0].set_ylabel('x deflection [m]')
    axs1[1,0].set_ylabel('y deflection [m]')
    axs1[2,0].set_ylabel('z deflection [m]')
    axs1[0,1].set_ylabel('z rotation [deg]')
    axs1[1,1].set_ylabel('Inplane force [kN/m]')
    axs1[2,1].set_ylabel('Out-of-plane force [kN/m]')
    axs1[0,0].legend(['CS','HS2'])
    axs1_2[0][0].plot(bem_data[1:-1,0],1e3*(np.interp(bem_data[1:-1,0],defl_data[:,0],defl_data[:,4])-hs2_ind[:,9]),'.-g')
    axs1_2[1][0].plot(bem_data[1:-1,0],1e3*(np.interp(bem_data[1:-1,0],defl_data[:,0],defl_data[:,5])-hs2_ind[:,10]),'.-g')
    axs1_2[2][0].plot(bem_data[1:-1,0],1e3*(np.interp(bem_data[1:-1,0],defl_data[:,0],defl_data[:,6])-hs2_ind[:,11]),'.-g')
    axs1_2[0][1].plot(bem_data[1:-1,0],np.interp(bem_data[1:-1,0],defl_data[:,0],defl_data[:,9])-np.degrees(hs2_ind[:,34]*hs2_ind[:,37]),'.-g')
    axs1_2[1][1].plot(bem_data[1:-1,0],bem_data[1:-1,5]-hs2_ind[:,6],'.-g')
    axs1_2[2][1].plot(bem_data[1:-1,0],bem_data[1:-1,6]-hs2_ind[:,7],'.-g')
    axs1_2[0][0].set_ylabel('Diff CS - HS2 [mm]')
    axs1_2[1][0].set_ylabel('Diff CS - HS2 [mm]')
    axs1_2[2][0].set_ylabel('Diff CS - HS2 [mm]')
    axs1_2[0][1].set_ylabel('Diff CS - HS2 [deg]')
    axs1_2[1][1].set_ylabel('Diff CS - HS2 [N/m]')
    axs1_2[2][1].set_ylabel('Diff CS - HS2 [N/m]')
    axs1[2,0].set_xlabel('Spanwise coordinate [m]')
    axs1[2,1].set_xlabel('Spanwise coordinate [m]')
    plt.tight_layout()
    # plt.show()
    fig1.savefig(os.path.join(work_folder, res_folder + '/deflforce{:d}ms.png'.format(int(wsp))),dpi=300)

    plt.close('all')

    veltri = models[iops].rotors[0].blades[0].states_and_forces()

    # Plot velocity triangles
    fig2=plt.figure(figsize=[12,6])
    axs2=fig2.subplots(3,2)
    axs2_2 = []
    for i in range(3):
        axdummy = []
        for j in range(2):
            axdummy.append(axs2[i,j].twinx())
        axs2_2.append(axdummy)
    axs2[0,0].plot(bem_data[:,0],bem_data[:,8],'.-', hs2_ind[:,0],hs2_ind[:, 16],'.-')
    axs2[1,0].plot(bem_data[:,0],bem_data[:,9],'.-', hs2_ind[:,0],hs2_ind[:, 17],'.-')
    axs2[2,0].plot(bem_data[:,0],bem_data[:,3],'.-' ,hs2_ind[:,0],np.degrees(hs2_ind[:, 4]),'.-')
    axs2[0,1].plot(bem_data[:,0],bem_data[:,4],'.-' ,hs2_ind[:,0],hs2_ind[:, 5],'.-')
    axs2[1,1].plot(bem_data[:,0],bem_data[:,1],'.-' ,hs2_ind[:,0],hs2_ind[:, 1],'.-')
    axs2[2,1].plot(bem_data[:,0],bem_data[:,2],'.-' ,hs2_ind[:,0],hs2_ind[:, 2],'.-')
    axs2[0,0].set_ylabel('CL [-]')
    axs2[1,0].set_ylabel('CD [-]')
    axs2[2,0].set_ylabel('AoA [deg]')
    axs2[0,1].set_ylabel('Relative speed [m/s]')
    axs2[1,1].set_ylabel('a [-]')
    axs2[2,1].set_ylabel('ap [-]')
    axs2[0,0].legend(['CS','HS2'])
    axs2_2[0][0].plot(bem_data[1:-1,0],bem_data[1:-1,8]-hs2_ind[:, 16],'.-g')
    axs2_2[1][0].plot(bem_data[1:-1,0],bem_data[1:-1,9]-hs2_ind[:, 17],'.-g')
    axs2_2[2][0].plot(bem_data[1:-1,0],bem_data[1:-1,3]-np.degrees(hs2_ind[:, 4]),'.-g')
    axs2_2[0][1].plot(bem_data[1:-1,0],bem_data[1:-1,4]-hs2_ind[:, 5],'.-g')
    axs2_2[1][1].plot(bem_data[1:-1,0],bem_data[1:-1,1]-hs2_ind[:, 1],'.-g')
    axs2_2[2][1].plot(bem_data[1:-1,0],bem_data[1:-1,2]-hs2_ind[:, 2],'.-g')
    axs2_2[0][0].set_ylabel('Diff CS - HS2 [-]')
    axs2_2[1][0].set_ylabel('Diff CS - HS2 [-]')
    axs2_2[2][0].set_ylabel('Diff CS - HS2 [deg]')
    axs2_2[0][1].set_ylabel('Diff CS - HS2 [m/s]')
    axs2_2[1][1].set_ylabel('Diff CS - HS2 [-]')
    axs2_2[2][1].set_ylabel('Diff CS - HS2 [-]')
    axs2[2,0].set_xlabel('Spanwise position [m]')
    axs2[2,1].set_xlabel('Spanwise position [m]')
    plt.tight_layout()
    # plt.show()
    fig2.savefig(os.path.join(work_folder,res_folder + '/veltri{:d}ms.png'.format(int(wsp))),dpi=300)

    plt.close('all')



hs2_pwr = np.loadtxt(os.path.join(work_folder,'HAWCStab2_results/hs2.pwr'),skiprows = 1)

# Plot power curve
fig = plt.figure()
ax = fig.gca()
ax2 = ax.twinx()

ax.plot(ops[:,0],pwr*1e-3,'o-',hs2_pwr[:,0],hs2_pwr[:,1],'o-')
ax2.plot(ops[:,0],pwr*1e-3-hs2_pwr[:,1],'o-g')

ax.set_ylabel('Power [kW]')
ax.legend(['CS','HS2'])
ax2.set_ylabel('Diff CS - HS2 [kW]')
ax.set_xlabel('Wind speed [m/s]')

ax.set_xlim([ops[0,0],ops[-1,0]])

plt.tight_layout()
#plt.show()
fig.savefig(os.path.join(work_folder,res_folder + '/power.png'),dpi=300)

# Plot thrust curve
fig = plt.figure()
ax = fig.gca()
ax2 = ax.twinx()

ax.plot(ops[:,0],thr*1e-3,'o-',hs2_pwr[:,0],hs2_pwr[:,2],'o-')
ax2.plot(ops[:,0],thr*1e-3-hs2_pwr[:,2],'o-g')

ax.set_ylabel('Thrust [kN]')
ax.legend(['CS','HS2'])
ax2.set_ylabel('Diff CS - HS2 [kN]')
ax.set_xlabel('Wind speed [m/s]')

ax.set_xlim([ops[0,0],ops[-1,0]])

plt.tight_layout()
#plt.show()
fig.savefig(os.path.join(work_folder,res_folder + '/thrust.png'),dpi=300)




pass

