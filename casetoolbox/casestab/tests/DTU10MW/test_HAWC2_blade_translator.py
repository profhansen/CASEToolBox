import numpy as np
import matplotlib.pyplot as plt
from casetoolbox.casestab import math_functions as mf
from casetoolbox.casestab import HAWC2_blade_translator as hbt

# HAWC2 input files
htcfile='DTU_10MW_RWT.htc'
ae_filename='DTU_10MW_RWT_ae.dat'
pc_filename='DTU_10MW_RWT_pc.dat'
st_filename='DTU_10MW_RWT_Blade_st.dat'
# HAWC2 main body name of blade
blade_name='blade1'
# HASEtool structural format choice
# st_format = 'ISO'
st_format = '6x6'

stru_sec_data,aero_sec_data = hbt.translate_HAWC2_blade_model(htcfile,ae_filename,pc_filename,st_filename,blade_name,st_format)


Nae = np.size(aero_sec_data,axis=0)
rle=np.zeros((3,Nae))
rte=np.zeros((3,Nae))
rac=np.zeros((3,Nae))
for isec in range(Nae):
    Ec = mf.rotmat_from_pseudovec(np.radians(aero_sec_data[isec,3:6]))
    rcref = np.array([aero_sec_data[isec,1],aero_sec_data[isec,2],aero_sec_data[isec,0]])
    racloc   = np.array([aero_sec_data[isec,8],aero_sec_data[isec,9],0.0])
    rle[:,isec] = rcref + Ec@(racloc + np.array([aero_sec_data[isec,10]*aero_sec_data[isec,6],0.0,0.0]))
    rte[:,isec] = rcref + Ec@(racloc - np.array([(1.0-aero_sec_data[isec,10])*aero_sec_data[isec,6],0.0,0.0]))
    rac[:,isec] = rcref + Ec@racloc

Nst = np.size(stru_sec_data,axis=0)
rref=np.zeros((3,Nst))
for isec in range(Nst):
    rref[:,isec] = np.array([stru_sec_data[isec,1],stru_sec_data[isec,2],stru_sec_data[isec,0]])
    # if st_format == 'ISO'



plt.figure(figsize=[8,8])
plt.subplot(2,1,1)
plt.plot(rle[2,:],rle[0,:],'-b',rte[2,:],rte[0,:],'-r',rac[2,:],rac[0,:],'-g')
plt.plot(rref[2,:],rref[0,:],'-k')
plt.ylabel('x-coordinate [m]')
plt.subplot(2,1,2)
plt.plot(rle[2,:],rle[1,:],'-b',rte[2,:],rte[1,:],'-r',rac[2,:],rac[1,:],'-g')
plt.plot(rref[2,:],rref[1,:],'-k')
plt.xlabel('z-coordinate [m]')
plt.ylabel('y-coordinate [m]')
plt.legend(['LE','TE','AC','Reference point'])

# Check interpolation on the pseudo vector for the CCS
z=np.linspace(0.0,aero_sec_data[-1,0],200)

plt.show()