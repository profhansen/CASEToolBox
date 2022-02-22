## Copyright 2021 Morten Hartvig Hansen
#
# This file is part of CASEToolBox/CASEStab.

# CASEToolBox/CASEStab is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# CASEToolBox/CASEStab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with CASEToolBox/CASEStab.  If not, see <https://www.gnu.org/licenses/>.
#
#
import os
import json
from copy import deepcopy
import numpy as np
from . import math_functions as mf
from . import model_assembler as ma
from matplotlib import pyplot as plt


## Class that computes the steady state of axis-symmetric rotor deflection
#
# Input:
#   filename: Name of json file with model input
#
#
#
#
#
class rotor_models():
    def __init__(self,filename):
        # Remember folder and filename
        self.folder = os.path.dirname(filename)
        self.filename = filename
        # Read json file        
        with open(filename) as json_file:
            input_para = json.load(json_file)
        # Define substructures =======================================================
        subs_para_set=[]
        # HUB Type and base data of substructure
        rigid_para={}
        rigid_para['nodes']=np.array([[0.0,0.0],[0.0,0.0],[0.0,input_para['hub']['radius']]])
        subs_para={}
        subs_para['isubs']=0
        subs_para['name']='hub1'
        subs_para['Sbase']=np.eye(3)
        subs_para['bearing']='constant_speed 2 rotor_speed rpm'
        subs_para['type']='rigid_vector'
        subs_para['isubs_connection'] = -1 
        subs_para['inode_connection'] = -1 
        subs_para['para']=rigid_para
        subs_para_set.append(subs_para)
        # ROTOR Type and base data of substructure
        corotbeam_para={}
        #----- Generated data
        if input_para['blade']['structure']['type'] == 'HAWC2elements':
            corotbeam_para['znode']=input_para['blade']['structure']['znode']
            corotbeam_para['type']=input_para['blade']['structure']['type']
            corotbeam_para['name']=input_para['blade']['structure']['name']
            corotbeam_para['bname'] = input_para['blade']['structure']['bname']
            corotbeam_para['norder']=input_para['blade']['structure']['norder']
        else:
            corotbeam_para['znode']=input_para['blade']['structure']['znode']
            corotbeam_para['type']=''
            corotbeam_para['name']=input_para['blade']['structure']['st_file']
            corotbeam_para['setno']=input_para['blade']['structure']['setno']
            corotbeam_para['subsetno']=input_para['blade']['structure']['subsetno']
            corotbeam_para['nintp']=input_para['blade']['structure']['nintp']
            corotbeam_para['norder']=input_para['blade']['structure']['norder']
        subs_para={}
        subs_para['isubs']=1
        subs_para['name']='blade1'
        subs_para['Sbase']=mf.R1(np.radians(input_para['hub']['cone'])) # Coning
        subs_para['bearing']='constant_angle -3 pitch deg'
        subs_para['type']='corotbeam'
        subs_para['isubs_connection'] = 0 # Substructure "below" subtructure
        subs_para['inode_connection'] = 1 # Node number for connection of the support substructures, here the last node of hub
        subs_para['para']=corotbeam_para
        subs_para_set.append(subs_para)
        # Define blades ==============================================================
        blade_para_set=[]
        blade_para={}
        blade_para['geo_file']=input_para['blade']['aerodynamics']['ae_file']
        blade_para['pro_file']=input_para['blade']['aerodynamics']['pc_file']
        blade_para['geo_set']=input_para['blade']['aerodynamics']['ae_setno']
        blade_para['zaero'] = input_para['blade']['aerodynamics']['zaero']
        blade_para['geo_inter'] = input_para['blade']['aerodynamics']['geo_inter']
        blade_para['ae_inter'] = input_para['blade']['aerodynamics']['ae_inter']
        blade_para['pro_inter'] = input_para['blade']['aerodynamics']['pc_inter']
        blade_para['substructures']=[1] 
        blade_para_set.append(blade_para)
        # Define wake
        wake_para_set=[]
        wake_para={}
        wake_para['type']=input_para['wake']['type']
        wake_para['a_of_CT_model']=input_para['wake']['a_of_CT_model']
        wake_para['tip_correction']=input_para['wake']['tip_correction']
        wake_para['number_of_blades']=input_para['rotor']['number_of_blades']
        wake_para_set.append(wake_para)
        # Define wind =================================================================
        wind_para_set=[]
        wind_para={}
        wind_para['windtype']='uniform'
        wind_para['umean']=0.0
        wind_para['density'] = input_para['wind']['density']
        wind_para_set.append(wind_para)
        # Define axissymmetric rotor ==================================================
        rotor_para_set=[]
        rotor_para={}
        rotor_para['isubs_rotorcenter']=0 
        rotor_para['iaxis_rotorcenter']=2 # Defines the constant 'nvec_rotor' in the initial call
        rotor_para['type'] = 'axissym'
        rotor_para['blades'] = [0]
        rotor_para['blades_isubs'] = [1]
        rotor_para['number_of_blades']=input_para['rotor']['number_of_blades']
        rotor_para['iwake']=0
        rotor_para['iwind']=0
        rotor_para_set.append(rotor_para)
        # Deflection flag
        if 'deflection' in input_para.keys():
            self.include_deflection = (input_para['deflection'] == 1)
        else:
            self.include_deflection = True
        # Assmeble model
        model = ma.model(subs_para_set,blade_para_set,rotor_para_set,wake_para_set,wind_para_set)
        # Compute steady state of rotor if a operational data file is given
        self.models = []
        if input_para['operation'].split()[0] == 'ops_file':
            self.ops = np.loadtxt(input_para['operation'].split()[1],skiprows=1)
            self.Nops = np.size(self.ops,axis=0)
            self.pwr = np.zeros(self.Nops)
            self.thr = np.zeros(self.Nops)
            self.CP  = np.zeros(self.Nops)
            for iops in range(self.Nops):
                model.substructures[0].bearing.state.speed = self.ops[iops,2]*np.pi/30.0
                model.substructures[1].bearing.state.angle = np.radians(self.ops[iops,1])
                model.rotors[0].wind.lookup.umean = self.ops[iops,0]
                self.models.append(deepcopy(model))
    #=============================================================================================================================================
    # Perform steady state computation
    #=============================================================================================================================================
    def steady_state_computation(self,ops_list=[]):
        if len(ops_list) == 0:
            ops_list = np.arange(self.Nops)
        for iops in ops_list:
            print('======================================================================================================')
            print('=========== Computing steady state for operation point {:2d} with wind speed {:4.1f} m/s ==================='.format(iops+1,self.ops[iops,0]))
            print('======================================================================================================')
            self.models[iops].compute_rotor_stationary_steady_state(0,1.0,self.include_deflection)
            self.pwr[iops] = self.models[iops].rotors[0].power
            self.thr[iops] = self.models[iops].rotors[0].thrust
            self.CP[iops]  = self.models[iops].rotors[0].CP
    #=============================================================================================================================================
    # Save steady state results
    #=============================================================================================================================================
    def save_steady_state_results(self,prefix=''):
        defl_header = ['z-coord. [m] 1','ux_blade [m] 2','uy_blade [m] 3','uz_blade [m] 4','ux_rotor [m] 5','uy_rotor [m] 6','uz_rotor [m] 7','rotx_blade [deg] 8','roty_blade [deg] 9','rotz_blade [deg] 10']
        bem_header  = ['z-coord. [m] 1','a [-] 2','ap [-] 3','AoA [deg] 4', 'Urel [m/s] 5','Inplane Fx [N/m] 6','Axial Fy [N/m] 7','Moment [Nm/m] 8', 
                       'CL [-] 9','CD [-] 10','CM [-] 11','CLp [1/rad] 12','CDp [1/rad] 13','vx [m/s] 14','vy [m/s] 15','CT [-] 16','CQ [-] 17']
        for iops in range(self.Nops):
            # Structural results
            defl_filename = self.folder + prefix + 'defl_wsp{:d}mms_pitch{:d}mdeg_spd{:d}mrpm'.format(int(1e3*self.ops[iops,0]),int(1e3*self.ops[iops,1]),int(1e3*self.ops[iops,2]))
            # Collect data
            data = self.models[iops].substructures[1].create_data_for_deflection_state()
            # Save data
            header_txt='# Steady state results for all structural nodes \n'+''.join('{:>20s} '.format(text) for text in defl_header)
            np.savetxt(defl_filename,data,fmt='%20.10e',header=header_txt,comments='',delimiter=' ')
            # Aerdynamic BEM results
            bem_filename  = self.folder + prefix +  'BEM_wsp{:d}mms_pitch{:d}mdeg_spd{:d}mrpm'.format(int(1e3*self.ops[iops,0]),int(1e3*self.ops[iops,1]),int(1e3*self.ops[iops,2]))
            # Create data for output
            data = self.models[iops].rotors[0].create_data_for_BEM_results()
            # Save results 
            header_txt='# Steady state results for all aerodynamic calculation points \n'+''.join('{:>20s} '.format(text) for text in bem_header)
            np.savetxt(bem_filename,data,fmt='%20.10e',header=header_txt,comments='',delimiter=' ')
    #=============================================================================================================================================
    # Pitch curve tuner
    #=============================================================================================================================================
    def tune_pitch_curve(self,Prated,Tlimit,StallMargin,reltol,max_pitch_increment,max_CP=0.55,Nmaxiter=3,prefix='',plot_flag=False):
        # Reasons for tuning
        tune_reasons=['No tuning','Power','Thrust','Stall margin']
        # Initiate all reasons to "no tuning"
        tune_reason_ops=[]
        for iops in range(self.Nops):
            tune_reason_ops.append(tune_reasons[0])
        # Array for remembering the CP gradients
        CPgrad = -np.ones(self.Nops)
        # Compute initial steady state of rotor models
        print('********************** Runing steady state computation with original pitch curve *****************')
        self.steady_state_computation()
        # Plot initial pitch, power, and thrust curves
        if Tlimit > 0.0:
            l = ['Initial pitch','Power and thrust limits']
        else:
            l = ['Initial pitch','Power limit']
        fig = plt.figure(figsize=(8,12))
        axs = fig.subplots(3,1)
        axs[0].plot(self.ops[:,0],self.ops[:,1],'.-')
        axs[1].plot(self.ops[:,0],self.pwr,'.-')
        axs[2].plot(self.ops[:,0],self.thr,'.-')
        axs[0].set_ylabel('Pitch angle [deg]')
        axs[1].plot([self.ops[0,0],self.ops[-1,0]],[Prated,Prated],'k--')
        axs[1].set_ylabel('Aero power [w]')
        axs[1].legend(l)
        if Tlimit > 0.0:
            axs[2].plot([self.ops[0,0],self.ops[-1,0]],[Tlimit,Tlimit],'k--')
        axs[2].set_xlabel('Wind speed [m/s]')
        axs[2].set_ylabel('Thrust [N]')
        plt.show(block=plot_flag)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Adjust pitch for the operation point where the power or the thrust is exceeded
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        icount = 0
        while icount < Nmaxiter:
            # Increment counter
            icount += 1 
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Find the operation points where the power or the thrust are exceeded and the errors are larger than 'reltol'
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            iops_to_adjust_power = []
            iops_to_adjust_thrust = []
            iops_to_adjust_stallmargin = []
            for iops in range(self.Nops):
                # Check power
                rho = self.models[iops].winds[0].rho
                R = self.models[iops].rotors[0].wake.model.R
                CP_rated = 2.0*Prated/(rho*np.pi*R**2*self.ops[iops,0]**3)
                if CP_rated < max_CP and np.abs(self.pwr[iops]-Prated)/Prated > reltol:
                    iops_to_adjust_power.append(iops)
                # Check thrust
                if Tlimit > 0.0 and (self.thr[iops]-Tlimit)/Tlimit > reltol:
                    iops_to_adjust_thrust.append(iops)
                # Check stall margin
                StallMarginFlag =  StallMargin.shape[0] > 0
                if StallMarginFlag: 
                    dat,iaero_dat = self.models[iops].rotors[0].blades[0].get_AoA_stall_margins(StallMargin[-1,0])
                    daoa = np.zeros(dat.shape[0])
                    for i in range(dat.shape[0]):
                        daoa[i] = dat[i,2] - np.interp(dat[i,1],StallMargin[:,0],StallMargin[:,1])
                    daoa_min = np.min(daoa)
                    ihigh_light = iaero_dat[np.argmin(daoa)]
                    if daoa_min < 0.2:
                        iops_to_adjust_stallmargin.append(iops)
                        if plot_flag:
                            self.models[iops].rotors[0].blades[0].plot_stall_margins([-10,40],[ihigh_light],'Stall margin at WSP = {:4.1f} m/s'.format(self.ops[iops,0]),plot_flag)
            # Number of operational points to adjust the pitch for
            iops_to_adjust = np.union1d(iops_to_adjust_power,np.union1d(iops_to_adjust_thrust,iops_to_adjust_stallmargin))
            iops_to_adjust = np.unique(np.array(iops_to_adjust,dtype=int))
            Nops = len(iops_to_adjust)
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Check if we can stop
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if Nops == 0:
                break
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Save results for last validation run
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            theta0 = self.ops[iops_to_adjust,1].copy()
            pwr0 = self.pwr[iops_to_adjust].copy()
            thr0 = self.thr[iops_to_adjust].copy()
            stm0 =  np.zeros(Nops)
            if StallMarginFlag: 
                for iops in range(Nops):
                    dat,iaero_dat = self.models[iops_to_adjust[iops]].rotors[0].blades[0].get_AoA_stall_margins(StallMargin[-1,0])
                    daoa = np.zeros(dat.shape[0])
                    for i in range(dat.shape[0]):
                        daoa[i] = dat[i,2] - np.interp(dat[i,1],StallMargin[:,0],StallMargin[:,1])
                    daoa_min = np.min(daoa)
                    stm0[iops] = daoa_min
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Make perturbation to pitch curve
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            theta1 = theta0 + np.ones(Nops)
            for iops in range(Nops):
                self.ops[iops_to_adjust[iops],1] = theta1[iops]
                self.models[iops_to_adjust[iops]].substructures[1].bearing.state.angle = np.radians(self.ops[iops_to_adjust[iops],1])
            # Compute steady state for perturbed pitch curve
            print('********************** Perturbation run in iteration no. {:d} *****************'.format(icount))
            self.steady_state_computation(iops_to_adjust)
            # Read results for perturbed pitch curve
            pwr1 = self.pwr[iops_to_adjust].copy()
            thr1 = self.thr[iops_to_adjust].copy()
            stm1 =  np.zeros(Nops)
            if StallMarginFlag: 
                for iops in range(Nops):
                    dat,iaero_dat = self.models[iops_to_adjust[iops]].rotors[0].blades[0].get_AoA_stall_margins(StallMargin[-1,0])
                    daoa = np.zeros(dat.shape[0])
                    for i in range(dat.shape[0]):
                        daoa[i] = dat[i,2] - np.interp(dat[i,1],StallMargin[:,0],StallMargin[:,1])
                    daoa_min = np.min(daoa)
                    stm1[iops] = daoa_min
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Compute the pitch angle that satisfies the limits if the CP gradient is not too low (numerically)
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            for iops in range(Nops):
                P0 = pwr0[iops]
                P1 = pwr1[iops]
                T0 = thr0[iops]
                T1 = thr1[iops]
                S0 = stm0[iops]
                S1 = stm1[iops]
                theta_Tlimit     = (T1*theta0[iops]-T0*theta1[iops]+Tlimit*theta1[iops]-Tlimit*theta0[iops])/(T1-T0)
                theta_Prated     = (P1*theta0[iops]-P0*theta1[iops]+Prated*theta1[iops]-Prated*theta0[iops])/(P1-P0)
                if StallMarginFlag: 
                    theta_StallMargin = (S1*theta0[iops]-S0*theta1[iops])/(S1-S0)
                    pitch_angles = np.array([theta_Prated,theta_Tlimit,theta_StallMargin])
                else:
                    pitch_angles = np.array([theta_Prated,theta_Tlimit])
                imax_pitch = np.argmax(pitch_angles)
                tuned_pitch = pitch_angles[imax_pitch]
                # Save reason
                tune_reason_ops[iops_to_adjust[iops]] = tune_reasons[imax_pitch+1]
                # Keep increment within limit
                if tuned_pitch - theta0[iops] > max_pitch_increment:
                    tuned_pitch = theta0[iops] + max_pitch_increment
                if tuned_pitch - theta0[iops] < -max_pitch_increment:
                    tuned_pitch = theta0[iops] - max_pitch_increment
                self.ops[iops_to_adjust[iops],1] = tuned_pitch
                self.models[iops_to_adjust[iops]].substructures[1].bearing.state.angle = np.radians(self.ops[iops_to_adjust[iops],1])
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Compute steady state for adjusted pitch curve
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            print('********************** Validation run in iteration no. {:d} *****************'.format(icount))
            self.steady_state_computation(iops_to_adjust)
            # Add to plot
            axs[0].plot(self.ops[:,0],self.ops[:,1],'.-')
            axs[1].plot(self.ops[:,0],self.pwr,'.-')
            axs[2].plot(self.ops[:,0],self.thr,'.-')
            l.append('Iteration no. {:d}'.format(icount) )
            axs[1].legend(l)
            plt.show(block=plot_flag)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Save plot of iterations
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        fig.savefig(prefix + '_pitch_tuning.png' ,dpi=300)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Save results with statement at the end on the validation
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        header_txt='{:4d} WSP [m/s] Pitch [deg] Speed [rpm]'.format(self.ops.shape[0])
        np.savetxt(prefix + '_tuned.opt',self.ops,fmt='%15.8f' ,header=header_txt,comments='',delimiter=' ')
        with open(prefix + '_tuned.opt', 'a') as f:
            f.write('# This operational data file was created with CASEStab \n')
            f.write('# \n')
            f.write('# Model file: {:s} \n'.format(self.filename))
            f.write('# \n')
            f.write('# Limits:\n')
            f.write('#          Prated = {:12.1f} kW \n'.format(1e-3*Prated))
            if Tlimit > 0.0:
                f.write('#          Tlimit = {:12.0f} N \n'.format(Tlimit))
            if StallMarginFlag: 
                for ism in range(StallMargin.shape[0]):
                    f.write('#          Stall margin for {:3.1f}% airfoil = {:3.1f} deg \n'.format(StallMargin[ism,0],StallMargin[ism,1]))
            f.write('# \n')
            f.write('# Used {:d} of maximum {:d} iterations \n'.format(icount,Nmaxiter))
            f.write('# \n')
            f.write('# Tuning reasons and values \n')
            f.write('# \n')
            f.write('#    WSP [m/s]         Reason        Description    \n')
            for iops in range(self.Nops):
                if tune_reason_ops[iops] == tune_reasons[0]:
                    f.write('# {:12.3f} {:>14s} \n'.format(self.ops[iops,0],tune_reason_ops[iops]))
                elif tune_reason_ops[iops] == tune_reasons[1]:
                    f.write('# {:12.3f} {:>14s}        Difference to rated power = {:5.1f} kW \n'.format(self.ops[iops,0],tune_reason_ops[iops],1e-3*(self.pwr[iops]-Prated)))
                elif tune_reason_ops[iops] == tune_reasons[2]:
                    f.write('# {:12.3f} {:>14s}        Difference to thrust limit = {:5.1f} kN \n'.format(self.ops[iops,0],tune_reason_ops[iops],1e-3*(self.thr[iops]-Tlimit)))
                else:
                    dat,iaero_dat = self.models[iops].rotors[0].blades[0].get_AoA_stall_margins(StallMargin[-1,0])
                    daoa = np.zeros(dat.shape[0])
                    for i in range(dat.shape[0]):
                        daoa[i] = dat[i,2] - np.interp(dat[i,1],StallMargin[:,0],StallMargin[:,1])
                    iaoa_min = np.argmin(daoa)
                    ihigh_light = iaero_dat[iaoa_min]
                    f.write('# {:12.3f} {:>14s}        Minimum stall margin = {:5.1f} deg for {:3.1f}% airfoil at z_blade = {:4.1f} m \n'.format(self.ops[iops,0], \
                        tune_reason_ops[iops],dat[iaoa_min,2],self.models[iops].rotors[0].blades[0].aero_point[ihigh_light].thk,self.models[iops].rotors[0].blades[0].zaero[ihigh_light]))


