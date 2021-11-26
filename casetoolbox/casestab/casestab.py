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
        # Remember folder 
        self.folder = os.path.dirname(filename)
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
            for iops in range(self.Nops):
                model.substructures[0].bearing.state.speed = self.ops[iops,2]*np.pi/30.0
                model.substructures[1].bearing.state.angle = np.radians(self.ops[iops,1])
                model.rotors[0].wind.lookup.umean = self.ops[iops,0]
                model.pwr = np.zeros(self.Nops)
                model.thr = np.zeros(self.Nops)
                self.models.append(deepcopy(model))
    # Perform steady state computation
    def steady_state_computation(self):
        for iops in range(self.Nops):
            print('======================================================================================================')
            print('=========== Computing steady state for operation point {:2d} with wind speed {:4.1f} m/s ==================='.format(iops+1,self.ops[iops,0]))
            print('======================================================================================================')
            self.models[iops].compute_rotor_stationary_steady_state(0,1.0,self.include_deflection)
            self.pwr[iops] = self.models[iops].rotors[0].power
            self.thr[iops] = self.models[iops].rotors[0].thrust
    # Save steady state results
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






