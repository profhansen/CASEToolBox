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
##  @package Generic_model_components
#
#
#
#
#
import numpy as np
from . import math_functions as mf
from . import model_precompiled_functions as mpf

## Inertia states of a substructure
class substructure_inertia():
    def __init__(self,ndofs):
        # Number of DOFs of substructure
        self.ndofs=ndofs
        # Total mass
        self.mass=0.0
        # Rotational inertias
        self.Ibase=np.zeros((3,3))
        # Local mass matrix
        self.M11=np.zeros((ndofs,ndofs))
        # Local gyroscopic and centrifugal stiffness matrix
        self.G11=np.zeros((ndofs,ndofs))
        self.Kc11=np.zeros((ndofs,ndofs))
        # Centrifugal force vector
        self.Fc1=np.zeros(ndofs)
        # Local nonlinear gyroscopic matrix
        self.H111=np.zeros((ndofs,ndofs,ndofs))
        # Mass times the vector to CG and other helping matrices and their derivatives
        self.nsym=int(ndofs*(ndofs+1)/2)
        self.ijsym=mf.generate_ijNsym(ndofs)
        self.m_rcg=np.zeros(3)
        self.m_drcg_dqi=np.zeros((3,ndofs))
        self.m_ddrcg_dqidqj=np.zeros((3,self.nsym))
        self.Abase_i=np.zeros((3,3,ndofs))
        self.Abase1_ij=np.zeros((3,3,self.nsym))
        self.Abase2_ij=np.zeros((3,3,self.nsym))
        self.jcol_nonzero_irow=np.zeros((2,ndofs),dtype=int)
    # Reset the inertia data for updating
    def reset_inertia(self):
        self.m_rcg=np.zeros(3)
        self.Ibase=np.zeros((3,3))
        self.M11=np.zeros((self.ndofs,self.ndofs))
        self.H111=np.zeros((self.ndofs,self.ndofs,self.ndofs))
        self.ijsym=mf.generate_ijNsym(self.ndofs)
        self.m_drcg_dqi=np.zeros((3,self.ndofs))
        self.m_ddrcg_dqidqj=np.zeros((3,self.nsym))
        self.Abase_i=np.zeros((3,3,self.ndofs))
        self.Abase1_ij=np.zeros((3,3,self.nsym))
        self.Abase2_ij=np.zeros((3,3,self.nsym))
    # Centrifugal forces and stiffness matrix
    def compute_local_centrifugal_forces_and_matrix(self,R0Tddr0,R0TddR0,R0TdR0):
        Fc,Gc,Kc=mpf.compute_local_centrifugal_forces_and_matrix(self.ndofs,self.jcol_nonzero_irow,self.ijsym, \
                                                                 self.m_drcg_dqi,self.m_ddrcg_dqidqj,self.Abase_i,self.Abase1_ij,self.Abase2_ij, \
                                                                 R0Tddr0,R0TddR0,R0TdR0)
        return Fc,Gc,Kc
## Aerodynamic forces from and on a substructure 
class initiate_acp_motion_state():
    def __init__(self,idofs):
        self.idofs=idofs.copy()
        self.ndofs=len(idofs)
        self.rtp=np.zeros(3)
        self.rcp=np.zeros(3)
        self.Ec=np.zeros((3,3))
        self.drtp_dqi=np.zeros((3,self.ndofs))
        self.drcp_dqi=np.zeros((3,self.ndofs))
        self.dec1_dqi=np.zeros((3,self.ndofs))
        self.dec2_dqi=np.zeros((3,self.ndofs))
        self.ddrtp_dqidqj=np.zeros((3,self.ndofs,self.ndofs))
        self.ddrcp_dqidqj=np.zeros((3,self.ndofs,self.ndofs))    
## Bearing class  
class bearing():
    def __init__(self,bearing_text):
        # Reset
        self.bear_flag=False
        self.bear_dof=0
        # Select the right bearing
        if len(bearing_text):
            self.bear_flag=True
            self.bear_type=bearing_text.split()[0]
            self.bear_axis=int(bearing_text.split()[1])
            self.bear_name=bearing_text.split()[2]
            self.bear_unit=bearing_text.split()[3]
            if self.bear_type == 'free':
                self.bear_dof=1
                self.state = self.free_bearing(self.bear_axis)
            elif self.bear_type == 'constant_angle':
                self.state = self.constant_angle_bearing(self.bear_axis)
            elif self.bear_type == 'constant_speed':
                self.state = self.constant_speed_bearing(self.bear_axis)
            else:
                self.bear_flag=False
                self.bear_dof=0
    # Bearing sub class for free bearing
    class free_bearing:
        def __init__(self,iaxis):
            self.B    =np.eye(3)
            self.dB   =np.zeros((3,3))
            self.ddB  =np.zeros((3,3))
            self.BTdB =np.zeros((3,3))
            self.BTddB=np.zeros((3,3))
            self.angle = 0.0
            self.iaxis = iaxis
            self.update(0.0)
        def update(self,t):
            self.B = mf.Ri(self.angle,self.iaxis)
    # Bearing sub class for constant angle bearing
    class constant_angle_bearing:
        def __init__(self,iaxis):
            self.B    =np.eye(3)
            self.dB   =np.zeros((3,3))
            self.ddB  =np.zeros((3,3))
            self.BTdB =np.zeros((3,3))
            self.BTddB=np.zeros((3,3))
            self.angle = 0.0
            self.iaxis = iaxis
            self.update(0.0)
        def update(self,t):
            self.B = mf.Ri(self.angle,self.iaxis)
    # Bearing sub class for constant speed bearing
    class constant_speed_bearing:
        def __init__(self,iaxis):
            self.B    =np.eye(3)
            self.dB   =np.zeros((3,3))
            self.ddB  =np.zeros((3,3))
            self.BTdB =np.zeros((3,3))
            self.BTddB=np.zeros((3,3))
            self.speed = 0.0
            self.iaxis = iaxis
            self.update(0.0)
        def update(self,t):
            self.B   = mf.Ri(self.speed*t,self.iaxis)
            N = mf.Smat[np.abs(self.iaxis)-1]*np.sign(self.iaxis)
            self.dB  = self.speed*self.B@N
            self.ddB = self.speed*self.speed*self.dB@N
            self.BTdB = self.speed*N
            self.BTddB= self.speed*self.speed*N@N
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        