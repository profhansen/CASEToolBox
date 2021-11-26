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
## @package Model assembler
#   
#
import numpy as np
from . import corotbeam
from . import rigidbody
from . import aerodynamics as aero
from . import wake_model
from . import wind_model 
from . import math_functions as mf
from . import generic_model_components as gmc
from matplotlib import pyplot as plt
from scipy.linalg import solve
from scipy.optimize import root

#==============================================================================================================================
#==============================================================================================================================
## Substructure class containing the different substructure models
#
#   'rigid_vector': A substructure that is rigid and has no mass
#   'corotbeam':    Co-rotational flexible beam 
#
class substructure():
    # Initialization of substructure of different kinds
    def __init__(self,para):
        # Save parameters
        self.para=para.copy()
        # Unique number of the substructure
        self.isubs=para['isubs']
        self.name =para['name']
        # Numbers of the substructure and node to which it is connected.
        # Fixed to ground if para['isubs_connection'] = 0
        self.isubs_connection=para['isubs_connection']
        self.inode_connection=para['inode_connection']
        # Chain of supporting substructures 
        self.isubs_chain=[]
        self.inode_chain=[]
        # Static (initial) orientation of substructure in ground-fixed frame
        self.S=para['Sbase']
        # Bearings
        if 'bearing' in para.keys():
            self.bearing = gmc.bearing(para['bearing'])
        else:
            self.bearing = gmc.bearing('')   
        # Index to a supporting substructure with a constant speed bearing (-1 = none)
        self.ksubs_constant_speed = -1
        # Dynamic position and orientation of substructure in ground-fixed frame
        self.r0=np.zeros(3)
        self.R0=np.eye(3)
        # Time derivatives of position and orientation of substructure in substructure frame
        self.R0Tdr0 =np.zeros(3)
        self.R0Tddr0=np.zeros(3)
        self.R0TdR0 =np.zeros((3,3))
        self.R0TddR0=np.zeros((3,3))
        # DOF derivatives of position and orientation of substructure in ground-fixed frame
        self.dr0dqi=[]       
        self.dR0dqi=[]        
        self.ddr0dqidqj=[]   
        self.ddR0dqidqj=[]    
        # Rigid massless body
        if para['type'] == 'rigid_vector':
            self.subs = rigidbody.rigidbody_substructure(para['para'])
        # Co-rotational flexible beam 
        if para['type'] == 'corotbeam':
            self.subs = corotbeam.corotbeam_substructure(para['para'])
        # Global DOF index vector of the DOFs inside the substructure including any bearing which is the first DOF in the vector
        self.ndofs=self.subs.ndofs+self.bearing.bear_dof
        self.idofs=np.arange(self.ndofs,dtype=int)


    # Structural modes of the substructure
    def compute_modes(self):
        # Point to substructure
        subs = self.subs
        # Only do something if the substructure is flexible
        if subs.type == 'corotbeam':
            # Positions of sections
            zpos=subs.pos[2,1:]
            # Setup matrices            
            # if subs.aero_part: # Add aerodynamic stiffness, not used for now where it is only structural modes
            #     # Stiffness matrix from stationary force
            #     Ksf=-subs.TKQf@force - subs.TKQm@moment
            #     K = subs.K+subs.inertia.Kc11+Ksf
            # else:
            K = subs.K+subs.inertia.Kc11
            M = subs.inertia.M11
            # Invert mass matrix
            Minv = np.linalg.inv(M)
            # Solve EVP
            A = np.concatenate((np.concatenate((np.zeros((subs.ndofs,subs.ndofs)),np.eye(subs.ndofs)),axis=1), \
                                np.concatenate((-Minv@K,-Minv@subs.inertia.G11),axis=1)))
            evals,evecs=np.linalg.eig(A)
            # Sort eigensolutions after frequency
            isort=np.argsort(np.imag(evals))[6*subs.nelem:]
            # Save frequencies
            freq=np.imag(evals[isort])/np.pi/2.0
            # Save mode shapes
            modeamp=np.zeros((subs.nelem,3,6*subs.nelem))
            modepha=np.zeros((subs.nelem,3,6*subs.nelem))
            for i in range(6*subs.nelem):
                modeamp[:,0,i]=np.abs(evecs[0:6*subs.nelem:6,isort[i]])
                modepha[:,0,i]=np.angle(evecs[0:6*subs.nelem:6,isort[i]])
                modeamp[:,1,i]=np.abs(evecs[1:6*subs.nelem:6,isort[i]])
                modepha[:,1,i]=np.angle(evecs[1:6*subs.nelem:6,isort[i]])
                # Convert Rodrigues' parameters to Euler angles (Note that this is a nonlinear function on the mode shape so the scale (norm) of "evecs" must be small)
                modeamp[:,2,i]=2.0*np.arctan(0.5*np.abs(evecs[5:6*subs.nelem:6,isort[i]])) # Crisfield (1990) Eq. (6)
                modepha[:,2,i]=np.angle(0.5*evecs[5:6*subs.nelem:6,isort[i]]) 
            # Name modes and normalize mode shapes
            name=[]
            ke=0
            kf=0
            kt=0
            for imode in range(6*subs.nelem):
                imax=np.argmax(np.abs(modeamp[-1,:,imode].reshape(3)))
                if imax==0:
                    ke+=1
                    mtype='{:d}E'.format(ke)
                elif imax==1:
                    kf+=1
                    mtype='{:d}F'.format(kf)
                else:
                    kt+=1
                    mtype='{:d}T'.format(kt)
                modeamp[:,:,imode]=modeamp[:,:,imode]/modeamp[-1,imax,imode]
                modepha[:,:,imode]=modepha[:,:,imode]-modepha[-1,imax,imode]
                name.append(mtype)
        else:
            freq=[]
            zpos=[]
            modeamp=[]
            modepha=[]
            name=[]
        # Save in dict
        subs_mode_solution={}
        subs_mode_solution['zpos']=zpos
        subs_mode_solution['freq']=freq
        subs_mode_solution['modeamp']=modeamp
        subs_mode_solution['modepha']=modepha
        subs_mode_solution['name']=name
        subs_mode_solution['solname']=''
        # Return the values
        return subs_mode_solution

    ## Routine that computes the 
    def plot_substructure_modes(self,subs_mode_solutions,nmodes,showflag,fname,dpi=300):
        # Plot
        for imode in range(nmodes):
            fig=plt.figure(figsize=[4,6])
            axs=fig.subplots(3,1)
            ltxt = []
            for sms in subs_mode_solutions:
                axs[0].plot(sms['zpos'],sms['modeamp'][:,0,imode]*np.cos(sms['modepha'][:,0,imode]),'.-')
                axs[1].plot(sms['zpos'],sms['modeamp'][:,1,imode]*np.cos(sms['modepha'][:,1,imode]),'.-')
                axs[2].plot(sms['zpos'],np.degrees(sms['modeamp'][:,2,imode]*np.cos(sms['modepha'][:,2,imode])),'.-')
                # Add axis labels
                axs[0].set_ylabel('Inplane [m]')
                axs[1].set_ylabel('Out-of-plane [m]')
                axs[2].set_ylabel('Rotation [deg]')
                axs[2].set_xlabel('Lengthwise coordinate [m]')
                fig.set_tight_layout(True)
                # Insert legend
                ltxt.append('{:s} {:4.2f} Hz'.format(sms['solname'],sms['freq'][imode]))
            # Add legends
            plt.legend(ltxt,loc='best')
            # Save plot
            if len(fname) > 0:
                fig.savefig('{:s}_mode_{:d}.png'.format(fname,imode+1),dpi=dpi)
            # Show plot
            plt.show(block = showflag)
        plt.close('all')

    def create_data_for_deflection_state(self):
        # Deflections in blade system
        defl_blade = np.zeros((self.subs.nelem,3))
        for i in range(self.subs.nelem):
            defl_blade[i,:] = self.subs.q[6*i:6*i+3]
        # Deflections in rotor plane
        defl_rotor = np.zeros((self.subs.nelem,3))
        for i in range(self.subs.nelem):
            q_subloc = self.R0@self.subs.q[6*i:6*i+3]
            defl_rotor[i,:] = q_subloc[0:3]
        # Rotations
        rot=np.degrees(self.subs.compute_rotation_peudo_vector().T)

        data = np.concatenate([np.array(self.subs.z).reshape((self.subs.nelem,1)),defl_blade,defl_rotor,rot],axis=1)
        return data
#==============================================================================================================================
#==============================================================================================================================
## Class containing the different rotor models
#
#   'axissymmetric': An axis-symmetric rotor with identical blades that deflect axis-symmetrically
#
# Input:
#
#
#
#
#
# Functions in the class:
#
#
class rotor():
    def __init__(self,para,blades,substructures,wake_para,wind):
        # Blades on the rotor 
        self.blades=blades
        # Type of rotor
        self.type = para['type']
        # Check that the blades have the same number of ACPs
        if len(self.blades) > 1:
            naero = []
            for b in self.blades:
                naero.append(b.naero)
            naero = np.array(naero)
            diff_naero = np.max(np.abs(np.diff(naero)))
            if diff_naero > 0:
                print('Error: Blades do not have the same number of aerodynamic calculation points.')
        self.naero = self.blades[0].naero
        # All substructures are included because we need the rotor center and the blade substructures
        self.substructures=substructures
        # Wind model
        self.wind = wind
        # Save the connection point of the rotor center
        self.isubs = para['isubs_rotorcenter']
        self.iaxis = para['iaxis_rotorcenter']
        # The constant normal vector of the rotor plane in the ground-fixed frame for the induced velocities 
        self.nvec = self.substructures[self.isubs].R0[:,self.iaxis-1]
        # Number of blades
        self.Nb = para['number_of_blades']
        # Current rotor speed for BEM calculations
        self.omega=0.0
        # Current center position and orientation matrix of rotor system
        self.rc0=self.substructures[self.isubs].r0
        self.Rc0=self.substructures[self.isubs].R0
        
        # Torque and thrust 
        self.power = 0.0
        self.torque= 0.0
        self.thrust= 0.0


        # Compute radial positions of the ACP on the blades and check that they are the same (diff < 1mm)
        radii = np.zeros((len(self.blades),self.naero))
        sigma = np.zeros((len(self.blades),self.naero))
        iblade=0
        for b in self.blades:
            for isubs in b.isubs:
                s=self.substructures[isubs]
                rc_0 = s.r0  - self.rc0
                for iacp in range(b.naero):
                    rc_tp = rc_0 + s.R0@s.subs.acp_motion_state[iacp].rtp
                    tvec = mf.crossproduct(self.nvec,rc_tp)
                    radii[iblade,iacp] = mf.vector_length(tvec)
                    sigma[iblade,iacp] = 0.5*b.aero_point[iacp].c/np.pi/radii[iblade,iacp]
            iblade+=1
        if np.max(np.std(radii,axis=0)) > 0.001:
            print('Error: Radial position of aerodynamic calculation points differs between blades.')
        self.radii = np.mean(radii,axis=0)
        self.sigma = np.mean(sigma,axis=0)*self.Nb


        # Create wake model
        self.wake = wake_model.wake(wake_para,self.radii)




    def update_rotor_kinematic_state(self):
        # Center position and 
        self.rc0=self.substructures[self.isubs].r0
        self.Rc0=self.substructures[self.isubs].R0
        # Compute rotor speed
        self.omega = mf.deskew(self.substructures[self.isubs].R0TdR0)[self.iaxis-1]
        



    def update_steady_blade_forces(self):
        # Update rotor kinematic states
        self.update_rotor_kinematic_state()
        # Compute steady state aerodynamic forces on each blade aassuming wake in balance
        for b in self.blades:
            # Update aeroelastic coupling for each substructure of the blade
            for isubs in b.isubs:
                s=self.substructures[isubs]
                # Update velocity triangles along the blade
                for iacp in s.subs.iacp_motion:
                    # Update chord coordinate systems
                    b.aero_point[iacp].Ec = s.subs.acp_motion_state[iacp].Ec
                    # Compute tangential vector at the section
                    rc_tp = s.r0+s.R0@s.subs.acp_motion_state[iacp].rtp - self.rc0
                    tvec = mf.crossproduct(self.nvec,rc_tp)
                    tvec = mf.unit_vector(tvec)
                    # Wind velocity at the TP which will also be used at the CP
                    w = self.wind.lookup.uvw_at_xyzt(s.subs.acp_motion_state[iacp].rtp,0.0)
                    # Wind speed normal to the rotor plane
                    wn = mf.innerproduct(self.nvec,w)
                    # BEM point setup
                    BEM_point = self.wake.model.momentum_balance_point(w,wn,self.wake.model.R,self.radii[iacp],s.R0,s.R0Tdr0,s.R0TdR0, \
                                                                       s.subs.acp_motion_state[iacp].rtp,s.subs.acp_motion_state[iacp].Ec, \
                                                                       self.nvec,tvec,self.omega,self.Nb,self.sigma[iacp],b.aero_point[iacp],self.wake.model.a_of_CT)
                    # Initial guess from the current values
                    x0 = np.array([self.wake.model.a[iacp],self.wake.model.ap[iacp]])
                    # Call root for solution until no error
                    solving = True
                    icount = 1
                    while(solving and icount < 2):
                        # Solving with root
                        xsol = root(BEM_point.f,x0=x0,jac=BEM_point.fprime,method='hybr')
                        # Check for success
                        solving = not xsol['success']
                        # Increase counter
                        icount += 1
                        # Error handling
                        if solving:
                            print('In BEM calculation for ACP number {:d} scipy.optimize.root comes with the message: \n {:s} '.format(iacp,xsol['message']))
                            print('Current function evaluations: ({:4.1e},{:4.1e})'.format(xsol['fun'][0],xsol['fun'][1]))
                            print('Restarting with new initial guess = half the current guess values.')
                            # New starting conditions
                            x0 = 0.5*xsol['x']
                            # # Plot for debugging
                            # if icount ==2:
                            #     NN = 101
                            #     MM = 101
                            #     xa  = np.linspace(xsol['x'][0]-0.1,xsol['x'][0]+0.1,MM)
                            #     xap = np.linspace(xsol['x'][1]-0.1,xsol['x'][1]+0.1,NN)
                            #     x,y = np.meshgrid(xa,xap)
                            #     za = np.zeros((NN,MM))
                            #     zap = np.zeros((NN,MM))
                            #     for i in range(MM):
                            #         for j in range(NN):
                            #             xx = np.array([xa[i],xap[j]])
                            #             xf = BEM_point.f(xx)
                            #             za[j,i] = xf[0]
                            #             zap[j,i] = xf[1]
                            #     plt.figure()
                            #     plt.contour(x,y,za,np.array([0.0]))
                            #     plt.contour(x,y,zap,np.array([0.0]))
                            #     plt.plot(xsol['x'][0],xsol['x'][1],'o')
                            #     plt.draw()
                            #     plt.show()
                    # Insert solution
                    self.wake.model.a[iacp]  = xsol['x'][0]
                    self.wake.model.ap[iacp] = xsol['x'][1]
                    # Compute the steady state forces at the section in substructure frame
                    f,m = b.aero_point[iacp].update_steady_aero_forces(self.wind.rho)
                    # Insert forces
                    b.f[3*iacp:3*iacp+3] = f.reshape(3)
                    b.m[3*iacp:3*iacp+3] = m.reshape(3)
                    # Save BEM solution
                    self.wake.model.CT[iacp]  = BEM_point.CT
                    self.wake.model.CQ[iacp]  = BEM_point.CQ
                    self.wake.model.sinphi[iacp] = BEM_point.sinphi
                    self.wake.model.local_TSR[iacp] = BEM_point.local_TSR
                    self.wake.model.ftip[iacp] = BEM_point.ftip
                    
    def create_data_for_BEM_results(self,iblade=0):
        # Compute forces in rotor plane
        fm_subloc = np.zeros((self.blades[iblade].naero,3))
        for i in range(self.blades[iblade].naero):
            f_subloc = self.substructures[1].R0@self.blades[iblade].f[3*i:3*i+3]
            fm_subloc[i,0:2] = f_subloc[0:2]
            fm_subloc[i,2] = self.blades[iblade].aero_point[i].M


        veltri = self.blades[iblade].states_and_forces()


        data = np.zeros((self.blades[0].naero,17))

        data[:, 0] = self.blades[0].zaero    # 'z-coord. [m] 1' 
        data[:, 1] = self.wake.model.a       # 'a [-] 2',
        data[:, 2] = self.wake.model.ap      # 'ap [-] 3',
        data[:, 3] = np.degrees(veltri[:,1]) # 'AoA [deg] 4'
        data[:, 4] = veltri[:,4]             # 'Urel [m/s] 5'
        data[:, 5] = fm_subloc[:,0]          # 'Inplane Fx [N/m] 6'
        data[:, 6] = fm_subloc[:,1]          # 'Axial Fy [N/m] 7'
        data[:, 7] = fm_subloc[:,2]          # 'Moment [Nm/m] 8' 
        data[:, 8] = veltri[:,5]             #  'CL [-] 9'
        data[:, 9] = veltri[:,6]             #  'CD [-] 10'
        data[:,10] = veltri[:,7]             #  'CM [-] 11'
        data[:,11] = veltri[:,11]            #  'CLp [1/rad] 12'
        data[:,12] = veltri[:,12]            #  'CDp [1/rad] 13'
        data[:,13] = veltri[:,2]             #  'vx [m/s] 14'
        data[:,14] = veltri[:,3]             #  'vy [m/s] 15'
        data[:,15] = self.wake.model.CT      #  'CT [-] 16'
        data[:,16] = self.wake.model.CQ      #  'CQ [-] 17'

        return data
#==============================================================================================================================
#==============================================================================================================================
## Model class that assembles the model from input parameters
#
# Input:
#   subs_para_set : 
#   blade_para_set:
#   rotor_para_set:
#   wake_para_set :
#   wind_para_set :
#
# Functions in the class:q
#
#
#
#
#
#
#
class model():
    def __init__(self,subs_para_set,blade_para_set,rotor_para_set={},wake_para_set={},wind_para_set={}):
        # Create substructures
        self.substructures=[]
        for subs_para in subs_para_set:
            self.substructures.append(substructure(subs_para))
            self.substructures[-1].subs.update_substructure()
        # Check for time dependent bearing (constant speed bearing)
        self.ndofs=0
        for s in self.substructures:
            # Create DOF index vector for substructure
            s.idofs += self.ndofs*np.ones(s.ndofs,dtype=int)
            # Sum up number of DOFs
            self.ndofs += s.ndofs
            # Create chain of substructures and check supporting substructures for constant speed bearing
            nbear_constant_speed=0
            icount=0
            isubs_con = s.isubs_connection
            inode_con = s.inode_connection
            while isubs_con > -1 and icount < 100:
                # Add to chain of substructures
                s.isubs_chain.append(isubs_con)
                s.inode_chain.append(inode_con)
                # Check of constant speed bearing
                if self.substructures[isubs_con].bearing.bear_flag:
                    if self.substructures[isubs_con].bearing.bear_type=='constant_speed':
                        nbear_constant_speed+=1
                        s.ksubs_constant_speed=isubs_con
                isubs_con = self.substructures[isubs_con].isubs_connection
                inode_con = self.substructures[isubs_con].inode_connection
            # Check if the substructure is connected to the ground
            if icount == 100:
                print('Substructure named ' + s.name + ' is not connected to a substructure connected to the ground')
            # Check the substructure itself
            if s.bearing.bear_flag:
                if s.bearing.bear_type=='constant_speed':
                    nbear_constant_speed+=1
                    s.ksubs_constant_speed=s.isubs
            # Write an error message if more than one is present                        
            if nbear_constant_speed > 1:
                print('More than one constant speed bearing rotates the substructure named ' + s.name)
        # # Allocate matrices
        # self.K=np.zeros((self.ndofs,self.ndofs))
        
        # Create blades
        iblade=0
        self.blades=[]
        for blade_para in blade_para_set:
            self.blades.append(aero.aero_blade(blade_para))
            # Link substructures to blades
            rbase=np.zeros(3)
            Sbase=np.eye(3) 
            S0 = self.substructures[blade_para['substructures'][0]].S # Blade is defined in the first structure frame
            base_subs_flag = True
            for isubs in blade_para['substructures']:
                # Save the substructure number
                self.blades[iblade].isubs.append(self.substructures[isubs].isubs)
                # Define the base orientation matrix in the first substructure frame
                Sbase=S0.T@self.substructures[isubs].S
                # Link the blade to the substructures
                self.substructures[isubs].subs.initiate_aeroelastic_coupling(iblade,self.blades[iblade].rac,self.blades[iblade].rcp,self.blades[iblade].Ec,rbase,Sbase,base_subs_flag)
                # The next substructure lies at the end of this one
                rbase=self.substructures[isubs].subs.pos[:,-1] 
                # Unset base substructure flag
                base_subs_flag = False
            # Increment the blade index
            iblade+=1
        # Initiate substructure positions and orientations
        for s in self.substructures:
            # Index to supporting substructure
            isubs_con = s.isubs_connection
            # Check if it is ground-fixed 
            if isubs_con > -1:
                # Local position vector and rotation matrix for supporting substructure
                r,R = self.substructures[isubs_con].subs.update_node_position_and_rotation(s.inode_connection)
                # Compute orientation matrix and position vector
                s.R0 = self.substructures[isubs_con].R0@R@self.substructures[isubs_con].S.T@s.S@s.bearing.state.B
                s.r0 = self.substructures[isubs_con].r0+self.substructures[isubs_con].R0@r
            else:
                s.r0 = np.zeros(3)
                if s.bearing.bear_flag:
                    s.R0 = s.S@s.bearing.state.B
                else:
                    s.R0 = np.eye(3)
        # Create wind fields 
        self.winds=[]
        for wind_para in wind_para_set:
            self.winds.append(wind_model.wind(wind_para))
        # Create rotors and wake models for each rotor
        self.rotors=[]
        for rotor_para in rotor_para_set:
            # Gather the blades and substructures for the aerodynamic part of rotor (excluding the hub)
            rotor_blades=[]
            for iblade in rotor_para['blades']:
                # Append the blades
                rotor_blades.append(self.blades[iblade])
            # Create rotor
            self.rotors.append(rotor(rotor_para,rotor_blades,self.substructures,wake_para_set[rotor_para['iwake']],self.winds[rotor_para['iwind']]))



#=====================================================================================================================================
    ## Routines that updates all substructures and their base motions
    def update_all_substructures(self,t):
        # Compute the current position vector and orientation matrix in the ground-fixed frame of each substructure
        for s in self.substructures:
            # Update the substructure
            s.subs.update_substructure()
            # Update bearing of each substructure of the model
            if s.bearing.bear_flag:
                s.bearing.state.update(t)
            # Update the positions and orientations of each substructure of the model and their DOF derivatives
            s.r0 = np.zeros(3)
            s.R0 = np.eye(3)
            # Index to supporting substructure
            isubs_con = s.isubs_connection
            # Check if it is ground-fixed 
            if isubs_con > -1:
                # Local position vector and rotation matrix for supporting substructure
                r,R = self.substructures[isubs_con].subs.update_node_position_and_rotation(s.inode_connection)
                # Compute orientation matrix and position vector
                s.R0 = self.substructures[isubs_con].R0@R@self.substructures[isubs_con].S.T@s.S@s.bearing.state.B
                s.r0 = self.substructures[isubs_con].r0+self.substructures[isubs_con].R0@r
                # First derivatives
                
                
                
            else:
                if s.bearing.bear_flag:
                    s.R0 = s.S@s.bearing.state.B
            # Time derivatives
            ksubs = s.ksubs_constant_speed
            if ksubs > -1:
                R0kTR0b = self.substructures[ksubs].R0.T@s.R0
                BTdB  = self.substructures[ksubs].bearing.state.BTdB
                BTddB = self.substructures[ksubs].bearing.state.BTddB
                r0_k_to_b = (s.r0 - self.substructures[ksubs].r0).T
                s.R0TdR0  = R0kTR0b.T@BTdB @R0kTR0b
                s.R0TddR0 = R0kTR0b.T@BTddB@R0kTR0b
                s.R0Tdr0  = R0kTR0b.T@BTdB @self.substructures[ksubs].R0.T@r0_k_to_b
                s.R0Tddr0 = R0kTR0b.T@BTddB@self.substructures[ksubs].R0.T@r0_k_to_b
            # Update elastic forces and stiffness matrix
            s.subs.update_elastic_internal_forces_and_stiffness()
            # Update inertia forces and states
            s.subs.update_inertia()
            # Update centrifugal forces, gyroscopic matrix and centrifugal stiffness matrix
            s.subs.inertia.Fc1,s.subs.inertia.G11,s.subs.inertia.Kc11=s.subs.inertia.compute_local_centrifugal_forces_and_matrix(s.R0Tddr0,s.R0TddR0,s.R0TdR0)
            # Update aeroelastic coupling if part of a blade
            if s.subs.aero_part:
                s.subs.update_aeroelastic_coupling()

            

#=====================================================================================================================================
    ## Routines that updates all rotors including the aerodynamic forces on them
    #
    #   Assume a call to update_all_substructures prior to calling this function
    #
    def update_steady_state_all_rotors(self):
        # Update each rotor
        for r in self.rotors:
            # Update aerodynamic forces on the blades
            r.update_steady_blade_forces()
           
            
        
        
        

#=====================================================================================================================================
    ## Routine that computes the stationary steady state (balance between static external and internal forces) for a rotor
    def compute_rotor_stationary_steady_state(self,irotor,Rnorm_limit=1.0,include_deform=True):
        # Update all substructures at t = 0.0
        self.update_all_substructures(0.0)
        # Point to the rotor
        r = self.rotors[irotor]
        # Stationary steady state implies the rotor is isotropic so we only use the first blade 
        b = r.blades[0]
        # Compute steady state aerodynamic forces on blade aassuming wake in balance
        Anorm = Rnorm_limit + 1.0
        jcount = 0
        while Anorm > Rnorm_limit:
            # Reset thrust and power
            r.power = 0.0
            r.thrust= 0.0
            # Update aeroelastic coupling for each substructure of the blade
            for isubs in b.isubs:
                # Latest force distribution for computation of force difference
                force0  = b.f.copy()
                # Update steady state aerodynamic force distribution
                self.update_steady_state_all_rotors()
                # Inner loop on blade internal force distribution convergence
                if include_deform:
                    Rnorm = Rnorm_limit + 1.0
                else:
                    Rnorm = 0.0
                icount = 0
                while Rnorm > Rnorm_limit:
                    # Update all substructures at t = 0.0
                    self.update_all_substructures(0.0)
                    # Get the aerodynamic forces and moment from the blade
                    if self.substructures[isubs].subs.aero_part:
                        force  = b.f[self.substructures[isubs].subs.iforces]
                        moment = b.m[self.substructures[isubs].subs.iforces]
                    # Aerodynamic generalized forces
                    if self.substructures[isubs].subs.aero_part:
                        faero=self.substructures[isubs].subs.TQf@force + self.substructures[isubs].subs.TQm@moment
                        fnode=faero-self.substructures[isubs].subs.inertia.Fc1
                        # Stiffness matrix from stationary force
                        Ksf=-self.substructures[isubs].subs.TKQf@force - self.substructures[isubs].subs.TKQm@moment
                        K  = self.substructures[isubs].subs.K+self.substructures[isubs].subs.inertia.Kc11+Ksf
                    else:
                        fnode=-self.substructures[isubs].subs.inertia.Fc1
                        K=self.substructures[isubs].subs.K+self.substructures[isubs].subs.inertia.Kc11
                    # Increment deflection 
                    R=fnode-self.substructures[isubs].subs.Fint
                    x=solve(K,R)
                    # Calculate the relaxation such that the maximum component of the increment is never larger then 10% of the substructure length
                    relaxation_factor = np.min([1.0,0.1*self.substructures[isubs].subs.z[-1]/np.max(np.abs(x))])
                    # Increment the positions
                    self.substructures[isubs].subs.q+=relaxation_factor*x
                    # Compute the 2-norm of the residuals
                    Rnorm=np.linalg.norm(R)
                    icount+=1
                    print('Inner iteration #{:d}: norm of force balance residual = {:10.3e}'.format(icount,Rnorm))
                # Add power and thrust values
                if self.substructures[isubs].subs.aero_part:
                    force  = b.f[self.substructures[isubs].subs.iforces]
                    moment = b.m[self.substructures[isubs].subs.iforces]
                    F = self.substructures[isubs].subs.Tf@force
                    M = self.substructures[isubs].subs.TMf@force + self.substructures[isubs].subs.TMm@moment
                    r.power  += self.substructures[isubs].R0Tdr0@F + mf.inner_matrix_product(self.substructures[isubs].R0TdR0,M)
                    r.thrust += mf.innerproduct(r.nvec,self.substructures[isubs].R0@F)
                # Compute norm of aerodynamic force distribution difference
                if include_deform:
                    Anorm=np.linalg.norm(b.f-force0)
                else:
                    Anorm = 0.0
                jcount+=1
                print('Outer iteration #{:d}: norm of external aerodynamic force change = {:10.3e}'.format(jcount,Anorm))
        # Multiply power and thrust with the number of blades if the rotor is axis-symmetric
        if 'axissym' in r.type:
            r.power = r.Nb*r.power
            r.thrust= r.Nb*r.thrust



#=====================================================================================================================================
    ## Routine that computes the steady state deformation (balance between static external and internal forces) for a single substructure
    def compute_substructure_steady_state_deformation(self,isubs,Rnorm_limit=1.0):

        Rnorm = Rnorm_limit + 1.0
        icount = 0
        while Rnorm > Rnorm_limit:
            # Update all substructures at t = 0.0
            self.update_all_substructures(0.0)
            # Get the aerodynamic forces and moment from the blade
            if self.substructures[isubs].subs.aero_part:
                force  = self.blades[self.substructures[isubs].subs.iblade].f[self.substructures[isubs].subs.iforces]
                moment = self.blades[self.substructures[isubs].subs.iblade].m[self.substructures[isubs].subs.iforces]
            # Aerodynamic generalized forces
            if self.substructures[isubs].subs.aero_part:
                faero=self.substructures[isubs].subs.TQf@force + self.substructures[isubs].subs.TQm@moment
                fnode=faero-self.substructures[isubs].subs.inertia.Fc1
                # Stiffness matrix from stationary force
                Ksf=-self.substructures[isubs].subs.TKQf@force - self.substructures[isubs].subs.TKQm@moment
                K  = self.substructures[isubs].subs.K+self.substructures[isubs].subs.inertia.Kc11+Ksf
            else:
                fnode=-self.substructures[isubs].subs.inertia.Fc1
                K=self.substructures[isubs].subs.K+self.substructures[isubs].subs.inertia.Kc11
            # Increment deflection 
            R=fnode-self.substructures[isubs].subs.Fint
            x=solve(K,R,assume_a='pos')
            self.substructures[isubs].subs.q+=x
            Rnorm=np.linalg.norm(R)
            icount+=1
            print('Iteration #{:d}: norm of force balance residual = {:10.3e}'.format(icount,Rnorm))
        

#=====================================================================================================================================
    ## Routine that plots the aerodynamic geometry data  
    def plot_aerodynamic_points(self,isubs,ielem,iblade):
        zetas=np.linspace(self.substructures[isubs].subs.elem_force[ielem].zeta_forces[0],self.substructures[isubs].subs.elem_force[ielem].zeta_forces[-1],200)
        xac=mf.piecewise_linear_function(zetas,self.substructures[isubs].subs.elem_force[ielem].zeta_forces,self.substructures[isubs].subs.elem_force[ielem].cfx)
        yac=mf.piecewise_linear_function(zetas,self.substructures[isubs].subs.elem_force[ielem].zeta_forces,self.substructures[isubs].subs.elem_force[ielem].cfy)
        rac=np.zeros((3,200))
        for i in range(200):
            rac[:,i]=self.substructures[isubs].subs.elem_state[ielem].rmid0 \
                    +self.substructures[isubs].subs.elem_state[ielem].E0@np.array([xac[i],yac[i],zetas[i]*0.5*self.substructures[isubs].subs.elem_model[ielem].l])
            
        
        # Plot nodes and ACPs
        rac_all=np.concatenate(self.blades[iblade].rac).reshape(len(self.blades[iblade].rac),3).T
        rcp_all=np.concatenate(self.blades[iblade].rcp).reshape(len(self.blades[iblade].rcp),3).T
        
        
        plt.figure(figsize=[12,8])
        plt.subplot(2,1,1)
        plt.plot(self.substructures[isubs].subs.pos[2,:],self.substructures[isubs].subs.pos[0,:],'o-')
        plt.plot(rac_all[2,:],rac_all[0,:],'x')
        plt.plot(rac[2,:],rac[0,:],'-')
        plt.plot(rcp_all[2,:],rcp_all[0,:],'v')
        for iacp in self.substructures[isubs].subs.acp_motion_state.keys():
            plt.plot(self.substructures[isubs].subs.acp_motion_state[iacp].rtp[2],self.substructures[isubs].subs.acp_motion_state[iacp].rtp[0],'rs')
        for iacp in self.substructures[isubs].subs.acp_motion_state.keys():
            plt.plot(self.substructures[isubs].subs.acp_motion_state[iacp].rcp[2],self.substructures[isubs].subs.acp_motion_state[iacp].rcp[0],'g^')
        plt.subplot(2,1,2)
        plt.plot(self.substructures[isubs].subs.pos[2,:],self.substructures[isubs].subs.pos[1,:],'o-')
        plt.plot(rac_all[2,:],rac_all[1,:],'x')
        plt.plot(rac[2,:],rac[1,:],'-')
        plt.plot(rcp_all[2,:],rcp_all[1,:],'v')
        for iacp in self.substructures[isubs].subs.acp_motion_state.keys():
            plt.plot(self.substructures[isubs].subs.acp_motion_state[iacp].rtp[2],self.substructures[isubs].subs.acp_motion_state[iacp].rtp[1],'rs')
        for iacp in self.substructures[isubs].subs.acp_motion_state.keys():
            plt.plot(self.substructures[isubs].subs.acp_motion_state[iacp].rcp[2],self.substructures[isubs].subs.acp_motion_state[iacp].rcp[1],'g^')
        plt.draw()
        plt.show()
        




#=====================================================================================================================================
    ## Routine that plots the input aerodynamic and structural data for a blade
    def plot_input_data_blade(self,iblade,fn=''):
        # Point to blade
        b = self.blades[0]
        Na = np.size(b.aeset,axis=0)
        # Compute AC and leading and trailing edge positions
        posac = np.zeros((3,Na))
        posle = np.zeros((3,Na))
        poste = np.zeros((3,Na))
        posae_ref = np.zeros((3,Na))
        for i in range(Na):
            posae_ref[:,i] = np.array([b.aeset[i,1],b.aeset[i,2],b.aeset[i,0]])
            # Chord coordinate system
            if np.abs(b.aeset[i,3])+np.abs(b.aeset[i,4]) < 1.0e-10:
                nc = np.ones(3)
                z = b.aeset[i,0]
                nc[0:2] = b.rc_curve.der(z)
                twist = np.radians(b.aeset[i,5])
                # Initiate Ec
                Ec = np.zeros((3,3))
                # Third vector is the vector between the nodes
                l = np.sqrt(nc[0]**2 + nc[1]**2 + nc[2]**2)
                Ec[:,2] = nc/l
                # its second  vector is defined as the unit-vector with zero x-component 
                Ec[1,1] =  Ec[2,2]/np.sqrt(Ec[1,2]**2 + Ec[2,2]**2)
                Ec[2,1] = -Ec[1,1]*Ec[1,2]/Ec[2,2]
                # and rotated by the twist angle
                R = mf.rotmat(Ec[:,2],twist)
                Ec[:,1] = R@Ec[:,1]
                # its first vector is the cross-product of the other two vectors
                Ec[:,0] = mf.crossproduct(Ec[:,1], Ec[:,2])
            else:
                Ec = mf.rotmat_from_pseudovec(np.radians(b.aeset[i,3:6]))
            # Position of AC in CCS
            rac = np.array([b.aeset[i,8],b.aeset[i,9],0.0])
            # AC, LE and TE positions in blade coordinate system
            posac[:,i] = posae_ref[:,i] + Ec@rac
            posle[:,i] = posac[:,i] +  b.aeset[i,10]     *b.aeset[i,6]*Ec[:,0]
            poste[:,i] = posac[:,i] + (b.aeset[i,10]-1.0)*b.aeset[i,6]*Ec[:,0]
        # # Compute the total number of structural center inputs
        # Ns = 0
        # for isubs in b.isubs:
        #     Ns += np.size(self.substructures[isubs].subs.stset,axis=0)
        #  Compute structural center positions
        posea = []
        possh = []
        poscg = []
        posst_ref = []
        for isubs in b.isubs:
            s = self.substructures[isubs].subs
            for i in range(np.size(s.stset,axis=0)):
                posst_ref.append(np.array([s.stset[i,1],s.stset[i,2],s.stset[i,0]]))
                if i<np.size(s.stset,axis=0)-1:
                    nvec = np.array([s.stset[i+1,1],s.stset[i+1,2],s.stset[i+1,0]])-posst_ref[-1]
                    nvec = mf.unit_vector(nvec)
                # nvec=np.array([0.0,0.0,1.0])
                R = mf.rotmat(nvec,np.radians(s.stset[i,3]))
                rea = np.array([s.stset[i,10],s.stset[i,11],0.0])
                rsh = np.array([s.stset[i,12],s.stset[i,13],0.0])
                rcg = np.array([s.stset[i, 5],s.stset[i, 6],0.0])
                
                at9 = np.array([[ 0.9744093 , -0.22478102],[ 0.22478102,  0.9744093 ]])
                
                posea.append(posst_ref[-1] + R@rea)
                possh.append(posst_ref[-1] + R@rsh)
                poscg.append(posst_ref[-1] + R@rcg)
        # Convert to numpy arrays    
        posea = np.array(posea).T
        possh = np.array(possh).T
        poscg = np.array(poscg).T
        posst_ref = np.array(posst_ref).T
        # Make plot        
        plt.figure(figsize=[7,7])
        plt.subplot(2,1,1)
        plt.plot(posac[2,:],posac[0,:],'k.-')
        plt.plot(poscg[2,:],poscg[0,:],'r.-')
        plt.plot(posea[2,:],posea[0,:],'b.-')
        plt.plot(possh[2,:],possh[0,:],'g.-')
        plt.plot(posst_ref[2,:],posst_ref[0,:],'m.-')
        plt.plot(posae_ref[2,:],posae_ref[0,:],'c.-')
        plt.plot(posle[2,:],posle[0,:],'k-',poste[2,:],poste[0,:],'k-')
        plt.ylabel('Inplane coordinate [m]')

        plt.subplot(2,1,2)
        plt.plot(posac[2,:],posac[1,:],'k.-')
        plt.plot(poscg[2,:],poscg[1,:],'r.-')
        plt.plot(posea[2,:],posea[1,:],'b.-')
        plt.plot(possh[2,:],possh[1,:],'g.-')
        plt.plot(posst_ref[2,:],posst_ref[1,:],'m.-')
        plt.plot(posae_ref[2,:],posae_ref[1,:],'c.-')
        plt.plot(posle[2,:],posle[1,:],'k-',poste[2,:],poste[1,:],'k-')
        plt.xlabel('Spanwise coordinate [m]')
        plt.ylabel('Out-of-plane coord. [m]')
        plt.legend(['AC','CG','EA','SC','ST ref','AE ref'])
        plt.tight_layout()

        if len(fn) > 0:
            plt.savefig(fn)

        return posac,posle,poste,posea,possh,poscg,posst_ref,posae_ref









