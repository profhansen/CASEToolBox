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
## @package corotbeam
#  The co-rotational beam element formulation 
#
# 
import numpy as np
from . import math_functions as mf
from . import generic_model_components as gmc
from .timoshenko_beam_section import isotropic_to_6x6_compliance_matrix
from .HAWC2_blade_translator import HAWC2_elements
from .timoshenko_beam_section import transform_reference_point_of_matrix
from matplotlib import pyplot as plt
from . import corotbeam_precompiled_functions as cpf
## Identity matrix
Imat=np.eye(3)
## 12x12 index matrix 
ij12sym=mf.generate_ijNsym(12)
## Routine that computes the initial element coordinate system (ECS) from initial node positions and average rotation
def element_coordinate_system(r0,angle):
    E = np.zeros((3,3))
    # Vector between nodes
    rvec=r0[3:]-r0[0:3]
    # Third vector is the vector between the nodes
    l = np.sqrt(rvec[0]**2 + rvec[1]**2 + rvec[2]**2)
    E[:,2] = rvec/l
    # its second  vector is defined as the unit-vector with zero x-component: e2 . e3 = 0, e21 = 0, |e2| = 1 
    E[1,1] =  E[2,2]/np.sqrt(E[1,2]**2 + E[2,2]**2)
    E[2,1] = -E[1,1]*E[1,2]/E[2,2]
    # and rotated by the twist angle
    R = mf.rotmat(E[:,2],angle)
    E[:,1] = R@E[:,1]
    # its first vector is the cross-product of the other two vectors
    E[:,0] = mf.crossproduct(E[:,1], E[:,2])
    return l,E
## Index function for integration of polynomials
def c_function(r):
    # r is a positive integer
    if r % 2:
        c = 0.0
    else:
        c = 2.0/(1.0+r)
    return c


## Krenk and Couturier equilibrium element with full 6x6 compliance matrix
#
#
class corotbeam_substructure:
    ## Initialization of body described by co-rotational equilibrium beam elements
    #  Structural input:
    #      znode: Node distribution which defines the discretization of the beam-like body
    #      ftype: String with file type ('HAWC2',...) for the file with the beam properties
    #      fname: String with file name of input for the beam properties
    #      para: Dict with parameters related to the file type:
    #          HAWC2elements: 
    #          'bname': Body name to extract data for.
    #          ISO: 
    #          'setno1': Set numbers (int) of the data in the file
    #          'setno2': Subset numbers (int) of the data in the file
    #          'refpt':  String with reference point method ('EA',...)
    #          'nintp':  Number of interpolation points (int) over the element in the input file
    #      norder: Order of element (int)                 
    #  Structural output:
    #
    #
    #
    #
    def __init__(self,para):
        # Define substructure type
        self.type='corotbeam'
        # Read structural data
        if para['type'] == 'HAWC2elements':
            # Read data
            mbdy_name=para['bname']
            elements=HAWC2_elements(para['name'],mbdy_name)
            norder=0
        else:
            setno=para['setno']
            subsetno=para['subsetno']
            nintp=para['nintp']
            norder=para['norder']
            elements=elements_created_from_file(para['name'],para['znode'],setno,subsetno,nintp,norder)
            self.stset=elements.stset
        # Insert data
        self.elem_model=[]
        self.elem_state=[]
        self.nelem=elements.nelem
        self.z=elements.z
        self.pos=elements.pos
        for i in range(self.nelem):
            # Create the element data structure
            elem_state=corotbeam_element_kinematics(elements.l[i],elements.r0s[i],elements.E0s[i],norder)
            elem_model=equilibrium_beam_element(elements.l[i],elements.Cs[i],elements.iner_pars[i])
            self.elem_model.append(elem_model)   
            self.elem_state.append(elem_state)   
        # Set up internal DOF vectors
        self.ndofs = 6*self.nelem
        self.q   =np.zeros(6*self.nelem)
        self.dqdt=np.zeros(6*self.nelem)
        # Initialize internal force vector and stiffness matrix
        self.Fint=np.zeros(6*self.nelem)
        self.K=np.zeros((6*self.nelem,6*self.nelem))
        # Set up inertia state variables
        self.inertia=gmc.substructure_inertia(6*self.nelem)
        for i in range(self.nelem):
            self.inertia.mass += self.elem_model[i].mass
        # Set up indices to non-zero elements of inertia matrices
        for i in range(self.nelem):
            idiag=6*i
            for idof in range(6):
                self.inertia.jcol_nonzero_irow[0,idof+idiag]=idof+idiag
                self.inertia.jcol_nonzero_irow[1,idof+idiag]=min([12+idiag,self.ndofs])
        # Element data for the force distribution
        self.elem_force=[]
        # Blade number to which the forces relate
        self.aero_part=False
        self.iblade=-1
        # Internal data for motion state computations of each ACP
        self.acp_data={}
        # External data with the motion state of each ACP 
        self.acp_motion_state={}
        # Initial update
        self.update_substructure()
    ## Update elements
    def update_substructure(self):
        # Loop over all elements
        for i in range(self.nelem):
            # Update body deflection
            if i > 0:
                q=self.q[6*(i-1):6*(i+1)]
            else:
                q=np.zeros(12)
                q[6:]+=self.q[0:6]
            self.elem_state[i].update_nodal_triads_and_position(q)
            # Update local defomations and their derivatives
            self.elem_state[i].update_local_nodal_rotations_elongation()
            self.elem_state[i].update_first_derivative_local_nodal_rotations_elongation()
            self.elem_state[i].update_second_derivative_local_nodal_rotations_elongation()
            # Update deflection sub-vector coefficients for each shape function order
            self.elem_state[i].update_element_deflection_subvectors_and_derivatives(self.elem_model[i].Nl)
    ## Compute the internal forces and setup the elastic stiffness matrix of the body
    def update_elastic_internal_forces_and_stiffness(self):
        # Reset
        self.Fint=np.zeros(6*self.nelem)
        self.K=np.zeros((6*self.nelem,6*self.nelem))
        # Loop over all elements
        for i in range(self.nelem):
            # Insert the initial force components and element stiffness matrix
            tmp1=self.elem_state[i].ql.T@self.elem_model[i].Kl
            tmp2=self.elem_state[i].dqldi0.T@self.elem_model[i].Kl
            felem=tmp1 @ self.elem_state[i].dqldi0
            kelem=tmp2 @ self.elem_state[i].dqldi0
            for k in range(7):
                kelem+=np.reshape(tmp1[k]*self.elem_state[i].ddqldidj0[k,:,:],(12,12))
            if i > 0:
                self.Fint[6*(i-1):6*(i+1)]+=felem
                self.K[6*(i-1):6*(i+1),6*(i-1):6*(i+1)]+=kelem
            else: # First clamped element
                self.Fint[0:6]+=felem[6:]
                self.K[0:6,0:6]+=kelem[6:,6:]
    ## Update current inertia states of a substructure 
    def update_inertia(self):
        # Reset inertia states
        self.inertia.reset_inertia()
        
        for i in range(self.nelem):
            idiag=6*(i-1)
            if i > 0:
                i1=0
            else:
                i1=6
                
            m_rcg,Ibase,m_drcg_dqi,Abase_i,m_ddrcg_dqidqj,M11,Abase1_ij,Abase2_ij = \
                cpf.update_element_inertia(self.elem_model[i].norder,self.elem_model[i].l,self.elem_model[i].iner_pars, \
                                           self.elem_state[i].ro,self.elem_state[i].rx,self.elem_state[i].ry,\
                                           self.elem_state[i].dro_dqi,self.elem_state[i].drx_dqi,self.elem_state[i].dry_dqi, \
                                           self.elem_state[i].ddro_dqidqj,self.elem_state[i].ddrx_dqidqj,self.elem_state[i].ddry_dqidqj)
                
            self.inertia.m_rcg += m_rcg
            self.inertia.Ibase+=Ibase
            self.inertia.m_drcg_dqi[:,idiag+i1:idiag+12] += m_drcg_dqi[:,i1:12]
            self.inertia.Abase_i[:,:,idiag+i1:idiag+12] += Abase_i[:,:,i1:12]
            for idof in range(i1,12):
                for jdof in range(idof,12):
                    self.inertia.m_ddrcg_dqidqj[:,self.inertia.ijsym[idof+idiag,jdof+idiag]] += m_ddrcg_dqidqj[:,ij12sym[idof,jdof]]
                    self.inertia.M11[idof+idiag,jdof+idiag]+=M11[ij12sym[idof,jdof]]
                    self.inertia.Abase1_ij[:,:,self.inertia.ijsym[idof+idiag,jdof+idiag]]+=Abase1_ij[:,:,ij12sym[idof,jdof]]
                    self.inertia.Abase2_ij[:,:,self.inertia.ijsym[idof+idiag,jdof+idiag]]+=Abase2_ij[:,:,ij12sym[idof,jdof]]
            # Insert symmetric elements in the local mass matrix
            for idof in range(i1,12):
                for jdof in range(idof+1,12):
                        self.inertia.M11[jdof+idiag,idof+idiag]=self.inertia.M11[idof+idiag,jdof+idiag]
    ## Routine that gives the current position and rotation matrix of a node
    def update_node_position_and_rotation(self,inode):
        ielem=inode
        #
        #
        # For now the base node can't be selected, it must be handled later
        #
        r = self.elem_state[ielem].r[3:]
        R = self.elem_state[ielem].Q@self.elem_state[ielem].Q0.T
        return r,R
    ## Routine that converts the rodrigues parameters to rotation peudo vector
    def node_rotations(self):
        phivec=np.zeros((self.nelem,3))
        for ielem in range(self.nelem): 
            R=mf.Rmat(self.q[6*ielem+3:6*ielem+6])
            q=mf.rotmat_to_quaternion(R)
            v,phi=mf.quaternion_to_vector_and_angle(q)
            phivec[ielem,:]=v*phi    
        return phivec
    ## Routine that initiates the integrations over the force and moment distributions
    #
    #  Input:
    #
    #
    #
    #
    #
    def initiate_aeroelastic_coupling(self,iblade,rforce,rmotion,Eforce,rbase,Sbase,base_subs_flag):
        # Blade number to which the forces relate
        self.aero_part=True
        self.iblade=iblade
        # Number of forcing points
        self.nforce=len(rforce)
        # First and last ACP 
        i1acp=self.nforce
        i2acp=0
        # Find the force points affecting each element
        for ielem in range(self.nelem):
            # Transform all aerodynamic calculation point positions and orientations to ECS
            zeta=[]
            rfelm=[]
            rcpelm=[]
            Efelm=[]
            for iforce in range(self.nforce):
                # Position of forcing point in the substructure frame
                rsub=Sbase.T@(rforce[iforce]-rbase)
                # Position of motion point in the substructure frame
                rcpsub=Sbase.T@(rmotion[iforce]-rbase)
                # Position of forcing point in the element frame
                rfelm.append( self.elem_state[ielem].E0.T@(  rsub-self.elem_state[ielem].rmid0))
                # Position of motion point in the element frame
                rcpelm.append(self.elem_state[ielem].E0.T@(rcpsub-self.elem_state[ielem].rmid0))
                # Non-dimensional element coordinate of point
                zeta.append(2.0*rfelm[-1][2]/self.elem_model[ielem].l)
                # Force orientation in the element frame
                Efelm.append(self.elem_state[ielem].E0.T@Eforce[iforce])
            # Make into numpy arrays
            zeta=np.array(zeta)
            rfelm=np.array(rfelm)
            rcpelm=np.array(rcpelm)
            Efelm=np.array(Efelm)
            # Find forcing and motion points on the element
            zeta_tol=1e-3
            iacp_forces=np.nonzero((zeta >= -1.0-zeta_tol) * (zeta <= 1.0+zeta_tol))[0]
            if base_subs_flag and ielem == 0 and 0 not in iacp_forces:
                iacp_forces = np.concatenate([np.array([0]),iacp_forces])
            iacp_motion=iacp_forces.copy()
            # If there are no forcing points on the element add the ends
            if iacp_forces.size == 0:
                i1=np.argmin(zeta[0:-1]*zeta[1:])
                iacp_forces = [i1,i1+1]
            else:
                # Check if the first and last are exact on the nodes
                if np.abs(zeta[iacp_forces[0]]+1.0) > zeta_tol and not (base_subs_flag and ielem == 0):
                    iacp_forces=np.concatenate((np.array([iacp_forces[0]-1]),iacp_forces))
                if np.abs(zeta[iacp_forces[-1]]-1.0) > zeta_tol and iacp_forces[-1]+1 < self.nforce:
                    iacp_forces=np.concatenate((iacp_forces,np.array([iacp_forces[-1]+1])))
            # Save the non-dimensional element coordinate of the forcing point
            zeta_forces=zeta[iacp_forces]
            # Save the 2D position of the forcing points
            rfelm_forces=rfelm[iacp_forces,0:2]
            # Save the orietation of the forces in ECS
            Efelm_forces=Efelm[iacp_forces]
            # Store the data for force transformation
            elem_force=forced_element(iacp_forces,zeta_forces,rfelm_forces,Efelm_forces,self.elem_model[ielem].norder)
            self.elem_force.append(elem_force)
            # Save element number of motion points
            for iacp in iacp_motion:
                # Check if the ACP is already in there
                if iacp in self.acp_data.keys():
                    print('ACP number {:d} is already linked to another element.'.format(iacp))
                # Save motion data for ACP
                self.acp_data[iacp]=acp_motion(ielem,zeta[iacp],rcpelm[iacp]-np.array([0.0,0.0,0.5*self.elem_model[ielem].l*zeta[iacp]]),Efelm[iacp])
                # Update first and last 
                i1acp=np.min([i1acp,iacp])
                i2acp=np.max([i1acp,iacp])
                # Initiate the motion states
                if ielem == 0:
                    idofs=np.arange(6)
                else:
                    idofs=np.arange(6*ielem-6,6*ielem+6)
                self.acp_motion_state[iacp]=gmc.initiate_acp_motion_state(idofs)
        # Save index vectors for forces and motion points on substructure
        self.iforces=np.arange(3*self.elem_force[0].iacp_forces[0],3*self.elem_force[-1].iacp_forces[-1]+3)
        # For each ACP, the motion state vectors are initiated
        self.iacp_motion=np.arange(i1acp,i2acp+1)
        # Update initial positions
        for iacp in self.iacp_motion:
            # Element number
            ielem=self.acp_data[iacp].ielem
            if ielem==0:
                i1=6
            else:
                i1=0
            # Torional point (TP) on the element axis
            relm=np.array([0.0,0.0,0.5*self.acp_data[iacp].zeta*self.elem_model[ielem].l])
            self.acp_motion_state[iacp].rtp=self.elem_state[ielem].rmid+self.elem_state[ielem].E@relm
            # Coallocation point 
            rcpelm=relm+self.acp_data[iacp].rcpelm
            self.acp_motion_state[iacp].rcp=self.elem_state[ielem].rmid+self.elem_state[ielem].E@rcpelm
            # The CCS in substructure frame
            self.acp_motion_state[iacp].Ec=self.elem_state[ielem].E@self.acp_data[iacp].Ecelm
        
    ## Routine that updates the integrations over the force and moment distributions
    #  including the transformation matrix Ta2q from distributed forces and moment to nodal forces
    #
    #  Input:
    #
    #
    #
    #
    #
    def update_aeroelastic_coupling(self):
        # Reset the substructure total force matrix
        self.Tf=np.zeros((3,3*self.nforce))             
        # Reset the substructure total moment matrix
        self.TMf=np.zeros((3,3,3*self.nforce))
        self.TMm=np.zeros((3,3,3*self.nforce))
        # Reset the substructure generalized force matrix
        self.TQf=np.zeros((6*self.nelem,3*self.nforce))
        self.TQm=np.zeros((6*self.nelem,3*self.nforce))
        # Reset the stationary force stiffness matrix for the substructure generalized force
        self.TKQf=np.zeros((6*self.nelem,6*self.nelem,3*self.nforce))
        self.TKQm=np.zeros((6*self.nelem,6*self.nelem,3*self.nforce))
        # Add the contributions from integration of forces and moment over each element
        for ielem in range(self.nelem):
            self.elem_force[ielem].update_forcing_point_position_and_moment_arm_vectors(self.elem_state[ielem],self.elem_model[ielem])
            Tf=     self.elem_force[ielem].compute_element_total_force_matrix( self.elem_model[ielem])
            TMf,TMm=self.elem_force[ielem].compute_element_total_moment_matrix(self.elem_model[ielem])
            TQf,TQm=self.elem_force[ielem].compute_element_generalized_force_matrix(self.elem_model[ielem])    
            TKQf,TKQm=self.elem_force[ielem].compute_element_stiffness_generalized_force_matrix(self.elem_model[ielem])    
            i1=3*self.elem_force[ielem].iacp_forces[0]
            i2=3*self.elem_force[ielem].iacp_forces[-1]+3
            self.Tf [ : ,i1:i2]+=Tf
            self.TMf[:,:,i1:i2]+=TMf
            self.TMm[:,:,i1:i2]+=TMm
            if ielem==0:
                self.TQf[6*ielem:6*ielem+6,i1:i2]+=TQf[6:,:]
                self.TQm[6*ielem:6*ielem+6,i1:i2]+=TQm[6:,:]
                idof1=6
            else:
                self.TQf[6*ielem-6:6*ielem+6,i1:i2]+=TQf
                self.TQm[6*ielem-6:6*ielem+6,i1:i2]+=TQm
                idof1=0
            for idof in range(idof1,12):
                for jdof in range(idof,12):
                    self.TKQf[6*(ielem-1)+idof,6*(ielem-1)+jdof,i1:i2]+=TKQf[ij12sym[idof,jdof],:]
                    self.TKQm[6*(ielem-1)+idof,6*(ielem-1)+jdof,i1:i2]+=TKQm[ij12sym[idof,jdof],:]
                    if jdof>idof:
                        self.TKQf[6*(ielem-1)+jdof,6*(ielem-1)+idof,i1:i2]+=TKQf[ij12sym[idof,jdof],:]
                        self.TKQm[6*(ielem-1)+jdof,6*(ielem-1)+idof,i1:i2]+=TKQm[ij12sym[idof,jdof],:]
        # Update the position vectors for the aerodynamic centers and coallocation points
        for iacp in self.iacp_motion:
            # Element number
            ielem=self.acp_data[iacp].ielem
            if ielem==0:
                i1=6
            else:
                i1=0
            # Shape function matrix
            N=self.elem_model[ielem].local_shape_function_matrix(self.acp_data[iacp].zeta)
            # Local deformation/position vector and its derivatives
            ul=N@self.elem_state[ielem].ql
            dul_dqi=N@self.elem_state[ielem].dqldi0
            # Local rotation matrix
            Rmat=(Imat+mf.Skew(ul[3:6]))
            # Torional point (TP) on the element axis
            relm=ul[0:3]+np.array([0.0,0.0,0.5*self.acp_data[iacp].zeta*self.elem_model[ielem].l])
            self.acp_motion_state[iacp].rtp=self.elem_state[ielem].rmid+self.elem_state[ielem].E@relm
            # Coallocation point 
            rcpelm=relm+Rmat@self.acp_data[iacp].rcpelm
            self.acp_motion_state[iacp].rcp=self.elem_state[ielem].rmid+self.elem_state[ielem].E@rcpelm
            # The CCS in substructure frame
            self.acp_motion_state[iacp].Ec=self.elem_state[ielem].E@Rmat@self.acp_data[iacp].Ecelm
            # First derivatives of 
            i=-1
            for idof in self.acp_motion_state[iacp].idofs:
                i+=1
                # Local deformation
                drelm_dqi=dul_dqi[0:3,i1+i]
                self.acp_motion_state[iacp].drtp_dqi[:,i]=self.elem_state[ielem].E@drelm_dqi+self.elem_state[ielem].dE_dqis[:,:,i]@relm
                drcpelm_dqi=dul_dqi[0:3,i1+i]+mf.Skew(dul_dqi[3:6,i1+i])@self.acp_data[iacp].rcpelm
                self.acp_motion_state[iacp].drcp_dqi[:,i]=self.elem_state[ielem].E@drcpelm_dqi+self.elem_state[ielem].dE_dqis[:,:,i]@rcpelm
                # Add the nodal motion
                if i in [0,1,2]:
                    self.acp_motion_state[iacp].drtp_dqi[:,i]+=0.5*Imat[:,i]
                    self.acp_motion_state[iacp].drcp_dqi[:,i]+=0.5*Imat[:,i]
                elif i in [6,7,8]:
                    self.acp_motion_state[iacp].drtp_dqi[:,i]+=0.5*Imat[:,i-6]
                    self.acp_motion_state[iacp].drcp_dqi[:,i]+=0.5*Imat[:,i-6]


    def compute_rotation_peudo_vector(self):
        # Compute rotations
        rot=np.zeros((3,self.nelem))
        for ielem in range(self.nelem): 
            rot[:,ielem] = mf.pseudo_vector_from_Rodrigues(self.q[6*ielem+3:6*ielem+6])
        return rot






#===============================================================================================
# Internal classes and routines
#===============================================================================================
## Function that changes the reference curve of an existing input file and save a new file
def change_reference_curve(fname,setno,subsetno,new_ref_tab,fname_new):
    # Read structural data format
    fd=open(fname,'r')
    txt=fd.read()
    stset = {}
    for datset in txt.split("#")[1:]:
        datset_nr = int(datset.strip().split()[0])
        subset = {}
        for set_txt in datset.split("@")[1:]:
            set_lines = set_txt.split("\n")
            set_nr, no_rows = map(int, set_lines[0].split()[:2])
            assert set_nr not in subset
            subset[set_nr] = np.array([set_lines[i].split() for i in range(1, no_rows + 1)], dtype=np.float)
        stset[datset_nr] = subset
    stru_sec_data = stset[setno][subsetno]
    # Check format 
    aniso_format = np.size(stru_sec_data,axis=1) == 31
    # Number of rows
    Nst = np.size(stru_sec_data,axis=0)
    # Most data is the same
    new_stru_sec_data = stru_sec_data.copy()
    # Common columns
    stheader = ['z [m]','x_ref [m]','y_ref [m]','angle_ref [deg]','m [kg/m]','x_cg [m]','y_cg [m]','ri_x [m]','ri_y [m]','angle_rix [deg]']
    # Add columns to header 
    if aniso_format:
        for i in range(6):
            for j in range(i,6):
                if i<3 and j<3:
                    stheader.append('C{:d}{:d} [1/N]'.format(i+1,j+1))
                elif i<3:
                    stheader.append('C{:d}{:d} [1/(Nm)]'.format(i+1,j+1))
                else:
                    stheader.append('C{:d}{:d} [1/(Nm^2)]'.format(i+1,j+1))
    else: 
        stheader.append('x_ea [m]')
        stheader.append('y_ea [m]')
        stheader.append('x_sc [m]')
        stheader.append('y_sc [m]')
        stheader.append('angle_bend [deg]')
        stheader.append('E [Pa]')
        stheader.append('G [Pa]')
        stheader.append('A [m^2]')
        stheader.append('Ix [m^4]')
        stheader.append('Iy [m^4]')
        stheader.append('K [m^4]')
        stheader.append('kx [-]')
        stheader.append('ky [-]')
    # Compute new positions
    for i in range(Nst):
        # Insert new reference curve by linear interpolation in the table "new_ref_tab"
        new_stru_sec_data[i,1] = np.interp(new_stru_sec_data[i,0],new_ref_tab[:,0],new_ref_tab[:,1])
        if np.size(new_ref_tab,axis=1) > 2:
            new_stru_sec_data[i,2] = np.interp(new_stru_sec_data[i,0],new_ref_tab[:,0],new_ref_tab[:,2])
        # Displacement of the centers
        cosref = np.cos(np.radians(stru_sec_data[i,3]))
        sinref = np.sin(np.radians(stru_sec_data[i,3]))
        T = np.array([[cosref,sinref],[-sinref,cosref]])
        rref = np.array([[    stru_sec_data[i,1]],[    stru_sec_data[i,2]]])
        rnew = np.array([[new_stru_sec_data[i,1]],[new_stru_sec_data[i,2]]])
        # Center displacement in reference frame
        rshift = T@(rref-rnew)
        # Mass center shift
        new_stru_sec_data[i,5] = stru_sec_data[i,5] + rshift[0]
        new_stru_sec_data[i,6] = stru_sec_data[i,6] + rshift[1]
        # Stiffness part
        if aniso_format:
            C = np.zeros((6,6))
            k=10
            for i in range(6):
                for j in range(i,6):
                    C[i,j] = stru_sec_data[i,k] 
                    C[j,i] = stru_sec_data[i,k]
                    k+=1
            Cnew = transform_reference_point_of_matrix(C,rshift[0],rshift[1],0.0)
            k=10
            for i in range(6):
                for j in range(i,6):
                    new_stru_sec_data[i,k] = Cnew[i,j]
                    k+=1
        else:
            # Centroid shift
            new_stru_sec_data[i,10] = stru_sec_data[i,10] + rshift[0]
            new_stru_sec_data[i,11] = stru_sec_data[i,11] + rshift[1]
            # Shear center shift 
            new_stru_sec_data[i,12] = stru_sec_data[i,12] + rshift[0]
            new_stru_sec_data[i,13] = stru_sec_data[i,13] + rshift[1]
    # Save to file
    header_txt='#1 Structural blade data file \n'+''.join('{:>16s} '.format(text) for text in stheader) + '\n'+'@1 {:d}'.format(Nst)
    np.savetxt(fname_new,new_stru_sec_data,fmt='%16.8e',header=header_txt,comments='',delimiter=' ')
## Class of element geometrical and structural data
class elements_created_from_file:
    ## Creates elements from the structural input data with the following format:
    #      Column 1:     z [m], Distance to section from blade root flange along the pitch axis of the blade frame
    #      Column 2-3:   x_ref and y_ref [m], 2-D vector from pitch axis to the origo (the reference point) of the cross-sectional reference frame
    #      Column 4:     angle_ref [deg], Angle of inplane rotation of the cross-sectional reference frame about the reference point
    #                    The normal to the plane of the cross-section is defined by the tangent of the reference point curve.
    #                    The inplane axes of the cross-sectional reference frame are defined such that the y-axis has zero x-component in the blade frame.
    # Mass properties
    #      Column 5:     m [kg/m], Mass per unit-length 
    #      Column 6-7:   x_cg and y_cg [m], 2-D cross-sectional position vector from ref point to mass center (CG)
    #      Column 8-9:   ri_x and ri_y [m], Radii gyration for rotations of the cross-section about CG in the cross-sectional reference frame
    #      Column 10:    angle_rix [deg] Angular offset of mass inertia x-axis from the cross-sectional reference frame x-axis
    # Stiffness properties (isotropic section = 13 columns or full 6x6 compliance matrix = 21 columns)
    # Isotropic section
    #      Column 11-12: x_ea and y_ea [m], 2D position of the "elastic axis", the centroid of bending in the cross-sectional reference frame
    #      Column 13-14: x_sc and y_sc [m], 2D position of the shear center in the cross-sectional reference frame
    #      Column 15:    angle_bend [deg], Structural twist of the neutral axis of bending about the principal x-axis in the cross-sectional reference frame
    #      Column 16:    E [Pa], Average elastic Young's modulus of the cross-section
    #      Column 17:    G [Pa], Average shear modulus of the cross-section
    #      Column 18:    A [m^2], Area of cross-section
    #      Column 19:    Ix [m^4], Moment of inertia for bending about the principal x-axis
    #      Column 20:    Iy [m^4], Moment of inertia for bending about the principal y-axis
    #      Column 21:    K [m^4], Moment of inertia for torsion
    #      Column 22-23: kx and ky [-], Shear correction factors in the direction of the principal x-axis and y-axis
    # Full 6x6 compliance matrix 
    #      Column 11-31: Upper symmetric elements of the 6x6 compliance matrix row by row [C11, C12, ..., C16, C22, ..., ... C66]
    def __init__(self,fname,znode,setno,subsetno,nintp,norder):
        # Node distribution
        self.nelem=len(znode)-1
        self.z=znode[1:]
        # Read structural data format
        fd=open(fname,'r')
        txt=fd.read()
        stset = {}
        for datset in txt.split("#")[1:]:
            datset_nr = int(datset.strip().split()[0])
            subset = {}
            for set_txt in datset.split("@")[1:]:
                set_lines = set_txt.split("\n")
                set_nr, no_rows = map(int, set_lines[0].split()[:2])
                assert set_nr not in subset
                subset[set_nr] = np.array([set_lines[i].split() for i in range(1, no_rows + 1)], dtype=np.float)
            stset[datset_nr] = subset
        stru_sec_data = stset[setno][subsetno]
        self.stset=stset[setno][subsetno]
        # Compute the 6x6 compliance matrix if the format is an isotropic input 
        Nst = np.size(stru_sec_data,axis=0)
        if np.size(stru_sec_data,axis=1) == 31:
            self.stru_sec_data = stru_sec_data
        else:
            self.stru_sec_data = np.zeros((Nst,31))
            self.stru_sec_data[:,0:10] = stru_sec_data[:,0:10]
            for isec in range(Nst):
                rsc=np.array([stru_sec_data[isec,12],stru_sec_data[isec,13]])
                theta=np.radians(stru_sec_data[isec,14])
                E    =stru_sec_data[isec,15]
                G    =stru_sec_data[isec,16]
                A    =stru_sec_data[isec,17]
                Ix=   stru_sec_data[isec,18]
                Iy=   stru_sec_data[isec,19]
                Iz=   stru_sec_data[isec,20]
                kx=   stru_sec_data[isec,21]
                ky=   stru_sec_data[isec,22]
                C = isotropic_to_6x6_compliance_matrix(np.zeros(2),np.zeros(2),rsc,theta,E,G,A,Ix,Iy,Iz,kx,ky)
                # Insert values
                k=10
                for i in range(6):
                    for j in range(i,6):
                        self.stru_sec_data[isec,k] = C[i,j]
                        k+=1
        # Create data for equilibrium beam elements
        pos_node=np.zeros((3,self.nelem+1))
        pos_node[0,:] = np.interp(znode,stru_sec_data[:,0],stru_sec_data[:,1])
        pos_node[1,:] = np.interp(znode,stru_sec_data[:,0],stru_sec_data[:,2])
        pos_node[2,:] = znode
        self.pos=pos_node
        self.l=np.zeros(self.nelem)
        self.r0s=[]
        self.E0s=[]
        r0=np.zeros(6)
        zeta=np.linspace(-1.0,1.0,nintp)
        # Allocate the compliance matrix and the inertia parameters
        self.Cs=[]
        self.iner_pars=[]
        for ielem in range(self.nelem):
            C=[]
            for i in range(norder+1):
                C.append(np.zeros((6,6)))
            self.Cs.append(C)
            iner_par=np.zeros((norder+1,6))
            self.iner_pars.append(iner_par)
        # Create polynomial fits to interepolated compliance matrices and inertia parameters for each element
        for ielem in range(self.nelem):
            # Interpolation points
            zintp=np.linspace(znode[ielem],znode[ielem+1],nintp)
            # Average the rotation angle of the reference coordinate system
            angle_ref_vec=np.radians(np.interp(zintp,stru_sec_data[:,0],stru_sec_data[:,3]))
            angle_elm = np.mean(angle_ref_vec)
            # Element length and element coordinate system (ECS)
            r0[0:3]=pos_node[:,ielem]
            r0[3: ]=pos_node[:,ielem+1]
            self.l[ielem],E0 = element_coordinate_system(r0,angle_elm)
            self.r0s.append(r0.copy())
            self.E0s.append(E0.copy())
            # Rotation angle of reference frame for each section of interpolation
            dangle_vec = angle_elm - angle_ref_vec 
            # Off-sets of reference point for each section of interpolation
            x_ref = np.interp(zintp,stru_sec_data[:,0],stru_sec_data[:,1])
            y_ref = np.interp(zintp,stru_sec_data[:,0],stru_sec_data[:,2])
            x_elm = 0.5*(r0[0]+r0[3]) + E0[0,2]*0.5*zeta*self.l[ielem]
            y_elm = 0.5*(r0[1]+r0[4]) + E0[1,2]*0.5*zeta*self.l[ielem]
            dx_vec = x_elm - x_ref
            dy_vec = y_elm - y_ref
            # Compliance matrix in element coordinate system for each section of interpolation
            Cintp=np.zeros((6,6,nintp))
            for iintp in range(nintp):
                Cnow=np.zeros((6,6))
                k=10
                for i in range(6):
                    for j in range(i,6):
                        cij = np.interp(zintp[iintp],stru_sec_data[:,0],self.stru_sec_data[:,k])
                        # Insert the coefficents
                        Cnow[i,j]=cij
                        if i != j:
                            Cnow[j,i]=cij
                        # Increment index
                        k+=1
                # Transform matrices
                Cnow = transform_reference_point_of_matrix(Cnow,dx_vec[i],dy_vec[i],dangle_vec[i])
                # Insert transformed matrices
                Cintp[:,:,iintp] = Cnow
            # Fit polynomial to the interpolated compliance matrix
            for i in range(6):
                for j in range(i,6):
                    cij_vec = Cintp[i,j,:]
                    cij_coeff = np.polyfit(zeta,cij_vec,norder)[::-1]
                    # Insert the coefficents
                    for n in range(norder+1):
                        self.Cs[ielem][n][i,j]=cij_coeff[n]
                        if i != j:
                            self.Cs[ielem][n][j,i]=cij_coeff[n]
            # Inertia parameters in element coordinate system for each section of interpolation
            Mintp=np.zeros((6,nintp))
            Mintp[0,:] = np.interp(zintp,stru_sec_data[:,0],stru_sec_data[:,4])
            xcg_ref = np.interp(zintp,stru_sec_data[:,0],stru_sec_data[:,5])
            ycg_ref = np.interp(zintp,stru_sec_data[:,0],stru_sec_data[:,6])
            xcg_elm = xcg_ref*np.cos(dangle_vec)+ycg_ref*np.sin(dangle_vec)-(dx_vec*np.cos(angle_elm)+dy_vec*np.sin(angle_elm))
            ycg_elm =-xcg_ref*np.sin(dangle_vec)+ycg_ref*np.cos(dangle_vec)+(dx_vec*np.sin(angle_elm)-dy_vec*np.cos(angle_elm))
            Mintp[1,:] = Mintp[0,:]*xcg_elm
            Mintp[2,:] = Mintp[0,:]*ycg_elm
            # Radius of gyration for uncoupled rotational inertia in reference system
            rix = np.interp(zintp,stru_sec_data[:,0],stru_sec_data[:,8]) # rotation about y-axis
            riy = np.interp(zintp,stru_sec_data[:,0],stru_sec_data[:,7]) # rotation about x-axis
            # Angle from uncoupled rotational inertia to element coordinate system
            beta = dangle_vec - np.interp(zintp,stru_sec_data[:,0],stru_sec_data[:,9])
            # Rotational inertia in element coordinate system
            
            # Det skal tjekkes. Her er byttet rundt
            
            # Mintp[3,:] = Mintp[0,:]*(0.5*(1.0+np.cos(2.0*beta))*rix**2+0.5*(1.0-np.cos(2.0*beta))*riy**2+xcg_elm**2)
            # Mintp[4,:] = Mintp[0,:]*(0.5*(1.0-np.cos(2.0*beta))*rix**2+0.5*(1.0+np.cos(2.0*beta))*riy**2+ycg_elm**2)
            
            Mintp[4,:] = Mintp[0,:]*(0.5*(1.0+np.cos(2.0*beta))*rix**2+0.5*(1.0-np.cos(2.0*beta))*riy**2+ycg_elm**2)
            Mintp[3,:] = Mintp[0,:]*(0.5*(1.0-np.cos(2.0*beta))*rix**2+0.5*(1.0+np.cos(2.0*beta))*riy**2+xcg_elm**2)
            
            
            
            
            Mintp[5,:] = Mintp[0,:]*(0.5*np.sin(2.0*beta)*(rix**2-riy**2)+xcg_elm*ycg_elm)
            # Fit polynomial to the interpolated inertia parameters
            for i in range(6):
                mpar_vec = Mintp[i,:]
                mpar_coeff = np.polyfit(zeta,mpar_vec,norder)[::-1]
                # Insert the coefficents
                for n in range(norder+1):
                    self.iner_pars[ielem][n,i]=mpar_coeff[n]
    # def plot_discretization():
        
        
## Class with shape functions and stiffness matrices of Krenk and Couturier equilibrium element
#
#
#
#
#
#
class equilibrium_beam_element:
    ## Routine that reads the input 
    def __init__(self,l,C,iner_pars):
        # Save the compliance matrix
        norder=len(C)-1
        self.norder=norder
        self.C=C.copy()
        self.l=l.copy()
        # Define the T1, P, B, and L matrices
        T1=np.zeros((6,6))
        T1[4,0]=-l/2.0
        T1[3,1]= l/2.0
        P=np.zeros((6,6))
        P[0,4]= 1.0
        P[1,3]=-1.0
        B=np.concatenate((np.eye(6),np.zeros((6,6))),1)
        L=np.zeros((12,7))
        L[3,0]=1.0
        L[4,1]=1.0
        L[5,2]=1.0
        L[9,3]=1.0
        L[10,4]=1.0
        L[11,5]=1.0
        L[2,6]=-0.5
        L[8,6]=0.5
        # Compute the H matrix
        self.H=np.zeros((6,6))
        for n in range(0,self.norder+1,2):
            self.H=self.H+l/(n+1.0)*self.C[n]
            tmp=self.C[n]@T1
            self.H=self.H+l/(n+3.0)*T1.T@tmp
        for n in range(1,self.norder+1,2):
            tmp=self.C[n]@T1
            self.H=self.H+l/(n+2.0)*(tmp.T+tmp)
        # Compute the inverse H matrix
        self.Hinv=np.linalg.inv(self.H)
        # Create the G^T matrix
        self.GT=np.concatenate((-np.eye(6)+T1,np.eye(6)+T1)).T
        # Element stiffness matrix
        self.K=self.GT.T@self.Hinv@self.GT
        # Local eLement stiffness matrix
        self.Kl=L.T@self.K@L
        # Compute the polynomial components of the strain functions
        self.Ns=[]
        # ... for 0'th order
        self.Ns.append(self.C[0]@self.Hinv@self.GT)
        # ... for 1 - norder'th order
        for n in range(1,self.norder+1):
            self.Ns.append(self.C[n]@T1@self.Hinv@self.GT-self.C[n]@self.Hinv@self.GT)
        # ... for norder+1'th order
        self.Ns.append(self.C[self.norder]@T1@self.Hinv@self.GT)
        # Create polunomial components of integrated strain functions
        # First integration
        Ns1=[]
        # ... for 0'th order
        Ns1.append(np.zeros((6,12)))
        for n in range(self.norder+2):
            Ns1[0]=Ns1[0]+l/2.0*(-1.0)**n/(n+1)*self.Ns[n]
        # ... for 1 - (norder+2)'th order
        for n in range(1,self.norder+3):
            Ns1.append(l/2.0/n*self.Ns[n-1])       
        # Second integration
        Ns2=[]
        # ... for 0'th order
        Ns2.append(np.zeros((6,12)))
        for n in range(0,self.norder+3):
            Ns2[0]=Ns2[0]+l/2.0*(-1.0)**n/(n+1)*Ns1[n]
        # ... for 1 - (norder+2)'th order
        for n in range(1,self.norder+4):
            Ns2.append(l/2.0/n*Ns1[n-1])
        # Compute the polynomial components of the shape functions
        self.N=[]
        # ... for 0'th order
        self.N.append(Ns1[0]+B+P@(Ns2[0]+l/2.0*B))
        # ... for 1'th order
        self.N.append(Ns1[1]+P@(Ns2[1]+l/2.0*B))
        # ... for 2 - (norder+2)'th order
        for n in range(2,self.norder+3):
            self.N.append(Ns1[n]+P@Ns2[n])
        # ... for (norder+3)'th order
        self.N.append(P@Ns2[self.norder+3])
        # Compute the polynomial components of the local shape functions
        self.Nl=np.zeros((6,7,self.norder+4))
        for n in range(self.norder+4):
            self.Nl[:,:,n]=self.N[n]@L
        # Insert inertia parameters 
        self.iner_pars=iner_pars
        # Compute element mass
        self.mass = 0.0
        for r in range(self.norder+1):
            self.mass += 0.5*self.l*c_function(r)*self.iner_pars[r,0]
    ## Full 6x12 Shape function evaluation for a single zeta
    def shape_function_matrix(self,zeta):
        N=np.zeros((6,12))
        for n in range(self.norder+4):
            N=N+self.N[n]*zeta**n
        return N
    ## Local 6x7 shape function evaluation for a single zeta
    def local_shape_function_matrix(self,zeta):
        Nl=np.zeros((6,7))
        for n in range(self.norder+4):
            Nl=Nl+self.Nl[:,:,n]*zeta**n
        return Nl
    ## Local element deflections and rotations for several zeta
    def local_beam_element_deflection(self,ql,zeta):
        npoints=zeta.size   
        ul=np.zeros((6,npoints))
        for i in range(npoints):
            N=self.local_shape_function_matrix(zeta[i])
            ul[:,i]=N@ql+np.array([0.0,0.0,0.5*zeta[i]*self.l,0.0,0.0,0.0])
        return ul

    # Function that prints the element properties
    def print_element_properties(self):
        print('Element properties:')
        print('    l = {:.2e} m  '.format(self.l))
        print('    m = {:.2e} kg '.format(self.mass))


## Class that contains the kinematics of the co-rotational element
#
#
#
#
#
class corotbeam_element_kinematics:
    ## Initialization of the co-rotational element kinematics. Requires order of the equilibrium_beam_element
    def __init__(self,l0,r0,E0,norder):
        # Length, position and orientation
        self.l0=l0.copy() 
        self.r0=r0.copy()
        self.T0=E0.copy()
        self.Q0=E0.copy()
        self.rvec0=r0[3:]-r0[0:3]
        # Dynamic values
        self.l=self.l0.copy()
        self.T=self.T0.copy()
        self.Q=self.Q0.copy()
        self.E=E0.copy()
        self.E0=E0.copy() # Saved for later
        self.r=r0.copy()
        self.dvec=np.zeros(3)
        self.rvec=r0[3:]-r0[0:3]
        self.rmid=0.5*(r0[3:]+r0[0:3])
        self.rmid0=0.5*(r0[3:]+r0[0:3])
        self.d=2.0
        # First derivatives
        self.dT=np.zeros((3,3,3))
        self.dQ=np.zeros((3,3,3))
        self.dd=np.zeros(12)
        # Second derivatives
        self.ddT=np.zeros((3,3,6))
        self.ddQ=np.zeros((3,3,6))
        # Derivatives of the element coordinate system
        self.dE_dqis=np.zeros((3,3,12))
        self.ddE_dqidqjs=np.zeros((3,3,78))
        # Initial triad and position derivatives
        self.compute_element_triad_and_position()
        # Nodal rotations and element elongation
        self.ql=np.zeros(7)
        self.dqldi0=np.zeros((7,12))
        self.ddqldidj0=np.zeros((7,12,12))
        # Deflection sub-vectors and their derivatives
        self.ro=np.zeros((3,norder+4))
        self.rx=np.zeros((3,norder+4))
        self.ry=np.zeros((3,norder+4))
        self.dro_dqi=np.zeros((3,12,norder+4))
        self.drx_dqi=np.zeros((3,12,norder+4))
        self.dry_dqi=np.zeros((3,12,norder+4))
        self.ddro_dqidqj=np.zeros((3,78,norder+4))
        self.ddrx_dqidqj=np.zeros((3,78,norder+4))
        self.ddry_dqidqj=np.zeros((3,78,norder+4))
        # Coefficients of shape functions multiplied by the element deformation and its derivatives
        self.Nlql=np.zeros((6,norder+4))
        self.Nldqldqi=np.zeros((6,12,norder+4))
        self.Nlddqldqidqj=np.zeros((6,78,norder+4))
    ## Update position and orientation of nodal triads
    def update_nodal_triads_and_position(self,q):
        self.r,self.dvec,self.T,self.Q,self.dT,self.dQ,self.ddT,self.ddQ=cpf.update_nodal_triads_and_position(q,self.r0,self.T0,self.Q0)
        # Update triad and position derivatives
        self.compute_element_triad_and_position()
    ## Compute element triad using the average of t2 and q2
    def compute_element_triad_and_position(self):
        self.l,self.E,self.rvec,self.rmid,self.d,self.dd,self.dE_dqis,self.ddE_dqidqjs = \
            cpf.compute_element_triad_and_position(self.r,self.T,self.Q,self.dT,self.dQ,self.ddT,self.ddQ)
    ## Compute local nodal rotations (assumed small) and elongation
    def update_local_nodal_rotations_elongation(self):
        self.ql=cpf.update_local_nodal_rotations_elongation(self.l0,self.l,self.rvec0,self.rvec,self.dvec,self.E,self.T,self.Q)
    ## Compute first derivatives of local nodal rotations (assumed small) and elongation
    def update_first_derivative_local_nodal_rotations_elongation(self):
        self.dqldi0=cpf.update_first_derivative_local_nodal_rotations_elongation(self.l,self.rvec,self.E,self.dE_dqis,self.T,self.Q,self.dT,self.dQ)
        
    ## Compute second derivatives of local nodal rotations (assumed small) and elongation
    def update_second_derivative_local_nodal_rotations_elongation(self):
        self.ddqldidj0=cpf.update_second_derivative_local_nodal_rotations_elongation(self.l,self.rvec,self.E,self.dE_dqis,self.ddE_dqidqjs, \
                                                                                     self.T,self.Q,self.dT,self.dQ,self.ddT,self.ddQ)
        
    ## Compute deflection sub-vector coefficients for each shape function order
    def update_element_deflection_subvectors_and_derivatives(self,Nl):
        self.Nlql,self.Nldqldqi,self.Nlddqldqidqj,self.ro,self.rx,self.ry,self.dro_dqi,self.drx_dqi,self.dry_dqi,self.ddro_dqidqj,self.ddrx_dqidqj,self.ddry_dqidqj = \
            cpf.update_element_deflection_subvectors_and_derivatives(self.l,self.rmid,self.E,self.dE_dqis,self.ddE_dqidqjs,Nl,self.ql,self.dqldi0,self.ddqldidj0)

        
## Class with coupling feedback of element motion 
#
#   Input:
#
#
#
#
class acp_motion:
    # Init
    def __init__(self,ielem,zeta,rcpelm,Ecelm):
        # Save input
        self.ielem=ielem
        self.zeta=zeta.copy()
        self.rcpelm=rcpelm.copy() 
        self.Ecelm=Ecelm.copy()
        
## Class with coupling of element node forces to external force distribution
#
#   Input:
#
#
#
#
class forced_element:
    # Init
    def __init__(self,iacp_forces,zeta_forces,rfelm_forces,Efelm_forces,norder):
        # Save input
        self.iacp_forces=iacp_forces.copy()
        self.zeta_forces=zeta_forces.copy()
        self.rfelm_forces=rfelm_forces.copy()
        self.Efelm_forces=Efelm_forces.copy()
        # Number of integration intervals
        self.ninterval=len(iacp_forces)-1
        # Compute the linear interpolation coefficients of forcing point position and vector arm for moment force in ECS
        self.a=np.zeros(self.ninterval)
        self.b=np.zeros(self.ninterval)
        self.w=np.zeros((self.ninterval,2,2))
        self.cfx=np.zeros((self.ninterval,2))
        self.cfy=np.zeros((self.ninterval,2))
        self.ce1=np.zeros((3,self.ninterval,2))
        for m in range(self.ninterval):
            self.a[m]=np.max([-1.0,zeta_forces[m]])
            self.b[m]=np.min([ 1.0,zeta_forces[m+1]])
            dzeta=self.zeta_forces[m+1]-self.zeta_forces[m]
            self.w[m,0,0]= zeta_forces[m+1]/dzeta
            self.w[m,0,1]=-zeta_forces[  m]/dzeta
            self.w[m,1,0]=-1.0/dzeta
            self.w[m,1,1]= 1.0/dzeta
            self.cfx[m,0]=rfelm_forces[m,0]*self.w[m,0,0]+rfelm_forces[m+1,0]*self.w[m,0,1]
            self.cfx[m,1]=rfelm_forces[m,0]*self.w[m,1,0]+rfelm_forces[m+1,0]*self.w[m,1,1]
            self.cfy[m,0]=rfelm_forces[m,1]*self.w[m,0,0]+rfelm_forces[m+1,1]*self.w[m,0,1]
            self.cfy[m,1]=rfelm_forces[m,1]*self.w[m,1,0]+rfelm_forces[m+1,1]*self.w[m,1,1]
            self.ce1[:,m,0]=Efelm_forces[m,0,:]*self.w[m,0,0]+Efelm_forces[m+1,0,:]*self.w[m,0,1]
            self.ce1[:,m,1]=Efelm_forces[m,0,:]*self.w[m,1,0]+Efelm_forces[m+1,0,:]*self.w[m,1,1]
        # Initialization of variables that need updating
        #
        # Coefficients of forcing point position and vector arm for moment force in substructure frame
        self.rf=np.zeros((3,self.ninterval,norder+4))
        self.e1=np.zeros((3,self.ninterval,norder+4))
        # and their first and second derivatives
        self.drf_dqi=np.zeros((3,12,self.ninterval,norder+4))
        self.de1_dqi=np.zeros((3,12,self.ninterval,norder+4))
        self.ddrf_dqidqj=np.zeros((3,78,self.ninterval,norder+4))
        self.dde1_dqidqj=np.zeros((3,78,self.ninterval,norder+4))
    # Compute element total force matrix
    def compute_element_total_force_matrix(self,elem_model):
        Tf=cpf.compute_element_total_force_matrix(self.ninterval,elem_model.l,self.a,self.b,self.w)
        return Tf
    # Compute element total moment matrix 
    def compute_element_total_moment_matrix(self,elem_model):
        TMf,TMm=cpf.compute_element_total_moment_matrix(self.ninterval,elem_model.norder,elem_model.l,self.a,self.b,self.w,self.rf,self.e1)
        return TMf,TMm
    # Compute element generalized force matrix
    def compute_element_generalized_force_matrix(self,elem_model):
        TQf,TQm=cpf.compute_element_generalized_force_matrix(self.ninterval,elem_model.norder,elem_model.l,self.a,self.b,self.w,self.drf_dqi,self.de1_dqi)
        return TQf,TQm
    # Compute element stiffness matrix from stationary generalized force 
    def compute_element_stiffness_generalized_force_matrix(self,elem_model):
        TKQf,TKQm=cpf.compute_element_stiffness_generalized_force_matrix(self.ninterval,elem_model.norder,elem_model.l,self.a,self.b,self.w,self.ddrf_dqidqj,self.dde1_dqidqj)
        return TKQf,TKQm
    # Update the forcing point position and vector arm for moment force and their derivatives
    def update_forcing_point_position_and_moment_arm_vectors(self,elem_state,elem_model):
        self.rf,self.e1,self.drf_dqi,self.de1_dqi,self.ddrf_dqidqj,self.dde1_dqidqj= \
            cpf.update_forcing_point_position_and_moment_arm_vectors(self.ninterval,elem_model.norder,elem_state.ro,elem_state.rx,elem_state.ry,\
                                                                     elem_state.dro_dqi,elem_state.drx_dqi,elem_state.dry_dqi,elem_state.ddro_dqidqj,elem_state.ddrx_dqidqj,elem_state.ddry_dqidqj, \
                                                                     self.cfx,self.cfy,self.ce1,elem_state.E,elem_state.dE_dqis,elem_state.ddE_dqidqjs,elem_state.Nlql,elem_state.Nldqldqi,elem_state.Nlddqldqidqj)
