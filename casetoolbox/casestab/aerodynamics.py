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
## @package aerodynamics
#  The aerodynamics of a wind turbine:
#      rotor_aero: 
#
# 
import numpy as np
from . import math_functions as mf

#================================================================================================
## Class containing the aero model and states of a blade.
#
#
#
#
class aero_blade:
    ## Init routine reads the file with the blade aero geometry and profile definition
    #  and generates a blade with its aero states and model properties
    # Input in para dict:
    #   geo_file:   File with blade geometry
    #      Column 1:     z [m], distance to section from blade root flange along the pitch axis
    #      Column 2-3:   x_ccs and y_ccs [m], 2-D vector from pitch axis to origo of chord coordinate system (CCS)
    #      Column 4-6:   phi_x, phi_y, phi_z [deg], pseudo vector of finite rotation of CCS from blade CS.
    #                    If phi_x = phi_y = 0 then the normal to the plane of the airfoil is defined by 
    #                    the tangent of the origo curve and the inplane axes of the CCS are defined such that 
    #                    the y-axis has zero x-component in the blade frame before rotating them about the tangent by phi_z. 
    #      Column 7:     c [m], chord length of airfoil or diameter of circle
    #      Column 8:     rel_thick [%], relative thickness of airfoil (100% = circle)
    #      Column 9-10:  x_ac and y_ac [m], 2-D cross-sectional position vector from origo of CCS to aerodynamic pressure center (AC)
    #      Column 11:    a_ac [-], non-dimensional chordwise position of the AC for visualization of the blade 
    #      Column 12:    pc_set [-], set number in PC file related to the airfoil type
    #   geo_set:   Set number to use for the blade geometry [string]
    #   pro_file:  File with airfoil polars in HAWC2 format [string]
    #   zaero:     Distribution type of aerodynamic calculation points along the pitch axis [np.ndarray, linear N, cosine N]
    #   ae_inter:  Method of interpolation in the AE data [linear, akima, pchip]
    #   geo_inter: Method of interpolation in the reference point for the CCS [linear, akima, pchip]
    #   pro_inter: Method and AoA step size of polar interpolation [linear da, akima da, pchip da]
    # 
    # Output 
    def __init__(self,para):
        # Status
        self.status = ''
        # Read blade geometry
        fd=open(para['geo_file'],'r')
        txt=fd.read()
        aeset = {}
        for set_txt in txt.split("@")[1:]:
            set_lines = set_txt.split("\n")
            set_nr, no_rows = map(int, set_lines[0].split()[:2])
            assert set_nr not in aeset
            aeset[set_nr] = np.array([set_lines[i].split() for i in range(1, no_rows + 1)], dtype=np.float)
        fd.close()
        self.aeheader = ['z [m]','x_ccs [m]','y_ccs [m]','phi_x [deg]','phi_y [deg]','phi_z [deg]','c [m]','rel_thick [%]','x_ac [m]','y_ac [m]','a_ac [-]','pc_set [-]']
        self.aeset=aeset[para['geo_set']]
        self.aeset_curve=mf.curve_interpolate(para['ae_inter'].split()[0],self.aeset[:,0], self.aeset[:,1:])
        # Create interpolation function of the ference curve of the CCS origin
        self.rc_curve = mf.curve_interpolate(para['geo_inter'].split()[0],self.aeset[:,0], self.aeset[:,1:3])
        # Read airfoil polars and interpolate them to equidistant arrays
        fd=open(para['pro_file'],'r')
        txt=fd.read()
        lines=txt.split("\n")
        daoa = np.float(para['pro_inter'].split()[1])
        self.naoa = int(360.0/daoa) + 1
        aoas_deg = np.linspace(-180.0,180.0,self.naoa)
        # All AoAs are in radians
        self.aoas = np.radians(aoas_deg)
        iline=0
        nset=int(lines[iline].split()[0])
        self.prosets = {}
        self.origprosets = {}
        for iset in range(nset):
            iline+=1
            nairfoil = int(lines[iline].split()[0])
            proairfoil = {}
            origproairfoil = {}
            for iairfoil in range(nairfoil):
                iline+=1
                airfoil_nr, nrows = map(int, lines[iline].split()[:2])
                thickness = np.float(lines[iline].split()[2])
                polar = np.array([lines[iline+i+1].split() for i in range(nrows)], dtype=np.float)
                origproairfoil[thickness] = polar
                iline+=nrows
                polar_curve = mf.curve_interpolate(para['pro_inter'].split()[0],polar[:,0], polar[:,1:])
                proairfoil[thickness] = polar_curve.fcn(aoas_deg)
            self.prosets[iset] = proairfoil
            self.origprosets[iset] = origproairfoil
        # Define the aerodynamic calculation points
        if isinstance(para['zaero'], np.ndarray):
            self.zaero = para['zaero']
            maxdiff = 1e-3
            if np.abs(self.zaero[-1]-self.aeset[-1,0]) > maxdiff:
                self.status = 'ERROR: End point of user-defined aero calculation points differ more than {:5.1e} m from last point in geometry file'.format(maxdiff)
                return
        elif para['zaero'].split()[0] == 'linear':
            self.zaero = np.linspace(0.0,np.float(para['zaero'].split()[2]),int(para['zaero'].split()[1]))
        elif para['zaero'].split()[0] == 'cosine':
            theta = np.linspace(0.0,np.pi,int(para['zaero'].split()[1]))
            if len(para['zaero'].split()) < 3:
                self.zaero = 0.5*self.aeset[-1,0]*(1.0 - np.cos(theta)) 
            else:
                self.zaero = 0.5*np.float(para['zaero'].split()[2])*(1.0 - np.cos(theta))
        else:
            self.zaero = self.aeset[:,0]
        # Compute the aero calculation point data
        self.naero=len(self.zaero)
        self.aero_point={}
        self.rac=[]
        self.rcp=[]
        self.Ec=[]
        for iaero in range(self.naero):
            # AE values at the current section
            aehere = self.aeset_curve.fcn(self.zaero[iaero])
            # Origin of coordinate system
            rc = self.rc_curve.fcn(self.zaero[iaero])
            # Chord coordinate system in the blade frame
            Ec = self.chord_coordinate_system(self.zaero[iaero])
            # Local vector to AC
            rac = np.array([aehere[7],aehere[8],0.0])
            # Chord length
            c = aehere[5]
            # Nondimensional chordwise position of the AC
            aac = aehere[9]
            # Polar interpolated on relative thickness and set number
            thk = aehere[6]
            iset = int(np.around(aehere[10])) - 1
            thk_list = np.array(list(self.prosets[iset]))
            if thk < thk_list[0]:
                self.status = 'WARNING: Airfoil thickness at aero calc point {:d} is {:6.2f} which is below the thinnest airfoil {:6.2f} of polar set {:d} which is then used.'.format(iaero+1,thk,thk_list[0],iset)
                clcdcm = self.prosets[iset][thk_list[0]]
            elif thk > thk_list[-1]:
                self.status = 'WARNING: Airfoil thickness at aero calc point {:d} is {:6.2f} which is above the thickness airfoil {:6.2f} of polar set {:d} which is then used.'.format(iaero+1,thk,thk_list[-1],iset)
                clcdcm = self.prosets[iset][thk_list[-1]]
            else:
                i1=np.nonzero(thk_list<=thk)[0][-1]
                i2=np.nonzero(thk_list>=thk)[0][0]
                if i1==i2:
                    clcdcm = self.prosets[iset][thk_list[i1]]
                else:
                    gamma=(thk-thk_list[i1])/(thk_list[i2]-thk_list[i1])
                    clcdcm = self.prosets[iset][thk_list[i1]]*(1.0-gamma)+self.prosets[iset][thk_list[i2]]*gamma
            # Create aero point object
            self.aero_point[iaero] = aero_calc_point(rc,Ec,rac,c,thk,aac,self.aoas,clcdcm)
            # Save in position vectors and orinetation matrices for substructure call
            self.rac.append(np.array([rc[0],rc[1],self.zaero[iaero]])+Ec@rac)
            self.rcp.append(np.array([rc[0],rc[1],self.zaero[iaero]])+Ec@(rac+np.array([(2.0*aac-1.0)*c,0.0,0.0])))
            self.Ec.append(Ec)
        # Save initial positions and orientations
        self.rac0 = self.rac.copy()
        self.Ec0 = self.Ec.copy()
            
        # Allocate index list for the substructure number for each ACP
        self.isubs=[]
        # Initiate force and moment vectors defined in the frames of each substructure
        self.f=np.zeros(3*self.naero)
        self.m=np.zeros(3*self.naero)

            
    ## Routine that computes the CCS in case only the aero twist is given (phi_x = phi_y = 0)
    def chord_coordinate_system(self,z):
        # AE values at the current section
        aehere = self.aeset_curve.fcn(z)
        # Chord coordinate system
        if np.abs(aehere[2])+np.abs(aehere[3]) < 1.0e-10:
            nc = np.ones(3)
            nc[0:2] = self.rc_curve.der(z)
            twist = np.radians(aehere[4])
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
            Ec = mf.rotmat_from_pseudovec(np.radians(aehere[2:5]))
        return Ec
    

    ## Routine that collects the aerodynamic states and forces for each ACP in a single array
    def states_and_forces(self):
        dat=np.zeros((self.naero,13))
        for iaero in range(self.naero):
            dat[iaero, 0] = self.aero_point[iaero].aoa_cp
            dat[iaero, 1] = self.aero_point[iaero].aoa_tp
            dat[iaero, 2] = self.aero_point[iaero].vrel[0]
            dat[iaero, 3] = self.aero_point[iaero].vrel[1]
            dat[iaero, 4] = self.aero_point[iaero].urel
            dat[iaero, 5] = self.aero_point[iaero].CL
            dat[iaero, 6] = self.aero_point[iaero].CD
            dat[iaero, 7] = self.aero_point[iaero].CM
            dat[iaero, 8] = self.aero_point[iaero].fx
            dat[iaero, 9] = self.aero_point[iaero].fy
            dat[iaero,10] = self.aero_point[iaero].M
            dat[iaero,11] = self.aero_point[iaero].cl.der(self.aero_point[iaero].aoa_tp)
            dat[iaero,12] = self.aero_point[iaero].cd.der(self.aero_point[iaero].aoa_tp)
        return dat


    def geometry(self):
        dat=np.zeros((self.naero,9))
        for iaero in range(self.naero):
            dat[iaero,0:3] = self.rac0[iaero]+ self.aero_point[iaero].aac     *self.aero_point[iaero].c*self.Ec0[iaero][:,0]
            dat[iaero,3:6] = self.rac0[iaero]+(self.aero_point[iaero].aac-1.0)*self.aero_point[iaero].c*self.Ec0[iaero][:,0]
            dat[iaero,6:9] = self.rac0[iaero]
        return dat




    
#================================================================================================
## Class containing the data and functions of an aerodynamic calculation point
#
# Parameters and variables:
#   z:          Position of aero point on pitch axis [m]
#   c:          Chord length
#   clcdcm:     Airfoil polar with AoA step size given by aero_blade.daoa
#   Ec0:        Initial chord coordinate system 
#   Ec:         Current chord coordinate system 
#   xa:         Current aerodynamic state variables
#
# Functions:
#   cl(aoa):    
#    
#    
class aero_calc_point:
    def __init__(self,rc,Ec,rac,c,thk,aac,aoas,clcdcm):
        # Status
        self.status = ''
        # Initial geometry
        self.rc0=rc.copy()
        self.Ec0=Ec.copy()
        self.rac=rac.copy()
        self.c=c.copy()
        self.thk=thk.copy()
        self.aac=aac.copy()
        # Updated geometry
        self.rc=rc.copy()
        self.Ec=Ec.copy()
        # Save polars and the step in AoAs
        self.aoas = aoas.copy()
        self.clcdcm = clcdcm.copy()
        # Initiate vectors of AoAs and relative velocities and speed 
        self.aoa_cp=0.0
        self.aoa_tp=0.0
        self.vrel=np.zeros(2)
        self.urel=0.0
        self.CL=0.0
        self.CD=0.0
        self.CM=0.0
        self.fx=0.0
        self.fy=0.0
        self.M=0.0
        # Create function for cl, cd, and cm evaluations
        self.cl = mf.quick_interpolation_periodic_function(self.aoas,self.clcdcm[:,0])
        self.cd = mf.quick_interpolation_periodic_function(self.aoas,self.clcdcm[:,1])
        self.cm = mf.quick_interpolation_periodic_function(self.aoas,self.clcdcm[:,2])
        
        

    
    ## Routine that computes the aerodynamic forces at the ACP
    def update_steady_aero_forces(self,rho):
        # Steady state aerodynamic coefficients
        self.CL = self.cl.fcn(self.aoa_cp)
        self.CD = self.cd.fcn(self.aoa_cp)
        self.CM = self.cm.fcn(self.aoa_cp)
        # CCS forces and moment
        a = 0.5*rho*self.c*self.urel
        self.fx = a*(self.vrel[1]*self.CL+self.vrel[0]*self.CD)
        self.fy = a*(self.vrel[1]*self.CD-self.vrel[0]*self.CL)
        self.M = a*self.c*self.urel*self.CM
        # Transform forces to substructure frame
        f = self.Ec@np.array([[self.fx],[self.fy],[0.0]])
        # Create moment vector cf. Eq. (2.17)
        m = self.M*self.Ec[:,1]
        return f,m


#===============================================================================================
# Internal classes and routines
#===============================================================================================
## Function that changes the reference curve of an existing input file and save a new file
def change_reference_curve(fname,setno,new_ref_tab,fname_new):
     # Read blade geometry
    fd=open(fname,'r')
    txt=fd.read()
    aeset = {}
    for set_txt in txt.split("@")[1:]:
        set_lines = set_txt.split("\n")
        set_nr, no_rows = map(int, set_lines[0].split()[:2])
        assert set_nr not in aeset
        aeset[set_nr] = np.array([set_lines[i].split() for i in range(1, no_rows + 1)], dtype=np.float)
    fd.close()
    aeheader = ['z [m]','x_ccs [m]','y_ccs [m]','phi_x [deg]','phi_y [deg]','phi_z [deg]','c [m]','rel_thick [%]','x_ac [m]','y_ac [m]','a_ac [-]','pc_set [-]']
    aero_sec_data=aeset[setno]       
    # Number of rows
    Nae = np.size(aero_sec_data,axis=0)
    # Most data is the same
    new_aero_sec_data = aero_sec_data.copy()
    # Compute new positions
    for i in range(Nae):
        # Insert new reference curve by linear interpolation in the table "new_ref_tab"
        new_aero_sec_data[i,1] = np.interp(new_aero_sec_data[i,0],new_ref_tab[:,0],new_ref_tab[:,1])
        if np.size(new_ref_tab,axis=1) > 2:
            new_aero_sec_data[i,2] = np.interp(new_aero_sec_data[i,0],new_ref_tab[:,0],new_ref_tab[:,2])
        # Displacement of the centers
        cosref = np.cos(np.radians(aero_sec_data[i,5]))
        sinref = np.sin(np.radians(aero_sec_data[i,5]))
        T = np.array([[cosref,sinref],[-sinref,cosref]])
        rref = np.array([[    aero_sec_data[i,1]],[    aero_sec_data[i,2]]])
        rnew = np.array([[new_aero_sec_data[i,1]],[new_aero_sec_data[i,2]]])
        # Center displacement in reference frame
        rshift = T@(rref-rnew)
        # aero center shift
        new_aero_sec_data[i,8] = aero_sec_data[i,8] + rshift[0]
        new_aero_sec_data[i,9] = aero_sec_data[i,9] + rshift[1]
    # Save to file
    header_txt='Aerodynamic blade data \n'+''.join('{:>16s} '.format(text) for text in aeheader) + '\n'+'@1 {:d}'.format(Nae)
    np.savetxt(fname_new,new_aero_sec_data,fmt='%16.8e',header=header_txt,comments='',delimiter=' ')



#================================================================================================
## Class containing the unsteady aerodynamic airfoil model (dynamic stall)
#
#
#
#
# class unsteady_airfoil_dynamics:


    
    
    ## Routine that initializes the Beddoes-Leishman dynamic stall model
    
    
    
    ## 




