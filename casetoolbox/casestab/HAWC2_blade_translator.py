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
## @package HAWC2_blade_translator
#  
#
# 
import numpy as np
import scipy as sp
from . import math_functions as mf
from .timoshenko_beam_section import isotropic_to_6x6_compliance_matrix
## Routine that reads and inserts the structural elements of a HAWC2 model 
class HAWC2_elements:
    def __init__(self,fname,mbdy_name):
        f=open(fname,'r')
        txt=f.readlines()
        readpos=False
        readrot=False
        readdat=False
        elempos=[]
        elemrot=[]
        elemdat=[]
        for word in txt:
            if mbdy_name in word:
                readpos=True
                i=-4
            if readpos:
                i+=1
                if 'Element orientation in body coordinates' in word:
                    self.nelem=i-1
                    i=-3
                    readpos=False
                    readrot=True
                if i > 0:
                    words=word.split()
                    tal=np.array([float(x) for x in words])
                    elempos.append(tal)
            if readrot:
                i+=1
                if 'Element orientation in global coordinates' in word:
                    i=-5-self.nelem
                    readrot=False
                    readdat=True
                if i > 0:
                    words=word.split()
                    tal=np.array([float(x) for x in words])
                    elemrot.append(tal)
            if readdat:
                i+=1
                if '______' in word: # Reached the next body
                    break
                if i > 0:
                    words=word.split()
                    tal=np.array([float(x) for x in words])
                    elemdat.append(tal)
        # Close file
        f.close()
        # Initiate the compliance matrix
        C=[]
        C.append(np.zeros((6,6)))
        # Insert compliance matrices
        self.l=np.zeros(self.nelem)
        self.z=np.zeros(self.nelem)
        self.pos=np.zeros((3,self.nelem+1))
        self.r0s=[]
        self.E0s=[]
        self.Cs=[]
        self.iner_pars=[]
        for i in range(self.nelem):
            ex=elemdat[i][ 7].copy()
            ey=elemdat[i][ 8].copy()
            E =elemdat[i][ 9].copy()
            G =elemdat[i][10].copy()
            Ix=elemdat[i][11].copy()
            Iy=elemdat[i][12].copy()
            Iz=elemdat[i][13].copy()
            kx=elemdat[i][14].copy()
            ky=elemdat[i][15].copy()
            A =elemdat[i][16].copy()
            # Insert the prismatic element data from the HAWC2 model
            rsc = np.array([ex,ey])
            theta = 0.0
            C[0] = isotropic_to_6x6_compliance_matrix(np.zeros(2),np.zeros(2),rsc,theta,E,G,A,Ix,Iy,Iz,kx,ky)
            self.Cs.append(C.copy())
            # Insert element positions and orientation
            r0=np.zeros(6)
            r0[0:3]=elempos[i][3:6]
            r0[3:]=elempos[i][3:6]+elempos[i][6:]
            self.r0s.append(r0)
            E0=elemrot[i][2:].reshape(3,3).T
            self.E0s.append(E0)
            self.l[i]=np.sqrt((r0[3]-r0[0])**2+(r0[4]-r0[1])**2+(r0[5]-r0[2])**2)
            self.pos[:,i+1]=elempos[i][3:6]+elempos[i][6:9]
            self.z[i]=self.pos[2,i+1]
            # Inertia parameters in element coordinate system for each section of interpolation
            Mpar = np.zeros(6)
            Mpar[0] = elemdat[i][ 2].copy()
            Mpar[1] = Mpar[0]*elemdat[i][ 3].copy()
            Mpar[2] = Mpar[0]*elemdat[i][ 4].copy()
            # Rotational inertia in element coordinate system
            Mpar[3] = Mpar[0]*elemdat[i][ 5].copy()**2
            Mpar[4] = Mpar[0]*elemdat[i][ 6].copy()**2
            Mpar[5] = 0.0
            # Fit polynomial to the interpolated inertia parameters
            self.iner_pars.append(np.zeros((1,6)))
            # Insert the coefficents
            self.iner_pars[i][0,:]=Mpar.copy()
## Routine that computes the chord coordinate system (CCS) of the blade
def chord_coordinate_system_in_HAWC2(c2def,z):
    # Slopes of half chord curve
    xc2 = sp.interpolate.Akima1DInterpolator(c2def[:,2], c2def[:,0]).__call__(z,nu=0,extrapolate=True)
    yc2 = sp.interpolate.Akima1DInterpolator(c2def[:,2], c2def[:,1]).__call__(z,nu=0,extrapolate=True)
    # Slopes of half chord curve
    xp = sp.interpolate.Akima1DInterpolator(c2def[:,2], c2def[:,0]).__call__(z,nu=1,extrapolate=True)
    yp = sp.interpolate.Akima1DInterpolator(c2def[:,2], c2def[:,1]).__call__(z,nu=1,extrapolate=True)
    # Aero twist
    twist = np.radians(np.interp(z, c2def[:,2], c2def[:,3]))
    # Loop through each section
    n = len(z)
    rc2=[]
    phi=[]
    ccs=[]
    for isec in range(n):
        # CCS in blade coordinate system
        Ec = np.zeros((3,3))
        # its third vector is the tangent
        l = np.sqrt(1.0 + xp[isec]**2 + yp[isec]**2)
        Ec[0,2] = xp[isec]/l
        Ec[1,2] = yp[isec]/l
        Ec[2,2] = 1.0/l
        # its second  vector is defined as the unit-vector with zero x-component 
        Ec[1,1] = Ec[2,2]/np.sqrt(Ec[1,2]**2 + Ec[2,2]**2)
        Ec[2,1] = -Ec[1,1]*Ec[1,2]/Ec[2,2]
        # and rotated by the twist angle
        R = mf.rotmat(Ec[:,2],twist[isec])
        Ec[:,1] = R@Ec[:,1]
        # its first vector is the cross-product of the other two vectors
        Ec[:,0] = mf.crossproduct(Ec[:,1], Ec[:,2])
        # Save
        ccs.append(Ec)
        rc2.append(np.array([xc2[isec],yc2[isec]]))
        # Translate to angle and vector of finite rotation
        q = mf.rotmat_to_quaternion(Ec)
        vec,ang = mf.quaternion_to_vector_and_angle(q)
        # Pseudo vector of finite rotation
        phi.append(ang*vec)
    return rc2,ccs,phi
## Class that read a HAWC2 model returning the c2def and st-file set for a main body
class read_HAWC2_main_body():
    def __init__(self,htcfile,st_filename,mbdy_name):
        # Extract set and subset number and c2def for "mbdy_name"
        fhtc = open(htcfile,'r')
        lhtc = fhtc.readlines()
        mbdy_found = False
        timo_found = False
        c2def_found = False
        self.c2def = []
        for line in lhtc:
            words = line.split(';')[0].split()
            if words:
                if words[0] == 'name' and words[1] == mbdy_name:
                    mbdy_found = True
                if mbdy_found and words[0] == 'end' and words[1] == 'main_body':
                    mbdy_found = False
                if mbdy_found and words[0] == 'begin' and words[1] == 'timoschenko_input':
                    timo_found = True
                if timo_found and words[0] == 'end' and words[1] == 'timoschenko_input':
                    timo_found = False
                if timo_found and words[0] == 'set':
                    set_no = int(words[1])
                    subset_no = int(words[2])
                if mbdy_found and words[0] == 'begin' and words[1] == 'c2_def':
                    c2def_found = True
                if c2def_found and words[0] == 'end' and words[1] == 'c2_def':
                    c2def_found = False
                if c2def_found and words[0] == 'sec':
                    self.c2def.append([float(word) for word in words[2:6]])
        # Convert to array
        self.c2def = np.array(self.c2def)
        # Read st-file
        fd=open(st_filename,'r')
        txt=fd.read()
        stset = {}
        for datset in txt.split("#")[1:]:
            datset_nr = int(datset.strip().split()[0])
            subset = {}
            for set_txt in datset.split("$")[1:]:
                set_lines = set_txt.split("\n")
                set_nr, no_rows = map(int, set_lines[0].split()[:2])
                assert set_nr not in subset
                subset[set_nr] = np.array([set_lines[i].split() for i in range(1, no_rows + 1)], dtype=np.float)
            stset[datset_nr] = subset
        self.stset=stset[set_no][subset_no]
## Class that read a HAWC2 model returning the ae-file set for a blade
class read_HAWC2_ae_set():
    def __init__(self,htcfile,ae_filename):
        # Extract set and subset number and c2def for "mbdy_name"
        fhtc = open(htcfile,'r')
        lhtc = fhtc.readlines()
        aero_found = False
        for line in lhtc:
            words = line.split(';')[0].split()
            if words:
                if words[0] == 'begin' and words[1] == 'aero':
                    aero_found = True
                if aero_found and words[0] == 'end' and words[1] == 'aero':
                    aero_found = False
                if aero_found and words[0] == 'ae_sets':
                    set_no = int(words[1])
        # Read ae-file
        fd=open(ae_filename,'r')
        lines=fd.readlines()
        aeset = {}
        nset = int(lines[0].split()[0])
        iline = 0
        for iset in range(nset):
            iline += 1
            ndat = int(lines[iline].split()[1])
            aedat = np.zeros((ndat,4))
            for idat in range(ndat):
                iline += 1
                aedat[idat,:] = np.array([float(x) for x in lines[iline].split()[:4]])
            aeset[iset + 1] = aedat
        self.aeset=aeset[set_no]
    def pc_set_nr(self,z):
        return np.interp(z,self.aeset[:,0],self.aeset[:,3])
## Class that read a HAWC2 model returning the ae-file set for a blade
class read_HAWC2_pc_file():
    def __init__(self,pc_filename):
        # Read airfoil polars and interpolate them to equidistant arrays
        fd=open(pc_filename,'r')
        txt=fd.read()
        lines=txt.split("\n")
        iline=0
        nset=int(lines[iline].split()[0])
        self.pcsets = {}
        for iset in range(nset):
            iline+=1
            nairfoil = int(lines[iline].split()[0])
            airfoil = {}
            for iairfoil in range(nairfoil):
                iline+=1
                airfoil_nr, nrows = map(int, lines[iline].split()[:2])
                thickness = np.float(lines[iline].split()[2])
                polar = np.array([lines[iline+i].split() for i in range(1, nrows + 1)], dtype=np.float)
                airfoil[thickness] = polar
                iline+=nrows
            self.pcsets[iset+1] = airfoil
## Translate a HAWC2 blade input model files to SDU format
def translate_HAWC2_blade_model(htcfile,ae_filename,pc_filename,st_filename,blade_name,st_format):
    # Read blade model using the WE Toolbox (wetb) from DTU
    bld_mbdy = read_HAWC2_main_body(htcfile,st_filename, blade_name)
    h2_ae_set = read_HAWC2_ae_set(htcfile,ae_filename)
    h2_pc_set = read_HAWC2_pc_file(pc_filename)
    ## Create structural input file
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
    # Header
    stheader = ['z [m]','x_ref [m]','y_ref [m]','angle_ref [deg]','m [kg/m]','x_cg [m]','y_cg [m]','ri_x [m]','ri_y [m]','angle_rix [deg]']
    if st_format=='ISO':
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
    if st_format=='6x6':
        for i in range(6):
            for j in range(i,6):
                if i<3 and j<3:
                    stheader.append('C{:d}{:d} [1/N]'.format(i+1,j+1))
                elif i<3:
                    stheader.append('C{:d}{:d} [1/(Nm)]'.format(i+1,j+1))
                else:
                    stheader.append('C{:d}{:d} [1/(Nm^2)]'.format(i+1,j+1))
    # Lengthwise coordinate
    zst = bld_mbdy.stset[:,0]
    Nst = len(zst)
    # Interpolate in the HAWC2 geometry definition 'c2def'
    rc2,ccs,phi = chord_coordinate_system_in_HAWC2(bld_mbdy.c2def,zst)
    twist = np.radians(np.interp(zst, bld_mbdy.c2def[:,2], bld_mbdy.c2def[:,3]))
    # Insert values in the structural data format
    stru_sec_data = np.zeros((Nst,31))
    stru_sec_data[:, 0] = zst
    stru_sec_data[:, 4] = bld_mbdy.stset[:,1]
    if st_format == 'ISO':
        stru_sec_data[:, 15] = bld_mbdy.stset[:, 8]
        stru_sec_data[:, 16] = bld_mbdy.stset[:, 9]
        stru_sec_data[:, 17] = bld_mbdy.stset[:,15]
        stru_sec_data[:, 18] = bld_mbdy.stset[:,10]
        stru_sec_data[:, 19] = bld_mbdy.stset[:,11]
        stru_sec_data[:, 20] = bld_mbdy.stset[:,12]
        stru_sec_data[:, 21] = bld_mbdy.stset[:,13]
        stru_sec_data[:, 22] = bld_mbdy.stset[:,14]
    for isec in range(Nst):
        rea_c2 = np.array([bld_mbdy.stset[isec,17],bld_mbdy.stset[isec,18]])
        Tc = np.array([[np.cos(twist[isec]),-np.sin(twist[isec])],
                       [np.sin(twist[isec]), np.cos(twist[isec])]])
        rea = rc2[isec] + Tc@rea_c2
        stru_sec_data[isec, 1] = rea[0]
        stru_sec_data[isec, 2] = rea[1]
        stpitch = np.radians(bld_mbdy.stset[isec,16])
        stru_sec_data[isec, 3] = np.degrees(twist[isec] + stpitch)
        rcg_c2 = np.array([bld_mbdy.stset[isec,2],bld_mbdy.stset[isec,3]])-rea_c2
        Ts = np.array([[ np.cos(stpitch),np.sin(stpitch)],
                       [-np.sin(stpitch),np.cos(stpitch)]])
        rcg_e = Ts@rcg_c2
        stru_sec_data[isec, 5] = rcg_e[0]
        stru_sec_data[isec, 6] = rcg_e[1]
        stru_sec_data[isec, 7] = np.sqrt(bld_mbdy.stset[isec,4]**2-rcg_e[1]**2)
        stru_sec_data[isec, 8] = np.sqrt(bld_mbdy.stset[isec,5]**2-rcg_e[0]**2)
        rsh_c2 = np.array([bld_mbdy.stset[isec,6],bld_mbdy.stset[isec,7]])-rea_c2
        rsh_e = Ts@rsh_c2
        #stru_sec_data[isec,11] = 0.0 The mass inertia axes are assumed aligned with the bending axes in HAWC2
        # ISO format
        if st_format == 'ISO':
            #stru_sec_data[isec,10] = 0.0 The reference point is chosen to be EA.
            #stru_sec_data[isec,11] = 0.0 So this vector is zero.
            stru_sec_data[isec,12] = rsh_e[0]
            stru_sec_data[isec,13] = rsh_e[1] 
            #stru_sec_data[isec,14] = 0.0 The cross-sectional reference frame is rotated according to the neutral axes of bending.
        if st_format == '6x6':
            rsc=rsh_e
            theta=0.0 # The cross-sectional reference frame is rotated according to the neutral axes of bending.
            E=bld_mbdy.stset[isec,8]
            G=bld_mbdy.stset[isec,9]
            A=bld_mbdy.stset[isec,15]
            Ix=bld_mbdy.stset[isec,10]
            Iy=bld_mbdy.stset[isec,11]
            Iz=bld_mbdy.stset[isec,12]
            kx=bld_mbdy.stset[isec,13]
            ky=bld_mbdy.stset[isec,14]
            C = isotropic_to_6x6_compliance_matrix(np.zeros(2),np.zeros(2),rsc,theta,E,G,A,Ix,Iy,Iz,kx,ky)
            # Insert values
            k=10
            for i in range(6):
                for j in range(i,6):
                    stru_sec_data[isec,k] = C[i,j]
                    k+=1
    # Reset size of isotropic data array
    if st_format == 'ISO':
        stru_sec_data = stru_sec_data[:,0:23]
    # Save to file
    header_txt='#1 Structural blade data file generated from HAWC2 data \n'+''.join('{:>16s} '.format(text) for text in stheader) + '\n'+'@1 {:d}'.format(Nst)
    np.savetxt('stru_' + st_format + '.dat',stru_sec_data,fmt='%16.8e',header=header_txt,comments='',delimiter=' ')
    ## Create aerodynamic input file
    #      Column 1:     z [m], distance to section from blade root flange along the pitch axis
    #      Column 2-3:   x_ccs and y_ccs [m], 2-D vector from pitch axis to origo of chord coordinate system (CCS)
    #      Column 4-6:   phi_x, phi_y, phi_z [deg], pseudo vector of finite rotation of CCS from blade CS.
    #                    If phi_x = phi_y = 0 then the normal to the plane of the airfoil is defined by 
    #                    the tangent of the origo curve and the inplane axes of the CCS are defined such that 
    #                    the y-axis has zero x-component in the blade frame before rotating them about the tangent by phi_y. 
    #      Column 7:     c [m], chord length of airfoil or diameter of circle
    #      Column 8:     rel_thick [%], relative thickness of airfoil (100% = circle)
    #      Column 9-10:  x_ac and y_ac [m], 2-D cross-sectional position vector from origo of CCS to aerodynamic pressure center (AC)
    #      Column 11:    a_ac [-], non-dimensional chordwise position of the pressure center
    #      Column 12:    pc_set [-], set number in PC file related to the airfoil type
    aeheader = ['z [m]','x_ccs [m]','y_ccs [m]','phi_x [deg]','phi_y [deg]','phi_z [deg]','c [m]','rel_thick [%]','x_ac [m]','y_ac [m]','a_ac [-]','pc_set [-]']
    zae = h2_ae_set.aeset[:,0]
    Nae = len(zae)
    # Interpolate in the HAWC2 geometry definition 'c2def'
    rc2,ccs,phi = chord_coordinate_system_in_HAWC2(bld_mbdy.c2def,zae)
    # Thickest airfoil 
    airfoil_thicknesses = np.array([float(x) for x in h2_pc_set.pcsets[1].keys()])
    thickest_airfoil = np.max(airfoil_thicknesses[np.nonzero(airfoil_thicknesses<100.0)])
    # Insert values in the aerodynamic data format
    aero_sec_data = np.zeros((Nae,12))
    aero_sec_data[:,0] = zae
    aero_sec_data[:,6] = h2_ae_set.aeset[:,1]
    aero_sec_data[:,7] = h2_ae_set.aeset[:,2]
    for isec in range(Nae):
        aero_sec_data[isec, 1] = rc2[isec][0]
        aero_sec_data[isec, 2] = rc2[isec][1]
        aero_sec_data[isec, 3] = np.degrees(phi[isec][0])
        aero_sec_data[isec, 4] = np.degrees(phi[isec][1])
        aero_sec_data[isec, 5] = np.degrees(phi[isec][2])
        if aero_sec_data[isec,7] > thickest_airfoil:
            airfoil_cylinder_ratio = (100.0 - aero_sec_data[isec,7])/(100.0-thickest_airfoil)
        else:
            airfoil_cylinder_ratio = 1.0
        aero_sec_data[isec, 8] = 0.25*airfoil_cylinder_ratio*aero_sec_data[isec,6]
        aero_sec_data[isec, 9] = 0.0
        aero_sec_data[isec,10] = 0.25*airfoil_cylinder_ratio + 0.5*(1.0-airfoil_cylinder_ratio)
        aero_sec_data[isec,11] = h2_ae_set.pc_set_nr(zae[isec])
    # Save to file
    header_txt='#1 Aerodynamic blade data file generated from HAWC2 data \n'+''.join('{:>16s} '.format(text) for text in aeheader) + '\n'+'@1 {:d}'.format(Nae)
    np.savetxt('aero.dat',aero_sec_data,fmt='%16.8e',header=header_txt,comments='',delimiter=' ')
    # Return SDU data format arrays
    return stru_sec_data,aero_sec_data
## Routine that reads and inserts the structural elements of a HAWC2 model as read by HAWC2
class HAWCStab2_blade:
    def __init__(self,fname):
        f=open(fname,'r')
        txt=f.readlines()
        readcho=False
        readrac=False
        self.rac=[]
        self.Ec=[]
        self.ct=[]
        for word in txt:
            if 'Thickness' in word:
                readcho=True
                i=-1
            if '-------' in word:
                readcho=False
            if 'orientation' in word:
                readrac=True
                rac=np.zeros(3)
                Ec=np.zeros((3,3))
                i=-2
            if readcho:
                i+=1
                if i > 0:
                    words=word.split()
                    tal=np.array([float(x) for x in words])
                    self.ct.append(tal[4:])
            if readrac:
                i+=1
                if '-------' in word:
                    i=0
                    rac=np.zeros(3)
                    Ec=np.zeros((3,3))
                if i > 0:
                    words=word.split()
                    tal=np.array([float(x) for x in words])
                    if i==1:
                        rac[0] = tal[2]
                        Ec[0,:] = tal[3:]
                    elif i==2:
                        rac[1] = tal[1]
                        Ec[1,:] = tal[2:]
                    else:
                        rac[2] = tal[1]
                        Ec[2,:] = tal[2:]
                        self.rac.append(rac)
                        self.Ec.append(Ec)
        # Close file
        f.close()

