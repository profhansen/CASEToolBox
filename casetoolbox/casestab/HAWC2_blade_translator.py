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
## Routine that reads and inserts the aerodynamic calculation points of a HAWCStab2 model 
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

