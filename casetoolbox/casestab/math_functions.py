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
## @package math_functions
#  
#
# 
from numba import njit
import numpy as np
from scipy import interpolate
## Constant helping matrix for Rodrigues rotation matrix
Smat=[]
Smat.append(np.array([[0.0, 0.0,0.0],[0.0,0.0,-1.0],[ 0.0,1.0,0.0]]))
Smat.append(np.array([[0.0, 0.0,1.0],[0.0,0.0, 0.0],[-1.0,0.0,0.0]]))
Smat.append(np.array([[0.0,-1.0,0.0],[1.0,0.0, 0.0],[ 0.0,0.0,0.0]]))
## Identity matrix
Imat=np.eye(3)
## Rotation matrix from rotation angle and vector
@njit
def rotmat(v,phi):
    cosphi=np.cos(phi)
    sinphi=np.sin(phi)
    R=np.zeros((3,3))
    R[0,0] =  cosphi+(1-cosphi)*v[0]**2
    R[0,1] = -sinphi*v[2]+(1-cosphi)*v[0]*v[1]
    R[0,2] =  sinphi*v[1]+(1-cosphi)*v[0]*v[2]
    R[1,0] =  sinphi*v[2]+(1-cosphi)*v[0]*v[1]
    R[1,1] =  cosphi+(1-cosphi)*v[1]**2
    R[1,2] = -sinphi*v[0]+(1-cosphi)*v[1]*v[2]
    R[2,0] = -sinphi*v[1]+(1-cosphi)*v[0]*v[2]
    R[2,1] =  sinphi*v[0]+(1-cosphi)*v[1]*v[2]
    R[2,2] =  cosphi+(1-cosphi)*v[2]**2
    return R
## Rotation matrix from rotation angle and vector
@njit
def rotmat_from_pseudovec(p):
    phi = np.sqrt(p[0]**2+p[1]**2+p[2]**2)
    if phi > 1.e-6:
        v = p/phi
        R = rotmat(v,phi)
    else:
        R = np.eye(3)
    return R
## Quaternion form rotation matrix
# Taken from http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
def rotmat_to_quaternion(R):
    tr=R[0,0]+R[1,1]+R[2,2]
    q=np.zeros(4)
    if tr>0:
        a=np.sqrt(tr+1)*2
        q[0]=0.25*a
        q[1]=(R[2,1]-R[1,2])/a
        q[2]=(R[0,2]-R[2,0])/a
        q[3]=(R[1,0]-R[0,1])/a
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        a=np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2
        q[0]=(R[2,1]-R[1,2])/a
        q[1]=0.25*a
        q[2]=(R[0,1]+R[1,0])/a
        q[3]=(R[0,2]+R[2,0])/a
    elif R[1,1] > R[2,2]:
        a=np.sqrt(1-R[0,0]+R[1,1]-R[2,2])*2
        q[0]=(R[0,2]-R[2,0])/a
        q[1]=(R[0,1]+R[1,0])/a
        q[2]=0.25*a
        q[3]=(R[1,2]+R[2,1])/a
    else:
        a=np.sqrt(1-R[0,0]-R[1,1]+R[2,2])*2
        q[0]=(R[1,0]-R[0,1])/a
        q[1]=(R[0,2]+R[2,0])/a
        q[2]=(R[1,2]+R[2,1])/a
        q[3]=0.25*a
    return q
## Rotation angle and vector from quaternion
# Taken from http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToAngle/index.htm
def quaternion_to_vector_and_angle(q):
    s=np.sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3])
    q=q/s
    phi=2*np.arccos(q[0])
    s=np.sqrt(1.0-q[0]**2)
    v=np.zeros(3)
    if s<1.0e-16:
        v[0]=q[1]
        v[1]=q[2]
        v[2]=q[3]
    else:
        v[0]=q[1]/s
        v[1]=q[2]/s
        v[2]=q[3]/s
    return v,phi
## Interpolation of rotation between two triads
def interpolate_rotmat(T,Q,ratio):
    R=Q@T.T
    q=rotmat_to_quaternion(R)
    v,phi=quaternion_to_vector_and_angle(q)
    dR=rotmat(v,ratio*phi)
    return dR   
## Inner product function
@njit
def innerproduct(v1,v2):
    x=v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]
    return x
## Inner product function
def transposed_innerproduct(v1,v2):
    x=np.zeros((3,3))
    x[0,0]=v1[0]*v2[0]
    x[0,1]=v1[0]*v2[1]
    x[0,2]=v1[0]*v2[2]
    x[1,0]=v1[1]*v2[0]
    x[1,1]=v1[1]*v2[1]
    x[1,2]=v1[1]*v2[2]
    x[2,0]=v1[2]*v2[0]
    x[2,1]=v1[2]*v2[1]
    x[2,2]=v1[2]*v2[2]
    return x
## Inner matrix product function
def inner_matrix_product(A1,A2):
    x=0.0
    for i in range(3):
        for j in range(3):
            x+=A1[i,j]*A2[i,j]
    return x
## Function that computes the length of a vector 
def vector_length(v):
    l=np.sqrt(v[0]**2+v[1]**2+v[2]**2)
    return l
## Function that normalizes a vector to a unit-vector
def unit_vector(v):
    l=vector_length(v)
    v=v/l
    return v
## Cross product function
def crossproduct(v1,v2):
    x=np.zeros(3)
    x[0]=v1[1]*v2[2]-v1[2]*v2[1]
    x[1]=v1[2]*v2[0]-v1[0]*v2[2]
    x[2]=v1[0]*v2[1]-v1[1]*v2[0]
    return x
## Rotation matrix for rotation about the first axis
def R1(phi):
    cosphi=np.cos(phi)
    sinphi=np.sin(phi)
    R=np.eye(3)
    R[1,1] = cosphi
    R[1,2] =-sinphi
    R[2,1] = sinphi
    R[2,2] = cosphi
    return R
## Rotation matrix for rotation about the second axis
def R2(phi):
    cosphi=np.cos(phi)
    sinphi=np.sin(phi)
    R=np.eye(3)
    R[0,0] = cosphi
    R[0,2] = sinphi
    R[2,0] =-sinphi
    R[2,2] = cosphi
    return R
## Rotation matrix for rotation about the third axis
def R3(phi):
    cosphi=np.cos(phi)
    sinphi=np.sin(phi)
    R=np.eye(3)
    R[0,0] = cosphi
    R[0,1] =-sinphi
    R[1,0] = sinphi
    R[1,1] = cosphi
    return R
## Rotation matrix for rotation about the i'th axis (can be negative)
def Ri(phi,iaxis):
    phi=phi*np.sign(iaxis)
    iaxis=np.abs(iaxis)
    cosphi=np.cos(phi)
    sinphi=np.sin(phi)
    R=np.eye(3)*cosphi    
    R+=Smat[iaxis-1]*sinphi
    R[iaxis-1,iaxis-1] = 1.0
    return R
## Variable helping matrix for Rodrigues rotation matrix and its derivative
@njit
def Skew(v):
    return np.array([[0.0,-v[2],v[1]],
                     [v[2],0.0,-v[0]],
                     [-v[1],v[0],0.0]])
## Variable helping matrix for Rodrigues rotation matrix and its derivative
def deskew(S):
    return np.array([S[2,1],S[0,2],S[1,0]])
## Rodrigues rotation matrix
def Rmat(v):
    # v is the pseudo vector of the Rodrigues parameters
    d=1.0+innerproduct(v,v)/4.0
    S=Skew(v)
    R=np.eye(3)+(S+0.5*S@S)/d # Crisfield (1990) Eq. (7)
    return R
## First derivative of Rodrigues rotation matrix
def dRmat(i,v):
    # v are the Rodrigues parameters
    d=1.0+innerproduct(v,v)/4.0
    S=Skew(v)
    Si=Smat[i]
    SiS=Si@S
    dR=(Si+0.5*(SiS+SiS.T))/d-0.5*v[i]*(S+0.5*S@S)/d**2
    return dR
## Second derivative of Rodrigues rotation matrix
def ddRmat(i,j,v):
    # v are the Rodrigues parameters
    d=1.0+innerproduct(v,v)/4.0
    S=Skew(v)
    Si=Smat[i]
    Sj=Smat[j]
    SiS=Si@S
    SjS=Sj@S
    SiSj=Si@Sj
    RmI=S+0.5*S@S
    ddR=0.5*(SiSj+SiSj.T)/d
    -0.5*(Si+0.5*(SiS+SiS.T))*v[j]/d**2
    -0.5*(Sj+0.5*(SjS+SjS.T))*v[i]/d**2
    -0.5*RmI*Imat[i,j]/d**2+0.5*v[i]*v[j]*RmI/d**3
    return ddR
## Compute pseudo vector from Rodrigues parameters
def pseudo_vector_from_Rodrigues(q):
    theta=np.zeros(3)
    R=Rmat(q)
    q=rotmat_to_quaternion(R)
    v,phi=quaternion_to_vector_and_angle(q)
    theta=v*phi
    return theta
## Compute pseudo vector from Rodrigues parameters
def small_rotation_pseudo_vector_from_Rodrigues(q):
    theta=np.zeros(3)
    qnorm2 = vector_length(q)
    if qnorm2 > 0.0:
        q = q/qnorm2*1.0e-3
        R=Rmat(q)
        q=rotmat_to_quaternion(R)
        v,phi=quaternion_to_vector_and_angle(q)
        theta=v*phi*qnorm2*1.0e3
    return theta
## Vector index to a N x N symmetric matrix
def generate_ijNsym(N):
    ijNsym=np.empty([N,N],dtype=int)
    k=0
    for i in range(N):
        for j in range(i+1):
            ijNsym[i,j]=k
            ijNsym[j,i]=k
            k+=1
    return ijNsym
## Class that provides the interpolation functions 
class curve_interpolate:
    def __init__(self,itype,x,y):
        self.itype=itype
        if itype == 'pchip':
            self.fcn = interpolate.PchipInterpolator(x,y)
            self.der = interpolate.PchipInterpolator(x,y).derivative()
        elif itype == 'akima':
            self.fcn = interpolate.Akima1DInterpolator(x,y)
            self.der = interpolate.Akima1DInterpolator(x,y).derivative()
        else: # linear is default
            self.fcn = interpolate.interp1d(x,y,axis=0)
            yp = np.zeros(np.shape(y))
            for icol in range(np.size(y,axis=1)):
                yp[1:,icol] = np.diff(y[:,icol])/np.diff(x)
            yp[0,:] = yp[1,:]
            self.der = interpolate.interp1d(x,yp,kind='next',axis=0)
## Piecewise linear function
def piecewise_linear_function(x,xbreak,coeffs):
    n=len(x)
    y=np.zeros(n)
    m=len(xbreak)
    for i in range(n):
        for k in range(m-1):
            if x[i] >= xbreak[k] and x[i] <= xbreak[k+1]:
                y[i] = coeffs[k,0]+coeffs[k,1]*x[i]
    return y


## Class for linear interpolation of equidistant points
class quick_interpolation_periodic_function:
    def __init__(self,x,y):
        self.x = x.copy()
        self.y = y.copy()
        self.n = len(x)
        self.dx = x[1]-x[0]
        self.xrange = x[-1] - x[0]
        if np.abs(y[0]-y[-1]) > 1e-6:
            print('In the class quick_interpolation_periodic_function: \n The end y-values differ by more than 1e-6 \n thus the function seems not to be periodic.')
            exit()
    def fcn(self,x):
        mx = np.remainder(x-self.x[0],self.xrange)
        fx = mx/self.dx
        ix = fx.astype(int)
        i1 = np.remainder(ix,self.n)
        i2 = np.remainder(i1+1,self.n)
        y = self.y[i1] + (self.y[i2] - self.y[i1])*(fx - i1)
        return y
    def der(self,x):
        mx = np.remainder(x-self.x[0],self.xrange)
        fx = mx/self.dx
        ix = fx.astype(int)
        i1 = np.remainder(ix,self.n)
        i2 = np.remainder(i1+1,self.n)
        yp = (self.y[i2] - self.y[i1])/self.dx
        return yp





















