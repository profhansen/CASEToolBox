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
import numpy as np
from numba import njit,float64,int32
from numba.pycc import CC

cc = CC('corotbeam_precompiled_functions')
cc.verbose = True

## Inner product function
@njit(float64(float64[:],float64[:]))
def innerproduct(v1,v2):
    x=v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]
    return x
## Cross product function
@njit(float64[:](float64[:],float64[:]))
def crossproduct(v1,v2):
    x=np.zeros(3)
    x[0]=v1[1]*v2[2]-v1[2]*v2[1]
    x[1]=v1[2]*v2[0]-v1[0]*v2[2]
    x[2]=v1[0]*v2[1]-v1[1]*v2[0]
    return x
## Variable helping matrix for Rodrigues rotation matrix and its derivative
@njit(float64[:](float64[:],float64[:]))
def skewmul(v,e):
    x=np.zeros(3)
    x[0]=-v[2]*e[1]+v[1]*e[2]
    x[1]= v[2]*e[0]-v[0]*e[2]
    x[2]=-v[1]*e[0]+v[0]*e[1]
    return x
## Product of 3x3 matrix and a 3x1 vector
@njit(float64[:](float64[:,:],float64[:]))
def matvec33(A,b):
    c=np.zeros(3)
    c[0]=A[0,0]*b[0]+A[0,1]*b[1]+A[0,2]*b[2]
    c[1]=A[1,0]*b[0]+A[1,1]*b[1]+A[1,2]*b[2]
    c[2]=A[2,0]*b[0]+A[2,1]*b[1]+A[2,2]*b[2]
    return c
## Product of two 3x3 matrix
@njit(float64[:,:](float64[:,:],float64[:,:]))
def matmul33(A,B):
    C=np.zeros((3,3))
    C[0,0]=A[0,0]*B[0,0]+A[0,1]*B[1,0]+A[0,2]*B[2,0]
    C[0,1]=A[0,0]*B[0,1]+A[0,1]*B[1,1]+A[0,2]*B[2,1]
    C[0,2]=A[0,0]*B[0,2]+A[0,1]*B[1,2]+A[0,2]*B[2,2]
    C[1,0]=A[1,0]*B[0,0]+A[1,1]*B[1,0]+A[1,2]*B[2,0]
    C[1,1]=A[1,0]*B[0,1]+A[1,1]*B[1,1]+A[1,2]*B[2,1]
    C[1,2]=A[1,0]*B[0,2]+A[1,1]*B[1,2]+A[1,2]*B[2,2]
    C[2,0]=A[2,0]*B[0,0]+A[2,1]*B[1,0]+A[2,2]*B[2,0]
    C[2,1]=A[2,0]*B[0,1]+A[2,1]*B[1,1]+A[2,2]*B[2,1]
    C[2,2]=A[2,0]*B[0,2]+A[2,1]*B[1,2]+A[2,2]*B[2,2]
    return C
## Product of 6x7 matrix and a 7x1 vector
@njit(float64[:](float64[:,:],float64[:]))
def matvec67(A,b):
    c=np.zeros(6)
    c[0]=A[0,0]*b[0]+A[0,1]*b[1]+A[0,2]*b[2]+A[0,3]*b[3]+A[0,4]*b[4]+A[0,5]*b[5]+A[0,6]*b[6]
    c[1]=A[1,0]*b[0]+A[1,1]*b[1]+A[1,2]*b[2]+A[1,3]*b[3]+A[1,4]*b[4]+A[1,5]*b[5]+A[1,6]*b[6]
    c[2]=A[2,0]*b[0]+A[2,1]*b[1]+A[2,2]*b[2]+A[2,3]*b[3]+A[2,4]*b[4]+A[2,5]*b[5]+A[2,6]*b[6]
    c[3]=A[3,0]*b[0]+A[3,1]*b[1]+A[3,2]*b[2]+A[3,3]*b[3]+A[3,4]*b[4]+A[3,5]*b[5]+A[3,6]*b[6]
    c[4]=A[4,0]*b[0]+A[4,1]*b[1]+A[4,2]*b[2]+A[4,3]*b[3]+A[4,4]*b[4]+A[4,5]*b[5]+A[4,6]*b[6]
    c[5]=A[5,0]*b[0]+A[5,1]*b[1]+A[5,2]*b[2]+A[5,3]*b[3]+A[5,4]*b[4]+A[5,5]*b[5]+A[5,6]*b[6]
    return c
# Constant helping matrix for Rodrigues rotation matrix
Smat=np.zeros((3,3,3))
Smat[:,:,0]=np.array([[0.0, 0.0,0.0],[0.0,0.0,-1.0],[ 0.0,1.0,0.0]])
Smat[:,:,1]=np.array([[0.0, 0.0,1.0],[0.0,0.0, 0.0],[-1.0,0.0,0.0]])
Smat[:,:,2]=np.array([[0.0,-1.0,0.0],[1.0,0.0, 0.0],[ 0.0,0.0,0.0]])
# Identity matrix
Imat=np.eye(3)
# 3x3 index matrix
ij3sym=np.array([[0, 1, 3],[1, 2, 4],[3, 4, 5]])
# 6x6 index matrix
ij6sym=np.array([[ 0,  1,  3,  6, 10, 15],[ 1,  2,  4,  7, 11, 16],[ 3,  4,  5,  8, 12, 17], \
                 [ 6,  7,  8,  9, 13, 18],[10, 11, 12, 13, 14, 19],[15, 16, 17, 18, 19, 20]])
# 12x12 index matrix 
ij12sym=np.array([[ 0,  1,  3,  6, 10, 15, 21, 28, 36, 45, 55, 66], \
                  [ 1,  2,  4,  7, 11, 16, 22, 29, 37, 46, 56, 67], \
                  [ 3,  4,  5,  8, 12, 17, 23, 30, 38, 47, 57, 68], \
                  [ 6,  7,  8,  9, 13, 18, 24, 31, 39, 48, 58, 69], \
                  [10, 11, 12, 13, 14, 19, 25, 32, 40, 49, 59, 70], \
                  [15, 16, 17, 18, 19, 20, 26, 33, 41, 50, 60, 71], \
                  [21, 22, 23, 24, 25, 26, 27, 34, 42, 51, 61, 72], \
                  [28, 29, 30, 31, 32, 33, 34, 35, 43, 52, 62, 73], \
                  [36, 37, 38, 39, 40, 41, 42, 43, 44, 53, 63, 74], \
                  [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 64, 75], \
                  [55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 76], \
                  [66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77]])
## Variable helping matrix for Rodrigues rotation matrix and its derivative
@njit(float64[:,:](float64[:]))
def Skew(v):
    return np.array([[0.0,-v[2],v[1]],
                     [v[2],0.0,-v[0]],
                     [-v[1],v[0],0.0]])
## Rodrigues rotation matrix
@njit(float64[:,:](float64[:]))
def Rmat(v):
    # v is the pseudo vector of the Rodrigues parameters
    d=1.0+innerproduct(v,v)/4.0
    S=Skew(v)
    R=np.eye(3)+(S+0.5*matmul33(S,S))/d # Crisfield (1990) Eq. (7)
    return R
## First derivative of Rodrigues rotation matrix
@njit(float64[:,:](int32,float64[:]))
def dRmat(i,v):
    # v are the Rodrigues parameters
    d=1.0+innerproduct(v,v)/4.0
    S=Skew(v)
    Si=np.zeros((3,3))
    Si[:,:]=Smat[:,:,i]
    SiS=matmul33(Si,S)
    dR=(Si+0.5*(SiS+SiS.T))/d-0.5*v[i]*(S+0.5*matmul33(S,S))/(d*d)
    return dR
## Second derivative of Rodrigues rotation matrix
@njit(float64[:,:](int32,int32,float64[:]))
def ddRmat(i,j,v):
    # v are the Rodrigues parameters
    d=1.0+innerproduct(v,v)/4.0
    S=Skew(v)
    Si=np.zeros((3,3))
    Si[:,:]=Smat[:,:,i]
    Sj=np.zeros((3,3))
    Sj[:,:]=Smat[:,:,j]
    SiS=matmul33(Si,S)
    SjS=matmul33(Sj,S)
    SiSj=matmul33(Si,Sj)
    RmI=S+0.5*matmul33(S,S)
    ddR=0.5*(SiSj+SiSj.T)/d
    -0.5*(Si+0.5*(SiS+SiS.T))*v[j]/(d*d)
    -0.5*(Sj+0.5*(SjS+SjS.T))*v[i]/(d*d)
    -0.5*RmI*Imat[i,j]/(d*d)+0.5*v[i]*v[j]*RmI/(d*d*d)
    return ddR
## Rotate triad
@njit(float64[:,:](float64[:,:],float64[:]))
def RotateTriad(T,v):
    return matmul33(Rmat(v),T)   
## First derivative of triad rotation
@njit(float64[:,:](int32,float64[:,:],float64[:]))
def dRotateTriad(i,T,v):
    return matmul33(dRmat(i,v),T)   
## Second derivative of triad rotation
@njit(float64[:,:](int32,int32,float64[:,:],float64[:]))
def ddRotateTriad(i,j,T,v):
    return matmul33(ddRmat(i,j,v),T)  
## Index function for integration of polynomials
@njit(float64(int32))
def c_function(r):
    # r is a positive integer
    if r % 2:
        c = 0.0
    else:
        c = 2.0/(1.0+r)
    return c
## Matrix function of the product of the unit-vector e_k with the transpose of the vector v
@njit(float64[:,:](float64[:],int32))
def ek_vT(v,k):
    mat=np.zeros((3,3))
    mat[k,:]=v
    return mat
## Generic operator for the inertia state computation
@njit(float64(float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:]))
def scalar_G_operator(a,uo,ux,uy,wo,wx,wy):
    G = a[0]*innerproduct(uo,wo)+a[1]*(innerproduct(uo,wx)+innerproduct(ux,wo)) \
       +a[2]*(innerproduct(uo,wy)+innerproduct(uy,wo))+a[3]*innerproduct(ux,wx) \
       +a[4]*innerproduct(uy,wy)+a[5]*(innerproduct(ux,wy)+innerproduct(uy,wx))
    return G
@njit(float64[:,:](float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:]))
def matrix_G_operator(a,uo,ux,uy,wo,wx,wy):
    G = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            G[i,j] = a[0]*uo[i]*wo[j]+a[1]*(uo[i]*wx[j]+ux[i]*wo[j])+a[2]*(uo[i]*wy[j]+uy[i]*wo[j]) \
                    +a[3]*ux[i]*wx[j]+a[4]*uy[i]*wy[j]+a[5]*(ux[i]*wy[j]+uy[i]*wx[j])
    return G

## Update position and orientation of nodal triads
#
# Inputs:
#  q   float64[:]
#  r0  float64[:]
#  T0  float64[:,:]
#  Q0  float64[:,:]
#
# Outputs
#  r    float64[:]
#  dvec float64[:]
#  T    float64[:,:]
#  Q    float64[:,:]
#  dT   float64[:,:,:]
#  dQ   float64[:,:,:]
#  ddT  float64[:,:,:]
#  ddQ  float64[:,:,:]
#
@cc.export('update_nodal_triads_and_position','(f8[:],f8[:],f8[:,:],f8[:,:])')
def update_nodal_triads_and_position(q,r0,T0,Q0):
    r=np.zeros(6)
    r[0:3]=q[0:3]+r0[0:3]
    T=RotateTriad(T0,q[3:6])
    r[3:]=q[6:9]+r0[3:]
    Q=RotateTriad(Q0,q[9:])
    # For elongation computation
    dvec=np.zeros(3)
    dvec=q[6:9]-q[0:3]
    # First derivative computations
    dT = np.zeros((3,3,3))
    dQ = np.zeros((3,3,3))
    for i in range(3):
        dT[:,:,i]=dRotateTriad(i,T0,q[3:6])
        dQ[:,:,i]=dRotateTriad(i,Q0,q[9:])
    # Second derivative computations
    ddT = np.zeros((3,3,6))
    ddQ = np.zeros((3,3,6))
    for i in range(3):
        for j in range(i+1):
            ddT[:,:,ij3sym[i,j]]=ddRotateTriad(i,j,T0,q[3:6])
            ddQ[:,:,ij3sym[i,j]]=ddRotateTriad(i,j,Q0,q[9:])
    return r,dvec,T,Q,dT,dQ,ddT,ddQ
## Compute element triad using the average of t2 and q2
#
# Inputs:
#  r   float64[:]
#  T   float64[:,:]
#  Q   float64[:,:]
#  dT  float64[:,:,:]
#  dQ  float64[:,:,:]
#  ddT float64[:,:,:]
#  ddQ float64[:,:,:]
#
# Outputs
#  l           float64
#  E           float64[:,:]
#  rvec        float64[:]
#  rmid        float64[:]
#  d           float64
#  dd          float64[:]
#  dE_dqis     float64[:,:,:]
#  ddE_dqidqjs float64[:,:,:]
#
@cc.export('compute_element_triad_and_position','(f8[:],f8[:,:],f8[:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:])')
def compute_element_triad_and_position(r,T,Q,dT,dQ,ddT,ddQ):
    # Element axis
    rvec=r[3:]-r[0:3]
    rmid=r[0:3]+0.5*rvec
    l=np.sqrt(innerproduct(rvec,rvec))
    # e3 is the unit-vecor between nodes
    E=np.zeros((3,3))
    E[:,2]=rvec/l
    # e2 is the t2 + q2 orthonormal to e3
    v=T[:,1]+Q[:,1]
    a=innerproduct(v,E[:,2])
    d=np.sqrt(2.0+2.0*innerproduct(T[:,1],Q[:,1])-(a*a))
    e2tilde=v-a*E[:,2]
    E[:,1]=e2tilde/d
    # e1 is the cross-product of e2 and e3
    E[:,0]=crossproduct(E[:,1],E[:,2])   

    #-------------------------------------------------------------------------
    # Allocation
    dd=np.zeros(12)
    dE_dqis=np.zeros((3,3,12))
    ddE_dqidqjs=np.zeros((3,3,78))
    #-------------------------------------------------------------------------
    # First derivative computations of e3
    for i in range(3):
        dE_dqis[:,2,i]  =-Imat[:,i]/l+rvec[i]*rvec/(l*l*l)
        dE_dqis[:,2,i+6]=-dE_dqis[:,2,i]
    # Second derivative computations of e3
    for i in range(3):
        for j in range(i+1):
            ddE_dqidqjs[:,2,ij12sym[i,j]]=rvec*(3.0*rvec[i]*rvec[j]/(l*l*l*l*l)-Imat[i,j]/(l*l*l)) \
                                   -(Imat[:,i]*rvec[j]+Imat[:,j]*rvec[i])/(l*l*l)
            ddE_dqidqjs[:,2,ij12sym[i+6,j+6]]=ddE_dqidqjs[:,2,ij12sym[i,j]]
    for i in range(3):
        for j in range(3):
            ddE_dqidqjs[:,2,ij12sym[i+6,j]]=-ddE_dqidqjs[:,2,ij12sym[i,j]]
    # Derivaties of inplane vectors e1 and e2
    for i in range(3):
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # First derivative computations for i = translations of node 1
        """ variations of e2 and e1 for q_m m=i """
        m=i
        dadi=innerproduct(v,dE_dqis[:,2,i])
        dd[m]=(-a*dadi)/d
        de2tilde_i=-dadi*E[:,2]-a*dE_dqis[:,2,i]
        dE_dqis[:,1,m]=de2tilde_i/d-e2tilde*dd[m]/(d*d)
        dE_dqis[:,0,m]=crossproduct(dE_dqis[:,1,m],E[:,2]) \
                    +crossproduct(E[:,1],dE_dqis[:,2,i])
        # Second derivative computations for i = translations at node 1 and j = translations at node 1
        """ variations of e2 and e1 for q_n n=j """
        for j in range(i+1):
            n=j
            dadj=innerproduct(v,dE_dqis[:,2,j])
            de2tilde_j=-dadj*E[:,2]-a*dE_dqis[:,2,j]
            ddadij=innerproduct(v,ddE_dqidqjs[:,2,ij12sym[i,j]])
            ddbdij=0.0
            ddddij=(ddbdij-ddadij*a-dadi*dadj-dd[m]*dd[n])/d
            dde2tilde_ij=-ddadij*E[:,2]-dadi*dE_dqis[:,2,j]-dadj*dE_dqis[:,2,i]-a*ddE_dqidqjs[:,2,ij12sym[i,j]]
            ddE_dqidqjs[:,1,ij12sym[m,n]]=dde2tilde_ij/d-(dd[m]*de2tilde_j+dd[n]*de2tilde_i)/(d*d) \
                                         +(2.0*dd[m]*dd[n]/(d*d*d)-ddddij/(d*d))*e2tilde
            ddE_dqidqjs[:,0,ij12sym[m,n]]=crossproduct(ddE_dqidqjs[:,1,ij12sym[m,n]],E[:,2]) \
                                    +crossproduct(dE_dqis[:,1,m],dE_dqis[:,2,j]) \
                                    +crossproduct(dE_dqis[:,1,n],dE_dqis[:,2,i]) \
                                    +crossproduct(E[:,1],ddE_dqidqjs[:,2,ij12sym[i,j]])
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    for i in range(3):
    # First derivative computations i = rotations at node 1
        """ variations of e2 and e1 for q_m m=i+3 """
        m=i+3
        dadi=innerproduct(dT[:,1,i],E[:,2])
        dd[m]=(innerproduct(dT[:,1,i],Q[:,1])-a*dadi)/d
        de2tilde_i=dT[:,1,i]-dadi*E[:,2]
        dE_dqis[:,1,m]=de2tilde_i/d-e2tilde*dd[m]/(d*d)
        dE_dqis[:,0,m]=crossproduct(dE_dqis[:,1,m],E[:,2])
        # Second derivative computations i = rotations at node 1 and j = translations at node 1
        """ variations of e2 and e1 for q_m m=i+3 and q_n n=j """
        for j in range(3):
            n=j
            dadj=innerproduct(v,dE_dqis[:,2,j])
            de2tilde_j=-dadj*E[:,2]-a*dE_dqis[:,2,j]
            ddadij=innerproduct(dT[:,1,i],dE_dqis[:,2,j])
            ddbdij=0.0
            ddddij=(ddbdij-ddadij*a-dadi*dadj-dd[m]*dd[n])/d
            dde2tilde_ij=-ddadij*E[:,2]-dadi*dE_dqis[:,2,j]
            ddE_dqidqjs[:,1,ij12sym[m,n]]=dde2tilde_ij/d-(dd[m]*de2tilde_j+dd[n]*de2tilde_i)/(d*d) \
                                            +(2.0*dd[m]*dd[n]/(d*d*d)-ddddij/(d*d))*e2tilde
            ddE_dqidqjs[:,0,ij12sym[m,n]]=crossproduct(ddE_dqidqjs[:,1,ij12sym[m,n]],E[:,2]) \
                                    +crossproduct(dE_dqis[:,1,m],dE_dqis[:,2,j])
        # Second derivative computations i = rotations at node 1 and j = rotations at node 1
        """ variations of e2 and e1 for q_m m=i+3 and q_n n=j+3 """
        for j in range(i+1):
            n=j+3
            dadj=innerproduct(dT[:,1,j],E[:,2])
            de2tilde_j=dT[:,1,j]-dadj*E[:,2]
            ddadij=innerproduct(ddT[:,1,ij3sym[i,j]],E[:,2])
            ddbdij=innerproduct(ddT[:,1,ij3sym[i,j]],Q[:,1])
            ddddij=(ddbdij-ddadij*a-dadi*dadj-dd[m]*dd[n])/d
            dde2tilde_ij=ddT[:,1,ij3sym[i,j]]-ddadij*E[:,2]
            ddE_dqidqjs[:,1,ij12sym[m,n]]=dde2tilde_ij/d-(dd[m]*de2tilde_j+dd[n]*de2tilde_i)/(d*d) \
                                            +(2.0*dd[m]*dd[n]/(d*d*d)-ddddij/(d*d))*e2tilde
            ddE_dqidqjs[:,0,ij12sym[m,n]]=crossproduct(ddE_dqidqjs[:,1,ij12sym[m,n]],E[:,2])
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    for i in range(3):
    # First derivative computations i = translations at node 2
        """ variations of e2 and e1 for q_m m=i+6 """
        m=i+6
        dadi=innerproduct(v,dE_dqis[:,2,m])
        dd[m]=(-a*dadi)/d
        de2tilde_i=-dadi*E[:,2]-a*dE_dqis[:,2,m]
        dE_dqis[:,1,m]=de2tilde_i/d-e2tilde*dd[m]/(d*d)
        dE_dqis[:,0,m]=crossproduct(dE_dqis[:,1,m],E[:,2]) \
                      +crossproduct(E[:,1],dE_dqis[:,2,m])
        # Second derivative computations i = translations at node 2 and j = translations at node 1
        """ variations of e2 and e1 for for q_m m=i+6 and q_n n=j """
        for j in range(3):
            n=j
            dadj=innerproduct(v,dE_dqis[:,2,j])
            de2tilde_j=-dadj*E[:,2]-a*dE_dqis[:,2,j]
            ddadij=innerproduct(v,ddE_dqidqjs[:,2,ij12sym[m,j]])
            ddbdij=0.0
            ddddij=(ddbdij-ddadij*a-dadi*dadj-dd[m]*dd[n])/d
            dde2tilde_ij=-ddadij*E[:,2]-dadi*dE_dqis[:,2,j]-dadj*dE_dqis[:,2,m]-a*ddE_dqidqjs[:,2,ij12sym[m,j]]
            ddE_dqidqjs[:,1,ij12sym[m,n]]=dde2tilde_ij/d-(dd[m]*de2tilde_j+dd[n]*de2tilde_i)/(d*d) \
                                             +(2.0*dd[m]*dd[n]/(d*d*d)-ddddij/(d*d))*e2tilde
            ddE_dqidqjs[:,0,ij12sym[m,n]]=crossproduct(ddE_dqidqjs[:,1,ij12sym[m,n]],E[:,2]) \
                                    +crossproduct(dE_dqis[:,1,m],dE_dqis[:,2,j]) \
                                    +crossproduct(dE_dqis[:,1,n],dE_dqis[:,2,m]) \
                                    +crossproduct(E[:,1],ddE_dqidqjs[:,2,ij12sym[m,j]])
        # Second derivative computations i = translations at node 2 and j = rotations at node 1
        """ variations of e2 and e1 for q_m m=i+6 and q_n n=j+3 """
        for j in range(3):
            n=j+3
            dadj=innerproduct(dT[:,1,j],E[:,2])
            de2tilde_j=dT[:,1,j]-dadj*E[:,2]
            ddadij=innerproduct(dT[:,1,j],dE_dqis[:,2,m])
            ddbdij=0.0
            ddddij=(ddbdij-ddadij*a-dadi*dadj-dd[m]*dd[n])/d
            dde2tilde_ij=-ddadij*E[:,2]-dadj*dE_dqis[:,2,m]
            ddE_dqidqjs[:,1,ij12sym[m,n]]=dde2tilde_ij/d-(dd[m]*de2tilde_j+dd[n]*de2tilde_i)/(d*d) \
                                            +(2.0*dd[m]*dd[n]/(d*d*d)-ddddij/(d*d))*e2tilde
            ddE_dqidqjs[:,0,ij12sym[m,n]]=crossproduct(ddE_dqidqjs[:,1,ij12sym[m,n]],E[:,2]) \
                                    +crossproduct(dE_dqis[:,1,n],dE_dqis[:,2,m])
        # Second derivative computations i = translations at node 2 and j = translations at node 2
        """ variations of e2 and e1 for q_m m=i+6 and q_n n=j+6 """
        for j in range(i+1):
            n=j+6
            dadj=innerproduct(v,dE_dqis[:,2,n])
            de2tilde_j=-dadj*E[:,2]-a*dE_dqis[:,2,n]
            ddadij=innerproduct(v,ddE_dqidqjs[:,2,ij12sym[m,n]])
            ddbdij=0.0
            ddddij=(ddbdij-ddadij*a-dadi*dadj-dd[m]*dd[n])/d
            dde2tilde_ij=-ddadij*E[:,2]-dadi*dE_dqis[:,2,n]-dadj*dE_dqis[:,2,m]-a*ddE_dqidqjs[:,2,ij6sym[i+3,j+3]]
            ddE_dqidqjs[:,1,ij12sym[m,n]]=dde2tilde_ij/d-(dd[m]*de2tilde_j+dd[n]*de2tilde_i)/(d*d) \
                                             +(2.0*dd[m]*dd[n]/(d*d*d)-ddddij/(d*d))*e2tilde
            ddE_dqidqjs[:,0,ij12sym[m,n]]=crossproduct(ddE_dqidqjs[:,1,ij12sym[m,n]],E[:,2]) \
                                    +crossproduct(dE_dqis[:,1,m],dE_dqis[:,2,n]) \
                                    +crossproduct(dE_dqis[:,1,n],dE_dqis[:,2,m]) \
                                    +crossproduct(E[:,1],ddE_dqidqjs[:,2,ij12sym[m,n]])
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    for i in range(3):
    # First derivative computations i = rotations at node 2
        """ variations of e2 and e1 for q_m m=i+9 """
        m=i+9
        dadi=innerproduct(dQ[:,1,i],E[:,2])
        dd[m]=(innerproduct(dQ[:,1,i],T[:,1])-a*dadi)/d
        de2tilde_i=dQ[:,1,i]-dadi*E[:,2]
        dE_dqis[:,1,m]=de2tilde_i/d-e2tilde*dd[m]/(d*d)
        dE_dqis[:,0,m]=crossproduct(dE_dqis[:,1,m],E[:,2])
        # Second derivative computations i = rotations at node 2 and j = translations at node 1
        """ variations of e2 and e1 for q_m m=i+9 and q_n n=j """
        for j in range(3):
            n=j
            dadj=innerproduct(v,dE_dqis[:,2,j])
            de2tilde_j=-dadj*E[:,2]-a*dE_dqis[:,2,j]
            ddadij=innerproduct(dQ[:,1,i],dE_dqis[:,2,j])
            ddbdij=0.0
            ddddij=(ddbdij-ddadij*a-dadi*dadj-dd[m]*dd[n])/d
            dde2tilde_ij=-ddadij*E[:,2]-dadi*dE_dqis[:,2,j]
            ddE_dqidqjs[:,1,ij12sym[m,n]]=dde2tilde_ij/d-(dd[m]*de2tilde_j+dd[n]*de2tilde_i)/(d*d) \
                                            +(2.0*dd[m]*dd[n]/(d*d*d)-ddddij/(d*d))*e2tilde
            ddE_dqidqjs[:,0,ij12sym[m,n]]=crossproduct(ddE_dqidqjs[:,1,ij12sym[m,n]],E[:,2]) \
                                    +crossproduct(dE_dqis[:,1,m],dE_dqis[:,2,j])
        # Second derivative computations i = rotations at node 2 and j = rotations at node 1
        """ variations of e2 and e1 for q_m m=i+9 and q_n n=j+3 """
        for j in range(3):
            n=j+3
            dadj=innerproduct(dT[:,1,j],E[:,2])
            de2tilde_j=dT[:,1,j]-dadj*E[:,2]
            ddadij=0.0
            ddbdij=innerproduct(dT[:,1,j],dQ[:,1,i])
            ddddij=(ddbdij-ddadij*a-dadi*dadj-dd[m]*dd[n])/d
            dde2tilde_ij=-ddadij*E[:,2]
            ddE_dqidqjs[:,1,ij12sym[m,n]]=dde2tilde_ij/d-(dd[m]*de2tilde_j+dd[n]*de2tilde_i)/(d*d) \
                                            +(2.0*dd[m]*dd[n]/(d*d*d)-ddddij/(d*d))*e2tilde
            ddE_dqidqjs[:,0,ij12sym[m,n]]=crossproduct(ddE_dqidqjs[:,1,ij12sym[m,n]],E[:,2])
        # Second derivative computations i = rotations at node 2 and j = translations at node 2
        """ variations of e2 and e1 for q_m m=i+9 and q_n n=j+6 """
        for j in range(3):
            n=j+6
            dadj=innerproduct(v,dE_dqis[:,2,n])
            de2tilde_j=-dadj*E[:,2]-a*dE_dqis[:,2,n]
            ddadij=innerproduct(dQ[:,1,i],dE_dqis[:,2,n])
            ddbdij=0.0
            ddddij=(ddbdij-ddadij*a-dadi*dadj-dd[m]*dd[n])/d
            dde2tilde_ij=-ddadij*E[:,2]-dadi*dE_dqis[:,2,n]
            ddE_dqidqjs[:,1,ij12sym[m,n]]=dde2tilde_ij/d-(dd[m]*de2tilde_j+dd[n]*de2tilde_i)/(d*d) \
                                            +(2.0*dd[m]*dd[n]/(d*d*d)-ddddij/(d*d))*e2tilde
            ddE_dqidqjs[:,0,ij12sym[m,n]]=crossproduct(ddE_dqidqjs[:,1,ij12sym[m,n]],E[:,2]) \
                                    +crossproduct(dE_dqis[:,1,m],dE_dqis[:,2,n])
        # Second derivative computations i = rotations at node 2 and j = rotations at node 2
        """ variations of e2 and e1 for q_m m=i+9 and q_n n=j+9 """
        for j in range(i+1):
            n=j+9
            dadj=innerproduct(dQ[:,1,j],E[:,2])
            de2tilde_j=dQ[:,1,j]-dadj*E[:,2]
            ddadij=innerproduct(ddQ[:,1,ij3sym[i,j]],E[:,2])
            ddbdij=innerproduct(ddQ[:,1,ij3sym[i,j]],T[:,1])
            ddddij=(ddbdij-ddadij*a-dadi*dadj-dd[m]*dd[n])/d
            dde2tilde_ij=ddQ[:,1,ij3sym[i,j]]-ddadij*E[:,2]
            ddE_dqidqjs[:,1,ij12sym[m,n]]=dde2tilde_ij/d-(dd[m]*de2tilde_j+dd[n]*de2tilde_i)/(d*d) \
                                             +(2.0*dd[m]*dd[n]/(d*d*d)-ddddij/(d*d))*e2tilde
            ddE_dqidqjs[:,0,ij12sym[m,n]]=crossproduct(ddE_dqidqjs[:,1,ij12sym[m,n]],E[:,2])
    return l,E,rvec,rmid,d,dd,dE_dqis,ddE_dqidqjs
## Compute local nodal rotations (assumed small) and elongation
#
# Inputs:
#  l0    float64
#  l     float64
#  rvec0 float64[:]
#  rvec  float64[:]
#  dvec  float64[:]
#  E     float64[:,:]
#  T     float64[:,:]
#  Q     float64[:,:]
#
# Outputs
#  ql    float64[:]
#
@cc.export('update_local_nodal_rotations_elongation','(f8,f8,f8[:],f8[:],f8[:],f8[:,:],f8[:,:],f8[:,:])')
def update_local_nodal_rotations_elongation(l0,l,rvec0,rvec,dvec,E,T,Q):
    ql=np.zeros(7)
    ql[0]=0.5*(innerproduct(T[:,1],E[:,2])-innerproduct(T[:,2],E[:,1]))
    ql[1]=0.5*(innerproduct(T[:,2],E[:,0])-innerproduct(T[:,0],E[:,2]))
    ql[2]=0.5*(innerproduct(T[:,0],E[:,1])-innerproduct(T[:,1],E[:,0]))
    ql[3]=0.5*(innerproduct(Q[:,1],E[:,2])-innerproduct(Q[:,2],E[:,1]))
    ql[4]=0.5*(innerproduct(Q[:,2],E[:,0])-innerproduct(Q[:,0],E[:,2]))
    ql[5]=0.5*(innerproduct(Q[:,0],E[:,1])-innerproduct(Q[:,1],E[:,0]))
    ql[6]=2.0*innerproduct(dvec,rvec0+0.5*dvec)/(l+l0)
    return ql
## Compute first derivatives of local nodal rotations (assumed small) and elongation
#
# Inputs:
#  l       float64
#  rvec    float64[:]
#  E       float64[:,:]
#  dE_dqis float64[:,:,:]
#  T       float64[:,:]
#  Q       float64[:,:]
#  dT      float64[:,:,:]
#  dQ      float64[:,:,:]
#
# Outputs
#  dqldi0 float64[:,:]
#
# @nb.njit(nb.float64[:,:](nb.float64,nb.float64[:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:], \
#                          nb.float64[:,:,:],nb.float64[:,:,:],nb.float64[:,:],nb.float64[:,:,:]))
@cc.export('update_first_derivative_local_nodal_rotations_elongation','(f8,f8[:],f8[:,:],f8[:,:,:],f8[:,:],f8[:,:],f8[:,:,:],f8[:,:,:])')
def update_first_derivative_local_nodal_rotations_elongation(l,rvec,E,dE_dqis,T,Q,dT,dQ):
    # Initial
    dqldi0=np.zeros((7,12))
    # First derivatives of local rotations of node 1
    for j in range(3): # Translations at node 1
        dqldi0[0,j  ]=0.5*( innerproduct(T[:,1],dE_dqis[:,2,j])-innerproduct(T[:,2],dE_dqis[:,1,j]))
        dqldi0[1,j  ]=0.5*( innerproduct(T[:,2],dE_dqis[:,0,j])-innerproduct(T[:,0],dE_dqis[:,2,j]))
        dqldi0[2,j  ]=0.5*( innerproduct(T[:,0],dE_dqis[:,1,j])-innerproduct(T[:,1],dE_dqis[:,0,j]))
    for j in range(3): # Rotations at node 1
        m=j+3
        dqldi0[0,m]=0.5*(innerproduct(dT[:,1,j],E[:,2])-innerproduct(dT[:,2,j],E[:,1])-innerproduct(T[:,2],dE_dqis[:,1,m]))
        dqldi0[1,m]=0.5*(innerproduct(dT[:,2,j],E[:,0])-innerproduct(dT[:,0,j],E[:,2])+innerproduct(T[:,2],dE_dqis[:,0,m]))
        dqldi0[2,m]=0.5*(innerproduct(dT[:,0,j],E[:,1])-innerproduct(dT[:,1,j],E[:,0])+innerproduct(T[:,0],dE_dqis[:,1,m])-innerproduct(T[:,1],dE_dqis[:,0,m]))
    for j in range(3): # Translations at node 2
        m=j+6
        dqldi0[0,m]=0.5*( innerproduct(T[:,1],dE_dqis[:,2,m])-innerproduct(T[:,2],dE_dqis[:,1,m]))
        dqldi0[1,m]=0.5*( innerproduct(T[:,2],dE_dqis[:,0,m])-innerproduct(T[:,0],dE_dqis[:,2,m]))
        dqldi0[2,m]=0.5*( innerproduct(T[:,0],dE_dqis[:,1,m])-innerproduct(T[:,1],dE_dqis[:,0,m]))
    for j in range(3): # Rotations at node 2
        m=j+9
        dqldi0[0,m]=0.5*(-innerproduct(T[:,2],dE_dqis[:,1,m]))
        dqldi0[1,m]=0.5*( innerproduct(T[:,2],dE_dqis[:,0,m]))
        dqldi0[2,m]=0.5*( innerproduct(T[:,0],dE_dqis[:,1,m])-innerproduct(T[:,1],dE_dqis[:,0,m]))
    # First derivatives of local rotations of node 2
    for j in range(3): # Translations at node 1
        dqldi0[3,j  ]=0.5*(innerproduct(Q[:,1],dE_dqis[:,2,j])-innerproduct(Q[:,2],dE_dqis[:,1,j]))
        dqldi0[4,j  ]=0.5*(innerproduct(Q[:,2],dE_dqis[:,0,j])-innerproduct(Q[:,0],dE_dqis[:,2,j]))
        dqldi0[5,j  ]=0.5*(innerproduct(Q[:,0],dE_dqis[:,1,j])-innerproduct(Q[:,1],dE_dqis[:,0,j]))
    for j in range(3): # Rotations at node 1
        m=j+3
        dqldi0[3,m]=0.5*(-innerproduct(Q[:,2],dE_dqis[:,1,m]))
        dqldi0[4,m]=0.5*( innerproduct(Q[:,2],dE_dqis[:,0,m]))
        dqldi0[5,m]=0.5*( innerproduct(Q[:,0],dE_dqis[:,1,m])-innerproduct(Q[:,1],dE_dqis[:,0,m]))
    for j in range(3): # Translations at node 2
        m=j+6
        dqldi0[3,m]=0.5*(innerproduct(Q[:,1],dE_dqis[:,2,m])-innerproduct(Q[:,2],dE_dqis[:,1,m]))
        dqldi0[4,m]=0.5*(innerproduct(Q[:,2],dE_dqis[:,0,m])-innerproduct(Q[:,0],dE_dqis[:,2,m]))
        dqldi0[5,m]=0.5*(innerproduct(Q[:,0],dE_dqis[:,1,m])-innerproduct(Q[:,1],dE_dqis[:,0,m]))
    for j in range(3): # Rotations at node 2
        m=j+9
        dqldi0[3,m]=0.5*(innerproduct(dQ[:,1,j],E[:,2])-innerproduct(dQ[:,2,j],E[:,1])-innerproduct(Q[:,2],dE_dqis[:,1,m]))
        dqldi0[4,m]=0.5*(innerproduct(dQ[:,2,j],E[:,0])-innerproduct(dQ[:,0,j],E[:,2])+innerproduct(Q[:,2],dE_dqis[:,0,m]))
        dqldi0[5,m]=0.5*(innerproduct(dQ[:,0,j],E[:,1])-innerproduct(dQ[:,1,j],E[:,0])+innerproduct(Q[:,0],dE_dqis[:,1,m])-innerproduct(Q[:,1],dE_dqis[:,0,m]))
    # First derivative of elongation
    for j in range(3):
        dqldi0[6,j]=-rvec[j]/l
    for j in range(3):
        dqldi0[6,j+6]=rvec[j]/l
    # Return first derivatives
    return dqldi0
## Compute second derivatives of local nodal rotations (assumed small) and elongation
# Input:
#  l      float64
#  rvec   float64[:]
#  E      float64[:,:]
#  dE_dqis     float64[:,:,:]
#  ddE_dqidqjs float64[:,:,:]
#  T      float64[:,:]
#  Q      float64[:,:]
#  dT     float64[:,:,:]
#  dQ     float64[:,:,:]
#  ddT    float64[:,:,:]
#  ddQ    float64[:,:,:]
#
# Outputs
#  ddqldidj0 float64[:,:,:]
#
@cc.export('update_second_derivative_local_nodal_rotations_elongation','(f8,f8[:],f8[:,:],f8[:,:,:],f8[:,:,:],f8[:,:],f8[:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:])')
def update_second_derivative_local_nodal_rotations_elongation(l,rvec,E,dE_dqis,ddE_dqidqjs,T,Q,dT,dQ,ddT,ddQ):
    # Allocation
    ddqldidj0=np.zeros((7,12,12))    
    # Second derivatives of local rotations of node 1
    for j in range(3): # Translations at node 1 = j
        for k in range(j+1): # Translations at node 1 = k
            ddqldidj0[0,j,k]=0.5*(innerproduct(T[:,1],ddE_dqidqjs[:,2,ij12sym[j,k]])-innerproduct(T[:,2],ddE_dqidqjs[:,1,ij12sym[j,k]]))
            ddqldidj0[1,j,k]=0.5*(innerproduct(T[:,2],ddE_dqidqjs[:,0,ij12sym[j,k]])-innerproduct(T[:,0],ddE_dqidqjs[:,2,ij12sym[j,k]]))
            ddqldidj0[2,j,k]=0.5*(innerproduct(T[:,0],ddE_dqidqjs[:,1,ij12sym[j,k]])-innerproduct(T[:,1],ddE_dqidqjs[:,0,ij12sym[j,k]]))
    for j in range(3): # Rotations at node 1 = j+3
        m=j+3
        for k in range(3): # Translations at node 1 = k
            ddqldidj0[0,m,k]=0.5*(innerproduct(dT[:,1,j],dE_dqis[:,2,k])-innerproduct(dT[:,2,j],dE_dqis[:,1,k])-innerproduct(T[:,2],ddE_dqidqjs[:,1,ij12sym[m,k]]))
            ddqldidj0[1,m,k]=0.5*(innerproduct(dT[:,2,j],dE_dqis[:,0,k])-innerproduct(dT[:,0,j],dE_dqis[:,2,k])+innerproduct(T[:,2],ddE_dqidqjs[:,0,ij12sym[m,k]]))
            ddqldidj0[2,m,k]=0.5*(innerproduct(dT[:,0,j],dE_dqis[:,1,k])-innerproduct(dT[:,1,j],dE_dqis[:,0,k])+innerproduct(T[:,0],ddE_dqidqjs[:,1,ij12sym[m,k]])-innerproduct(T[:,1],ddE_dqidqjs[:,0,ij12sym[m,k]]))
        for k in range(j+1): # Rotations at node 1 = k+3
            n=k+3
            ddqldidj0[0,m,n]=0.5*(innerproduct(ddT[:,1,ij3sym[j,k]],E[:,2])-innerproduct(ddT[:,2,ij3sym[j,k]],E[:,1]) \
                                 -innerproduct(dT[:,2,j],dE_dqis[:,1,n])-innerproduct(dT[:,2,k],dE_dqis[:,1,m]) \
                                 -innerproduct(T[:,2],ddE_dqidqjs[:,1,ij12sym[m,n]]))
            ddqldidj0[1,m,n]=0.5*(innerproduct(ddT[:,2,ij3sym[j,k]],E[:,0])+innerproduct(dT[:,2,j],dE_dqis[:,0,n]) \
                                 +innerproduct(dT[:,2,k],dE_dqis[:,0,m])-innerproduct(ddT[:,0,ij3sym[j,k]],E[:,2]) \
                                 +innerproduct(T[:,2],ddE_dqidqjs[:,0,ij12sym[m,n]]))
            ddqldidj0[2,m,n]=0.5*(innerproduct(ddT[:,0,ij3sym[j,k]],E[:,1]) \
                                 +innerproduct(dT[:,0,j],dE_dqis[:,1,n])+innerproduct(dT[:,0,k],dE_dqis[:,1,m]) \
                                 +innerproduct(T[:,0],ddE_dqidqjs[:,1,ij12sym[m,n]]) \
                                 -innerproduct(ddT[:,1,ij3sym[j,k]],E[:,0]) \
                                 -innerproduct(dT[:,1,j],dE_dqis[:,0,n])-innerproduct(dT[:,1,k],dE_dqis[:,0,m]) \
                                 -innerproduct(T[:,1],ddE_dqidqjs[:,0,ij12sym[m,n]]))
    for j in range(3): # Translations at node 2 = j+6
        m=j+6
        for k in range(3): # Translations at node 1 = k
            ddqldidj0[0,m,k]=0.5*(innerproduct(T[:,1],ddE_dqidqjs[:,2,ij12sym[m,k]])-innerproduct(T[:,2],ddE_dqidqjs[:,1,ij12sym[m,k]]))
            ddqldidj0[1,m,k]=0.5*(innerproduct(T[:,2],ddE_dqidqjs[:,0,ij12sym[m,k]])-innerproduct(T[:,0],ddE_dqidqjs[:,2,ij12sym[m,k]]))
            ddqldidj0[2,m,k]=0.5*(innerproduct(T[:,0],ddE_dqidqjs[:,1,ij12sym[m,k]])-innerproduct(T[:,1],ddE_dqidqjs[:,0,ij12sym[m,k]]))
        for k in range(3): # Rotations at node 1 = k+3
            n=k+3
            ddqldidj0[0,m,n]=0.5*(innerproduct(dT[:,1,k],dE_dqis[:,2,m])-innerproduct(dT[:,2,k],dE_dqis[:,1,n])-innerproduct(T[:,2],ddE_dqidqjs[:,1,ij12sym[m,n]]))
            ddqldidj0[1,m,n]=0.5*(innerproduct(dT[:,2,k],dE_dqis[:,0,m])-innerproduct(dT[:,0,k],dE_dqis[:,2,m])+innerproduct(T[:,2],ddE_dqidqjs[:,0,ij12sym[m,n]]))
            ddqldidj0[2,m,n]=0.5*(innerproduct(dT[:,0,k],dE_dqis[:,1,m])-innerproduct(dT[:,1,k],dE_dqis[:,0,m])+innerproduct(T[:,0],ddE_dqidqjs[:,1,ij12sym[m,n]])-innerproduct(T[:,1],ddE_dqidqjs[:,0,ij12sym[m,n]]))
        for k in range(j+1): # Translations at node 2 = k
            n=k+6
            ddqldidj0[0,m,n]=0.5*(innerproduct(T[:,1],ddE_dqidqjs[:,2,ij12sym[m,n]])-innerproduct(T[:,2],ddE_dqidqjs[:,1,ij12sym[m,n]]))
            ddqldidj0[1,m,n]=0.5*(innerproduct(T[:,2],ddE_dqidqjs[:,0,ij12sym[m,n]])-innerproduct(T[:,0],ddE_dqidqjs[:,2,ij12sym[m,n]]))
            ddqldidj0[2,m,n]=0.5*(innerproduct(T[:,0],ddE_dqidqjs[:,1,ij12sym[m,n]])-innerproduct(T[:,1],ddE_dqidqjs[:,0,ij12sym[m,n]]))
    for j in range(3): # Rotations at node 2
        m=j+9
        for k in range(3): # Translations at node 1 = k
            ddqldidj0[0,j+9,k]=0.5*(-innerproduct(T[:,2],ddE_dqidqjs[:,1,ij12sym[j+9,k]]))
            ddqldidj0[1,j+9,k]=0.5*( innerproduct(T[:,2],ddE_dqidqjs[:,0,ij12sym[j+9,k]]))
            ddqldidj0[2,j+9,k]=0.5*( innerproduct(T[:,0],ddE_dqidqjs[:,1,ij12sym[j+9,k]]) \
                                    -innerproduct(T[:,1],ddE_dqidqjs[:,0,ij12sym[j+9,k]]))
        for k in range(3): # Rotations at node 1 = k
            n=k+3
            ddqldidj0[0,m,n]=0.5*(-innerproduct(dT[:,2,k],dE_dqis[:,1,m])-innerproduct(T[:,2],ddE_dqidqjs[:,1,ij12sym[m,n]]))
            ddqldidj0[1,m,n]=0.5*( innerproduct(dT[:,2,k],dE_dqis[:,0,m])+innerproduct(T[:,2],ddE_dqidqjs[:,0,ij12sym[m,n]]))
            ddqldidj0[2,m,n]=0.5*( innerproduct(dT[:,0,k],dE_dqis[:,1,m])+innerproduct(T[:,0],ddE_dqidqjs[:,1,ij12sym[m,n]]) \
                                  -innerproduct(dT[:,1,k],dE_dqis[:,0,m])-innerproduct(T[:,1],ddE_dqidqjs[:,0,ij12sym[m,n]]))
        for k in range(3): # Translations at node 2 = k
            n=k+6
            ddqldidj0[0,m,n]=0.5*(-innerproduct(T[:,2],ddE_dqidqjs[:,1,ij12sym[m,n]]))
            ddqldidj0[1,m,n]=0.5*( innerproduct(T[:,2],ddE_dqidqjs[:,0,ij12sym[m,n]]))
            ddqldidj0[2,m,n]=0.5*( innerproduct(T[:,0],ddE_dqidqjs[:,1,ij12sym[m,n]]) \
                                  -innerproduct(T[:,1],ddE_dqidqjs[:,0,ij12sym[m,n]]))
        for k in range(j+1): # Rotations at node 2 = k
            n=k+9
            ddqldidj0[0,m,n]=0.5*(-innerproduct(T[:,2],ddE_dqidqjs[:,1,ij12sym[m,n]]))
            ddqldidj0[1,m,n]=0.5*( innerproduct(T[:,2],ddE_dqidqjs[:,0,ij12sym[m,n]]))
            ddqldidj0[2,m,n]=0.5*( innerproduct(T[:,0],ddE_dqidqjs[:,1,ij12sym[m,n]]) \
                                  -innerproduct(T[:,1],ddE_dqidqjs[:,0,ij12sym[m,n]]))
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Second derivatives of local rotations of node 2
    for j in range(3): # Qranslations at node 1 = j
        for k in range(j+1): # Translations at node 1 = k
            ddqldidj0[3,j,k]=0.5*(innerproduct(Q[:,1],ddE_dqidqjs[:,2,ij12sym[j,k]])-innerproduct(Q[:,2],ddE_dqidqjs[:,1,ij12sym[j,k]]))
            ddqldidj0[4,j,k]=0.5*(innerproduct(Q[:,2],ddE_dqidqjs[:,0,ij12sym[j,k]])-innerproduct(Q[:,0],ddE_dqidqjs[:,2,ij12sym[j,k]]))
            ddqldidj0[5,j,k]=0.5*(innerproduct(Q[:,0],ddE_dqidqjs[:,1,ij12sym[j,k]])-innerproduct(Q[:,1],ddE_dqidqjs[:,0,ij12sym[j,k]]))
    for j in range(3): # Rotations at node 1 = j+3
        m=j+3
        for k in range(3): # Translations at node 1 = k
            ddqldidj0[3,m,k]=0.5*(innerproduct(dQ[:,1,j],dE_dqis[:,2,k])-innerproduct(dQ[:,2,j],dE_dqis[:,1,k])-innerproduct(Q[:,2],ddE_dqidqjs[:,1,ij12sym[m,k]]))
            ddqldidj0[4,m,k]=0.5*(innerproduct(dQ[:,2,j],dE_dqis[:,0,k])-innerproduct(dQ[:,0,j],dE_dqis[:,2,k])+innerproduct(Q[:,2],ddE_dqidqjs[:,0,ij12sym[m,k]]))
            ddqldidj0[5,m,k]=0.5*(innerproduct(dQ[:,0,j],dE_dqis[:,1,k])-innerproduct(dQ[:,1,j],dE_dqis[:,0,k])+innerproduct(Q[:,0],ddE_dqidqjs[:,1,ij12sym[m,k]])-innerproduct(Q[:,1],ddE_dqidqjs[:,0,ij12sym[m,k]]))
        for k in range(j+1): # Rotations at node 1 = k+3
            n=k+3
            ddqldidj0[3,m,n]=0.5*(innerproduct(ddQ[:,1,ij3sym[j,k]],E[:,2])-innerproduct(ddQ[:,2,ij3sym[j,k]],E[:,1]) \
                                 -innerproduct(dQ[:,2,j],dE_dqis[:,1,n])-innerproduct(dQ[:,2,k],dE_dqis[:,1,m]) \
                                 -innerproduct(Q[:,2],ddE_dqidqjs[:,1,ij12sym[m,n]]))
            ddqldidj0[4,m,n]=0.5*(innerproduct(ddQ[:,2,ij3sym[j,k]],E[:,0])+innerproduct(dQ[:,2,j],dE_dqis[:,0,n]) \
                                 +innerproduct(dQ[:,2,k],dE_dqis[:,0,m])-innerproduct(ddQ[:,0,ij3sym[j,k]],E[:,2]) \
                                 +innerproduct(Q[:,2],ddE_dqidqjs[:,0,ij12sym[m,n]]))
            ddqldidj0[5,m,n]=0.5*(innerproduct(ddQ[:,0,ij3sym[j,k]],E[:,1]) \
                                 +innerproduct(dQ[:,0,j],dE_dqis[:,1,n])+innerproduct(dQ[:,0,k],dE_dqis[:,1,m]) \
                                 +innerproduct(Q[:,0],ddE_dqidqjs[:,1,ij12sym[m,n]]) \
                                 -innerproduct(ddQ[:,1,ij3sym[j,k]],E[:,0]) \
                                 -innerproduct(dQ[:,1,j],dE_dqis[:,0,n])-innerproduct(dQ[:,1,k],dE_dqis[:,0,m]) \
                                 -innerproduct(Q[:,1],ddE_dqidqjs[:,0,ij12sym[m,n]]))
    for j in range(3): # Translations at node 2 = j+6
        m=j+6
        for k in range(3): # Translations at node 1 = k
            ddqldidj0[3,m,k]=0.5*(innerproduct(Q[:,1],ddE_dqidqjs[:,2,ij12sym[m,k]])-innerproduct(Q[:,2],ddE_dqidqjs[:,1,ij12sym[m,k]]))
            ddqldidj0[4,m,k]=0.5*(innerproduct(Q[:,2],ddE_dqidqjs[:,0,ij12sym[m,k]])-innerproduct(Q[:,0],ddE_dqidqjs[:,2,ij12sym[m,k]]))
            ddqldidj0[5,m,k]=0.5*(innerproduct(Q[:,0],ddE_dqidqjs[:,1,ij12sym[m,k]])-innerproduct(Q[:,1],ddE_dqidqjs[:,0,ij12sym[m,k]]))
        for k in range(3): # Rotations at node 1 = k+3
            n=k+3
            ddqldidj0[3,m,n]=0.5*(innerproduct(dQ[:,1,k],dE_dqis[:,2,m])-innerproduct(dQ[:,2,k],dE_dqis[:,1,n])-innerproduct(Q[:,2],ddE_dqidqjs[:,1,ij12sym[m,n]]))
            ddqldidj0[4,m,n]=0.5*(innerproduct(dQ[:,2,k],dE_dqis[:,0,m])-innerproduct(dQ[:,0,k],dE_dqis[:,2,m])+innerproduct(Q[:,2],ddE_dqidqjs[:,0,ij12sym[m,n]]))
            ddqldidj0[5,m,n]=0.5*(innerproduct(dQ[:,0,k],dE_dqis[:,1,m])-innerproduct(dQ[:,1,k],dE_dqis[:,0,m])+innerproduct(Q[:,0],ddE_dqidqjs[:,1,ij12sym[m,n]])-innerproduct(Q[:,1],ddE_dqidqjs[:,0,ij12sym[m,n]]))
        for k in range(j+1): # Qranslations at node 2 = k
            n=k+6
            ddqldidj0[3,m,n]=0.5*(innerproduct(Q[:,1],ddE_dqidqjs[:,2,ij12sym[m,n]])-innerproduct(Q[:,2],ddE_dqidqjs[:,1,ij12sym[m,n]]))
            ddqldidj0[4,m,n]=0.5*(innerproduct(Q[:,2],ddE_dqidqjs[:,0,ij12sym[m,n]])-innerproduct(Q[:,0],ddE_dqidqjs[:,2,ij12sym[m,n]]))
            ddqldidj0[5,m,n]=0.5*(innerproduct(Q[:,0],ddE_dqidqjs[:,1,ij12sym[m,n]])-innerproduct(Q[:,1],ddE_dqidqjs[:,0,ij12sym[m,n]]))
    for j in range(3): # Rotations at node 2
        m=j+9
        for k in range(3): # Translations at node 1 = k
            ddqldidj0[3,j+9,k]=0.5*(-innerproduct(Q[:,2],ddE_dqidqjs[:,1,ij12sym[j+9,k]]))
            ddqldidj0[4,j+9,k]=0.5*( innerproduct(Q[:,2],ddE_dqidqjs[:,0,ij12sym[j+9,k]]))
            ddqldidj0[5,j+9,k]=0.5*( innerproduct(Q[:,0],ddE_dqidqjs[:,1,ij12sym[j+9,k]]) \
                                    -innerproduct(Q[:,1],ddE_dqidqjs[:,0,ij12sym[j+9,k]]))
        for k in range(3): # Rotations at node 1 = k
            n=k+3
            ddqldidj0[3,m,n]=0.5*(-innerproduct(dQ[:,2,k],dE_dqis[:,1,m])-innerproduct(Q[:,2],ddE_dqidqjs[:,1,ij12sym[m,n]]))
            ddqldidj0[4,m,n]=0.5*( innerproduct(dQ[:,2,k],dE_dqis[:,0,m])+innerproduct(Q[:,2],ddE_dqidqjs[:,0,ij12sym[m,n]]))
            ddqldidj0[5,m,n]=0.5*( innerproduct(dQ[:,0,k],dE_dqis[:,1,m])+innerproduct(Q[:,0],ddE_dqidqjs[:,1,ij12sym[m,n]]) \
                                  -innerproduct(dQ[:,1,k],dE_dqis[:,0,m])-innerproduct(Q[:,1],ddE_dqidqjs[:,0,ij12sym[m,n]]))
        for k in range(3): # Translations at node 2 = k
            n=k+6
            ddqldidj0[3,m,n]=0.5*(-innerproduct(Q[:,2],ddE_dqidqjs[:,1,ij12sym[m,n]]))
            ddqldidj0[4,m,n]=0.5*( innerproduct(Q[:,2],ddE_dqidqjs[:,0,ij12sym[m,n]]))
            ddqldidj0[5,m,n]=0.5*( innerproduct(Q[:,0],ddE_dqidqjs[:,1,ij12sym[m,n]]) \
                                  -innerproduct(Q[:,1],ddE_dqidqjs[:,0,ij12sym[m,n]]))
        for k in range(j+1): # Rotations at node 2 = k
            n=k+9
            ddqldidj0[3,m,n]=0.5*(-innerproduct(Q[:,2],ddE_dqidqjs[:,1,ij12sym[m,n]]))
            ddqldidj0[4,m,n]=0.5*( innerproduct(Q[:,2],ddE_dqidqjs[:,0,ij12sym[m,n]]))
            ddqldidj0[5,m,n]=0.5*( innerproduct(Q[:,0],ddE_dqidqjs[:,1,ij12sym[m,n]]) \
                                  -innerproduct(Q[:,1],ddE_dqidqjs[:,0,ij12sym[m,n]]))
    # Second derivative of elongation of lower triangle row >= col
    for j in range(3):
        for k in range(j+1):
            ddqldidj0[6,j,k]  = Imat[j,k]/l-rvec[j]*rvec[k]/(l*l*l)
            ddqldidj0[6,j+6,k+6]=ddqldidj0[6,j,k]
    for j in range(3):
        for k in range(3):
            ddqldidj0[6,j+6,k]=-Imat[j,k]/l+rvec[j]*rvec[k]/(l*l*l)
    # Insert symmetric parts
    for k in range(7):
        for j in range(12):
            for i in range(j):
                ddqldidj0[k,i,j]=ddqldidj0[k,j,i]
    # Return second derivatives
    return ddqldidj0


## Compute deflection sub-vector coefficients for each shape function order
#
# Inputs:
#  l           float64
#  rmid        float64[:]
#  E           float64[:,:]
#  dE_dqis     float64[:,:,:]
#  ddE_dqidqjs float64[:,:,:]
#  Nl          float64[:,:,:]
#  ql          float64[:]
#  dqldi0      float64[:,:]
#  ddqldidj0   float64[:,:,:]
#
# Outputs
#  Nlql,Nldqldqi,Nlddqldqidqj,ro,rx,ry,dro_dqi,drx_dqi,dry_dqi,ddro_dqidqj,ddrx_dqidqj,ddry_dqidqj
#
@cc.export('update_element_deflection_subvectors_and_derivatives','(f8,f8[:],f8[:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:],f8[:,:],f8[:,:,:])')
def update_element_deflection_subvectors_and_derivatives(l,rmid,E,dE_dqis,ddE_dqidqjs,Nl,ql,dqldi0,ddqldidj0):
    # Order of shape functions
    P=Nl.shape[2]
    # Deflection subvectors and the derivatives
    EPx=np.zeros((3,3))
    EPy=np.zeros((3,3))
    dEPx_dqi=np.zeros((3,3))
    dEPy_dqi=np.zeros((3,3))
    dEPx_dqj=np.zeros((3,3))
    dEPy_dqj=np.zeros((3,3))
    ddE_dqidqj=np.zeros((3,3))
    ddEPx_dqidqj=np.zeros((3,3))
    ddEPy_dqidqj=np.zeros((3,3))
    EPx[:,1]=-E[:,2]
    EPx[:,2]= E[:,1]
    EPy[:,0]= E[:,2]
    EPy[:,2]=-E[:,0]
    ro=np.zeros((3,P))
    rx=np.zeros((3,P))
    ry=np.zeros((3,P))
    dro_dqi=np.zeros((3,12,P))
    drx_dqi=np.zeros((3,12,P))
    dry_dqi=np.zeros((3,12,P))
    ddro_dqidqj=np.zeros((3,78,P))
    ddrx_dqidqj=np.zeros((3,78,P))
    ddry_dqidqj=np.zeros((3,78,P))
    Nlql=np.zeros((6,P+4))
    Nldqldqi=np.zeros((6,12,P+4))
    Nlddqldqidqj=np.zeros((6,78,P+4))
    for p in range(P):
        Nlql[:,p]=matvec67(Nl[:,:,p],ql)
        ro[:,p]=matvec33(E  ,Nlql[0:3,p])
        rx[:,p]=matvec33(EPx,Nlql[3:6,p])
        ry[:,p]=matvec33(EPy,Nlql[3:6,p])
        for i in range(12):
            Nldqldqi[:,i,p]=matvec67(Nl[:,:,p],dqldi0[:,i])
            for j in range(12):
                Nlddqldqidqj[:,ij12sym[i,j],p]=matvec67(Nl[:,:,p],ddqldidj0[:,i,j])
    # Add rigid body motion of the element
    ro[:,0]+=rmid
    ro[:,1]+=0.5*l*E[:,2]
    rx[:,0]+=      E[:,0]
    ry[:,0]+=      E[:,1]
    # First derivatives of translations at node 1 = i
    for i in range(12):
        dEPx_dqi[:,1]=-dE_dqis[:,2,i]
        dEPx_dqi[:,2]= dE_dqis[:,1,i]
        dEPy_dqi[:,0]= dE_dqis[:,2,i]
        dEPy_dqi[:,2]=-dE_dqis[:,0,i]
        for p in range(P):
            dro_dqi[:,i,p]=matvec33(dE_dqis[:,:,i],Nlql[0:3,p])+matvec33(E,  Nldqldqi[0:3,i,p])
            drx_dqi[:,i,p]=matvec33(dEPx_dqi      ,Nlql[3:6,p])+matvec33(EPx,Nldqldqi[3:6,i,p])
            dry_dqi[:,i,p]=matvec33(dEPy_dqi      ,Nlql[3:6,p])+matvec33(EPy,Nldqldqi[3:6,i,p])
        # Second derivatives of translations at node 1 (i) and translations at node 1 (j)
        for j in range(i,12):
            ddE_dqidqj=ddE_dqidqjs[:,:,ij12sym[i,j]]
            dEPx_dqj[:,1]=-dE_dqis[:,2,j]
            dEPx_dqj[:,2]= dE_dqis[:,1,j]
            dEPy_dqj[:,0]= dE_dqis[:,2,j]
            dEPy_dqj[:,2]=-dE_dqis[:,0,j]
            ddEPx_dqidqj[:,1]=-ddE_dqidqj[:,2]
            ddEPx_dqidqj[:,2]= ddE_dqidqj[:,1]
            ddEPy_dqidqj[:,0]= ddE_dqidqj[:,2]
            ddEPy_dqidqj[:,2]=-ddE_dqidqj[:,0]
            for p in range(P):
                ddro_dqidqj[:,ij12sym[i,j],p]=matvec33(ddE_dqidqj  ,Nlql[0:3,p])+matvec33(E  ,Nlddqldqidqj[0:3,ij12sym[i,j],p]) \
                                             +matvec33(dE_dqis[:,:,i],Nldqldqi[0:3,j,p])+matvec33(dE_dqis[:,:,j],Nldqldqi[0:3,i,p])
                ddrx_dqidqj[:,ij12sym[i,j],p]=matvec33(ddEPx_dqidqj,Nlql[3:6,p])+matvec33(EPx,Nlddqldqidqj[3:6,ij12sym[i,j],p]) \
                                             +matvec33(dEPx_dqi      ,Nldqldqi[3:6,j,p])+matvec33(dEPx_dqj      ,Nldqldqi[3:6,i,p])
                ddry_dqidqj[:,ij12sym[i,j],p]=matvec33(ddEPy_dqidqj,Nlql[3:6,p])+matvec33(EPy,Nlddqldqidqj[3:6,ij12sym[i,j],p]) \
                                             +matvec33(dEPy_dqi      ,Nldqldqi[3:6,j,p])+matvec33(dEPy_dqj      ,Nldqldqi[3:6,i,p])
    # Add rigid body motion of the element
    for i in range(12):
        dro_dqi[:,i,1]+=0.5*l*dE_dqis[:,2,i]
        drx_dqi[:,i,0]+=      dE_dqis[:,0,i]
        dry_dqi[:,i,0]+=      dE_dqis[:,1,i]
        if i in [0,1,2]:
            dro_dqi[:,i,0]+=0.5*Imat[:,i]
        elif i in [6,7,8]:
            dro_dqi[:,i,0]+=0.5*Imat[:,i-6]
        for j in range(i,12):
            ddro_dqidqj[:,ij12sym[i,j],1]+=0.5*l*ddE_dqidqjs[:,2,ij12sym[i,j]]
            ddrx_dqidqj[:,ij12sym[i,j],0]+=      ddE_dqidqjs[:,0,ij12sym[i,j]]
            ddry_dqidqj[:,ij12sym[i,j],0]+=      ddE_dqidqjs[:,1,ij12sym[i,j]]

    return Nlql,Nldqldqi,Nlddqldqidqj,ro,rx,ry,dro_dqi,drx_dqi,dry_dqi,ddro_dqidqj,ddrx_dqidqj,ddry_dqidqj

## Compute deflection sub-vector coefficients for each shape function order
#
# Inputs:
#  norder      int32
#  l           float64
#  iner_pars   float64[:,:]
#  ro
#  rx
#  ry
#  dro_dqi
#  drx_dqi
#  dry_dqi
#  ddro_dqidqj
#  ddrx_dqidqj
#  ddry_dqidqj
#  
#  
#  
#  
#  
#
# Outputs
#  m_rcg          float64[:]
#  Ibase          float64[:,:]
#  m_drcg_dqi     float64[:,:]
#  Abase_i        float64[:,:,:]
#  m_ddrcg_dqidqj float64[:,:]
#  M11            float64[:]
#  Abase1_ij      float64[:,:,:]
#  Abase2_ij      float64[:,:,:]
#  
#
@cc.export('update_element_inertia','(i4,f8,f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:])')
def update_element_inertia(norder,l,iner_pars,ro,rx,ry,dro_dqi,drx_dqi,dry_dqi,ddro_dqidqj,ddrx_dqidqj,ddry_dqidqj):
    
    m_rcg = np.zeros(3)
    Ibase = np.zeros((3,3))
    m_drcg_dqi = np.zeros((3,12))
    Abase_i = np.zeros((3,3,12))
    m_ddrcg_dqidqj = np.zeros((3,78))
    M11 = np.zeros(78)
    Abase1_ij = np.zeros((3,3,78))
    Abase2_ij = np.zeros((3,3,78))
    
    for r in range(norder+1):
        for p in range(norder+4):
            m_rcg += 0.5*l*c_function(p+r)*(iner_pars[r,0]*ro[:,p]+iner_pars[r,1]*rx[:,p]+iner_pars[r,2]*ry[:,p])
            for q in range(norder+4):
                Ibase+=0.5*l*c_function(q+p+r)*matrix_G_operator(iner_pars[r,:],ro[:,q],rx[:,q],ry[:,q],ro[:,p],rx[:,p],ry[:,p])
            # Clamped first node of body
            for idof in range(12):
                m_drcg_dqi[:,idof]   += 0.5*l*c_function(p+r)*(iner_pars[r,0]*dro_dqi[:,idof,p]+iner_pars[r,1]*drx_dqi[:,idof,p]+iner_pars[r,2]*dry_dqi[:,idof,p])
                for q in range(norder+4):
                    Abase_i[:,:,idof]+= 0.5*l*c_function(q+p+r)*matrix_G_operator(iner_pars[r,:],dro_dqi[:,idof,p],drx_dqi[:,idof,p],dry_dqi[:,idof,p],ro[:,q],rx[:,q],ry[:,q])
                for jdof in range(idof,12):
                    m_ddrcg_dqidqj[:,ij12sym[idof,jdof]] += 0.5*l*c_function(p+r)*(iner_pars[r,0]*ddro_dqidqj[:,ij12sym[idof,jdof],p] \
                                                                                 +iner_pars[r,1]*ddrx_dqidqj[:,ij12sym[idof,jdof],p] \
                                                                                 +iner_pars[r,2]*ddry_dqidqj[:,ij12sym[idof,jdof],p])
                    for q in range(norder+4):
                        M11[ij12sym[idof,jdof]]+=0.5*l*c_function(q+p+r)*scalar_G_operator(iner_pars[r,:],dro_dqi[:,idof,q],drx_dqi[:,idof,q],dry_dqi[:,idof,q], \
                                                                                                          dro_dqi[:,jdof,p],drx_dqi[:,jdof,p],dry_dqi[:,jdof,p])
                        Abase1_ij[:,:,ij12sym[idof,jdof]]+=0.5*l*c_function(q+p+r)*matrix_G_operator(iner_pars[r,:],dro_dqi[:,jdof,p],drx_dqi[:,jdof,p],dry_dqi[:,jdof,p], \
                                                                                                                    dro_dqi[:,idof,q],drx_dqi[:,idof,q],dry_dqi[:,idof,q])
                        Abase2_ij[:,:,ij12sym[idof,jdof]]+=0.5*l*c_function(q+p+r) \
                            *matrix_G_operator(iner_pars[r,:],ddro_dqidqj[:,ij12sym[idof,jdof],p],ddrx_dqidqj[:,ij12sym[idof,jdof],p],ddry_dqidqj[:,ij12sym[idof,jdof],p], \
                                               ro[:,q],rx[:,q],ry[:,q])
    # Return values
    return m_rcg,Ibase,m_drcg_dqi,Abase_i,m_ddrcg_dqidqj,M11,Abase1_ij,Abase2_ij






# Update the forcing point position and vector arm for moment force and their derivatives
#
# Inputs:
#  ninterval   int32
#  norder      int32
#  ro
#  rx
#  ry
#  dro_dqi
#  drx_dqi
#  dry_dqi
#  ddro_dqidqj
#  ddrx_dqidqj
#  ddry_dqidqj
#  cfx
#  cfy
#  ce1
#  E
#  dE_dqis
#  ddE_dqidqjs
#  Nlql
#  Nldqldqi
#  Nlddqldqidqj
#
# Outputs
#  rf,e1,drf_dqi,de1_dqi,ddrf_dqidqj,dde1_dqidqj
#
@cc.export('update_forcing_point_position_and_moment_arm_vectors','(i4,i4,f8[:,:],f8[:,:],f8[:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:,:],f8[:,:],f8[:,:,:],f8[:,:],f8[:,:,:],f8[:,:,:],f8[:,:],f8[:,:,:],f8[:,:,:])')
def update_forcing_point_position_and_moment_arm_vectors(ninterval,norder,ro,rx,ry,dro_dqi,drx_dqi,dry_dqi,ddro_dqidqj,ddrx_dqidqj,ddry_dqidqj, \
                                                         cfx,cfy,ce1,E,dE_dqis,ddE_dqidqjs,Nlql,Nldqldqi,Nlddqldqidqj):
    # Coefficients of forcing point position and vector arm for moment force in substructure frame
    rf=np.zeros((3,ninterval,norder+4))
    e1=np.zeros((3,ninterval,norder+4))
    # and their first and second derivatives
    drf_dqi=np.zeros((3,12,ninterval,norder+4))
    de1_dqi=np.zeros((3,12,ninterval,norder+4))
    ddrf_dqidqj=np.zeros((3,78,ninterval,norder+4))
    dde1_dqidqj=np.zeros((3,78,ninterval,norder+4))
    # Compute for each point
    for m in range(ninterval):
        # Position vector coefficients
        rf[:,m,0]       =ro[:,0]+cfx[m,0]*rx[:,0]       +cfy[m,0]*ry[:,0]
        rf[:,m,norder+3]=        cfx[m,1]*rx[:,norder+2]+cfy[m,1]*ry[:,norder+2]
        for p in range(1,norder+3):
            rf[:,m,p]=ro[:,p]+cfx[m,0]*rx[:,p]+cfy[m,0]*ry[:,p]+cfx[m,1]*rx[:,p-1]+cfy[m,1]*ry[:,p-1]
        # Moment arm vector coefficients
        e1[:,m,0]       =matvec33(E,ce1[:,m,0]+skewmul(Nlql[3:,0]       ,ce1[:,m,0]))
        e1[:,m,norder+3]=matvec33(E,           skewmul(Nlql[3:,norder+2],ce1[:,m,1]))
        for p in range(1,norder+3):
            e1[:,m,p]=matvec33(E,skewmul(Nlql[3:,p],ce1[:,m,0])+skewmul(Nlql[3:,p-1],ce1[:,m,1]))
        # Coefficients of first derivatives of position vector 
        for idof in range(12):
            drf_dqi[:,idof,m,0]=dro_dqi[:,idof,0]+cfx[m,0]*drx_dqi[:,idof,0]       +cfy[m,0]*dry_dqi[:,idof,0]
            drf_dqi[:,idof,m,norder+3]=           cfx[m,1]*drx_dqi[:,idof,norder+2]+cfy[m,1]*dry_dqi[:,idof,norder+2]
            for p in range(1,norder+3):
                drf_dqi[:,idof,m,p]=dro_dqi[:,idof,p]+cfx[m,0]*drx_dqi[:,idof,p]  +cfy[m,0]*dry_dqi[:,idof,p] \
                                                     +cfx[m,1]*drx_dqi[:,idof,p-1]+cfy[m,1]*dry_dqi[:,idof,p-1]
        # Coefficients of the first derivatives of moment arm vector 
        for idof in range(12):
            de1_dqi[:,idof,m,0]       =matvec33(dE_dqis[:,:,idof],ce1[:,m,0]+skewmul(Nlql[3:,0]       ,ce1[:,m,0]))+matvec33(E,skewmul(Nldqldqi[3:,idof,0]       ,ce1[:,m,0]))
            de1_dqi[:,idof,m,norder+3]=matvec33(dE_dqis[:,:,idof],           skewmul(Nlql[3:,norder+2],ce1[:,m,0]))+matvec33(E,skewmul(Nldqldqi[3:,idof,norder+2],ce1[:,m,0]))
            for p in range(1,norder+3):
                de1_dqi[:,idof,m,p]=matvec33(dE_dqis[:,:,idof],skewmul(Nlql[3:,p]         ,ce1[:,m,0])+skewmul(Nlql[3:,p-1]         ,ce1[:,m,1])) \
                                                   +matvec33(E,skewmul(Nldqldqi[3:,idof,p],ce1[:,m,0])+skewmul(Nldqldqi[3:,idof,p-1],ce1[:,m,1]))
        # Coefficients of second derivatives of position vector 
        for idof in range(12):
            for jdof in range(idof,12):
                ddrf_dqidqj[:,ij12sym[idof,jdof],m,0]=ddro_dqidqj[:,ij12sym[idof,jdof],0]+cfx[m,0]*ddrx_dqidqj[:,ij12sym[idof,jdof],0] \
                                                                                         +cfy[m,0]*ddry_dqidqj[:,ij12sym[idof,jdof],0]
                ddrf_dqidqj[:,ij12sym[idof,jdof],m,norder+3]=                             cfx[m,1]*ddrx_dqidqj[:,ij12sym[idof,jdof],norder+2] \
                                                                                         +cfy[m,1]*ddry_dqidqj[:,ij12sym[idof,jdof],norder+2]
                for p in range(1,norder+3):
                    ddrf_dqidqj[:,ij12sym[idof,jdof],m,p]=ddro_dqidqj[:,ij12sym[idof,jdof],p] \
                                                +cfx[m,0]*ddrx_dqidqj[:,ij12sym[idof,jdof],p]  +cfy[m,0]*ddry_dqidqj[:,ij12sym[idof,jdof],p] \
                                                +cfx[m,1]*ddrx_dqidqj[:,ij12sym[idof,jdof],p-1]+cfy[m,1]*ddry_dqidqj[:,ij12sym[idof,jdof],p-1]
        # Coefficients of the second derivatives of moment arm vector 
        for idof in range(12):
            for jdof in range(idof,12):
                dde1_dqidqj[:,ij12sym[idof,jdof],m,0]=matvec33(ddE_dqidqjs[:,:,ij12sym[idof,jdof]],ce1[:,m,0]+skewmul(Nlql[3:,0],ce1[:,m,0])) \
                                                                                  +matvec33(dE_dqis[:,:,idof],skewmul(Nldqldqi[3:,jdof,0],ce1[:,m,0])) \
                                                                                  +matvec33(dE_dqis[:,:,jdof],skewmul(Nldqldqi[3:,idof,0],ce1[:,m,0])) \
                                                                                                  +matvec33(E,skewmul(Nlddqldqidqj[3:,ij12sym[idof,jdof],0],ce1[:,m,0]))
                dde1_dqidqj[:,ij12sym[idof,jdof],m,norder+3]=matvec33(ddE_dqidqjs[:,:,ij12sym[idof,jdof]],skewmul(Nlql[3:,norder+2],ce1[:,m,0])) \
                                                                              +matvec33(dE_dqis[:,:,jdof],skewmul(Nldqldqi[3:,idof,norder+2],ce1[:,m,0])) \
                                                                              +matvec33(dE_dqis[:,:,idof],skewmul(Nldqldqi[3:,jdof,norder+2],ce1[:,m,0])) \
                                                                                              +matvec33(E,skewmul(Nlddqldqidqj[3:,ij12sym[idof,jdof],norder+2],ce1[:,m,0]))
                for p in range(1,norder+3):
                    dde1_dqidqj[:,ij12sym[idof,jdof],m,p]=matvec33(ddE_dqidqjs[:,:,ij12sym[idof,jdof]],skewmul(Nlql[3:,p],ce1[:,m,0])+skewmul(Nlql[3:,p-1],ce1[:,m,1])) \
                                                                           +matvec33(dE_dqis[:,:,idof],skewmul(Nldqldqi[3:,jdof,p],ce1[:,m,0])+skewmul(Nldqldqi[3:,idof,p-1],ce1[:,m,1])) \
                                                                           +matvec33(dE_dqis[:,:,jdof],skewmul(Nldqldqi[3:,idof,p],ce1[:,m,0])+skewmul(Nldqldqi[3:,jdof,p-1],ce1[:,m,1])) \
                                                                                           +matvec33(E,skewmul(Nlddqldqidqj[3:,ij12sym[idof,jdof],p],ce1[:,m,0])+skewmul(Nlddqldqidqj[3:,ij12sym[idof,jdof],p-1],ce1[:,m,1]))
    # Return values
    return rf,e1,drf_dqi,de1_dqi,ddrf_dqidqj,dde1_dqidqj

## Compute element total force matrix
#
# Input:
#  ninterval,l,a,b,w
#
#
@cc.export('compute_element_total_force_matrix','(i4,f8,f8[:],f8[:],f8[:,:,:])')
def compute_element_total_force_matrix(ninterval,l,a,b,w):
    Tf=np.zeros((3,3*ninterval+3))
    # Loop over each integration interval of the element
    for m in range(ninterval):
        for r in range(2):
            tmp=(np.power(b[m],(r+1))-np.power(a[m],(r+1)))/(r+1.0)
            for j in range(2):
                for k in range(3):
                    Tf[:,3*(m+j)+k]+=0.5*l*tmp*w[m,r,j]*Imat[:,k]
    return Tf
## Compute element total moment matrix 
#
# Input:
#  ninterval,norder,l,a,b,w,ddrf_dqidqj,dde1_dqidqj
#
#
@cc.export('compute_element_total_moment_matrix','(i4,i4,f8,f8[:],f8[:],f8[:,:,:],f8[:,:,:],f8[:,:,:])')
def compute_element_total_moment_matrix(ninterval,norder,l,a,b,w,rf,e1):
    TMf=np.zeros((3,3,3*ninterval+3))
    TMm=np.zeros((3,3,3*ninterval+3))
    for m in range(ninterval):
        for r in range(2):
            for p in range(norder+4):
                tmp=(np.power(b[m],(p+r+1))-np.power(a[m],(p+r+1)))/(p+r+1.0)
                for j in range(2):
                    for k in range(3):
                        TMf[:,:,3*(m+j)+k]+=0.5*l*tmp*w[m,r,j]*ek_vT(rf[:,m,p],k)
                        TMm[:,:,3*(m+j)+k]+=0.5*l*tmp*w[m,r,j]*ek_vT(e1[:,m,p],k)
    return TMf,TMm
## Compute element generalized force matrix
#
# Input:
#  ninterval,norder,l,a,b,w,ddrf_dqidqj,dde1_dqidqj
#
#
@cc.export('compute_element_generalized_force_matrix','(i4,i4,f8,f8[:],f8[:],f8[:,:,:],f8[:,:,:,:],f8[:,:,:,:])')
def compute_element_generalized_force_matrix(ninterval,norder,l,a,b,w,drf_dqi,de1_dqi):
    TQf=np.zeros((12,3*ninterval+3))
    TQm=np.zeros((12,3*ninterval+3))
    for m in range(ninterval):
        for r in range(2):
            for p in range(norder+4):
                tmp=(np.power(b[m],(p+r+1))-np.power(a[m],(p+r+1)))/(p+r+1.0)
                for j in range(2):
                    for k in range(3):
                        for i in range(12):
                            TQf[i,3*(m+j)+k]+=0.5*l*tmp*w[m,r,j]*drf_dqi[k,i,m,p]
                            TQm[i,3*(m+j)+k]+=0.5*l*tmp*w[m,r,j]*de1_dqi[k,i,m,p]
    return TQf,TQm
## Compute element stiffness matrix from stationary generalized force 
#
# Input:
#  ninterval,norder,l,a,b,w,ddrf_dqidqj,dde1_dqidqj
#
#
@cc.export('compute_element_stiffness_generalized_force_matrix','(i4,i4,f8,f8[:],f8[:],f8[:,:,:],f8[:,:,:,:],f8[:,:,:,:])')
def compute_element_stiffness_generalized_force_matrix(ninterval,norder,l,a,b,w,ddrf_dqidqj,dde1_dqidqj):
    TKQf=np.zeros((78,3*ninterval+3))
    TKQm=np.zeros((78,3*ninterval+3))
    for m in range(ninterval):
        for r in range(2):
            for p in range(norder+4):
                tmp=(np.power(b[m],(p+r+1))-np.power(a[m],(p+r+1)))/(p+r+1.0)
                for j in range(2):
                    for k in range(3):
                        for idof in range(12):
                            for jdof in range(idof,12):
                                TKQf[ij12sym[idof,jdof],3*(m+j)+k]+=0.5*l*tmp*w[m,r,j]*ddrf_dqidqj[k,ij12sym[idof,jdof],m,p]
                                TKQm[ij12sym[idof,jdof],3*(m+j)+k]+=0.5*l*tmp*w[m,r,j]*dde1_dqidqj[k,ij12sym[idof,jdof],m,p]
    return TKQf,TKQm



#===============================================================================================
# Compilation 
#===============================================================================================
if __name__ == "__main__":
    cc.compile()