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
from numba import njit,float64
from numba.pycc import CC

cc = CC('model_precompiled_functions')
cc.verbose = True

## Inner product function
@njit(float64(float64[:],float64[:]))
def innerproduct(v1,v2):
    x=v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]
    return x
## Inner matrix product function
@njit(float64(float64[:,:],float64[:,:]))
def inner_matrix_product(A1,A2):
    x=0.0
    for i in range(3):
        for j in range(3):
            x+=A1[i,j]*A2[i,j]
    return x
## Centrifugal forces and stiffness matrix
#
# Inputs:
#  ndofs             int32
#  jcol_nonzero_irow int32[:,:]
#  ijsym             int32[:,:]
#  m_drcg_dqi        float64[:,:]
#  m_ddrcg_dqidqj    float64[:,:]
#  Abase_i           float64[:,:,:]
#  Abase1_ij         float64[:,:,:]
#  Abase2_ij         float64[:,:,:]
#  R0Tddr0           float64[:]
#  R0TddR0           float64[:,:]
#  R0TdR0            float64[:,:]
#
# Outputs
#  Fc          float64[:]
#  Gc          float64[:,:]
#  Kc          float64[:,:]
#  
#
@cc.export('compute_local_centrifugal_forces_and_matrix','(i4,i4[:,:],i4[:,:],f8[:,:],f8[:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:],f8[:,:],f8[:,:])')
def compute_local_centrifugal_forces_and_matrix(ndofs,jcol_nonzero_irow,ijsym,m_drcg_dqi,m_ddrcg_dqidqj,Abase_i,Abase1_ij,Abase2_ij,R0Tddr0,R0TddR0,R0TdR0):
    ddR0TR0=R0TddR0.T
    Fc=np.zeros(ndofs)
    Gc=np.zeros((ndofs,ndofs))
    Kc=np.zeros((ndofs,ndofs))
    for i in range(ndofs):
        Fc[i]= innerproduct(R0Tddr0,m_drcg_dqi[:,i]) \
              +inner_matrix_product(ddR0TR0,Abase_i[:,:,i])
        for j in range(jcol_nonzero_irow[0,i],jcol_nonzero_irow[1,i]):
            # Upper triangle
            Gc[i,j]= inner_matrix_product(R0TdR0,Abase1_ij[:,:,ijsym[i,j]])
            Kc[i,j]= innerproduct(R0Tddr0,m_ddrcg_dqidqj[:,ijsym[i,j]]) \
                    +inner_matrix_product(R0TddR0,Abase1_ij[:,:,ijsym[i,j]]) \
                    +inner_matrix_product(ddR0TR0,Abase2_ij[:,:,ijsym[i,j]])
            # Lower triangle
            if j!=i:
                Gc[i,j]= inner_matrix_product(R0TdR0,Abase1_ij[:,:,ijsym[i,j]].T)
                Kc[j,i]= innerproduct(R0Tddr0,m_ddrcg_dqidqj[:,ijsym[i,j]]) \
                        +inner_matrix_product(R0TddR0,Abase1_ij[:,:,ijsym[i,j]].T) \
                        +inner_matrix_product(ddR0TR0,Abase2_ij[:,:,ijsym[i,j]])

    return Fc,Gc,Kc

#===============================================================================================
# Compilation 
#===============================================================================================
if __name__ == "__main__":
    cc.compile()