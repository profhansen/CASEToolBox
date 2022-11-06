## Copyright 2021 Morten Hartvig Hansen
#
# This file is part of CASEToolBox/CASEDamp.

# CASEToolBox/CASEDamp is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# CASEToolBox/CASEDamp is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with CASEToolBox/CASEDamp.  If not, see <https://www.gnu.org/licenses/>.
#
#
import numpy as np
from numba.pycc import CC
cc = CC('casedamp_precompiled_functions')
cc.verbose = True
# Function for computing the damping coefficients
@cc.export('compute_damping_terms','(f8[:],f8[:],f8[:,:])')
def compute_damping_terms(aoas,psis,clcd_clpcdp):
    # Polars and their gradient
    cl  = clcd_clpcdp[:,0]
    cd  = clcd_clpcdp[:,1]
    clp = clcd_clpcdp[:,2]
    cdp = clcd_clpcdp[:,3]
    # Compute the damping coefficient
    npsis = len(psis)
    naoas = len(aoas)
    W_tran1 = np.zeros((naoas,npsis))
    W_tran2 = np.zeros((naoas,npsis))
    W_tors1 = np.zeros((naoas,npsis))
    W_tors2 = np.zeros((naoas,npsis))
    for j in range(npsis):
        psi=psis[j]
        for i in range(naoas):
            aoa=aoas[i]
            v = np.radians(psi+aoa)
            c2v = np.cos(2.0*v)
            s2v = np.sin(2.0*v)
            cv = np.cos(v)
            sv = np.sin(v)
            W_tran1[i,j] = 0.5*(cd[i]*(3.0+c2v)+clp[i]*(1.0-c2v)-(cl[i]+cdp[i])*s2v)
            W_tran2[i,j] = 0.5*(cd[i]*(3.0-c2v)+clp[i]*(1.0+c2v)+(cl[i]+cdp[i])*s2v)
            W_tors1[i,j] = 0.5*(cdp[i]*cv-clp[i]*sv)
            W_tors2[i,j] = 0.5*(cdp[i]*sv+clp[i]*cv)
    return W_tran1,W_tran2,W_tors1,W_tors2

# Function for computing the damping coefficients
@cc.export('compute_damping_eta','(f8,f8,f8,f8,f8[:,:],f8[:,:],f8[:,:],f8[:,:])')
def compute_damping_eta(ured,gama,beta,phi,W_tran1,W_tran2,W_tors1,W_tors2):
    eta = W_tran1 + beta**2*W_tran2 + ured*gama*np.sin(phi)*W_tors1 + ured*gama*beta*np.cos(phi)*W_tors2
    return eta

#===============================================================================================
# Compilation 
#===============================================================================================
if __name__ == "__main__":
    cc.compile()