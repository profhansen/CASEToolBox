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
#===============================================================================================
# Compilation 
#===============================================================================================
if __name__ == "__main__":
    cc.compile()