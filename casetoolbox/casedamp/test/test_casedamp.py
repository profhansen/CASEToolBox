import numpy as np
from casetoolbox.casedamp import casedamp

# Main for testing
fn = 'DTU_10MW_RWT_pc.dat'
itype='pchip'
iset = 0
iairfoil = 1
naoas=721   
npsis=361
aoas = np.linspace(-180.0,180.0,naoas)
psis = np.linspace(-90.0,90.0,npsis)
beta=0.0
gama=0.0
phi=90.0
k=0.1
casedamp.casedamp(fn,itype,iset,iairfoil,aoas,psis,beta,gama,phi,k)