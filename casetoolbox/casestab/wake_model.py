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
## @package wake_model
#  The wake model for the induction on a wind turbine rotor:
#      wake: 
import numpy as np
from . import math_functions as mf

## Class that gives the induced velocities in the polar coordinates of rotor in its initial position
#
#
#
#
#
#
class wake:
    def __init__(self,para,wake_radii):
        # Save parameters
        self.para = para.copy()
        # Define the model type
        if para['type'] == 'none':
            self.model = no_induction(para,wake_radii)
        if para['type'] == 'axissym':
            self.model = axissym_induction(para,wake_radii)
        




                # # Compute tangent vector
                # tvec=mf.crossproduct(self.substructures[isubs].subs.acp_motion_state[iacp].rtp,self.nvec)
                # # Get current induced velocities       
                # vi2d=self.wake.model.vi[:,iacp]
                # vi=vi2d[0]*self.nvec+vi2d[1]*tvec




## Void wake induction class
#
# Input: CT vector 
#
class no_induction:
    def __init__(self,para,radii):
        self.a      = np.zeros(len(radii))
        self.ap     = np.zeros(len(radii))
        self.vi     = np.zeros((2,len(radii)))
        self.dvidt  = np.zeros((2,len(radii)))
        self.radii = radii.copy()
        self.tip_correction = para['tip_correction']
        self.nblades = para['number_of_blades']
        self.R = self.radii[-1]+0.001
        self.a_of_CT=[]
        self.CT  = np.zeros(len(radii))
        self.CQ  = np.zeros(len(radii))
        self.sinphi = np.zeros(len(radii))
        self.local_TSR = np.zeros(len(radii))
        self.ftip = np.zeros(len(radii))
        
      
    class momentum_balance_point:
        def __init__(self,w,wn,R,r,R0,R0Tdr0,R0TdR0,rtp,Ec,nvec,tvec,omega,Nb,sigma,aero_point,a_of_CT=[]):
            self.local_TSR = r*omega/wn
            self.aero_point = aero_point
            E = R0@Ec
            reT = rtp.reshape(3,1)@Ec[:,0].reshape(1,3)
            self.vx0 = mf.innerproduct(E[:,0],w)-mf.innerproduct(Ec[:,0],R0Tdr0)+mf.inner_matrix_product(R0TdR0,reT)
            self.vx1 = 0.0
            self.vx2 = 0.0
            reT = rtp.reshape(3,1)@Ec[:,1].reshape(1,3)
            self.vy0 = mf.innerproduct(E[:,1],w)-mf.innerproduct(Ec[:,1],R0Tdr0)+mf.inner_matrix_product(R0TdR0,reT)
            self.vy1 = 0.0
            self.vy2 = 0.0
            self.aTx = sigma/wn**2*mf.innerproduct(nvec,E[:,0])
            self.aTy = sigma/wn**2*mf.innerproduct(nvec,E[:,1])
            self.aQx = sigma/wn**2*mf.innerproduct(tvec,E[:,0])
            self.aQy = sigma/wn**2*mf.innerproduct(tvec,E[:,1])
            self.CT = 0.0
            self.CQ = 0.0
            self.sinphi = 1.0
            self.ftip = 1.0
            # Velocities
            vx=self.vx0
            vy=self.vy0
            self.aero_point.vrel[0]=vx 
            self.aero_point.vrel[1]=vy
            # Relative speed
            self.aero_point.urel=np.sqrt(vx**2+vy**2)
            # AoA
            self.aero_point.aoa_tp=np.arctan2(vy,-vx)
            self.aero_point.aoa_cp=self.aero_point.aoa_tp
            # Lift and drag coefficients
            self.aero_point.CL = self.aero_point.cl.fcn(self.aero_point.aoa_cp)
            self.aero_point.CD = self.aero_point.cd.fcn(self.aero_point.aoa_cp)
            self.aero_point.CM = self.aero_point.cm.fcn(self.aero_point.aoa_cp)
            # Thrust and torque force coefficients
            self.CT = self.aero_point.urel*(self.aTx*self.aero_point.CD*vx+self.aTy*self.aero_point.CD*vy+self.aTx*self.aero_point.CL*vy-self.aTy*self.aero_point.CL*vx)
            self.CQ = self.aero_point.urel*(self.aQx*self.aero_point.CD*vx+self.aQy*self.aero_point.CD*vy+self.aQx*self.aero_point.CL*vy-self.aQy*self.aero_point.CL*vx)


        def f(self,x):
            values = np.zeros(2)
            return values
        def fprime(self,x):
            values = np.zeros((2,2))
            return values



## Axis-symmetric wake induction class
#
# Input: CT vector 
#
class axissym_induction:
    def __init__(self,para,radii):
        self.a      = np.zeros(len(radii))
        self.ap     = np.zeros(len(radii))
        self.vi     = np.zeros((2,len(radii)))
        self.dvidt  = np.zeros((2,len(radii)))
        self.radii = radii.copy()
        self.tip_correction = para['tip_correction']
        self.nblades = para['number_of_blades']
        self.R = self.radii[-1]+0.001
        if para['a_of_CT_model'] == 'HAWC2':
            self.a_of_CT=HAWC2_a_of_CT()
        self.CT  = np.zeros(len(radii))
        self.CQ  = np.zeros(len(radii))
        self.sinphi = np.zeros(len(radii))
        self.local_TSR = np.zeros(len(radii))
        self.ftip = np.zeros(len(radii))
        


        
    class momentum_balance_point:
        def __init__(self,w,wn,R,r,R0,R0Tdr0,R0TdR0,rtp,Ec,nvec,tvec,omega,Nb,sigma,aero_point,a_of_CT):
            self.local_TSR = r*omega/wn
            self.aero_point = aero_point
            E = R0@Ec
            reT1 = rtp.reshape(3,1)@Ec[:,0].reshape(1,3)
            self.vx0 = mf.innerproduct(E[:,0],w)-mf.innerproduct(Ec[:,0],R0Tdr0)+mf.inner_matrix_product(R0TdR0,reT1)
            self.vx1 =      -wn*mf.innerproduct(E[:,0],nvec)
            self.vx2 = -r*omega*mf.innerproduct(E[:,0],tvec)
            reT2 = rtp.reshape(3,1)@Ec[:,1].reshape(1,3)
            self.vy0 = mf.innerproduct(E[:,1],w)-mf.innerproduct(Ec[:,1],R0Tdr0)+mf.inner_matrix_product(R0TdR0,reT2)
            self.vy1 =      -wn*mf.innerproduct(E[:,1],nvec)
            self.vy2 = -r*omega*mf.innerproduct(E[:,1],tvec)
            self.aTx = sigma/wn**2*mf.innerproduct(nvec,E[:,0])
            self.aTy = sigma/wn**2*mf.innerproduct(nvec,E[:,1])
            self.aQx = sigma/wn**2*mf.innerproduct(tvec,E[:,0])
            self.aQy = sigma/wn**2*mf.innerproduct(tvec,E[:,1])

            self.beta = -0.5*Nb*(R-r)/r
 
            self.a_of_CT = a_of_CT
            self.ap_of_CQ_a=HAWC2_ap_of_CQ(self.local_TSR)
                       
            self.CT = 0.0
            self.CQ = 0.0
            self.sinphi = 1.0
            self.ftip = 1.0

            
        def f(self,x):
            a =x[0]
            ap=x[1]
            values = np.zeros(2)
            # Velocities
            vx=self.vx0+self.vx1*a+self.vx2*ap
            vy=self.vy0+self.vy1*a+self.vy2*ap
            self.aero_point.vrel[0]=vx 
            self.aero_point.vrel[1]=vy
            # Relative speed
            self.aero_point.urel=np.sqrt(vx**2+vy**2)
            # AoA
            self.aero_point.aoa_tp=np.arctan2(vy,-vx)
            self.aero_point.aoa_cp=self.aero_point.aoa_tp
            # Lift and drag coefficients
            self.aero_point.CL = self.aero_point.cl.fcn(self.aero_point.aoa_cp)
            self.aero_point.CD = self.aero_point.cd.fcn(self.aero_point.aoa_cp)
            # Thrust and torque force coefficients
            self.CT = self.aero_point.urel*(self.aTx*self.aero_point.CD*vx+self.aTy*self.aero_point.CD*vy+self.aTx*self.aero_point.CL*vy-self.aTy*self.aero_point.CL*vx)
            self.CQ = self.aero_point.urel*(self.aQx*self.aero_point.CD*vx+self.aQy*self.aero_point.CD*vy+self.aQx*self.aero_point.CL*vy-self.aQy*self.aero_point.CL*vx)
            # Tip correction 
            t1 = np.abs(1.0 - a)
            t2 = t1**2
            t3 = self.local_TSR**2
            t5 = (1 + ap)**2
            t8 = np.sqrt(t5 * t3 + t2)
            self.sinphi = 1.0 / t8 * t1
            self.ftip = 2.0/np.pi*np.arccos(np.exp(self.beta/self.sinphi))
            # Balance equations
            values[0] = a -self.a_of_CT.fcn(self.CT/self.ftip)
            values[1] = ap-self.ap_of_CQ_a.fcn(self.CQ,a)
            # Return values
            return values
        def fprime(self,x):
            a =x[0]
            ap=x[1]
            values = np.zeros((2,2))
            # Velocities
            vx=self.vx0+self.vx1*a+self.vx2*ap
            vy=self.vy0+self.vy1*a+self.vy2*ap
            self.aero_point.vrel[0]=vx 
            self.aero_point.vrel[1]=vy
            # Relative speed
            self.aero_point.urel=np.sqrt(vx**2+vy**2)
            durel_da  = (vx * self.vx1 + vy * self.vy1)/self.aero_point.urel
            durel_dap = (vx * self.vx2 + vy * self.vy2)/self.aero_point.urel
            # AoA
            self.aero_point.aoa_tp=np.arctan2(vy,-vx)
            self.aero_point.aoa_cp=self.aero_point.aoa_tp
            t3 = vy * vx
            t1 = vy ** 2 * vx ** 2 + 1.0
            daoa_da  = vx * (t3 * self.vx1 - self.vy1) / t1
            daoa_dap = vx * (t3 * self.vx2 - self.vy2) / t1
            # Lift and drag coefficients
            self.aero_point.CL = self.aero_point.cl.fcn(self.aero_point.aoa_cp)
            self.aero_point.CD = self.aero_point.cd.fcn(self.aero_point.aoa_cp)
            CLp = self.aero_point.cl.der(self.aero_point.aoa_cp)
            CDp = self.aero_point.cd.der(self.aero_point.aoa_cp)
            # Thrust and torque force coefficients
            t1 = self.aTx * vx + self.aTy * vy
            t2 = self.aTx * vy - self.aTy * vx
            t3 = CDp * t1 + CLp * t2
            self.CT = self.aero_point.urel*((self.aero_point.CD*vx+self.aero_point.CL*vy)*self.aTx+(self.aero_point.CD*vy-self.aero_point.CL*vx)*self.aTy)
            dCT_da  =(( self.vx1*self.aTx+self.vy1*self.aTy)*self.aero_point.urel+durel_da*t1)*self.aero_point.CD\
                    -((-self.vy1*self.aTx+self.vx1*self.aTy)*self.aero_point.urel-durel_da*t2)*self.aero_point.CL+self.aero_point.urel*daoa_da*t3
            dCT_dap =( t1*durel_dap+( self.vx2*self.aTx+self.vy2*self.aTy)*self.aero_point.urel)*self.aero_point.CD\
                    -(-t2*durel_dap+(-self.vy2*self.aTx+self.vx2*self.aTy)*self.aero_point.urel)*self.aero_point.CL+self.aero_point.urel*daoa_dap*t3
            t1 = self.aQx * vx + self.aQy * vy
            t2 = self.aQx * vy - self.aQy * vx
            t3 = CDp * t1 + CLp * t2
            self.CQ = self.aero_point.urel*(self.aero_point.CD*t1+self.aero_point.CL*t2)
            dCQ_da  = ((self.vx1*self.aQx+self.vy1*self.aQy)*self.aero_point.urel+durel_da*t1)*self.aero_point.CD \
                    -((-self.vy1*self.aQx+self.vx1*self.aQy)*self.aero_point.urel-durel_da*t2)*self.aero_point.CL+self.aero_point.urel*daoa_da*t3
            dCQ_dap = (t1*durel_dap+( self.vx2*self.aQx+self.vy2*self.aQy)*self.aero_point.urel)*self.aero_point.CD \
                    -(-t2*durel_dap+(-self.vy2*self.aQx+self.vx2*self.aQy)*self.aero_point.urel)*self.aero_point.CL+self.aero_point.urel*daoa_dap*t3
            # Tip correction 
            t1 = np.abs(1.0 - a)
            t2 = 1.0 + ap
            t3 = self.local_TSR ** 2
            t4 = t3 * t2 ** 2
            t5 = t1 ** 2 + t4
            t7 = 1.0/np.sqrt(t5)
            t6 = t7 ** 3
            sinphi = t1 * t7
            if a < 0.0:
                dsinphi_da  =  t4 * t6
            else:
                dsinphi_da  = -t4 * t6
            dsinphi_dap = -t1 * t3 * t2 * t6
            t1 = 1.0 / sinphi
            t2 = self.beta * t1
            t3 = np.exp(t2)
            t2 = 1 - np.exp(2 * t2)
            t2 = 1.0 / np.sqrt(t2)
            t4 = 1.0 / np.pi
            t1 = t1 ** 2
            t5 = 2 * self.beta
            ftip = 2 * t4 * np.arccos(t3)
            dftip_da  = t5 * dsinphi_da  * t3 * t2 * t4 * t1
            dftip_dap = t5 * dsinphi_dap * t3 * t2 * t4 * t1
            # Gradients of induction functions
            da_dCT         = self.a_of_CT.der(self.CT/self.ftip)
            dap_dCQ,dap_da = self.ap_of_CQ_a.der(self.CQ,a)
            # Balance equations
            values[0,0] = 1.0 - da_dCT*(dCT_da*ftip -self.CT*dftip_da )/ftip**2
            values[0,1] =     - da_dCT*(dCT_dap*ftip-self.CT*dftip_dap)/ftip**2
            values[1,0] =     - dap_dCQ*dCQ_da - dap_da
            values[1,1] = 1.0 - dap_dCQ*dCQ_dap
            # Return values
            return values



## Class with the piecewise polynomial fit to the local relationship 
#  between axial induction factor and thrust coefficient described in HAWC2
class HAWC2_a_of_CT():
    def __init__(self):
        self.k1 = 0.2460
        self.k2 = 0.0586
        self.k3 = 0.0883
        self.CT1 = -2.5
        self.CT2 =  2.5
        self.a1 = self.k3*self.CT1**3+self.k2*self.CT1**2+self.k1*self.CT1
        self.a2 = self.k3*self.CT2**3+self.k2*self.CT2**2+self.k1*self.CT2
        self.aslope1 = 3.0*self.k3*self.CT1**2+2.0*self.k2*self.CT1+self.k1
        self.aslope2 = 3.0*self.k3*self.CT2**2+2.0*self.k2*self.CT2+self.k1
    def inner_fcn(self,CT):
        a = self.k3*CT**3+self.k2*CT**2+self.k1*CT
        return a
    def inner_der(self,CT):
        a = 3.0*self.k3*CT**2+2.0*self.k2*CT+self.k1
        return a
    def fcn(self,CT):
        # Limit -4 < CT < 4 
        CT=np.min([np.abs(CT),4.0])*np.sign(CT)
        # Piece linear function
        if CT < self.CT1:
            a = self.a1+self.aslope1*(CT-self.CT1)
        elif CT > self.CT2:
            a = self.a2+self.aslope2*(CT-self.CT2)
        else:
            a = self.inner_fcn(CT)
        return a
    def der(self,CT):
        # Piece linear function
        if np.abs(CT) > 4.0:
            da_dCT = 0.0
        else:
            if CT < self.CT1:
                da_dCT = self.aslope1
            elif CT > self.CT2:
                da_dCT = self.aslope2
            else:
                da_dCT = self.inner_der(CT)
        return da_dCT
class HAWC2_ap_of_CQ():
    def __init__(self,local_TSR):
        self.rotating = np.abs(local_TSR) > 1.0e-5
        if self.rotating:
            self.factor = 0.25/local_TSR
    def fcn(self,CQ,a):
        if self.rotating:
            if a > 0.9:              
                ap = self.factor*CQ/0.1
            else:
                ap = self.factor*CQ/(1.0-a)
        else:
            ap = 0.0
        return ap
    def der(self,CQ,a):
        if self.rotating:
            if a > 0.9:              
                dap_dCQ = self.factor/0.1
                dap_da = 0.0
            else:
                dap_dCQ = self.factor/(1.0-a)
                dap_da  = self.factor*CQ/(1.0-a)**2
        else:
            dap_dCQ = 0.0
            dap_da = 0.0
        return dap_dCQ,dap_da


