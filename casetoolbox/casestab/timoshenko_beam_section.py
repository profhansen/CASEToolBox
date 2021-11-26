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
## @package timoshenko_beam_section
import numpy as np
## Transform a 6x6 displacement and rotation matrix 
def transform_reference_point_of_matrix(S,dx,dy,theta):
    ## Input:
    #    S:     Matrix to transform defined in system 2
    #    dx:    Position of origo of system 2 on the x-axis of system 1 
    #    dy:    Position of origo of system 2 on the y-axis of system 1 
    #    theta: Rotation angle of system 2 from system 1
    # Output: The rotated matrix defined in system 1
    #         
    # Transformation of the compliance into the reference frame
    c = np.cos(theta)
    s = np.sin(theta)
    T12 = np.array([[c,-s,0,0,0,       dy],[ s,c,0,0,0,      -dx],[0,0,1,dx*s-dy*c,dx*c+dy*s,0],[0,0,0,c,-s,0],[0,0,0, s,c,0],[0,0,0,0,0,1]])
    T21 = np.array([[c, s,0,0,0,dx*s-dy*c],[-s,c,0,0,0,dx*c+dy*s],[0,0,1,       dy,      -dx,0],[0,0,0,c, s,0],[0,0,0,-s,c,0],[0,0,0,0,0,1]])
    return T21@S@T12
## Translation of conventional Timoshenko cross-section properties to 6x6 stiffness matrix
def isotropic_to_6x6_compliance_matrix(rref,rea,rsc,theta,E,G,A,Ix,Iy,Iz,kx,ky):
    ## Input:
    #  rref:  2D position of the reference point in the cross-sectional reference frame
    #  rea:   2D position of the "elastic axis", the centroid of bending in the cross-sectional reference frame
    #  rsc:   2D position of the shear center in the cross-sectional reference frame
    #  theta: structural twist of the principal axes in the cross-sectional reference frame
    #  E:     average elastic Young's modulus of the cross-section
    #  G:     average shear modulus of the cross-section
    #  A:     area of cross-section
    #  Ix:    moment of inertia for bending about the principal x-axis
    #  Iy:    moment of inertia for bending about the principal y-axis
    #  Iz:    moment of inertia for torsion
    #  kx:    shear correction factor in the direction of the principal x-axis
    #  ky:    shear correction factor in the direction of the principal y-axis
    #
    #  Output:
    #  C: cross-sectional 6x6 compliance matrix that relates the generalized force vector 
    #     consisting of the two inplane shear forces, the axial normal force, 
    #     the two bending moments, and the twisting moment to the generalized strain vector
    #     consisting of the two inplane shear strains, the axial elongation strain, 
    #     the two out-of-plane curvatures, and the rate of torsion in the cross-section. 
    #     The forces and displacements are all defined in the reference point. 
    #
    # Distances from EA to SC in the coordinate system of the principal axes
    ex = (rsc[0]-rea[0])*np.cos(theta)+(rsc[1]-rea[1])*np.sin(theta)
    ey =-(rsc[0]-rea[0])*np.sin(theta)+(rsc[1]-rea[1])*np.cos(theta)
    # Stiffness matrix in the frame of the principal axes according to Hodges (2006), Nonlinear composite beam theory, see 
    S=np.zeros((6,6))
    S[0,0]=kx*G*A
    S[1,1]=ky*G*A
    S[2,2]=E*A
    S[3,3]=E*Ix
    S[4,4]=E*Iy
    S[5,5]=G*Iz+kx*G*A*ey**2+ky*G*A*ex**2
    S[0,5]=-kx*G*A*ey
    S[1,5]= ky*G*A*ex
    S[5,0]=-kx*G*A*ey
    S[5,1]= ky*G*A*ex
    # Distances from reference point (origin) to EA in the reference frame
    xref = rref[0] - rea[0]
    yref = rref[1] - rea[1]
    # Transformation of the compliance into the reference frame
    S = transform_reference_point_of_matrix(S,xref,yref,-theta) # minus 'theta' because the principal axes is "system 1" 
    # Compute compliance matrix as the inverse stiffness matrix
    C = np.linalg.inv(S)
    # Return the matrix
    return C
