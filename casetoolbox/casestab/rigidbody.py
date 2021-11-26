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
from . import generic_model_components as gmc

## Rigid body substructures
#
#
class rigidbody_substructure:
    ## Initialization of body described by co-rotational equilibrium beam elements
    #
    #
    #
    def __init__(self,para):
        # Define substructure type
        self.type='rigid'
        # Number of DOFs
        self.ndofs=0
        # Get definition of nodes 
        self.rnode=para['nodes']
        # Blade number to which the forces relate
        self.aero_part=False
        self.iblade=-1
        # Initiate inertia states
        self.inertia=gmc.substructure_inertia(self.ndofs)

    ##  Routine that gives the current position and rotation matrix of a node
    def update_node_position_and_rotation(self,inode):
        r=self.rnode[:,inode]
        R=np.eye(3)
        return r,R
    ## Update substructure routine that does nothing
    def update_substructure(self):
        pass
    ## Update elastic forces and stiffness matrix does nothing
    def update_elastic_internal_forces_and_stiffness(self):
        pass
    ## Update inertia forces and matrices does nothing for now
    def update_inertia(self):
        pass