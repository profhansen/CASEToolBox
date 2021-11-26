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
## @package wind_model
#  The wind field for a wind turbine:
#      wind: 
import numpy as np

## Class that gives the wind field to a wind turbine rotor in meteorological carthesian coordinates
#
#
#
#
#
#
class wind:
    def __init__(self,para):
        # Save parameters
        self.para = para.copy()
        # Save density
        self.rho = para['density']
        # Set wind field type
        if para['windtype'] == 'uniform':
            self.lookup = uniform_wind(para['umean'])


## Uniform wind field class
#
# Input: umean
#
class uniform_wind:
    def __init__(self,umean):
        self.umean=umean
    def uvw_at_xyzt(self,pos,t):
        w = np.zeros(3)
        w[1]=self.umean
        return w

