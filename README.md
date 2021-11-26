# CASEToolBox

CASEToolBox is a software package for performing Computational Aero-Servo-Elastic analysis of wind turbines. The code is still under development. The package currently only has the tool CASEStab for power and stability analysis. Currently (November 2021), CASEStab can only be used for computing the stationary steady state of a wind turbine rotor with identical blades in a uniform flow without gravity, and for computing the structural blade modal frequencies and mode shapes. See a [presentation](./CASEStab/docs/CASEStab.pdf) and an unfinished draft of a [Theory manual](./CASEStab/docs/theory_manual.pdf).

## Installation

To install CASEToolBox into your Python environment run this command in the base folder with the setup.py file
```
pip install -e .
```
CASEStab is currently (November 2021) using the following packages numpy, numba, scipy, and matplotlib.

## Precompilation

CASEStab currently contains two modules that you should precompile to get the speed-up benefits of the numba CC precompilation. The precompilation is done by running the following commands in the folder CASEStab:
```
python corotbeam_precompiled_functions.py
python model_precompiled_functions.py
```
which create two associated pyd-files.

## Running tests

The plots shown in the [presentation](./CASEStab/docs/CASEStab.pdf) and many more should be created when you run the scripts in the test folder. 
