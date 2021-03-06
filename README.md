# CASEToolBox

CASEToolBox is a software package for performing Computational Aero-Servo-Elastic analysis of wind turbines. The code is still under development. The package has the tool CASEStab for power and stability analysis and the tool CASEDamp for analysing the aerodynamic damping of 2D airfoil translation and twist. Currently (February 2022), CASEStab can only be used for computing the stationary steady state of a wind turbine rotor with identical blades in a uniform flow without gravity, and for computing the structural blade modal frequencies and mode shapes. See a [presentation](./casetoolbox/casestab/docs/CASEStab.pdf) and an unfinished draft of a [Theory manual](./casetoolbox/casestab/docs/theory_manual.pdf).

## Installation

To install CASEToolBox into your Python environment run this command in the base folder with the setup.py file
```
pip install -e .
```
CASEStab is currently (November 2021) using the following packages [numpy](https://github.com/numpy/numpy), [numba](https://github.com/numba/numba), [scipy](https://github.com/scipy/scipy), and [matplotlib](https://github.com/matplotlib/matplotlib).

## Precompilation

CASEStab currently contains two modules that you should precompile to get the speed-up benefits of the numba CC precompilation. The precompilation is done by running the following commands in the folder CASEStab:
```
python corotbeam_precompiled_functions.py
python model_precompiled_functions.py
```
which create two associated pyd-files.

CASEDamp contains a single module that needs CC numba precompiling. 

## Running tests

The plots shown in the [presentation](./casetoolbox/casestab/docs/CASEStab.pdf) and many more plots can be created when you run the scripts in the test folder. Not all functionalities are currently (November 2021) tested with the uploaded tests.
