from casetoolbox.casestab import casestab
import os
import numpy as np
# Get current work directory
work_folder = os.getcwd()
# Setup rotor models for each operational point
rotor_models = casestab.rotor_models(os.path.join(work_folder,'DTU10MW_6x6.json'))
# Tune pitch curve to different power setting and thrust limit
Prated = 12.0e6
Tlimit = 1.4e6
StallMargin = np.array([])
reltol=0.01
max_pitch_increment=4.0
min_CP_gradient=0.01
Nmaxiter=3
prefix = 'test_higher_power'
plot_flag=False
rotor_models.tune_pitch_curve(Prated,Tlimit,StallMargin,reltol,max_pitch_increment,min_CP_gradient,Nmaxiter,prefix,plot_flag)

