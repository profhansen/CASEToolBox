from casetoolbox.casestab import casestab
import os
import numpy as np
# Get current work directory
work_folder = os.getcwd()
# Setup rotor models for each operational point
rotor_models = casestab.rotor_models(os.path.join(work_folder,'DTU10MW_6x6.json'))
# Tune pitch curve to different power setting and thrust limit
Prated = 9.0e6
Tlimit = 1.6e6
StallMargin = np.array([[24.1,10.0],
                        [30.1,8.0],
                        [36.0,4.0]])
reltol=0.01
max_pitch_increment=4.0
min_CP_gradient=0.01
Nmaxiter=3
prefix = 'test_stall_margin'
plot_flag=False
rotor_models.tune_pitch_curve(Prated,Tlimit,StallMargin,reltol,max_pitch_increment,min_CP_gradient,Nmaxiter,prefix,plot_flag)

