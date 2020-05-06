import numpy as np
import pandas as pd 
import scipy as sp
import scipy.io
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import os
import glob
import fnmatch
import re

from ArapucaRoutineMainFunctions import *

base_path = '/Users/bordoni/protoDUNE/XeDoping/testfiles'
file_path = os.path.join(base_path,'*.dat')

file_name_list =  glob.glob(file_path) 
print(file_name_list)

file_name_dict = [parse_file_name(base_path, f) for f in file_name_list]

runlist = [ r['run_number'] for r in file_name_dict ]
runlist = list(set(runlist))

print('Full list of runs to process : \n', runlist )

for r in runlist:
    print('loading files for run ', r)
    df_list = create_dataset_list(file_name_dict, r)

    df_list_proc = doPreProcessing(df_list)

    # compute average waveform (raw! )
    df_av_wf = do_average_wf(df_list_proc)


    # compute single pe for calibration
    df_average_spe = do_average_singlepe(df_list_proc).to_frame()
    df_average_spe.reset_index(inplace = True)
    tmp = df_average_spe.pivot(index='Run number', columns='Ch', values='pe area')
    


    df_calib = tmp.div(tmp[0], axis = 0)
    df_calib = df_calib.T.stack(level=0).to_frame(name='f_cal').reset_index()


    #combine datasets (done in two steps)
    df_av_wf = pd.merge(df_av_wf, df_average_spe, left_on=['Run number', 'Ch'], right_on=['Run number', 'Ch'])
    df_av_wf = pd.merge(df_av_wf, df_calib, left_on=['Run number', 'Ch'], right_on=['Run number', 'Ch'])

    #produce a calibrated dataset 
    df_av_wf_cal = calibrate_av_wf(df_av_wf)

    df_integral_calib = df_av_wf_cal.groupby(['Run number']).sum()
    df_integral_calib = tmp.sum(axis=1).to_frame().reset_index()
    
    print(df_integral_calib)
    
    outputname_integral = 'CalibratedIntegral_run'+r+'.csv'
    outputname_dataframe = 'Waveforms_run'+r+'.csv' 
    
    df_integral_calib.to_csv(outputname_integral)
    df_av_wf.to_csv(outputname_dataframe)
    #df_av_wf_cal.to_csv('./CalibratedWaveforms.csv')

    fig = df_av_wf.groupby(['Run number']).sum().T.plot().get_figure()
    fig.savefig('AverageWf_raw_run'+r+'.pdf')


    df_wf_calibrated = df_av_wf_cal.groupby(['Run number']).sum().T
    df_wf_calibrated[3:1300].plot().get_figure()
    fig.savefig('AverageWf_calib'+r+'.pdf')

    print('processing files for run ', r, '  done! ')

print('Loop over all run completed! ')
