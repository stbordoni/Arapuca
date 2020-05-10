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
    df_average_spe = do_average_singlepe_new(df_list_proc)
    df_average_spe.reset_index(inplace = True)

    df_average_spe['f_cal'] = df_average_spe['pe area_mean']/df_average_spe['pe area_mean'][0]    


    #combine datasets (done in two steps)
    df_av_wf = pd.merge(df_av_wf, df_average_spe, left_on=['Run number', 'Ch'], right_on=['Run number', 'Ch'])

    #produce a calibrated dataset 
    df_av_wf_cal = calibrate_av_wf(df_av_wf)
    
    df_integral_calib = df_av_wf_cal[wf].sum(axis=1).to_frame().rename(columns={0:'Integral'})
    df_integral_calib = pd.concat([df_av_wf_cal[['Run number', 'Ch']], df_integral_calib], axis=1)

    df_integral_calib = pd.merge(df_integral_calib, df_average_spe, left_on=['Run number', 'Ch'], right_on=['Run number', 'Ch'])
    df_integral_calib = pd.concat([df_integral_calib, df_av_wf_cal['n good evts']], axis=1)


        
    outputname_integral = 'CalibratedIntegral_run'+r+'_newsig.csv'
    outputname_dataframe = 'Waveforms_run'+r+'_newsig.csv' 
    outputname_dataframe_cal = 'CalibWaveforms_run'+r+'_newsig.csv' 
    
    df_integral_calib.to_csv(outputname_integral)
    df_av_wf.to_csv(outputname_dataframe)
    df_av_wf_cal.to_csv(outputname_dataframe_cal)
    
    fig = df_av_wf.groupby(['Ch'])[wf].sum().T.plot().get_figure()
    fig.savefig('AverageWfs_raw_run'+r+'_newsig.pdf')


    fig2 = df_av_wf_cal.groupby(['Ch'])[wf].sum().T.plot().get_figure()
    fig2.savefig('AverageWfs_calib_run'+r+'_newsig.pdf')

    fig3 = df_av_wf_cal.groupby(['Run number'])[wf].sum().T.plot().get_figure()
    fig3.savefig('AvWf_calib_run'+r+'_newsig.pdf')

    plt.close('all')

    print('processing files for run ', r, '  done! ')

print('Loop over all run completed! ')
