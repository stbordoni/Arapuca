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
import argparse

import sys, getopt


from ArapucaRoutineMainFunctions import *

def main(par1, par2):

    sample = 'mod304'    

    print('The starting element of the run list is ', par1)
    print('The last element of the run list is ', par2)

    #base_path = '/dune/data/users/sbordoni/Arapuca/mod304'
    base_path = '/Users/bordoni/protoDUNE/XeDoping/testfiles'
    file_path = os.path.join(base_path,'*.dat')

    file_name_list =  glob.glob(file_path) 
    print(file_name_list)

    file_name_dict = [parse_file_name(base_path, f) for f in file_name_list]

    runlist = [ r['run_number'] for r in file_name_dict ]
    runlist = list(set(runlist))

    print('Full list of runs to process : \n', runlist )

    for r in runlist[par1:par2]:
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


        #outputbasedir = '/dune/app/users/sbordoni/workarea/ArapucaAna/results'
        outputbasedir = './results/'
        outputdir = os.path.join(outputbasedir, sample)
        outputname_integral = os.path.join(outputdir, 'CalibratedIntegral_run'+r+'_newsig.csv')
        outputname_dataframe = os.path.join(outputdir, 'Waveforms_run'+r+'_newsig.csv' )
        outputname_dataframe_cal = os.path.join(outputdir, 'CalibWaveforms_run'+r+'_newsig.csv' )
    
        df_integral_calib.to_csv(outputname_integral)
        df_av_wf.to_csv(outputname_dataframe)
        df_av_wf_cal.to_csv(outputname_dataframe_cal)
    
        fig = df_av_wf.groupby(['Ch'])[wf].sum().T.plot().get_figure()
        outname_av_wf_raw_plt = os.path.join(outputdir,'AverageWfs_raw_run'+r+'_newsig.pdf') 
        fig.savefig(outname_av_wf_raw_plt)


        fig2 = df_av_wf_cal.groupby(['Ch'])[wf].sum().T.plot().get_figure()
        outname_av_wf_calib_plt = os.path.join(outputdir,'AverageWfs_calib_run'+r+'_newsig.pdf') 
        fig2.savefig(outname_av_wf_calib_plt)

        fig3 = df_av_wf_cal.groupby(['Run number'])[wf].sum().T.plot().get_figure()
        outname_final_av_wf = os.path.join(outputdir,'AvWf_calib_run'+r+'_newsig.pdf') 
        fig3.savefig(outname_final_av_wf)

        plt.close('all')

        print('processing files for run ', r, '  done! ')

    print('Loop over all run completed! ')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-f', '--first', type = int, required=False, default = 0, help='first element of the runlist to run the code')
    parser.add_argument('-l', '--last', type = int, required=False, default = -1, help='last element of the runlist to run the code')
    args = parser.parse_args()
    
    main(args.first, args.last) 
    