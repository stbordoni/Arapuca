import numpy as np
import pandas as pd
import os
import re


from scipy.signal import find_peaks
from joblib import Parallel, delayed

from ArapucaRoutineUtils import *
#from RoutineDrawUtils import *


def parse_file_name( base_path, file_name):
    p = re.compile(os.path.join(base_path,     
    'run(?P<run_number>\d+)_evt(?P<run_part>\d+)_mod(?P<module>\d+)_ch(?P<channel>\d+).dat') )


    m = p.match(file_name)
    d = m.groupdict()
    d['file_name'] = file_name
    return d



def file_for_run_and_channel(dictt, run=1234567, ch=4):
    return [ f['file_name'] for f in dictt if f['run_number'] == str(run) and f['channel'] == str(ch)]


def file_for_channel(dictt, ch=4):
    return [ f['file_name'] for f in dictt if f['channel'] == str(ch)]


def file_for_run(dictt, run=123445):
    return [ f['file_name'] for f in dictt if f['run_number'] == str(run)]



def readfile_list(filename_list):       
    
    def read_single_file(f):
        #load the file as a np array
        data = np.loadtxt(f)
        #data = pd.read_fwf(f, header=None)
        nevts = int(len(data)/2024)
        #reshape the array as 10000 events and 2024 columns each
        data = data.reshape(nevts, 2024)
        #create a dataframe from the numpy array
        return pd.DataFrame(data, columns=range(0,2024))
    
    #data_list = [read_single_file(f) for f in filename_list]
    data_list = Parallel(n_jobs=-1)(delayed(read_single_file)(file) for file in filename_list)
    return pd.concat(data_list, axis=0)




#def create_dataset_list(file_name_dict, run):
#    
#    df_list = []
#    tmp_filelist = file_for_run(file_name_dict, run)
#    print('Run ', run , '  has ', len(tmp_filelist), 'files'  ) 
#    df_tmp = readfile_list(tmp_filelist)
#    df_list.append(df_tmp)
#    return df_list

def create_dataset_list(file_name_dict, run=1234567):
    
    df_list = []
    for c in range(0,12):
        tmp_filelist = file_for_run_and_channel(file_name_dict, run, c)
        
        print('Run ', run , '  has for channel ', c, '  ', len(tmp_filelist), 'files'  )
        df_tmp = readfile_list(tmp_filelist)
        df_list.append(df_tmp)

    print('loaded ', len(df_list), 'channels for this run' )    
    return df_list




def doPreProcessing(df_list):
    df_list.copy()
    
    # pre-processing
    df_list = prepare_dataset(df_list)
    
    print('preprocessing done!')

    return df_list




def doEventSelection(df_list, rundir, csvdir):

    df_list.copy()
    
    #create a unique data-set
    df_allch = pd.concat(df_list, axis=0)
    
    #sort values per event and channel
    df_allch.sort_values(by=['evt number', 'Ch'], inplace=True)
    
    #create a new column on the dataframe to check if the event is saturated (the event NOT the waveform)
    df_allch['isSaturatedEvent'] = (df_allch.groupby(['evt number'])['Saturated'].sum() != 0)
    
    # determine if an event has signal
    df_allch['EventhasSignal'] = (df_allch.groupby(['evt number'])['hasSignal'].sum() != 0)
    
    
    #flag events which are not saturated and have signals has 'Good Events'
    # these are the events on which we will perform the analysis
    df_allch['isGoodEvt'] = (df_allch['isSaturatedEvent'] == False) & (df_allch['EventhasSignal'] == True)
    
    #print how many events have been passing the Event Selection
    print('Percentage of events passing the selection', 
          df_allch.groupby(['evt number'])['isGoodEvt'].first().value_counts()[1] / df_list[0].shape[0] )
    
    
    #save the processed dataframe in csv file
    #df_allch_packed = pack_waveform(df_allch)

    #suffix = '.csv'
    #base_filename = 'df_allch_' + rundir+'_processed'
    #filename = os.path.join(csvdir, base_filename + suffix)
    #print(filename)
    
    #df_allch_packed.to_csv(filename)

    #print('Event selection : done!')
    return df_allch

