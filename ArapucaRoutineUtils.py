import numpy as np
import pandas as pd 
import os

from scipy.signal import find_peaks

ped   = list(range(100, 200))   # to estimate the pedestal
rowin = list(range(24, 2024))   # all readout window
wf    = list(range(700, 2000))  # waveform 
tail  = list(range(1700, 2000)) # to estimate the single p.e.
pe    = list(range(100, 800)) # to estimate the single p.e.

#
#
#
##
def defineRoI():
    
    print('Defined Region of Interest: ' )
    print('Pedestal region (ped)  : [', min(ped)  , ',' , max(ped)  , ']'  ) 
    print('Readout window (rowin) : [', min(rowin), ',' , max(rowin), ']'  ) 
    print('Waveform (wf)          : [', min(wf)   , ',' , max(wf)   , ']'  ) 
    print('single p.e (pe)        : [', min(pe)   , ',' , max(pe)   , ']'  ) 
    print('Waveform tail (tail)   : [', min(tail) , ',' , max(tail) , ']'  ) 

    return ped, rowin, wf, pe




#
# Routine to prepare datasets: check for possible saturation and compute, subtract pedestal 
##
def prepare_dataset(df_list):
    
    mydflist = [] 
    ich = 0
    
    for _df in df_list: 
       
        ich = ich+1 
        print('preparing dataframe for channel : ', ich)

        _df = define_channel(_df)
        _df = do_reindex(_df)
        _df = flag_saturated(_df, 16383)
        _df = compute_pedestal(_df)
        _df = subtract_pedestal(_df)
        _df = remove_noise(_df) 
        _df = has_signal(_df)
        _df = compute_singlepe(_df)

        mydflist.append(_df)
        

    print('done!')
        
        
    return mydflist



def define_channel(df):
    df = df.copy()
    df.insert(0, 'Ch', int(df.iloc[0,5]) )   # column 5 in each dataset is the channel number
    return df



#
#  reindex the dataframe to prepare the final concat
## 
def do_reindex(df):
    df=df.copy()
    df.insert(1, 'Evt number', df[2].astype(int) )  # --> it's the 2nd entry of the header 
    df.insert(1, 'Run number', df[0].astype(int) )  # --> it's the 2nd entry of the header 

    return df
    #return df.reset_index().rename(columns={"index": "evt number"})



#
# Flag the saturated waveform
###
def flag_saturated(df, val=10000):
    df=df.copy()
    #df['Saturated']=(df.iloc[:, rowin].max(axis=1) >= val)
    df['Saturated']=(df[rowin].max(axis=1) >= val)
    return df



def compute_pedestal(df):
    df=df.copy()
    df['Pedestal']=df[ped].sum(axis=1)/len(ped)
    return df


def subtract_pedestal(df):
    df.copy()
    df[rowin] = df[rowin].subtract(df['Pedestal'], axis=0)
    return df


#
#  Apply rolling mean to remove noise. A winodow of 30 seems to work fine for the Arapucas 304
##
def remove_noise(df):
    df=df.copy()
    df[rowin]= df[rowin].rolling(window=30,  axis=1).mean()
    return df


def has_peak(x):
    peaks, properties = find_peaks(x, height=[15,800], width=10)
    return (len(peaks) > 0)



def has_signal(df):
    df.copy()
    df['hasSignal'] = df.apply(lambda x: has_peak(x[wf]), axis=1)
    return df


def find_singlePE(x, myrange):

    #rowin = list(range(200,1500))
    #rowin = list(range(200, 800))

    x = x[min(myrange) : max(myrange)]
    
    #peaks, properties = find_peaks(x, height=[10,30], width=30, distance = 50)    
    peaks, properties = find_peaks(x, height=[5,20], prominence=[5,15], width=20, distance = 30)
    peaks = peaks+min(myrange)
    
    npeaks = len(peaks)  
    if (npeaks > 0):
        height = properties['peak_heights'][0]
        width  = properties['widths'][0]
        xlow   = int(properties['left_ips'][0])
        xhigh  = int(properties['right_ips'][0])
        area   = x[xlow :xhigh].sum() 
    else :
        height = 0
        width  = 0
        area   = 0
        
    return pd.Series([npeaks, height, width, area], index=['n pe', 'pe height', 'pe width', 'pe area'])



def compute_singlepe(df):
    df.copy()

    df_pe = df.apply(lambda x: find_singlePE(x, pe), axis=1)    
    df = pd.concat([df, df_pe], axis =1)
    
    return df

