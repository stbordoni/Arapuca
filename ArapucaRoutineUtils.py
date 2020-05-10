import numpy as np
import pandas as pd 
import os

from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.signal import find_peaks

import matplotlib.pyplot as plt

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
        #_df = has_signal(_df)
        _df = has_signal_new(_df)
        _df = compute_singlepe(_df)
        _df = select_singlepe(_df)
        _df = tagGoodwf(_df)

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
    df = df.copy()
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
    peaks, properties = find_peaks(x, height=[15,2000], width=10)
    return (len(peaks) > 0)

def has_signal(df):
    df = df.copy()
    df['hasSignal'] = df.apply(lambda x: has_peak(x[wf]), axis=1)
    
    return df


def has_signal_new(df):
    df = df.copy()
    
    df_sig = df.apply(lambda x: find_signal(x, wf), axis=1)    
    df = pd.concat([df, df_sig], axis =1)

    return df


def find_signal(x, myrange):
    

    x = x[min(myrange) : max(myrange)]

    peaks, properties = find_peaks(x, height=[15,2000], width=10)
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

    return pd.Series([(npeaks>0), height, area], index=['hasSignal', 'signal height', 'signal area'])



def find_singlePE(x, myrange):

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
    df = df.copy()

    df_pe = df.apply(lambda x: find_singlePE(x, pe), axis=1)    
    df = pd.concat([df, df_pe], axis =1)
    
    return df



def select_singlepe(df):
    df = df.copy()

    X_pe=df.loc[(df['Saturated'] == False ) & 
                (df['hasSignal'] == True ) &
                (df['pe height']>0)&
                (df['pe width']>0),['pe height', 'pe width']].values

    mu_pe, cov_pe = estimate_gaus_param(X_pe,True)

    df['spe 1sig'] = df.apply(lambda x: select_wf(x[['pe height','pe width']], mu_pe, cov_pe, 1), axis=1)
    df['spe 2sig'] = df.apply(lambda x: select_wf(x[['pe height','pe width']], mu_pe, cov_pe, 2), axis=1)

    return df


def do_average_wf(dflist):
    df_av_wf = []
    
    for df in dflist:

        #df_tmp = (df.loc[ (df['Saturated'] == False ) & (df['hasSignal'] == True )].groupby(['Run number', 'Ch'])[rowin].mean() )
        df_tmp = (df.loc[ (df['Saturated'] == False ) & (df['isGoodwf'] == True )].groupby(['Run number', 'Ch'])[rowin].mean() )
        
        df_av_wf.append(df_tmp)
       
   
    return pd.concat(df_av_wf)




def do_average_singlepe(dflist):
    df_av_spe = []
    
    for df in dflist:

        df_tmp = df.loc[ (df['n pe'] >0) ].groupby(['Run number', 'Ch'])['pe area'].mean() 
        
        df_av_spe.append(df_tmp)
       
   
    return pd.concat(df_av_spe)



def do_average_singlepe_new(dflist):
    df_av_spe = []
    
    for df in dflist:

        area = df.loc[(df['spe 1sig']==True) ].groupby(['Run number', 'Ch'])['pe area'].mean()
        std  = df.loc[(df['spe 1sig']==True) ].groupby(['Run number', 'Ch'])['pe area'].std()
        
        df_tmp=pd.merge(left = area, right=std, on=['Run number', 'Ch'],suffixes = ('_mean', '_std') )
        df_av_spe.append(df_tmp)

    return pd.concat(df_av_spe)


def calibrate_av_wf(df):
    df = df.copy()
    df[rowin] = df[rowin].divide(df['f_cal'], axis=0)
    
    return df


def estimate_gaus_param(X, multivar=False):
    mean = np.mean(X, axis=0)
    
    if multivar:
        cov = 1/float(len(X)) * np.dot( (X - mean).T , X-mean)
    else:
        cov = np.diag(np.var(X, axis=0))
    return mean,cov



def plot_contours(X, mean, cov, stepx1, stepx2):
    
    X1range = np.arange(np.min(X[:,0]),np.max(X[:,0]),stepx1)
    X2range = np.arange(np.min(X[:,1]),np.max(X[:,1]),stepx2)

    X1mesh, X2mesh = np.meshgrid(X1range, X2range)
    
    coord_list = [ np.array([X0,X1]) for X0, X1 in zip(np.ravel(X1mesh), np.ravel(X2mesh)) ]
    Z = multivariate_normal.pdf( coord_list , mean=mean, cov=cov)
    Z = Z.reshape(X1mesh.shape)
    
    #cont_levels = [10**exp for exp in range(-50,0,3)]
    sigma = np.sqrt(np.diag(cov))
    nsigma = np.array([mean+n*sigma for n in range(1,3)[::-1]])
    
    cont_levels = multivariate_normal.pdf( nsigma , mean=mean, cov=cov)    
    cs = plt.contour(X1mesh, X2mesh, Z, 10, levels=cont_levels, colors=('blue', 'yellow', 'orange', 'red' ))
   
    #plt.clabel(cs, fmt='%g', inline=1, fontsize=10)

    fmt = {}
    strs = ['$1 \sigma$', '$2 \sigma$', '$3 \sigma$', '$4 \sigma$', '$5 \sigma$', '$6 \sigma$', '$7 \sigma$']
    for l, s in zip(cs.levels[::-1], strs):
        fmt[l] = s

    # Label every other level using strings
    plt.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=18)


def tagGoodwf(df):
    df = df.copy()

    pe_h_mu = df.loc[(df['spe 1sig']==True)]['pe height'].mean()
    pe_h_std  = df.loc[(df['spe 1sig']==True)]['pe height'].std()

    pe_a_mu = df.loc[(df['spe 1sig']==True)]['pe area'].mean()
    pe_a_std  = df.loc[(df['spe 1sig']==True)]['pe area'].std()


    X=df.loc[(df['hasSignal']==True) 
            & (df['signal height'] > (3* pe_h_mu) )
            & (df['signal area'] > (pe_a_mu +3* pe_a_std)) ,['signal height', 'signal area']].values




    mu_s, cov_s = estimate_gaus_param(X,True)

    df['isGoodwf'] = df.apply(lambda x: select_wf(x[['signal height', 'signal area']], mu_s, cov_s, 1), axis=1)
    

    return df



def select_wf(xy, mean, cov, n_sigma):
 
    Z = multivariate_normal.pdf( xy , mean=mean, cov=cov)
    #print(Z)
    
    sigma = np.sqrt(np.diag(cov))
    limit = n_sigma * sigma     
        
    thrsld = multivariate_normal.pdf( limit, mean=mean, cov=cov)
    #print(thrsld)
    
    return Z > thrsld