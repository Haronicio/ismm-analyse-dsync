'''
    Welcome to analyse dsync tool's main script

    Bienvenue dans le script principale de analyse dsync tool

    Section

    - imports
    - build
    - debug plot
    - signal
    - stats
    - data preprocess
    - data process personnal
    - data process interpersonnal
    - main

    by Haron DAUVET-DIAKHATE 31 août 2024
'''






#                            __      ___      ___    _______       ______      _______    ___________    ________  
#                           |" \    |"  \    /"  |  |   __ "\     /    " \    /"      \  ("     _   ")  /"       ) 
#                           ||  |    \   \  //   |  (. |__) :)   // ____  \  |:        |  )__/  \\__/  (:   \___/  
#                           |:  |    /\\  \/.    |  |:  ____/   /  /    ) :) |_____/   )     \\_ /      \___  \    
#                           |.  |   |: \.        |  (|  /      (: (____/ //   //      /      |.  |       __/  \\   
#                           /\  |\  |.  \    /:  | /|__/ \      \        /   |:  __   \      \:  |      /" \   :)  
#                          (__\_|_) |___|\__/|___|(_______)      \"_____/    |__|  \___)      \__|     (_______/   
#                         
                                                                                  
# imports



import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler

import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_theme(style="darkgrid")
# sns.color_palette(palette="colorblind")

import scipy as sp
from scipy.stats import zscore,permutation_test,kendalltau
from scipy import signal
from scipy.signal import hilbert
# from scipy.ndimage import shift
from scipy.fft import fft,fftfreq
# from scipy.fftpack import *
from tsfresh import extract_features
import librosa
import librosa.display
import pycwt as wavelet #wavelet
# import pywt
# from dtw import *
from functools import partial
import itertools
from typing import Any, Dict, Optional, Tuple, Union
from typing_extensions import Self
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.ar_model import AutoReg
from collections import deque


from typing import Callable


import glob
import os
import sys
import pickle
import warnings

import gc
import re

from tqdm import tqdm
import ipywidgets as widgets 
from ipywidgets import interact, interactive, fixed, interact_manual
from IPython.display import Audio, display, clear_output

# from numba import jit,njit

def import_external():
    global MutualInformation,WindowMutualInformation,GC,SGC,Normalize

    sys.path.insert(0, '../libs/python/SyncPy/src/')   # To be able to import packages from parent directory
    sys.path.insert(0, '../libs/python/SyncPy/src/Methods')
    sys.path.insert(0, '../libs/python/SyncPy/src/Methods/utils')  

    from DataFrom2Persons.Univariate.Continuous.Linear import Coherence
    import DataFrom2Persons.Univariate.Continuous.Nonlinear.MutualInformation as MutualInformation
    import DataFrom2Persons.Univariate.Continuous.Nonlinear.WindowMutualInformation as WindowMutualInformation
    import DataFrom2Persons.Univariate.Continuous.Linear.GrangerCausality as GC
    import DataFrom2Persons.Univariate.Continuous.Linear.SpectralGrangerCausality as SGC

    import utils.Standardize
    from utils.ExtractSignal import ExtractSignalFromCSV
    from utils import PeakDetect,ConvertContinueToBinary,Normalize

import_external()



#                                                         .=-.-.                           
#                              _..---.    .--.-. .-.-.   /==/_ /    _.-.       _,..---._   
#                            .' .'.-. \  /==/ -|/=/  |            .-,.'|     /==/,   -  \  
#                           /==/- '=' /  |==| ,||=| -|  |==|  |  |==|, |     |==|   _   _\ 
#                           |==|-,   '   |==|- | =/  |  |==|- |  |==|- |     |==|  .=.   | 
#                           |==|  .=. \  |==|,  \/ - |  |==| ,|  |==|, |     |==|,|   | -| 
#                           /==/- '=' ,| |==|-   ,   /  |==|- |  |==|- `-._  |==|  '='   / 
#                          |==|   -   /  /==/ , _  .'   /==/. /  /==/ - , ,/ |==|-,   _`/  
#                          `-._`.___,'   `--`..---'     `--`-`   `--`-----'  `-.`.____.'   

# build


MAIN_DATA_FOLDER = "../data/restart"
CLICK_TEMPO_FOLDER = "data-2023-09-26"
MASK_ATTACK_FOLDER = "data-2023-09-27"
CHANGE_FOLDER = "data-2023-09-28"

# Logics for my project

def trials_folder_to_sig_df(trial_folder : str) -> pd.DataFrame:

    def extract_trial_riot(filename):
        # trial_AnySTR-riot trial and str are int
        match = re.match(r'(\d+)_.*-(\d+)\.txt', filename)
        if match:
            trial, riot = match.groups()
            return int(trial), int(riot)
        else:
            return None, None

    def parse_file(filepath):
        data = []
        with open(filepath, 'r') as file:
            for line in file:
                values = line.split(' ')
                data.append(values)
        return data

    def create_dataframe(data, trial, riot):
        columns = [
            't', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z',
            'mag_x', 'mag_y', 'mag_z', 'orientation_x', 'orientation_y', 'orientation_z',
            'accfilt_x', 'accfilt_y', 'accfilt_z', 'intensity'
        ]
        df = pd.DataFrame(data, columns=columns)
        df = df.apply(pd.to_numeric, errors='coerce')
        df['trial_n'] = trial
        df['riot_n'] = riot
        return df


    all_data = []
    for filename in os.listdir(trial_folder):
        if filename.endswith(".txt"):
            trial, riot = extract_trial_riot(filename)
            if trial is not None and riot is not None:
                filepath = os.path.join(trial_folder, filename)
                data = parse_file(filepath)
                df = create_dataframe(data, trial, riot)
                all_data.append(df)
    
    compiled_df = pd.concat(all_data, ignore_index=True)
    return compiled_df
def infos_folder_to_complete_sig_df(info_folder : str,df : pd.DataFrame) -> pd.DataFrame:

    
    musicians_info_path = os.path.join(info_folder, 'riots-musicians.txt')
    musician_info = {}
    musician_riots = {}

    trials_info_path = os.path.join(info_folder, 'trials-info.txt')
    trials_info = {}

    # Load musician and riot info
    with open(musicians_info_path, 'r') as file:
        for line in file:
            riot_number, musician, riot_type = line.strip().split()
            musician_info[int(riot_number)] = (musician, riot_type)
            
           # musician already in dict
            if musician not in musician_riots:
                musician_riots[musician] = (None, None)
            
            if riot_type == 'head':
                musician_riots[musician] = (int(riot_number), musician_riots[musician][1])
            elif riot_type in ['arm', 'leg']:
                if musician_riots[musician][1] is None:
                    musician_riots[musician] = (musician_riots[musician][0], int(riot_number)-1)

     # Load trials info
    with open(trials_info_path, 'r') as file:
        n = 1
        for line in file:
            factors = line.strip().split()
            trials_info[n] = (tuple(factors))
            n += 1

    # Extract musician and riot_type from musician_info based on riot_n
    df['musician'] = df['riot_n'].map(lambda x: musician_info[x][0] if x in musician_info else None) # type: ignore
    df['riot_type'] = df['riot_n'].map(lambda x: musician_info[x][1] if x in musician_info else None)# type: ignore

    # Extract factor1 and factor2 from trials_info based on trial_n
    df['factor1'] = df['trial_n'].map(lambda x: trials_info[x][0] if x in trials_info else None)# type: ignore
    df['factor2'] = df['trial_n'].map(lambda x: trials_info[x][1] if x in trials_info else None)# type: ignore

    # change have a therd info : timecode of change prompt
    if len(trials_info[1]) > 2:
        df['change'] = df['trial_n'].map(lambda x: trials_info[x][2] if x in trials_info else None)# type: ignore

    return df

def onsets_folder_to_onset_df(onsets_folder : str) -> pd.DataFrame:

    def onset_csvname2musician(csvname):
        if("Pno" in csvname or "piano" in csvname):
            return "Piano"
        if("Sax" in csvname or "sax" in csvname):
            return "Sax"
        if("Vlc" in csvname or "cello" in csvname):
            return "Cello"
        if("MD" in csvname or "MG" in csvname):
            return "Accordion"
        if("Clar" in csvname):
            return "Clarinet"
        if("Tom" in csvname or "peau" in csvname or "OH" in csvname or "Snare" in csvname or "Kick" in csvname or "drums" in csvname):
            return "Drum"
        return None
        
    def extract_trial_number(csvname):
        match = re.match(r'(\d+)_.*\.csv', csvname)
        if match:
            return int(match.group(1))
        return None
    
    all_data = []

    for filename in os.listdir(onsets_folder):
        if filename.endswith(".csv"):
            filepath = os.path.join(onsets_folder, filename)
            trial_n = extract_trial_number(filename)
            musician = onset_csvname2musician(filename)

            if trial_n is not None and musician != "Unknown":
                df = pd.read_csv(filepath)
                df.rename(columns={'Absolute Time (s)': 'onsets'}, inplace=True)
                df['trial_n'] = trial_n
                df['musician'] = musician
                all_data.append(df)
            else:
                print('error reading trial and musician',trial_n,musician)

    compiled_df = pd.concat(all_data, ignore_index=True)
    compiled_df['onsets'] *= 1000.0 #convert s to ms
    return compiled_df
def infos_folder_to_complete_onset_df(info_folder : str,df : pd.DataFrame) -> pd.DataFrame:

    trials_info_path = os.path.join(info_folder, 'trials-info.txt')
    trials_info = {}

     # Load trials info
    with open(trials_info_path, 'r') as file:
        n = 1
        for line in file:
            factors = line.strip().split()
            trials_info[n] = (tuple(factors))
            n += 1

    # Extract factor1 and factor2 from trials_info based on trial_n
    df['factor1'] = df['trial_n'].map(lambda x: trials_info[x][0] if x in trials_info else None)# type: ignore
    df['factor2'] = df['trial_n'].map(lambda x: trials_info[x][1] if x in trials_info else None)# type: ignore
    # change have a therd info : timecode of change prompt
    if len(trials_info[1]) > 2:
        df['change'] = df['trial_n'].map(lambda x: trials_info[x][2] if x in trials_info else None)# type: ignore

    return df

def trunc_dfs(dfs: Dict[str, pd.DataFrame], truncs: Dict[Any, Tuple[Optional[float], Optional[float]]] = {None:(None, None)},**kwargs : Any) -> Dict[str, pd.DataFrame]:
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # always return a copy
    def trunc_mask(df, trunc_col, indexer_col, indexer, trunc_start, trunc_end):
        # Apply truncation mask if start and/or end is specified based on trunc_col following indexer rule
        mask = (df[indexer_col] == indexer)
        
        if trunc_start is not None:
            mask &= (df[trunc_col].astype(float) >= trunc_start)
            
        if trunc_end is not None:
            mask &= (df[trunc_col].astype(float) <= trunc_end)
        
        return mask

    sig_trunc_mask = pd.Series([False] * len(dfs['sig']))
    onset_trunc_mask = pd.Series([False] * len(dfs['onset']))

    for indexer, trunc_values in truncs.items():
        trunc_start, trunc_end = trunc_values

        # Update the mask for signal and onset DataFrames
        sig_trunc_mask |= trunc_mask(dfs['sig'], 't', 'trial_n', indexer, trunc_start, trunc_end)
        onset_trunc_mask |= trunc_mask(dfs['onset'], 'onsets', 'trial_n', indexer, trunc_start, trunc_end)

    # Create masks for rows where indexer_col values are not in truncs keys
    sig_not_in_truncs = ~dfs['sig']['trial_n'].isin(truncs.keys())
    onset_not_in_truncs = ~dfs['onset']['trial_n'].isin(truncs.keys())

    # Combine the masks to include all relevant data
    sig_combined_mask = sig_trunc_mask | sig_not_in_truncs
    onset_combined_mask = onset_trunc_mask | onset_not_in_truncs

    # Apply the masks to create truncated DataFrames
    sig_trunc : pd.DataFrame = dfs['sig'][sig_combined_mask].copy().reset_index(drop=True) # type: ignore
    onset_trunc : pd.DataFrame = dfs['onset'][onset_combined_mask].copy().reset_index(drop=True) # type: ignore

    warnings.filterwarnings("default")
    return {'sig': sig_trunc, 'onset': onset_trunc}

#logic to create a filter's mask
def get_data_mask(df: pd.DataFrame, **kwargs: Any) -> pd.Series:
    # Start with a mask of all True values
    mask = pd.Series([True] * len(df))
    
    # Iterate over the keyword arguments to update the mask
    for key, value in kwargs.items():
        if key in df.columns and value is not None:
            mask &= (df[key] == value)
    
    return mask

def compiling_exp_to_df(    trial_folder : str,
                            info_folder : str,
                            trial_logic : Callable[[str],pd.DataFrame], 
                            info_logic :  Callable[[str,pd.DataFrame],pd.DataFrame]) -> pd.DataFrame:
    
    warnings.simplefilter(action='ignore', category=FutureWarning)
    df = pd.DataFrame()


    #first construct df with signal/onset
    df = trial_logic(trial_folder)

    #complete df with id from info

    df = info_logic(info_folder,df)

    warnings.filterwarnings("default")

    return df

#standalone function to filter a df
def get_data(df: pd.DataFrame, **kwargs : Any) -> pd.DataFrame:
    # Iterate over the keyword arguments to filter the DataFrame
    for key, value in kwargs.items():
        if key in df.columns and value is not None:
            # always return a copy with filter in function
            df = df.loc[df[key] == value].copy().reset_index(drop=True)
    
    return df

# TODO : make Dataset Generic ? (to also handle NdArray rather than Dataframe)
# Dataset is the class used to store data like input data, output data from compute or batch datas on a sort or trunc 
# user give to this class logics 
# Philosophy : Dataset containing RAW datas (untouched), every data coming out of a Dataset is a copy
class Dataset:
    def __init__(self,dataframes : Dict[str,pd.DataFrame] = {},trunc_logic : Callable = lambda x:x,exp_name : str = "") -> None:
        self.dataframes = dataframes
        self.exp_name = exp_name
        self.truncs = {}
        self.trunc_func = trunc_logic

    def load(self,path : str) -> Self:
        with open(path, 'rb') as file:
            data : Dataset = pickle.load(file)
            self.dataframes = data.dataframes
            self.exp_name = data.exp_name
            self.truncs = data.truncs
            self.trunc_func = data.trunc_func

        return self

    def save(self,path : str) -> None:
        #  with open(os.path.join(path,self.exp_name), 'wb') as file:
         with open(path, 'wb') as file:
            pickle.dump(self, file)
    
    def set_trunc(self,indexer : Any,trunc_values : Optional[Tuple[Optional[float],Optional[float]]] ) -> None:
        self.truncs.update({indexer:trunc_values})
    
    def get_truncate_dfs(self,**kwargs : Any) -> Dict[str,pd.DataFrame]:
        return self.trunc_func(self.dataframes,self.truncs)
    
    def add_df(self,df_key : str,df : pd.DataFrame)-> None:
        self.dataframes.update({df_key:df})

    def add_dfs(self,dfs : Dict[str,pd.DataFrame]) -> None:
        self.dataframes = self.dataframes | dfs

    def get_df(self,df_key:str) -> pd.DataFrame:
        if df_key not in self.dataframes:
                raise ValueError(f"Key '{df_key}' not found in Dataset {self.exp_name}.")
        return self.dataframes[df_key].copy()

    def build_add_df(self,df_key : str,build_logic : Callable[...,pd.DataFrame],*args : Any)-> None :
        self.add_df(df_key,build_logic(*args))

    def get_from_mask(self, mask_logic: Callable[..., pd.Series], **kwargs: Any) -> Dict[str, pd.DataFrame]:
        filtered_dataframes = {}
        for key, df in self.dataframes.items():
            mask = mask_logic(df, **kwargs)
            filtered_dataframes[key] = df[mask].copy().reset_index(drop=True)
        return filtered_dataframes
    
    def get_df_unique_values(self,df_key : str,*args : str)-> Dict[str,set]:
        df = self.get_df(df_key)
        unique_values = {}
        for arg in args:
            if arg in df.columns:
                unique_values[arg] = set(df[arg].unique())
            else:
                raise ValueError(f"Column '{arg}' does not exist in the DataFrame {df_key}.")
        return unique_values
    
    def iterate_df(self, df_key: str, *keys: str):
        df = self.dataframes[df_key]
        # Ensure the keys are columns in the DataFrame
        for key in keys:
            if key not in df.columns:
                raise ValueError(f"Key '{key}' not found in DataFrame columns.")

        # Get unique values for each key
        key_values = [df[key].unique() for key in keys]
        
        # Iterate over all combinations of key values
        for combination in itertools.product(*key_values):
            mask = pd.Series([True] * len(df))
            for key, value in zip(keys, combination):
                mask &= (df[key] == value)
            subset = df[mask]
            if not subset.empty:
                yield combination, subset

    def iterate_dfs(self, *keys: str):
        key_values = {key: set() for key in keys}
        
        # Collect unique values for each key across all dataframes
        for df in self.dataframes.values():
            for key in keys:
                if key in df.columns:
                    key_values[key].update(df[key].unique())

        # Generate all combinations of key values
        all_combinations = list(itertools.product(*[key_values[key] for key in keys]))

        # Iterate over all combinations and yield corresponding DataFrame rows
        for combination in all_combinations:
            filtered_rows = {key: pd.Series() for key in self.dataframes.keys()}
            for df_key, df in self.dataframes.items():
                mask = pd.Series([True] * len(df))
                for key, value in zip(keys, combination):
                    if key in df.columns:
                        mask &= (df[key] == value)
                filtered_df = df[mask].copy().reset_index(drop=True)
                filtered_rows[df_key] = filtered_df # type: ignore
            yield combination, filtered_rows

    def export_dataframes_to_csv(self, path, prefix = None):
        if not os.path.exists(path):
            os.makedirs(path)

        if prefix == None:
            prefix = self.exp_name

        for name, df in self.dataframes.items():
            filename = f"{prefix}_{name}.csv"
            filepath = os.path.join(path, filename)
            df.to_csv(filepath, index=False)

    def get_data_in_df(self,df_key:str, **kwargs : Any) -> pd.DataFrame:
        return self.get_data(self.get_df(df_key),**kwargs)
    
    @staticmethod
    def build_df(build_logic : Callable[...,pd.DataFrame],*args : Any)-> pd.DataFrame :
        return build_logic(*args)

    @staticmethod
    #standalone function to filter a df
    def get_data(df: pd.DataFrame, **kwargs : Any) -> pd.DataFrame:
        # Iterate over the keyword arguments to filter the DataFrame
        for key, value in kwargs.items():
            if key in df.columns and value is not None:
                # always return a copy with filter in function
                df = df.loc[df[key] == value].copy().reset_index(drop=True)
        
        return df




#                ______        _                   
#                |  _  \      | |                  
#                | | | |  ___ | |__   _   _   __ _ 
#                | | | | / _ \| '_ \ | | | | / _` |
#                | |/ / |  __/| |_) || |_| || (_| |
#                |___/   \___||_.__/  \__,_| \__, |
#                                             __/ |
#                                            |___/ 
 # debug plot


def plot_signal(x, y, ax=None, title=None, xlabel=None, ylabel=None, ylim=None, show=False, **plot_kwargs):
    """
    Plot a signal on a given axis.
    """
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(x, y, **plot_kwargs)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)

    if show:
        plt.show()

    return ax







#                        ______   __                                __ 
#                       /      \ |  \                              |  \
#                      |  $$$$$$\ \$$  ______   _______    ______  | $$
#                      | $$___\$$|  \ /      \ |       \  |      \ | $$
#                       \$$    \ | $$|  $$$$$$\| $$$$$$$\  \$$$$$$\| $$
#                       _\$$$$$$\| $$| $$  | $$| $$  | $$ /      $$| $$
#                      |  \__| $$| $$| $$__| $$| $$  | $$|  $$$$$$$| $$
#                       \$$    $$| $$ \$$    $$| $$  | $$ \$$    $$| $$
#                        \$$$$$$  \$$ _\$$$$$$$ \$$   \$$  \$$$$$$$ \$$
#                                    |  \__| $$                        
#                                     \$$    $$                        
#                                      \$$$$$$                         

# signal


class NoFrequencyError(Exception):
    """Exception personnalisée pour une situation spécifique."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def advanced_cfd_autoperiod(sig : np.ndarray,iter_perm  : int = 100,adjust_treshold :int = 0,tolerance : float = 0.2,w_threshold_rate : float = 1/3,
                            fs  : int = 100,nperseg  : int = 2192,noverlap  : int = 2048 ,nfft : int = 8192,psd_window_type  : str = 'hamming',debug : bool = False)-> list[Dict]:

    """
    Advanced Clustered Filtered Detrended Autoperiod determines the frequencies and amplitudes that compose a signal
    by comparing the Fourier Transform, Power Spectral Density (PSD), and the detrended autocorrelation of the filtered signal
    over each period interval of each potential frequency.

    ## Parameters:
    * sig (np.ndarray): The input signal to be analyzed.
    * iter_perm (int): The number of iterations to find the threshold.
    * adjust_treshold (int): An integer value between [-iter_perm, iter_perm] to adjust the threshold during the iterations.
    * tolerance (float): The frequency tolerance used for clustering the hints and validating a hint.
    * w_threshold_rate (float) : The minimum rate of hint redundancy through the signal to be validate
    * fs (int): The sampling frequency of the signal.
    * nperseg (int): The number of samples per segment for calculating the PSD.
    * noverlap (int): The number of samples that overlap between segments for the PSD calculation.
    * nfft (int): The total number of samples taken for the threshold calculation and the PSD.
    * psd_window_type (str): The type of window used for calculating the PSD. Refer to scipy's window types.
    * debug (bool): If set to True, the function will display debugging graphs of the algorithm.

    ## Returns:
    List[Dict]: A list of dictionaries, where each dictionary contains:
        - frequency: The frequency of the hint.
        - amplitude: The amplitude of the hint.
        - power: The square root of the power of the hint.
        - centroid: The spectral centroid of the cluster.
        - range: The frequency interval of the power.
        - weight : redundancy rate of the hint (higher is better)
    """

    warnings.simplefilter(action='ignore', category=FutureWarning)

    def _acfda_find_treshold(sig : np.ndarray,fs,nfft = 8192,iter_perm = 100) -> list[float]:
        sig_p = np.ndarray(sig.shape)
        _max = 0
        max_psd : list[float] = []
        for i in range(iter_perm):
            np.random.seed()
            sig_p = np.random.permutation(sig)
            _max = np.max(np.sqrt(signal.welch(sig_p, fs=fs, nperseg=sig.size,nfft=nfft,noverlap=sig.size - 1,scaling='spectrum',average='mean',window='hamming')[1]))
            max_psd.append(_max)
        max_psd.sort()
        # return 2*max_psd[0]-max_psd[-1] # return max_psd[-1] but I want a lower baseline ton ensure to get low amplitude
        return max_psd

    def _acfda_find_peaks(sig : np.ndarray, threshold : float,fs,tolerance = 0.05,nperseg = 2192,noverlap=2190,nfft=8192,psd_window_type='hamming',debug = False ):
        # Perform FFT
        fft_freqs,fft_amplitudes = dft(sig,fs)
        fft_res = fft_freqs[1] - fft_freqs[0]
        
        # Perform Welch PSD
        frequencies, psd = signal.welch(sig, fs=fs, nperseg=nperseg, nfft=nfft, noverlap=noverlap, scaling='spectrum',average='mean',window=psd_window_type)
        psd_sqrt = np.sqrt(psd)
        frequency_res = fft_res # maximal frequency resolution to estimate welch lobe
        
        # Find peaks in FFT above threshold
        peaks, _ = signal.find_peaks(fft_amplitudes, height=threshold)

        #find at least a frequency
        tmp = 0.0
        while len(peaks) == 0:
            peaks, _ = signal.find_peaks(fft_amplitudes, height=threshold - threshold * tmp)
            tmp += 0.1
            if  threshold - threshold * tmp <= 0: raise NoFrequencyError("pas de fréquence détecté, le signal doit être plat")

        # find where sqrt psd cross treshold
        idx =  np.where(np.diff(np.sign(psd_sqrt - threshold)))[0]
        idx_adjust = []
        # Prefer the index where the value is under the threshold
        for idx in idx:
            while idx < len(psd_sqrt) and psd_sqrt[idx] > threshold:
                idx += 1
            idx_adjust.append(idx)


        # Prepare hints
        hints = []
        clusters = []
        cluster = []


        # find first true potential peak
        for peak in peaks:
            f_peak = fft_freqs[peak]
            a_peak = fft_amplitudes[peak]
            power_peak = psd_sqrt[np.argmin(np.abs(frequencies - fft_freqs[peak]))]
            welch_range = [None,None]
            # avoid fake positive
            if (power_peak < threshold):
                    continue
            else : break
        

        # Find range in Welch PSD
        welch_xzero_f = frequencies[idx_adjust]
        
        mask_left = welch_xzero_f < f_peak
        mask_right = welch_xzero_f > f_peak

        if np.any(mask_left):
            left_value = np.max(welch_xzero_f[mask_left])
        else:
            left_value = None

        if np.any(mask_right):
            right_value = np.min(welch_xzero_f[mask_right])
        else:
            right_value = None

        welch_range = [left_value, right_value]

        cluster.append({
                'frequency': f_peak,
                'amplitude': a_peak,
                'power': power_peak,
                'range': welch_range
                })
        
        e = fft_freqs[peak] + frequency_res + tolerance

        for peak in peaks[1:]:
            f_peak = fft_freqs[peak]
            a_peak = fft_amplitudes[peak]
            power_peak = psd_sqrt[np.argmin(np.abs(frequencies - fft_freqs[peak]))]
            welch_range = [None,None]

            #fake positive
            if (power_peak < threshold):
                continue
            
            # Find range in Welch PSD
            welch_xzero_f = frequencies[idx_adjust]
            
            mask_left = welch_xzero_f < f_peak
            mask_right = welch_xzero_f > f_peak

            if np.any(mask_left):
                left_value = np.max(welch_xzero_f[mask_left])
            else:
                left_value = None

            if np.any(mask_right):
                right_value = np.min(welch_xzero_f[mask_right])
            else:
                right_value = None

            welch_range = [left_value, right_value]

            
            # Density Clustering
            # to close and same welch range TODO this last can be problematic for a low treshold and avoid true peak
            previous_welch_range = cluster[-1]['range']
            if(f_peak <= e and (previous_welch_range[0] == welch_range[0] and previous_welch_range[1] == welch_range[1])):
            # if(f_peak <= e ):
                # same welch range

                cluster.append({
                'frequency': f_peak,
                'amplitude': a_peak,
                'power': power_peak,
                'range': welch_range
                })
                e = fft_freqs[peak] + frequency_res + tolerance


            else:
                clusters.append(cluster)
                cluster = []
                cluster.append({
                'frequency': f_peak,
                'amplitude': a_peak,
                'power': power_peak,
                'range': welch_range
                })
                e = fft_freqs[peak] + frequency_res + tolerance

        # last cluster
        if cluster:
            clusters.append(cluster)
        
        if debug : 
            _, ax = plt.subplots( figsize=(12, 6))
            plot_signal(fft_freqs,fft_amplitudes,ax=ax,color='red',label='dft')
            plot_signal(frequencies,psd_sqrt,ax=ax,color='green',label='sqrt of psd')
            ax.axhline(threshold,color='black',linestyle='--',label='treshold')

            for peak in peaks:
                peak_idx = np.argmin(np.abs(frequencies - fft_freqs[peak]))
                ax.plot(fft_freqs[peak],fft_amplitudes[peak], 'ro',label='frequency peak')
                ax.plot(frequencies[peak_idx],psd_sqrt[peak_idx], 'gx',label='power peak')

            for i in idx_adjust:
                ax.plot(frequencies[i],psd_sqrt[i], 'bo',label='power range')

            ax.set_xscale('log')
            plt.tight_layout()

        # cluster selection : use queue to avoid unexcepted behavior
        queue = deque(clusters)
        while queue:
            cluster = queue.popleft()

            max_amp_peak : Dict = max(cluster, key=lambda x: x['amplitude'])

            max_peak = max(cluster, key=lambda x: x['frequency'])
            min_peak = min(cluster, key=lambda x: x['frequency'])

            #compute centroids
            indx = np.where((fft_freqs >= min_peak['frequency']) & (fft_freqs <= max_peak['frequency']))[0]
            if len(indx):
                freqs_in_range = fft_freqs[indx]
                amps_in_range = fft_amplitudes[indx]
                centroid = np.sum(freqs_in_range * amps_in_range) / np.sum(amps_in_range)
            else:
                centroid = max_amp_peak['frequency']

            # solves the problem of close lobes
            if(abs(max_amp_peak['frequency']-centroid) > tolerance):
                near_lobes = [
                    x for x in cluster
                    if abs(frequencies[np.argmin(np.abs(frequencies - fft_freqs[np.argmin(np.abs(fft_freqs - x['frequency']))]))] - centroid) <= tolerance
                ]
                max_power_peak_near_centroid = max(near_lobes, key=lambda x: x['power'])

                if debug :print('new cluster ->',max_power_peak_near_centroid['frequency'])

                # add to re iterate hint
                queue.append([max_power_peak_near_centroid]) 

            max_amp_peak.update({'centroid' : centroid})

            hints.append(max_amp_peak)

            if debug : 
                ax.plot(max_amp_peak['frequency'],max_amp_peak['amplitude'], 'y*',label="dominent cluster's peak")
                ax.axvline(max_amp_peak['centroid'],color='maroon',linestyle=':',linewidth=0.5,label='cluster centroid')

        if debug :
            plt.show()


        return hints

    def _acfda_validate_hint(x,y,hint,k_period=0,debug=False):
        # find kth range of periodicity
        period = 1/hint
        hint_range =  [(k_period * period + period / 2),((k_period + 1) * period + period / 2)]
        idx = np.where((x >= hint_range[0]) & (x <= hint_range[1]))

        # plot_signal(x,y,show=True)

        x_range = x[idx]
        y_range = y[idx]

        # _max = x[idx][np.argmax(y[idx])]
        # print("max : " + str(_max))

        # fit a quadratic function

        nb_polynome = 2

        # Fit the polynomial
        coeffs = np.polyfit(x_range, y_range, nb_polynome)
        polynomial = np.poly1d(coeffs)

        # Compute the derivative of the polynomial
        polynomial_derivative = np.polyder(polynomial)

        # Evaluate the derivative over a fine grid within the same range
        x_der = x.copy()
        y_der = polynomial_derivative(x_der)
        
        # Find where the derivative changes sign
        sign_changes = np.where(np.diff(np.sign(y_der)))[0]

        sign_in_interval : bool = len(np.where((x_der[sign_changes] >= hint_range[0]) & (x_der[sign_changes] <= hint_range[1]))[0]) > 0
        negative_coeff : bool = (np.sign(coeffs[0]) < 0)

        # print(np.sign(coeffs[0]), np.where((x_der[sign_changes] >= hint_range[0]) & (x_der[sign_changes] <= hint_range[1]))[0] )

        if debug :
            fig, ax = plt.subplots( figsize=(12, 6))
            ax.set_xlim(x_range[0],x_range[-1])
            ax.set_ylim(-1,1)
            plot_signal(x,np.polyval(coeffs, x),ax=ax,color="green",linestyle="--",label='quadratic fit')
            plot_signal(x_der,y_der,ax=ax,color='y',linestyle='-.',label='derivative of fit')
            for s in sign_changes:
                ax.plot(x_der[s],y_der[s],"ro",label='sign change')
            plot_signal(x,y,ax=ax,color='b',label='autocorrelation')
            plot_signal(x_range,y_range,ax=ax,label='autocorrelation')
            plt.tight_layout()
            plt.show()

        return negative_coeff and sign_in_interval

    # TODO validation_window_type for better accuracy when filtering

    res = []
    # compute tresholds
    thresholds = _acfda_find_treshold(sig,fs=fs,iter_perm=iter_perm)
    
    # if abs(adjust_treshold)-1 > iter_perm-1 : adjust_treshold =  iter_perm-1
    # if abs(adjust_treshold)-1 < -(iter_perm-1) : adjust_treshold =  -(iter_perm-1)

    threshold = thresholds[0] + np.sign(adjust_treshold) * abs(thresholds[abs(adjust_treshold)-1]-thresholds[0])
    if threshold < 0 : threshold = 1E-5

    # hints detect
    # hints = _acfda_find_peaks(sig,threshold[-1],fs=fs,nperseg=nperseg,noverlap=noverlap,nfft=nfft,psd_window_type=psd_window_type,debug=debug)
    hints = _acfda_find_peaks(sig,threshold,fs=fs,nperseg=nperseg,tolerance=tolerance,noverlap=noverlap,nfft=nfft,psd_window_type=psd_window_type,debug=debug)

    # Sorting the list of dictionaries by 'frequency' in ascending order
    hints = sorted(hints, key=lambda x: x['frequency'])
    # hints = sorted(hints, key=lambda x: x['frequency'],reverse=True)

    _debug = debug
    debug = False

    for hint in range(len(hints)):
        # hints frequencies filtering
        hintf = hints[hint]['frequency']

        # high frequencies : itered through a low pass
        cutoff = hintf + 1.0/fs  + tolerance # next frequency bin with some tolerance
        # TODO CAUTION Need to be validate before filter ?!
        if (hint == 0): filtered_sig = sos_filter(sig ,sos_butter_lowpass(cutoff,fs)) # first : low pass
        elif (hint > len(hints)-2): filtered_sig = sig # last : nothing
        else : filtered_sig = sos_filter(sig ,sos_butter_lowpass(cutoff,fs))
        # else : filtered_sig = sos_filter(sig ,sos_butter_bandpass(hints[hint-1]['frequency'],hints[hint+1]['frequency'],fs)) # band

        # if (hint == 0 or len(res) == 0): filtered_sig = sig # first : nothing
        # if (hint == 0 or len(res) == 0): filtered_sig = sos_filter(sig ,sos_butter_lowpass(hintf,fs))
        # else : filtered_sig = sos_filter(sig ,sos_butter_lowpass(res[-1]['frequency'] + 1.0/fs  + tolerance,fs)) #  low pass

        # low frequencies : detrend autocorrelation
        autoc_x,autoc_y = autocorrelation(filtered_sig,lag_seconde=True)
        # interval = (1/(hintf - 1/fs + tolerance) ) # next frequency bin with some tolerance to period
        period = 1/(hintf - 1/fs + tolerance)

        max_period = int(round(hintf*autoc_x[-1] + 1/2)) - 1
        nb_validate = 0

        for k_period in range(max_period):
            #detrend auto-correlation on the kth period
            hint_range =  [(k_period * period + period / 2),((k_period + 1) * period + period / 2)]

            mask = np.where((autoc_x >= hint_range[0]) & (autoc_x <= hint_range[1]))[0]
            last_idx = mask[-1]
            autoc_interval_x = autoc_x[mask]
            autoc_interval_y = autoc_y[mask]
            coeffs = np.polyfit(autoc_interval_x, autoc_interval_y, 1)
            trend = np.poly1d(coeffs)(autoc_x)
            autoc_detrend_y = autoc_y - trend
    
            if debug :
                fig, ax = plt.subplots( figsize=(12, 6))
                ax.set_xlim(autoc_x[0],autoc_x[mask[-1]])
                ax.set_ylim(-1,1)
                plot_signal(autoc_x,autoc_y,ax=ax,color="blue",label='autocorrelation')
                plot_signal(autoc_interval_x,autoc_interval_y,ax=ax,color="red")
                plot_signal(autoc_x,trend,ax=ax,color="red",linestyle="--")
                plot_signal(autoc_x,autoc_detrend_y,ax=ax,color="green")
                # plot_signal(autoc_x,autoc_detrend_y2,ax=ax,color="pink")
                plt.show()

                # hints validation
            if _acfda_validate_hint(autoc_x,autoc_detrend_y,hintf,k_period=k_period,debug=debug): 
                # print('f',hintf,'period',k_period,"validate")
                nb_validate += 1
                # res.append(hints[hint])
            # else :
                # print('f',hintf,'period',k_period,"not validate")

            # if nb_validate >= nb_period_to_validate: 
            #     print('f',hintf,"validate")
            #     res.append(hints[hint])
            #     break

        w = nb_validate/max_period
        hints[hint].update({'weight':w})

        if w >= w_threshold_rate : 
            # print('f:',hintf,"validate")
            res.append(hints[hint])
        # else:
        #     print('f:',hintf,"not validate")

    if _debug:
        plt.figure(figsize=(12, 6))
        plt.title("Frequencies and Amplitudes with Power Curves")

        # Itérer sur les hints pour afficher les résultats
        for hint in res:
            freq = hint['frequency']
            amplitude = hint['amplitude']
            power_sqrt = hint['power']
            centroid = hint['centroid']
            freq_range = hint['range']
            weight = hint['weight']
            
            
            l = plt.stem(freq, amplitude, linefmt='red', markerfmt='.')  
            plt.setp(l, 'alpha', weight)
            plt.text(freq, amplitude, f'{freq} Hz', fontsize=12, ha='center')
            plt.axvline(centroid,color='maroon',linestyle=':',linewidth=0.5,label='cluster centroid')


            window = signal.get_window(psd_window_type, nfft)
            power_curve = power_sqrt * window
            x = np.linspace(freq_range[0], freq_range[1], nfft)
            y =  power_curve
            
            plt.plot(x, y, 'g--', alpha=0.7)

        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.xscale('log')
        plt.xlim(1E-2,50)
        plt.show()

    warnings.filterwarnings("default")

    return res

def get_principal_frequency(hints : Union[pd.DataFrame, Dict[str, list]]) -> Tuple[float,float,float]:
    '''
    OUTDATED : principal frequency is the frequency from hint where fp = max(amplitude * weight) and the relevency (>2 strong >1.2 medium >= 1 weak)

    on pondère l'amplitude des fréquences par rapport a sa probabilité si une fréquence a une grande amplitude mais qu'elle est jamais là
    elle ne devrait pas influencé nos autres fréquence
    , on prend le max pour ne garder que l'amplitude la plus "présente" dans le signal donc la force de l'amplitude qui représente le mieux notre fréquence
    ce sera donc la fréquence la plus présente de notre signal
    le signal contient évidemment d'autre fréquence, la relvancy estime si la fréquence la plus présente est plus importante que l'enssemble des autres
    donc en excluant cette fréquence de la moyenne on situe les autres force de fréquence, si le profile spectrale ne contient pas de véritable peak
    alors la différence de cette moyenne et la fréquence sont très proche et la relavancy est proche de 1 en revanche plus le peak se démarque vraiment des autres
    plus la relavancy tend vers l'infini
    pour pallier a ce problème on prendra plutôt le logarithme pour atténuer la variance et introduirons un epsilon pour éviter le logarithme (en base 10) 
    '''
    if isinstance(hints, dict):
        hints = pd.DataFrame(hints)
    
    products = hints['amplitude'] * hints['weight']
    
    max_index = products.idxmax()

    product_max = np.max(products)
    products_without_max = products.drop(max_index).fillna(0)
    product_mean_without_max = np.mean(products_without_max)
    # relevancy = product_max / product_mean_without_max
    e = 0.1
    relevancy = np.log10((product_max + e) / (product_mean_without_max + e)) #revient à faire log(pmax) - log(pmean_n_max)

    
    return hints.loc[max_index, 'amplitude'], hints.loc[max_index, 'frequency'] , relevancy # type: ignore

fs = 100

def dft(sig,fs=100):
    N = len(sig)
    yf = fft(sig)
    xf = fftfreq(N, 1 / fs)

    # Conserver uniquement la partie positive du spectre
    pos_indices = xf >= 0
    xf = xf[pos_indices]
    yf = np.abs(yf)[pos_indices] * 2/N # type: ignore # Absolute amplitude and normalize with 2/len(sig) to get the right amps

    return xf,yf

def autocorrelation(sig,lag_seconde=False,norm=True,fs=100) -> tuple[np.ndarray,np.ndarray]:
    autocorr = np.correlate(sig, sig,mode='full')
    autocorr = autocorr[autocorr.size // 2:]  # Conserver seulement la moitié positive
    lags = np.arange(autocorr.size)
    if lag_seconde : lags = lags / fs  # Convertir les lags en secondes
    # Normaliser l'autocorrélation par la première valeur (=1)
    if norm : autocorr /= autocorr[0]
    return lags,autocorr

def sos_butter_highpass(cutoff,fs=fs,order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return signal.butter(order,normal_cutoff,'high',output='sos')
def sos_butter_lowpass(cutoff,fs=fs,order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return signal.butter(order,normal_cutoff,'low',output='sos')
def sos_butter_bandpass(lowcut, highcut, fs=fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return signal.butter(N=order, Wn=[low, high],btype='band', output='sos')

def sos_filter(data,sos):
    return signal.sosfiltfilt(sos, data)

def center_sig(data):
    return (data - data.mean()).copy()

def apodize(sig, window_size, fs, window_type='taylor'):
    """
    Apodize the given signal using a specified window type.

    Parameters:
    signal (numpy array): The input binary signal.
    window_size (float): The size of the window in seconds.
    fs (int): The sampling frequency in Hz.
    window_type (str): The type of window to use ('gaussian', 'hann', 'hamming', 'blackmanharris',
                                                  'exponential', 'chebwin', 'flattop', 'kaiser', 'nuttall', 'taylor').

    Returns:
    numpy array: The apodized signal.
    """
    if window_size == 0:
        return sig
    # Convert window size from seconds to samples
    window_width = int(window_size * fs)
    
    # Create the specified window
    if window_type == 'gaussian':
        window = signal.windows.gaussian(window_width, std=window_width/6)
    elif window_type == 'hann':
        window = signal.windows.hann(window_width)
    elif window_type == 'hamming':
        window = signal.windows.hamming(window_width)
    elif window_type == 'blackmanharris':
        window = signal.windows.blackmanharris(window_width)
    elif window_type == 'exponential':
        window = signal.windows.exponential(window_width, center=None, tau=window_width/6)
    elif window_type == 'chebwin':
        window = signal.windows.chebwin(window_width, at=100)
    elif window_type == 'flattop':
        window = signal.windows.flattop(window_width)
    elif window_type == 'kaiser':
        window = signal.windows.kaiser(window_width, beta=14)
    elif window_type == 'nuttall':
        window = signal.windows.nuttall(window_width)
    elif window_type == 'taylor':
        window = signal.windows.taylor(window_width, nbar=4, sll=100)
    elif window_type == 'cosine':
        window = signal.windows.cosine(window_width)
    else:
        raise ValueError(f"Unsupported window type: {window_type}")
    
    # Find the indices where the signal has onsets (different than 0)
    # onset_indices = np.where(sig != 0)[0]
    
    # # Create a copy of the signal to apply the apodization
    # apodized_signal = np.copy(sig)
    
    # # Apply the specified window around each onset
    # for idx in onset_indices:
    #     start = max(0, idx - window_width//2)
    #     end = min(len(sig), idx + window_width//2)
    #     # apodized_signal[start:end] = np.maximum(apodized_signal[start:end], window[:end-start])
    #     apodized_signal[start:end] += apodized_signal[start:end] * window

    apodized_signal = np.convolve(sig,window,mode='same')
    
    return apodized_signal


def xcorrelation(sig1,sig2,lag_seconde=False,norm=True,fs=100):
    '''
    compute the signal cross-correlation of 2 signals, is simply the sum of dot product for each lag
    the result is the value of similarity between shape of both signales for each lag between each other 
    higly dependent of the amplitude, then ,need to be normalized to be interpreted, does not work with very short signals
    both signal need to be centered
    '''
    corr = np.correlate(sig1, sig2,mode='full')
    # corr = corr[corr.size // 2:]  # Conserver seulement la moitié positive
    # lags = np.arange(corr.size)
    lags = np.arange(-len(sig1) + 1, len(sig2))
    if lag_seconde : lags = lags / fs  # Convertir les lags en secondes
    # Normaliser la corrélation
    if norm:
        sumsig1 = np.sum((sig1 - np.mean(sig1))**2)
        sumsig2 = np.sum((sig2 - np.mean(sig2))**2)
        corr /= np.sqrt(sumsig1 * sumsig2)

    return lags,corr

def shift(arr, num, fill_value):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

# def dwt_alignement(sig1,sig2):
#     alignment = dtw(sig1,sig2,keep_internals=True)
#     slope, intercept = np.polyfit(alignment.index1, alignment.index2, 1)

#     x_fit = np.linspace(alignment.index1.min(), alignment.index1.max(), alignment.index1.size)
#     y_fit = slope * x_fit + intercept

#     # tolerance = 100.0 # ou 50.0 basé sur la résolution fréquentiel
#     # within_tolerance = np.abs(alignment.index2 - y_fit) <= tolerance
#     # alignement_rate = np.sum(within_tolerance) / len(alignment.index2) * 100   # pourcentage des points en dehors de la courbe (avec une tolérnce)

#     sst = np.sum((alignment.index2 - np.mean(alignment.index2))**2)
#     sse =  np.sum((alignment.index2 - y_fit)**2)
#     ssr = np.sum((y_fit - np.mean(y_fit))**2)

#     # alignement_rate = (1 - sse/sst) * 100 # coefficient de détermination R2
#     alignement_rate = ssr/sst

#     return (alignment.index1,alignment.index2), slope, alignement_rate

def jitter(onset_dev):
    '''
    Jitter computes the root mean square of deviations.
    quantifies the variability of a series of deviations,
    to capture the extent of deviations regardless of their direction (leading or lagging),
    very sensitive to outliers
    '''
    return np.sqrt(np.sum(onset_dev**2) / len(onset_dev))

def calculate_tempo(onsets_df : pd.DataFrame,onset_timescale: float=0.001 , window_size:int=8, bpm:bool=True) -> Optional[Tuple[float,np.ndarray,np.ndarray,np.ndarray]]:
    '''
    ## return: 
    average tempo through window, deviations from average tempo through window, tempi through window, interbeat intervals
    '''

    onsets = onsets_df['onsets'].to_numpy()
    if onsets.size < window_size:
        return None
    
    
    if bpm:
        # Calcul des intervalles et conversion en tempi (BPM)
        onsets *= onset_timescale
        intervals = 60.0 / np.diff(onsets)
    else:
        intervals = np.diff(onsets)
    
    # Calcul de la moyenne glissante des tempi the 'local average tempo/period'
    tempi = np.convolve(intervals, np.ones(window_size)/window_size, mode='valid')
    win_tempo  = float(np.mean(tempi))

    # Calcul des écarts par rapport à la moyenne glissante
    deviations = intervals - win_tempo

    return win_tempo, deviations,  tempi, intervals

def frequency_to_bpm(frequency) -> float:
    """Convert frequency in Hz to beats per minute (BPM)."""
    return frequency * 60

def bpm_to_frequency(bpm) -> float:
    """Convert beats per minute (BPM) to frequency in Hz."""
    return bpm / 60

def frequency_to_period(frequency) -> float:
    """Convert frequency in Hz to period in seconds."""
    # if frequency == 0:
    #     return None  # Avoid division by zero
    return 1 / frequency

def bpm_to_period(bpm) -> float:
    """Convert beats per minute (BPM) to period in seconds."""
    frequency = bpm_to_frequency(bpm)
    return frequency_to_period(frequency)

def period_to_frequency(period) -> float:
    """Convert period in seconds to frequency in Hz."""
    if period == 0:
        return 0.0  # Avoid division by zero
    return 1 / period

def period_to_bpm(period) -> float:
    """Convert period in seconds to beats per minute (BPM)."""
    frequency = period_to_frequency(period)
    if frequency is None:
        return None
    return frequency_to_bpm(frequency)


def onset_to_sig(onsets_df, signal_t: Optional[np.ndarray] = None, fs = 100,onset_timescale=0.001,value=1.0):

        if onsets_df.empty:
            return None
        
        if signal_t is not None:
            duration = len(signal_t)  # samples
            signal_firstidx = int(round(signal_t[0] * fs * onset_timescale))  # assuming signal_t is in ms
        else:
            duration = int(np.max(onsets_df['onsets']) * fs * onset_timescale)
            signal_firstidx = 0
        
        signal = np.zeros(duration)
        
        for idx in onsets_df['onsets']:
            onset_index = int(round(idx * fs * onset_timescale)) - signal_firstidx
            if 0 <= onset_index < duration:
                signal[onset_index] = value
        
        return signal

def xcorrelation_metrics(sig1,sig2,sample,lag_seconde: bool = False,norm: bool = True,fs: int = 100):
    '''
        return the xcorrelation value at lag 0, and the max between [-sample,sample] and the corresponding lag
    '''
    lags, corr = xcorrelation(sig1, sig2,lag_seconde,norm,fs)
    lag_zero_index = len(sig1) - 1  # lag 0

    indices = np.where((lags >= -sample) & (lags <= sample))[0]
    selected_corr = corr[indices]
    selected_lags = lags[indices]
    max_index = np.argmax(selected_corr)
    max_corr = selected_corr[max_index]
    max_lag = selected_lags[max_index]

    return corr[lag_zero_index],max_lag,max_corr

def asynchrony_rate(intervals_x,tempo_x,tempo_y):
    '''
    instantaneous rate of asynchronism from x to y
    calculate the ratio between the deviation of the intervals from its tempo and the deviation of the intervals from another tempo
    are x's intervals closer to its tempo or to the tempo of the other series?

    if deviation_of_x_from_x > deviation_of_x_from_y :
        if deviation_of_x_from_y -> 0 :
        ratio -> infinity, This would indicate total decorrelation of x from its own tempo and almost perfect synchronisation with y
        if deviation_of_x_from_y -> deviation_of_x_from_x(-) :
        ratio -> 1(+), x deviates as much from its own tempo as from that of y

    if deviation_of_x_from_x < deviation_of_x_from_y :
        if deviation_of_x_from_y -> infinity :
        ratio -> 0, This would indicate almost perfect synchronisation of x with its own tempo and almost total independence from y
        if deviation_of_x_from_y -> deviation_of_x_from_x(+) :
        ratio -> 1(-), x deviates as much from its own tempo as from that of y

    if deviation_of_x_from_x = deviation_of_x_from_y :
        ratio = 1
        This means that the intervals of x are equally distant from their own tempo and from the tempo of y
        Like the special case where x = y
        x favours neither its own nor y's tempo
    

    '''
    deviation_of_x_from_x = np.abs(intervals_x - tempo_x)
    deviation_of_x_from_y = np.abs(intervals_x - tempo_y)
    return deviation_of_x_from_x/deviation_of_x_from_y

def async_threshold_crossings(time_series, lower_threshold = 0.9, upper_threshold=1.2,keep_relevant=True):
    """
    Detects the time intervals where the time series crosses above the upper threshold and then back below the lower threshold.
    
    Args:
    - time_series (pd.Series): The input time series with datetime index.
    - lower_threshold (float): The lower threshold for detecting descents.
    - upper_threshold (float): The upper threshold for detecting ascents.
    
    Returns:
    - crossings (list of tuples): A list of tuples where each tuple contains (start_time, end_time,max) of the interval.
    - total_time_above_threshold (float): The total time the series spent above the lower threshold between detected crossings.
    """
    
    crossings = []
    total_time_above_threshold = 0
    start_time = None
    max_values = []
    
    for time, value in time_series.items():
        if value > upper_threshold and start_time is None:
            start_time = time  # Record the time when it first crosses the upper threshold
        elif value < lower_threshold and start_time is not None:
            end_time = time  # Record the time when it crosses back below the lower threshold
            max_value = time_series[start_time:end_time].max()
            max_values.append(max_value)
            crossings.append((start_time, end_time,max_value))
            total_time_above_threshold += end_time - start_time
            start_time = None  # Reset start_time for the next possible crossing
    
    if keep_relevant:
        crossings_tmp = []
        tmp = np.mean(max_values)
        for c in crossings:
            if c[2] > tmp : crossings_tmp.append(c)
        crossings = crossings_tmp

    return crossings, total_time_above_threshold


def phase_synchrony(sig1: np.ndarray, sig2 : np.ndarray,fs=100) -> tuple[np.ndarray,np.ndarray]:
    '''
    compute synchrony of Hilbert phase between sig 1 and sig 2
    indicate variation of cycle of phase independently of amplitude. Signals need to be filtered
    1 : angles of sigs are the same and there are superposed
    0 : angles are note the same, there no synchrony between theme
    '''

    hilbert_angle_1 = np.angle(hilbert(sig1),deg=False) # type: ignore
    hilbert_angle_2 = np.angle(hilbert(sig2),deg=False) # type: ignore

    n = len(sig1)
    t = (n - 1) / fs

    return np.linspace(0,t, n),1-np.sin(np.abs(hilbert_angle_1-hilbert_angle_2)/2)

def sliding_window_xcorr(datax: Union[pd.Series, np.ndarray], datay: Union[pd.Series, np.ndarray], window_size: int, range_s: float = 0, step: int = 1, fs: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the signal processing's cross-correlation on a sliding window with a range of lags.
    
    Parameters
    ----------
    datax, datay : pandas.Series or numpy.ndarray objects of equal length
    window_size : int
        Size of the sliding window (in number of samples).
    range_s : float
        Range of lags (in seconds).
    step : int, optional, default 1
        Step size for the sliding window (in number of samples).
    fs : int, optional, default 100
        Sampling frequency of the signals.
    
    Returns
    ----------
    times : np.ndarray
        Array of time indices (center of each window).
    lags : np.ndarray
        Array of lag values.
    heatmap_data : np.ndarray
        2D array of Pearson correlation values, where rows correspond to lags and columns to time.
    """
    assert len(datax) == len(datay)
    # if isinstance(datax, np.ndarray) or isinstance(datay, np.ndarray):
    #     datax = pd.Series(datax)
    #     datay = pd.Series(datay)
    
    n_samples = len(datax)

    lags, _ = xcorrelation(datax[0:window_size], datay[0:window_size], lag_seconde= True,fs=fs)
    lags_idx = np.where((lags >= -range_s) & (lags <= range_s))[0]
    
    heatmap_data = []
    times = []
    
    for start in range(0, n_samples - window_size + 1, step):
        window_datax = datax[start:start + window_size]
        window_datay = datay[start:start + window_size]
        _, window_corr = xcorrelation(window_datax, window_datay, lag_seconde= True,fs=fs)
        heatmap_data.append(window_corr[lags_idx])
        times.append(start + window_size // 2) # the mid-point of the current time window
    
    heatmap_data = np.array(heatmap_data).T  # Transpose to have lags on rows and time on columns
    times = np.array(times) / fs * 1000 # in ms
    lags = np.array(lags[lags_idx])
    
    return times, lags, heatmap_data

def max_correlation_lags(times: np.ndarray, lags: np.ndarray, heatmap_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the lag at which the maximum cross-correlation occurs for each time point.

    Parameters
    ----------
    times : np.ndarray
        Array of time indices (center of each window).
    lags : np.ndarray
        Array of lag values.
    heatmap_data : np.ndarray
        2D array of Pearson correlation values, where rows correspond to lags and columns to time.

    Returns
    ----------
    t : np.ndarray
        Array of time indices.
    lag_max : np.ndarray
        Array of lag values corresponding to the maximum correlation for each time point.
    """
    lag_max = []
    
    # Parcours de chaque colonne de la matrice heatmap_data
    for i in range(heatmap_data.shape[1]):
        # Trouver l'indice du lag qui correspond à la corrélation maximale pour chaque instant t
        max_index = np.argmax(heatmap_data[:, i])
        
        # Récupérer la valeur du lag correspondant
        lag_max.append(lags[max_index])
    
    # Convertir lag_max en un tableau numpy pour être cohérent avec times
    lag_max = np.array(lag_max)
    
    return times, lag_max

def max_correlation_lags_mean(times: np.ndarray, lags: np.ndarray, heatmap_data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the lag at which the maximum cross-correlation occurs for each time point,
    and apply a moving average over the results.

    Parameters
    ----------
    times : np.ndarray
        Array of time indices (center of each window).
    lags : np.ndarray
        Array of lag values.
    heatmap_data : np.ndarray
        2D array of Pearson correlation values, where rows correspond to lags and columns to time.
    window_size : int
        Size of the window over which to apply the moving average (in number of time points).

    Returns
    ----------
    t : np.ndarray
        Array of time indices.
    lag_max : np.ndarray
        Array of lag values corresponding to the maximum correlation for each time point,
        after applying the moving average.
    """
    # Filtrer les lags pour ne garder que les lags positifs
    positive_lags_idx = np.where(lags >= 0)[0]
    lags = lags[positive_lags_idx]
    heatmap_data = heatmap_data[positive_lags_idx, :]
    
    # Calculer les lags maximaux pour chaque temps
    lag_max = []
    
    for i in range(heatmap_data.shape[1]):
        max_index = np.argmax(heatmap_data[:, i])
        lag_max.append(lags[max_index])
    
    lag_max = np.array(lag_max)
    
    # Appliquer une moyenne glissante sur les lags maximaux
    smoothed_lag_max = np.convolve(lag_max, np.ones(window_size)/window_size, mode='valid')
    
    # Ajuster le vecteur temps pour correspondre à la taille de la fenêtre
    t = times[(window_size-1)//2 : -(window_size//2)]
    
    return t, smoothed_lag_max

def phase_locking_value(signal_x_raw, signal_y_raw):
    # signal_x_analytic = hilbert(signal_x_raw)
    # signal_y_analytic = hilbert(signal_y_raw)

    # phase_x = np.angle(signal_x_analytic)
    # phase_y = np.angle(signal_y_analytic)

    phase_x = np.angle(hilbert(signal_x_raw),deg=False) # type: ignore
    phase_y = np.angle(hilbert(signal_y_raw),deg=False) # type: ignore

    phase_diff = phase_x - phase_y

    complex_phase_diff = np.exp(1j * phase_diff)
    # Calculate the mean vector length (PLV or SI)
    # plv = np.mean(np.abs(complex_phase_diff))
    plv = np.abs(np.mean(complex_phase_diff))

    return plv



#                                ..######..########....###....########..######.
#                                .##....##....##......##.##......##....##....##
#                                .##..........##.....##...##.....##....##......
#                                ..######.....##....##.....##....##.....######.
#                                .......##....##....#########....##..........##
#                                .##....##....##....##.....##....##....##....##
#                                ..######.....##....##.....##....##.....######.


# stats

def pearson_corr(datax : Union[pd.Series,np.ndarray], datay : Union[pd.Series,np.ndarray], lag=0) -> float:
    """ 
    Pearson coefficient i.e. linear dependency between two time series shifted with a lag 
    
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    assert len(datax) == len(datay)
    if isinstance(datax,np.ndarray) or isinstance(datay,np.ndarray):
        datax = pd.Series(datax)
        datay = pd.Series(datay)

    return datax.corr(datay.shift(lag))

def pearson_xcorr(datax : Union[pd.Series,np.ndarray], datay : Union[pd.Series,np.ndarray], range_s : float, fs : int = 100) -> tuple[list[int],list[float]]:
    """ 
    Pearson coefficient i.e. linear dependency between two time series depending of lag between range (in second)

    Parameters
    ----------
    range_s : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : list[float]
    """
    assert len(datax) == len(datay)
    if isinstance(datax,np.ndarray) or isinstance(datay,np.ndarray):
        datax = pd.Series(datax)
        datay = pd.Series(datay)

    lags = range(-int(range_s*fs),int(range_s*fs+1))
    return list(lags),[pearson_corr(datax,datay, lag) for lag in lags]

def sliding_window_pearson_xcorr(datax: Union[pd.Series, np.ndarray], datay: Union[pd.Series, np.ndarray], window_size: int, range_s: float, step: int = 1, fs: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Pearson correlation on a sliding window with a range of lags.
    
    Parameters
    ----------
    datax, datay : pandas.Series or numpy.ndarray objects of equal length
    window_size : int
        Size of the sliding window (in number of samples).
    range_s : float
        Range of lags (in seconds).
    step : int, optional, default 1
        Step size for the sliding window (in number of samples).
    fs : int, optional, default 100
        Sampling frequency of the signals.
    
    Returns
    ----------
    times : np.ndarray
        Array of time indices (center of each window).
    lags : np.ndarray
        Array of lag values.
    heatmap_data : np.ndarray
        2D array of Pearson correlation values, where rows correspond to lags and columns to time.
    """
    print('len datax',len(datax),'len datay',len(datay))
    assert len(datax) == len(datay)
    if isinstance(datax, np.ndarray) or isinstance(datay, np.ndarray):
        datax = pd.Series(datax)
        datay = pd.Series(datay)
    
    n_samples = len(datax)
    lags, _ = pearson_xcorr(datax, datay, range_s, fs)
    
    heatmap_data = []
    times = []
    
    for start in range(0, n_samples - window_size + 1, step):
        print(str(start)+':'+str(start + window_size))
        window_datax = datax.iloc[start:start + window_size]
        window_datay = datay.iloc[start:start + window_size]
        _, window_corr = pearson_xcorr(window_datax, window_datay, range_s, fs)
        heatmap_data.append(window_corr)
        times.append(start + window_size // 2) # the mid-point of the current time window
    
    heatmap_data = np.array(heatmap_data).T  # Transpose to have lags on rows and time on columns
    times = np.array(times)
    lags = np.array(lags) / fs  # Convert lags to seconds if necessary
    
    return times, lags, heatmap_data

def rolling_correlation(data, wrap=False, *args, **kwargs):
    '''
    Intersubject rolling correlation.
    Data is dataframe with observations in rows, subjects in columns.
    Calculates pairwise rolling correlation at each time.
    Grabs the upper triangle, at each timepoints returns dataframe with
    observation in rows and pairs of subjects in columns.
    *args:
        window: window size of rolling corr in samples
        center: whether to center result (Default: False, so correlation values are listed on the right.)
    '''

    def get_triangle(df,k=0):
        '''
        This function grabs the upper triangle of a correlation matrix
        by masking out the bottom triangle (tril) and returns the values.

        df: pandas correlation matrix
        '''
        x = np.hstack(df.mask(np.tril(np.ones(df.shape),k=k).astype(bool)).values.tolist())
        x = x[~np.isnan(x)]
        return x  

    data_len = data.shape[0]
    half_data_len = int(data.shape[0]/2)
    start_len = data.iloc[half_data_len:].shape[0]
    if wrap:
        data = pd.concat([data.iloc[half_data_len:],data,data.iloc[:half_data_len]],axis=0).reset_index(drop=True)
    _rolling = data.rolling(*args, **kwargs).corr()        # type: ignore
    rs=[]
    for i in np.arange(0,data.shape[0]):
        rs.append(get_triangle(_rolling.loc[i]))
    rs = pd.DataFrame(rs)   
    rs = rs.iloc[start_len:start_len+data_len].reset_index(drop=True)
    return rs  

def granger_causality(signal1, signal2, max_lag, n_permutations=1) -> pd.DataFrame:
    """
    Calcule la causalité de Granger entre deux signaux pour un nombre maximum de lags spécifié.
    La causalité de Granger ne test pas la causlité a proprement parlé mais est plutôt une prédiction :
    Comment signal1 peut prédire (ou peut intéragir avec)  signal2 en prenant en compte un nombre lag de point en arrière 
    les résultats sont souvent comme une valeur de 'force' de prédiction et la probabilité que cette mesure pour ce lag spécifique soit du au hasard (hypothèse nulle)
    Retourne un dictionnaire contenant pour chaque lag :
    - 'F-value': La valeur F du test de Granger.
    - 'p-value': La p-value associée à la F-value.
    - 'log_ratio': Le log des variances des résidus entre X et X + Y.
    - 'log_ratio_p-value': La p-value empirique associée au log ratio, calculée via un test de permutation.
    
    :param signal1: La première série temporelle (X).
    :param signal2: La deuxième série temporelle (Y).
    :param max_lag: Le nombre maximum de lags à tester.
    :param n_permutations: Le nombre de permutations pour calculer la p-value empirique du log ratio.
    :return: Un dictionnaire contenant les résultats pour chaque lag.
    """
    # to shutup grangercausalitytest
    warnings.filterwarnings("ignore")

    def _compute_log_ratio_from_residuals(residuals_X, residuals_XY):
        """
        Calcule le log des rapports variances des résidus entre deux modèles AR.
        Attention X est la variable dépendante et Y le prédicateur (noms inversés pa rapport à la fonction parent)
        
        :param residuals_X: Les résidus du modèle AR sur X.
        :param residuals_XY: Les résidus du modèle AR sur X + Y.
        :return: Le log ratio des variances des résidus.
        """
        var_X = np.var(residuals_X)
        var_XY = np.var(residuals_XY)
        log_ratio = np.log(var_X / var_XY)
        return log_ratio
    
    def _compute_log_ratio_statistics(signal1, signal2, lag):
        '''
        grangercausalitytests(data[['Y', 'X']], [lag], verbose=False) does X granger cause Y ?
        signal1 (model X) is the predicator and signal2 Y is the dependant variable
        results[lag][1][0].resid residuals of OLS estimation of the restricted model (Y) 
        results[lag][1][1].resid  residuals of OLS estimation of the unrestricted model (XY)
        '''
        data = pd.DataFrame({'X': signal1, 'Y': signal2})
        results = grangercausalitytests(data[['Y', 'X']], [lag], verbose=False)
        return _compute_log_ratio_from_residuals(results[lag][1][0].resid, results[lag][1][1].resid)
    
    def _permutation_test(signal1, signal2, lag, n_resamples=1000, alternative='greater'):
        """
        Custom permutation test to assess the statistical significance of the log ratio statistic
        by permuting signal1 while keeping signal2 fixed.
        permute the predictor variable (i.e., the independent variable, X) while keeping the dependent variable (Y) unchanged.
        
        :param signal1: The first signal (which will be permuted).
        :param signal2: The second signal (kept fixed).
        :param lag: The lag value for Granger causality test.
        :param n_resamples: The number of permutations to perform.
        :param alternative: The type of test: 'greater', 'less', or 'two-sided'.
        :return: The p-value from the permutation test.
        """
        # Calculate the original log ratio statistic
        original_statistic = _compute_log_ratio_statistics(signal1, signal2, lag)

        # Initialize a list to store the statistics from the permutations
        permuted_statistics = []

        for _ in range(n_resamples):
            # gc.collect()
            # Permute only signal1
            permuted_signal1 = np.random.permutation(signal1)
            
            # Compute the log ratio statistic for the permuted data
            permuted_statistic = _compute_log_ratio_statistics(permuted_signal1, signal2, lag)
            
            # Append the result to the list of permuted statistics
            permuted_statistics.append(permuted_statistic)

        # Convert the list to a numpy array for easy calculation
        permuted_statistics = np.array(permuted_statistics)

        # Calculate the p-value based on the specified alternative hypothesis
        if alternative == 'greater':
            p_value = np.mean(permuted_statistics >= original_statistic)
        elif alternative == 'less':
            p_value = np.mean(permuted_statistics <= original_statistic)
        elif alternative == 'two-sided':
            p_value = np.mean(np.abs(permuted_statistics) >= np.abs(original_statistic))
        else:
            raise ValueError("Invalid alternative hypothesis. Choose from 'greater', 'less', or 'two-sided'.")

        return original_statistic, p_value


    # Combiner les deux signaux dans une seule DataFrame
    data = pd.DataFrame({'X': signal1, 'Y': signal2})
    
    # Tester la causalité de Granger pour les lags jusqu'à max_lag
    results = grangercausalitytests(data[['Y', 'X']], max_lag, verbose=False)
    
    # Préparer les listes pour stocker les résultats
    lags = []
    f_values = []
    p_values = []
    log_ratios = []
    log_ratio_p_values = []
    
    for lag, result in results.items():
        # gc.collect()
        # print(lag)

        # Récupérer les valeurs F et p du test de Granger
        f_value = result[0]['ssr_ftest'][0]
        p_value = result[0]['ssr_ftest'][1]

        # res = _permutation_test(
        #     signal1,
        #     signal2,
        #     lag,
        #     n_resamples=n_permutations,
        # )
        # log_ratio ,log_ratio_p_value = res
        if n_permutations > 1: #avoid to calculate log_ratio if no p-value
            def permutation_func(x,y):
                #signal2 is the dependant variable
                return _compute_log_ratio_statistics(x,signal2, lag)
            res = permutation_test(
                data=(signal1, signal2),
                statistic=permutation_func,
                vectorized=False, 
                n_resamples=n_permutations, 
                alternative='greater'
            )
            log_ratio = res.statistic
            log_ratio_p_value = res.pvalue
        else:
            log_ratio = 0
            log_ratio_p_value = 1

        
        # Stocker les résultats
        lags.append(lag)
        f_values.append(f_value)
        p_values.append(p_value)
        log_ratios.append(log_ratio)
        log_ratio_p_values.append(log_ratio_p_value)
    
    # Créer le DataFrame final avec les colonnes appropriées
    summary_df = pd.DataFrame({
        'lag': lags,
        'F-value': f_values,
        'p-value': p_values,
        'log_ratio': log_ratios,
        'log_ratio_p-value': log_ratio_p_values
    })
    
    # to shutup grangercausalitytest
    warnings.filterwarnings("default")

    return summary_df

def sliding_window_gc(datax: Union[pd.Series, np.ndarray], datay: Union[pd.Series, np.ndarray], window_size: int, range_sample: Union[int,list] = 1, step: int = 1, fs: int = 100) -> Tuple[np.ndarray, np.ndarray, Any,Any]:
    """
    Compute the gc on a sliding window with a range of lags.
    
    Parameters
    ----------
    datax, datay : pandas.Series or numpy.ndarray objects of equal length
    window_size : int
        Size of the sliding window (in number of samples).
    range_s : float
        Range of lags (in seconds).
    step : int, optional, default 1
        Step size for the sliding window (in number of samples).
    fs : int, optional, default 100
        Sampling frequency of the signals.
    
    Returns
    ----------
    times : np.ndarray
        Array of time indices (center of each window).
    lags : np.ndarray
        Array of lag values.
    heatmap_data : np.ndarray
        2D array of Pearson correlation values, where rows correspond to lags and columns to time.
    """
    assert len(datax) == len(datay)
    # if isinstance(datax, np.ndarray) or isinstance(datay, np.ndarray):
    #     datax = pd.Series(datax)
    #     datay = pd.Series(datay)
    
    n_samples = len(datax)
    # range_sample = range_s * fs
    
    heatmap_data = {'F-value': [],'log_ratio':[]}
    probability_value = {'p-value':[],'log_ratio_p-value':[]}
    times = []
    
    for start in range(0, n_samples - window_size + 1, step):
        print(str(start)+':'+str(start + window_size))
        window_datax = datax[start:start + window_size]
        window_datay = datay[start:start + window_size]
        gc_df = granger_causality(window_datax,window_datay,range_sample)
        heatmap_data['F-value'].append(gc_df['F-value'])
        heatmap_data['log_ratio'].append(gc_df['log_ratio'])
        probability_value['p-value'].append(gc_df['p-value'])
        # probability_value['log_ratio_p-value'].append(gc_df['log_ratio_p-value'])
        times.append(start + window_size // 2) # the mid-point of the current time window
    
     # Transpose to have lags on rows and time on columns
    heatmap_data['F-value'] = np.array(heatmap_data['F-value']).T.tolist()
    heatmap_data['log_ratio'] = np.array(heatmap_data['log_ratio']).T.tolist()
    probability_value['p-value'] = np.array(probability_value['p-value']).T.tolist()
    probability_value['log_ratio_p-value'] = np.array(probability_value['log_ratio_p-value']).T.tolist()

    times = np.array(times) / fs * 1000 # in ms
    if isinstance(range_sample,int):
        lags = np.arange(0,range_sample,1) / fs #to second
    else:
        lags = np.array(range_sample) / fs #to second
    
    return times, lags, heatmap_data,probability_value

def clean_gc_heatmap(heatmap_dict,probability_dict,seuil = 0.05):
    p_value_matrix = probability_dict['p-value']
    f_value_matrix = heatmap_dict['F-value']
    log_value_matrix = heatmap_dict['log_ratio']

    for i in range(len(p_value_matrix)):
        for j in range(len(p_value_matrix[i])):
            if f_value_matrix[i][j] < 0 :
                f_value_matrix[i][j] = 0
            if log_value_matrix[i][j] < 0 :
                log_value_matrix[i][j] = 0
            if p_value_matrix[i][j] > seuil:
                f_value_matrix[i][j] = 0
                log_value_matrix[i][j] = 0
def clean_gc(summary_df,p_value_threshold = 0.05):
    return summary_df[summary_df['p-value'] < p_value_threshold].copy()

def analyze_granger_causality(summary_df, p_value_threshold = 0.05) -> Dict[str,Union[float,pd.DataFrame]]:
    '''
    analyse summary df of granger causality
    filter by p-value treshold
    summary_df : lag F-value p-value log_ratio log_ratio_p-value
    res : Dict
    gc_X_lag : le lag correspondant au maximum X
    gc_X_lag_max : X maximum
    gc_X_mean: moyenne des X
    gc_X_std : std des X
    # gc_X_df : le dataframe originale

    '''
    # Filter the DataFrame based on the p-value threshold
    filtered_df = summary_df[summary_df['p-value'] < p_value_threshold].copy()
    
    # Define the results dictionary
    results = {}

    # Analyze F-value
    if not filtered_df.empty:
        max_f_value_row = filtered_df.loc[filtered_df['F-value'].idxmax()]
        results['gc_F_lag'] = max_f_value_row['lag']
        results['gc_F_lag_max'] = max_f_value_row['F-value']
        results['gc_F_mean'] = filtered_df['F-value'].mean()
        results['gc_F_std'] = filtered_df['F-value'].std()
        # results['gc_F_df'] = summary_df[['lag', 'F-value', 'p-value']]
    else:
        results['gc_F_lag'] = None
        results['gc_F_lag_max'] = None
        results['gc_F_mean'] = None
        results['gc_F_std'] = None
        # results['gc_F_df'] = pd.DataFrame(columns=['lag', 'F-value', 'p-value'])

    # Analyze log-ratio
    filtered_log_ratio_df = summary_df[summary_df['log_ratio_p-value'] < p_value_threshold].copy()
    
    if not filtered_log_ratio_df.empty:
        max_log_ratio_row = filtered_log_ratio_df.loc[filtered_log_ratio_df['log_ratio'].idxmax()]
        results['gc_log_ratio_lag'] = max_log_ratio_row['lag']
        results['gc_log_ratio_lag_max'] = max_log_ratio_row['log_ratio']
        results['gc_log_ratio_mean'] = filtered_log_ratio_df['log_ratio'].mean()
        results['gc_log_ratio_std'] = filtered_log_ratio_df['log_ratio'].std()
        # results['gc_log_ratio_df'] = summary_df[['lag', 'log_ratio', 'log_ratio_p-value']]
    else:
        results['gc_log_ratio_lag'] = None
        results['gc_log_ratio_lag_max'] = None
        results['gc_log_ratio_mean'] = None
        results['gc_log_ratio_std'] = None
        # results['gc_log_ratio_df'] = pd.DataFrame(columns=['lag', 'log_ratio', 'log_ratio_p-value'])
    
    return results

def zscore_column(df : pd.DataFrame, column_to_normalize : str, *columns :str, inplace : bool=False):
    """
    Calcule le z-score d'une colonne en regroupant selon les colonnes spécifiées dans args.
    
    Parameters:
    - df : DataFrame à utiliser.
    - column_to_normalize : colonne a normaliser
    - *args : colonnes à utiliser pour le regroupement avant le calcul du z-score.
    - inplace : booléen. Si True, modifie le DataFrame d'origine. Sinon, retourne un nouveau DataFrame.
    
    Returns:
    - Si inplace=False, retourne un nouveau DataFrame avec la colonne normalisée.
    """
    if not inplace:
        df = df.copy()
    # Crée une colonne zscore_intensity qui contient les z-scores normalisés par les colonnes spécifiées dans args
    df['zscore_'+ column_to_normalize] = df.groupby(list(columns))[column_to_normalize].transform(zscore)
    
    if inplace:
        return None
    else:
        return df.copy()


#  ______  _______ _______ _______                                                     
#  |     \ |_____|    |    |_____|                                                     
#  |_____/ |     |    |    |     |                                                     
                                                                                     
#   _____   ______ _______  _____  _______  ______ _______ _______ _____  _____  __   _
#  |_____] |_____/ |______ |_____] |_____| |_____/ |_____|    |      |   |     | | \  |
#  |       |    \_ |______ |       |     | |    \_ |     |    |    __|__ |_____| |  \_|
                                                                                     

# data preparation

def trunc_before_first_onset(dataset : Dataset) -> None:

    tqdm_progressbar = tqdm( dataset.iterate_df('onset','trial_n'))

    for k,df in tqdm_progressbar:

        trial_n = k[0]
        tqdm_progressbar.set_postfix({'trial' : trial_n})

        min_onset = min(df['onsets'])
        max_onset = max(df['onsets'])
        # print(k,min_onset)
        dataset.set_trunc(trial_n,(min_onset,max_onset+200))

columns_to_drop = ['acc_x' , 'acc_y' , 'acc_z' , 'gyro_x' , 'gyro_y' , 'gyro_z' , 'mag_x' , 'mag_y' , 'mag_z' , 'orientation_x' , 'orientation_y' , 'orientation_z' , 'accfilt_x' , 'accfilt_y' , 'accfilt_z']
# columns_to_drop = ['acc_x' , 'acc_y' , 'acc_z' , 'gyro_x' , 'gyro_y' , 'gyro_z' , 'mag_x' , 'mag_y' , 'mag_z' , 'intensity', 'accfilt_x' , 'accfilt_y' , 'accfilt_z']



def drop_column_sig_dataset(dataset:Dataset,columns : list[str]):
    dataset.dataframes['sig'].drop(columns, axis=1, inplace=True)



                                                                                                              

                                                                                                                       
        #              ***** **                                                                                          
        #           ******  ***                   *                                                                      
        #         **    *  * ***                 **                                                                      
        #        *     *  *   ***                **                                                                      
        #             *  *     ***             ********                                                                  
        #            ** **      **    ****    ********     ****                                                          
        #            ** **      **   * ***  *    **       * ***  *                                                       
        #            ** **      **  *   ****     **      *   ****                                                        
        #            ** **      ** **    **      **     **    **                                                         
        #            ** **      ** **    **      **     **    **                                                         
        #            *  **      ** **    **      **     **    **                                                         
        #               *       *  **    **      **     **    **                                                         
        #          *****       *   **    **      **     **    **                                                         
        #         *   *********     ***** **      **     ***** **                                                        
        #        *       ****        ***   **             ***   **                                                       
        #        *                                                                                                       
        #         **                                                                                                     
                                                                                                                       
                                                                                                                       
                                                                                                                       
                                                                                                                       
        #             ***** **                                                                                           
        #          ******  ****                                                                                          
        #         **   *  *  ***                                                                                         
        #        *    *  *    ***                                                                                        
        #            *  *      ** ***  ****       ****                            ****       ****                        
        #           ** **      **  **** **** *   * ***  *    ****       ***      * **** *   * **** *                     
        #           ** **      **   **   ****   *   ****    * ***  *   * ***    **  ****   **  ****                      
        #         **** **      *    **         **    **    *   ****   *   ***  ****       ****                           
        #        * *** **     *     **         **    **   **         **    ***   ***        ***                          
        #           ** *******      **         **    **   **         ********      ***        ***                        
        #           ** ******       **         **    **   **         *******         ***        ***                      
        #           ** **           **         **    **   **         **         ****  **   ****  **                      
        #           ** **           ***         ******    ***     *  ****    * * **** *   * **** *                       
        #           ** **            ***         ****      *******    *******     ****       ****                        
        #      **   ** **                                   *****      *****                                             
        #     ***   *  *                                                                                                 
        #      ***    *                                                                                                  
        #       ******                                                                                                   
        #         ***                                                                                                    
                                                                                                                       
                                                                                                                       
        #          ***** **                                                                                      ***     
        #       ******  ****                                                                                      ***    
        #      **   *  *  ***                                                                                      **    
        #     *    *  *    ***                                                                                     **    
        #         *  *      **           ***  ****       ****       ****                                           **    
        #        ** **      **    ***     **** **** *   * **** *   * ***  * ***  ****    ***  ****       ****      **    
        #        ** **      **   * ***     **   ****   **  ****   *   ****   **** **** *  **** **** *   * ***  *   **    
        #      **** **      *   *   ***    **         ****       **    **     **   ****    **   ****   *   ****    **    
        #     * *** **     *   **    ***   **           ***      **    **     **    **     **    **   **    **     **    
        #        ** *******    ********    **             ***    **    **     **    **     **    **   **    **     **    
        #        ** ******     *******     **               ***  **    **     **    **     **    **   **    **     **    
        #        ** **         **          **          ****  **  **    **     **    **     **    **   **    **     **    
        #        ** **         ****    *   ***        * **** *    ******      **    **     **    **   **    **     **    
        #        ** **          *******     ***          ****      ****       ***   ***    ***   ***   ***** **    *** * 
        #   **   ** **           *****                                         ***   ***    ***   ***   ***   **    ***  
        #  ***   *  *                                                                                                    
        #   ***    *                                                                                                     
        #    ******                                                                                                      
        #      ***                                                                                                       
                                                                                                                       
                                                                                                       
                                                                                                              


# data process personnal


# tsfresh features extract
def extract_motion_features(df):
    # Create id
    df['id'] = df['trial_n'].astype(str) + '_' + df['riot_n'].astype(str)

    # config tsfresh
    custom_fc_parameters = {
        #distribution and shape
        'standard_deviation': None,
        'mean': None,
        'median': None,
        'kurtosis': None,
        'skewness': None,
        #energy
        'root_mean_square': None,
        #complexity/regularity
        'cid_ce': [{'normalize': True}],
        'sample_entropy': None,
        'permutation_entropy': [{'tau': 1, 'dimension': 7}]
    }
    extracted_features = extract_features(df, column_id='id',column_sort='t', column_value='intensity_filtered', default_fc_parameters=custom_fc_parameters)
    return extracted_features

def extract_deviation_features(df):
    # Create id
    df['id'] = df['trial_n'].astype(str) + '_' + df['musician'].astype(str)

    # config tsfresh
    custom_fc_parameters = {
        #distribution and shape
        'standard_deviation': None,
        'mean': None,
        'median': None,
        'kurtosis': None,
        'skewness': None,
    }
    extracted_features = extract_features(df, column_id='id', column_value='deviations', default_fc_parameters=custom_fc_parameters)
    return extracted_features

def extract_onset_features(df):
    # Create id
    df['id'] = df['trial_n'].astype(str) + '_' + df['musician'].astype(str)

    # config tsfresh
    custom_fc_parameters = {
        #complexity/regularity
        'cid_ce': [{'normalize': True}],
        'sample_entropy': None,
        'permutation_entropy': [{'tau': 1, 'dimension': 7}]
    }

    #TODO
    # extracted_features = extract_features(df, column_id='id',column_sort='t', column_value='intensity_filtered', default_fc_parameters=custom_fc_parameters)
    return None

# process funcs

def process_motion(motion_dataset : Dataset, motion_dfname:str = 'sig') -> Dict[str, pd.DataFrame] :
    
    
    warnings.simplefilter(action='ignore', category=FutureWarning)

    low_cutoff = 0.2
    high_cutoff = 8.5

    filtered_sig = pd.DataFrame()
    processed_sig = pd.DataFrame()
    f_info_sig = pd.DataFrame()
    f_main_sig = pd.DataFrame()
    id_sig = pd.DataFrame()
    
    for k,df in tqdm(motion_dataset.iterate_df(motion_dfname,'trial_n','riot_n'),desc = 'motion'):
        
        if df['intensity'].size == 0:
            print(k,'no sig')
            continue

        df.reset_index(drop=True,inplace=True)

        # remove DC detrend and filter 
        sig_motion = df['intensity'].to_numpy()
        sig_motion = center_sig(sig_motion)
        sig_motion = signal.detrend(sig_motion)
        sos = sos_butter_bandpass(low_cutoff, high_cutoff)
        sig_motion = sos_filter(sig_motion, sos)

        if len(sig_motion) == 0:
            print(k,'no sig')
            continue

        # calculate frequency hints
        try:
            hints = pd.DataFrame.from_records(
                advanced_cfd_autoperiod(sig_motion,iter_perm=1000,adjust_treshold=1000,tolerance=0.1,w_threshold_rate=0.1,psd_window_type='hamming',debug=False))
        except NoFrequencyError as e:
            print(k,'something wrong with this signal')
            continue

        main_amplitude,main_frequency, relevancy = get_principal_frequency(hints)

        # filter noise with low pass
        filtered_df = pd.DataFrame ({
            't': df['t'].values,
            'intensity_filtered' : sos_filter(df['intensity'],sos_butter_lowpass(high_cutoff, fs))
        })
        
        processed_df = pd.DataFrame({
            't': df['t'].values,
            'intensity': sig_motion
        })

        f_main_sig_df = pd.DataFrame({

            'main_amplitude' : [main_amplitude],
            'main_frequency' : [main_frequency],
            'relevancy' : [relevancy]
        })

        id_sig_df =  pd.DataFrame({

            'id' : [df['trial_n'].astype(str).values[0] + '_' + df['riot_n'].astype(str).values[0]],

        })

        # adding id column
        columns_to_add = ['trial_n', 'riot_n', 'musician', 'riot_type', 'factor1', 'factor2','change']
        for col in columns_to_add:
            if col in df.columns:
                filtered_df[col] = df[col].values[0]
                processed_df[col] = df[col].values[0]
                hints[col] = df[col].values[0]
                f_main_sig_df[col] = df[col].values[0]
                id_sig_df[col] = df[col].values[0]
        
        filtered_sig = pd.concat([filtered_sig, filtered_df], ignore_index=True)
        processed_sig = pd.concat([processed_sig, processed_df], ignore_index=True)
        f_info_sig = pd.concat([f_info_sig, hints], ignore_index=True)
        f_main_sig = pd.concat([f_main_sig, f_main_sig_df], ignore_index=True)
        id_sig = pd.concat([id_sig, id_sig_df], ignore_index=True)

    # ts fresh to get features from filtered sig
    features : pd.DataFrame = extract_motion_features(filtered_sig.copy()) # type: ignore
    features = pd.concat([features.reset_index(drop=True),id_sig.reset_index(drop=True)], axis=1)

    return {'filtered_sig':filtered_sig,'processed_sig':processed_sig,'f_info_sig':f_info_sig,'f_main_sig' : f_main_sig,'id_sig': id_sig, 'motion_features' : features}


def process_onset(onset_dataset : Dataset, onset_dfname:str = 'onset') -> Dict[str, pd.DataFrame]:
    
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    dev_onset = pd.DataFrame()
    info_onset = pd.DataFrame()
    id_onset = pd.DataFrame()

    for k,df in tqdm(onset_dataset.iterate_df(onset_dfname,'trial_n','musician'),desc = 'onset :'):

        df.reset_index(drop=True,inplace=True)

        tempo,deviations,tempi,ioi =  calculate_tempo(df,window_size=4,bpm=False) # type: ignore
        jitter_value = jitter(deviations)

        dev_onset_df = pd.DataFrame({
            'deviations': deviations,
            'ioi': ioi
        })

        info_onset_df = pd.DataFrame({
            'tempo': [tempo],
            'jitter': [jitter_value],
            'nb': [len(df['onsets'])]
        })

        id_onset_df =  pd.DataFrame({

            'id' : [df['trial_n'].astype(str).values[0] + '_' + df['musician'].astype(str).values[0]],

        })

        columns_to_add = ['trial_n', 'musician', 'riot_type', 'factor1', 'factor2','change']
        for col in columns_to_add:
            if col in df.columns:
                dev_onset_df[col] = df[col].values[0]
                info_onset_df[col] = df[col].values[0]
                id_onset_df[col] = df[col].values[0]

        dev_onset = pd.concat([dev_onset, dev_onset_df], ignore_index=True)
        info_onset = pd.concat([info_onset, info_onset_df], ignore_index=True)
        id_onset = pd.concat([id_onset, id_onset_df], ignore_index=True)

    # ts fresh to get features from deviations
    dev_features : pd.DataFrame = extract_deviation_features(dev_onset.copy()) # type: ignore
    dev_features = pd.concat([dev_features.reset_index(drop=True),id_onset.reset_index(drop=True)],axis=1)


    # ts fresh to get features from onsets
    #TODO

    return {'dev_onset':dev_onset,'info_onset' : info_onset,'id_onset':id_onset,'deviation_features':dev_features}


def compute_onset_motion_features(onset_motion_dataset : Dataset,onset_dfname:str = 'onset',motion_dfname:str = 'processed_sig',
                                   deviation_dfname:str = 'dev_onset',onset_info_dfname:str = 'info_onset',
                                   frequency_dfname:str = 'f_main_sig',gc_lag : int = 80):
    
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    onset_motion_features = pd.DataFrame()
    gc_om_details = pd.DataFrame()
    gc_mo_details = pd.DataFrame()

    # avoid compute complexity
    gc_lag = 80

    tqdm_progressbar = tqdm(onset_motion_dataset.iterate_dfs('trial_n','musician','riot_type'),desc = 'onset motion')
    for k,dfs in tqdm_progressbar:
        tqdm_progressbar.set_postfix({'trial': k[0],'musician': k[1], 'riot_type': k[2]})
            
        if(dfs[onset_dfname].empty):
            print(k,'no onset')
            continue

        if(dfs[motion_dfname].empty):
            print(k,'no signal')
            continue

        if(len(dfs[motion_dfname]['intensity']) == 0):
            print(k,'no signal')
            continue

        #avoid unique couple of key
        for df in dfs.values():
            if(df.empty):
                continue


        dfs[onset_dfname].reset_index(drop=True,inplace=True)
        dfs[motion_dfname].reset_index(drop=True,inplace=True)

        sig_motion = dfs[motion_dfname]['intensity'].to_numpy()
        t = dfs[motion_dfname]['t'].to_numpy() #in ms 

        if sig_motion.size == 0:
            print(k,'no sig')
            continue

        sig_onset = onset_to_sig(dfs[onset_dfname],t,value=max(sig_motion))
        tempo = dfs[onset_info_dfname]['tempo'].values[0] #in ms
        sig_onset_apodize = center_sig(apodize(sig_onset,tempo/1000,fs=100))
        sine_onset =  (max(sig_motion)) * np.sin(2 * np.pi * period_to_frequency(tempo) * t )

        # onset_sample = tempo*fs
        # onset_frequency = period_to_frequency(tempo)

        # ideal lag is between max (max ioi (in s) , motion frequency) to observe 1 interact TODO
        nT = 1 # number of period to look out
        range_s = nT * max(tempo/1000,frequency_to_period(dfs[frequency_dfname]['main_frequency'].values[0]))
        range_sample = int(range_s * fs)
        range_s = np.max(dfs[deviation_dfname]['ioi'])/1000 # onset's shift is only constrained by is tempo
        # print(range_sample)

        # signal cross-correlation max and at 0
        xcorr_0,lag,xcorr_lag = xcorrelation_metrics(sig_motion,sig_onset_apodize,range_s,lag_seconde=True)

        # coherence max
        f,coh = signal.coherence(sig_motion,sine_onset,fs=100,nperseg=2048,noverlap=2047,window='hamming',nfft=sig_motion.size)
        coh_f = f[np.argmax(coh)]
        coh_max = np.max(coh)

        #pearson correlation coeff (lag same has xcorr ? yes)
        # pcc = pearson_corr(sig_motion,sig_onset_apodize)

        # SI
        si_onsets = phase_locking_value(sig_motion,sig_onset_apodize)
        # mean phasesync
        psync_onsets = np.mean(phase_synchrony(sig_motion,sig_onset_apodize)[1])

        # granger causality
        if range_sample > gc_lag : range_sample = gc_lag # avoid computentational complexity
        #gc onset->motion
        # gc_om_df = granger_causality(sig_onset_apodize,sig_motion,range_sample,n_permutations=9999) #extremly slow
        gc_om_df = granger_causality(sig_onset_apodize,sig_motion,range_sample)
        gc_onset_motion = analyze_granger_causality(gc_om_df)
        #gc motion->onset
        # gc_mo_df = granger_causality(sig_motion,sig_onset_apodize,range_sample,n_permutations=9999) #extremly slow
        gc_mo_df = granger_causality(sig_motion,sig_onset_apodize,range_sample)
        gc_motion_onset = analyze_granger_causality(gc_mo_df)
        
        #dtw distance
        #TODO

        onset_motion_features_df = pd.DataFrame({
        #xc
        'xcorr_0' : [xcorr_0],
        'lag' : [lag],
        'xcorr_lag' : [xcorr_lag],
        #coh
        'coh_f' : [coh_f],
        'coh_max' : [coh_max],
        #pcc
        # 'pcc' : [pcc],

        #Synchronisation Index = Phase Locking Value
        'si' : [si_onsets],
        #Phase synchrony
        'psync' : [psync_onsets],

        #gc onset->motion
        'gc_om_F_lag' : gc_onset_motion['gc_F_lag'],
        'gc_om_F_lag_max' : gc_onset_motion['gc_F_lag_max'],
        'gc_om_F_mean' : gc_onset_motion['gc_F_mean'],
        'gc_om_F_std' : gc_onset_motion['gc_F_std'],
        #gc motion->onset
        'gc_mo_F_lag' : gc_motion_onset['gc_F_lag'],
        'gc_mo_F_lag_max' : gc_motion_onset['gc_F_lag_max'],
        'gc_mo_F_mean' : gc_motion_onset['gc_F_mean'],
        'gc_mo_F_std' : gc_motion_onset['gc_F_std'],
        #gc onset->motion log-ratio
        # 'gc_om_log-ratio_lag' : gc_onset_motion['gc_log_ratio_lag'],
        # 'gc_om_log-ratio_lag_max' : gc_onset_motion['gc_log_ratio_lag_max'],
        # 'gc_om_log-ratio_mean' : gc_onset_motion['gc_log_ratio_mean'],
        # 'gc_om_log-ratio_std' : gc_onset_motion['gc_log_ratio_std'],
        # #gc motion->onset log-ratio
        # 'gc_mo_log-ratio_lag' : gc_motion_onset['gc_log_ratio_lag'],
        # 'gc_mo_log-ratio_lag_max' : gc_motion_onset['gc_log_ratio_lag_max'],
        # 'gc_mo_log-ratio_mean' : gc_motion_onset['gc_log_ratio_mean'],
        # 'gc_mo_log-ratio_std' : gc_motion_onset['gc_log_ratio_std'],
        })

        columns_to_add = ['trial_n', 'riot_n', 'musician', 'riot_type', 'factor1', 'factor2','change']
        for col in columns_to_add:
            if col in dfs[motion_dfname].columns:
                onset_motion_features_df[col] = dfs[motion_dfname][col].values[0]
                gc_om_df[col] = dfs[motion_dfname][col].values[0]
                gc_mo_df[col] = dfs[motion_dfname][col].values[0]

        # warnings.filterwarnings("ignore")
        onset_motion_features = pd.concat([onset_motion_features, onset_motion_features_df], ignore_index=True)
        gc_om_details = pd.concat([gc_om_details, gc_om_df], ignore_index=True)
        gc_mo_details = pd.concat([gc_mo_details, gc_mo_df], ignore_index=True)
        # warnings.filterwarnings("default")

    return  onset_motion_features



                                                                                                                                                                                                                    
#                                _____          ____    _________________        ____                                                                                                                              
#                            ___|\    \    ____|\   \  /                 \  ____|\   \                                                                                                                             
#                           |    |\    \  /    /\    \ \______     ______/ /    /\    \                                                                                                                            
#                           |    | |    ||    |  |    |   \( /    /  )/   |    |  |    |                                                                                                                           
#                           |    | |    ||    |__|    |    ' |   |   '    |    |__|    |                                                                                                                           
#                           |    | |    ||    .--.    |      |   |        |    .--.    |                                                                                                                           
#                           |    | |    ||    |  |    |     /   //        |    |  |    |                                                                                                                           
#                           |____|/____/||____|  |____|    /___//         |____|  |____|                                                                                                                           
#                           |    /    | ||    |  |    |   |`   |          |    |  |    |                                                                                                                           
#                           |____|____|/ |____|  |____|   |____|          |____|  |____|                                                                                                                           
#                             \(    )/     \(      )/       \(              \(      )/                                                                                                                             
#                              '    '       '      '         '               '      '                                                                                                                              
                                                                                                                                                                                                                    
                                                                                                                                                                                                                    
#                          _____        _____           _____          _____        ______            ______           ______                                                                                      
#                      ___|\    \   ___|\    \     ____|\    \     ___|\    \   ___|\     \       ___|\     \      ___|\     \                                                                                     
#                     |    |\    \ |    |\    \   /     /\    \   /    /\    \ |     \     \     |    |\     \    |    |\     \                                                                                    
#                     |    | |    ||    | |    | /     /  \    \ |    |  |    ||     ,_____/|    |    |/____/|    |    |/____/|                                                                                    
#                     |    |/____/||    |/____/ |     |    |    ||    |  |____||     \--'\_|/ ___|    \|   | | ___|    \|   | |                                                                                    
#                     |    ||    |||    |\    \ |     |    |    ||    |   ____ |     /___/|  |    \    \___|/ |    \    \___|/                                                                                     
#                     |    ||____|/|    | |    ||\     \  /    /||    |  |    ||     \____|\ |    |\     \    |    |\     \                                                                                        
#                     |____|       |____| |____|| \_____\/____/ ||\ ___\/    /||____ '     /||\ ___\|_____|   |\ ___\|_____|                                                                                       
#                     |    |       |    | |    | \ |    ||    | /| |   /____/ ||    /_____/ || |    |     |   | |    |     |                                                                                       
#                     |____|       |____| |____|  \|____||____|/  \|___|    | /|____|     | / \|____|_____|    \|____|_____|                                                                                       
#                       \(           \(     )/       \(    )/       \( |____|/   \( |_____|/     \(    )/         \(    )/                                                                                         
#                        '            '     '         '    '         '   )/       '    )/         '    '           '    '                                                                                          
#                                                                        '             '                                                                                                                           
                                                                                                                                                                                                                    
#    ____  _____   ______    _________________      ______        _____        _____        ______        _____            ______          _____     _____   ______    _____   ______          ____    ____        
#   |    ||\    \ |\     \  /                 \ ___|\     \   ___|\    \   ___|\    \   ___|\     \   ___|\    \       ___|\     \    ____|\    \   |\    \ |\     \  |\    \ |\     \    ____|\   \  |    |       
#   |    | \\    \| \     \ \______     ______/|     \     \ |    |\    \ |    |\    \ |     \     \ |    |\    \     |    |\     \  /     /\    \   \\    \| \     \  \\    \| \     \  /    /\    \ |    |       
#   |    |  \|    \  \     |   \( /    /  )/   |     ,_____/||    | |    ||    | |    ||     ,_____/||    | |    |    |    |/____/| /     /  \    \   \|    \  \     |  \|    \  \     ||    |  |    ||    |       
#   |    |   |     \  |    |    ' |   |   '    |     \--'\_|/|    |/____/ |    |/____/||     \--'\_|/|    |/____/  ___|    \|   | ||     |    |    |   |     \  |    |   |     \  |    ||    |__|    ||    |  ____ 
#   |    |   |      \ |    |      |   |        |     /___/|  |    |\    \ |    ||    |||     /___/|  |    |\    \ |    \    \___|/ |     |    |    |   |      \ |    |   |      \ |    ||    .--.    ||    | |    |
#   |    |   |    |\ \|    |     /   //        |     \____|\ |    | |    ||    ||____|/|     \____|\ |    | |    ||    |\     \    |\     \  /    /|   |    |\ \|    |   |    |\ \|    ||    |  |    ||    | |    |
#   |____|   |____||\_____/|    /___//         |____ '     /||____| |____||____|       |____ '     /||____| |____||\ ___\|_____|   | \_____\/____/ |   |____||\_____/|   |____||\_____/||____|  |____||____|/____/|
#   |    |   |    |/ \|   ||   |`   |          |    /_____/ ||    | |    ||    |       |    /_____/ ||    | |    || |    |     |    \ |    ||    | /   |    |/ \|   ||   |    |/ \|   |||    |  |    ||    |     ||
#   |____|   |____|   |___|/   |____|          |____|     | /|____| |____||____|       |____|     | /|____| |____| \|____|_____|     \|____||____|/    |____|   |___|/   |____|   |___|/|____|  |____||____|_____|/
#     \(       \(       )/       \(              \( |_____|/   \(     )/    \(           \( |_____|/   \(     )/      \(    )/          \(    )/         \(       )/       \(       )/    \(      )/    \(    )/   
#      '        '       '         '               '    )/       '     '      '            '    )/       '     '        '    '            '    '           '       '         '       '      '      '      '    '    
#                                                      '                                       '                                                                                                                   



# data process interperonnal

def compute_entrainment_features(sig1,sig2,range_s,range_sample,min_len,avoid_invert = False):

    warnings.simplefilter(action='ignore', category=FutureWarning)

    low_cutoff = 0.2
    high_cutoff = 8.5

    sig1 = sig1[:min_len]
    sig2 = sig2[:min_len]

    lags, corr = xcorrelation(sig1, sig2,True,True,fs)

                # cc
    xcorr_0_12,lag_12,xcorr_lag_12 = xcorrelation_metrics(sig2,sig1,range_s,lag_seconde=True)
                # coherence max[0.2:8Hz]
    f,coh = signal.coherence(sig1,sig2,fs=100,nperseg=2048,noverlap=2047,window='hamming',nfft=sig1.size)
    indices = np.where((f >= low_cutoff) & (f <= high_cutoff))
    coh_f = f[indices][np.argmax(coh)]
    coh_max = np.max(coh[indices])
                # SI
    si = phase_locking_value(sig1,sig2)
                # GC
    gc_res = granger_causality(sig1,sig2,range_sample)
    gc_12 = analyze_granger_causality(gc_res)


    res_1_on_2 = pd.DataFrame({
        'xcorr_0' : [xcorr_0_12],
        'lag' : [lag_12],
        'xcorr_lag' : [xcorr_lag_12],
        'coh_max' : [coh_max],
        'coh_f' : [coh_f],
        'si' : [si],
        'gc_F_lag' : gc_12['gc_F_lag'],
        'gc_F_lag_max' : gc_12['gc_F_lag_max'],
        'gc_F_mean' : gc_12['gc_F_mean'],
        'gc_F_std' : gc_12['gc_F_std'],
    })

    if avoid_invert:
        return res_1_on_2
    
    res_2_on_1 = pd.DataFrame()

                # cc
    xcorr_0_21,lag_21,xcorr_lag_21 = xcorrelation_metrics(sig1,sig2,range_s,lag_seconde=True)
                # coh [0.2:8Hz]
    f,coh = signal.coherence(sig1,sig2,fs=100,nperseg=2048,noverlap=2047,window='hamming',nfft=sig1.size)
    indices = np.where((f >= low_cutoff) & (f <= high_cutoff))
    coh_f = f[indices][np.argmax(coh)]
    coh_max = np.max(coh[indices])
                # SI
    si = phase_locking_value(sig1,sig2)
                # GC
    gc_res = granger_causality(sig2,sig1,range_sample)
    gc_21 = analyze_granger_causality(gc_res)

    res_2_on_1 = pd.DataFrame({
    'xcorr_0' : [xcorr_0_21],
    'lag' : [lag_21],
    'xcorr_lag' : [xcorr_lag_21],
    'coh_max' : [coh_max],
    'coh_f' : [coh_f],
    'si' : [si],
    'gc_F_lag' : gc_21['gc_F_lag'],
    'gc_F_lag_max' : gc_21['gc_F_lag_max'],
    'gc_F_mean' : gc_21['gc_F_mean'],
    'gc_F_std' : gc_21['gc_F_std'],
    })

    return res_1_on_2,res_2_on_1

def compute_static_entrainment(dataset : Dataset,tempo_window_size : int = 4,gc_sample : int = 80):
    
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    element_to_visit = dataset.get_df_unique_values('sig_df','trial_n','musician','riot_type')
    # sig_df = dataset.get_df('sig_df')

    trials : list[int] = list(element_to_visit['trial_n'])
    musicians : list[str] = list(element_to_visit['musician'])

    riot_types = list(element_to_visit['riot_type']) #TODO for generic purpose
    riot_head = 'head' #TODO
    riot_limb = list({x for x in riot_types if x != riot_head})[0] #TODO

    # parameters
    # tempo_window_size = 4
    apodize_height = 0.01
    # gc_sample = 80
    # gc_sample = 200 ideal to get entrainment at the musical phrase scale

    res_motion = pd.DataFrame()
    res_onset_motion = pd.DataFrame()
    res_onset = pd.DataFrame()

    tqdm_progress_main = tqdm(trials)
    for trial_n in tqdm_progress_main:

        # print(trial_n)
        # if trial_n > 1 : continue

        #parcoure triangulaire
        tqdm_progress = tqdm(range(len(musicians)),leave=False)
        for m in tqdm_progress:
            # get all what we want
            musician = musicians[m]
            onset_df = dataset.get_data_in_df('onset_df',trial_n=trial_n,musician=musician)
            #si pas d'onset on abandonne
            if onset_df.empty:
                print(trial_n,musician,'no onset')
                continue

            head_sig = dataset.get_data_in_df('sig_df',trial_n=trial_n,musician=musician,riot_type=riot_head)
            t = head_sig['t'].to_numpy()

            head_sig = head_sig['intensity'].to_numpy()
            limb_sig = dataset.get_data_in_df('sig_df',trial_n=trial_n,musician=musician,riot_type=riot_limb)['intensity'].to_numpy()
            try:
                #TODO GC and CC lag depend of this
                head_main_frequency = dataset.get_data_in_df('frequency',trial_n=trial_n,musician=musician,riot_type=riot_head).values[0]
                limb_main_frequency = dataset.get_data_in_df('sig_df',trial_n=trial_n,musician=musician,riot_type=riot_limb).values[0]
            except IndexError:
                print(trial_n,musician,'something wrong with frequencies of this signal ! both signals are excluded from analysis')
                continue 

            sig_onset = onset_to_sig(onset_df,t,value=apodize_height) #onset based on head
            tempo,_,_,ioi = calculate_tempo(onset_df,window_size=tempo_window_size,bpm=False) # type: ignore
            sig_onset_apodize = center_sig(apodize(sig_onset,tempo/1000,fs=100))

            for m_next in range(m+1,len(musicians)):

                # get all what we want
                musician_next = musicians[m_next]
                onset_df_next  = dataset.get_data_in_df('onset_df',trial_n=trial_n,musician=musician_next)
                #si pas d'onset on abandonne
                if onset_df_next .empty:
                    print(trial_n,musician_next ,'no onset')
                    continue
                head_sig_next  = dataset.get_data_in_df('sig_df',trial_n=trial_n,musician=musician_next ,riot_type=riot_head)
                t_next  = head_sig_next['t'].to_numpy()
                
                head_sig_next  = dataset.get_data_in_df('sig_df',trial_n=trial_n,musician=musician_next ,riot_type=riot_head)['intensity'].to_numpy()
                limb_sig_next  = dataset.get_data_in_df('sig_df',trial_n=trial_n,musician=musician_next,riot_type=riot_limb)['intensity'].to_numpy()
                try:
                    #TODO GC and CC lag depend of this
                    head_main_frequency_next  = dataset.get_data_in_df('frequency',trial_n=trial_n,musician=musician_next ,riot_type=riot_head).values[0]
                    limb_main_frequency_next  = dataset.get_data_in_df('sig_df',trial_n=trial_n,musician=musician_next ,riot_type=riot_limb).values[0]
                except IndexError:
                    print(trial_n,musician_next,'something wrong with frequencies of this signal ! both signals are excluded from analysis')
                    continue 
                sig_onset_next = onset_to_sig(onset_df_next,t_next,value=apodize_height)
                tempo_next,_,_,ioi_next  = calculate_tempo(onset_df_next ,window_size=tempo_window_size,bpm=False) # type: ignore
                sig_onset_apodize_next = center_sig(apodize(sig_onset_next,tempo_next/1000,fs=100))

                # print(trial_n,musician_next,'->',musician)

                # avoid 1 sample len diff
                min_len = min(len(limb_sig),len(limb_sig_next),len(head_sig),len(head_sig_next),len(sig_onset_apodize),len(sig_onset_apodize_next))
                sig_onset_apodize = sig_onset_apodize[:min_len]
                sig_onset_apodize_next = sig_onset_apodize_next[:min_len]

                tqdm_progress.set_postfix({'trial':trial_n,'musician':musician +' '+musician_next })

                # onset_next <-> onset
                        # cc
                range_s = max(np.max(ioi),np.max(ioi_next))/1000 #en seconde
                xcorr_0_onsets,lag_onsets,xcorr_lag_onsets = xcorrelation_metrics(sig_onset_apodize,sig_onset_apodize_next,range_s,lag_seconde=True)
                        # async
                async_mm_next = pd.Series(data=asynchrony_rate(ioi,tempo,tempo_next),index=onset_df['onsets'].to_numpy()[1:])
                async_m_nextm = pd.Series(data=asynchrony_rate(ioi_next,tempo_next,tempo),index=onset_df_next['onsets'].to_numpy()[1:])
                async_mm_next_details,async_mm_next_time = async_threshold_crossings(async_mm_next,0.9,1.2,False)
                async_m_nextm_details,async_m_nextm_time = async_threshold_crossings(async_m_nextm,0.9,1.2,False)
                        # SI
                si_onsets = phase_locking_value(sig_onset_apodize,sig_onset_apodize_next)
                        # mean phasesync
                psync_onsets = np.mean(phase_synchrony(sig_onset_apodize,sig_onset_apodize_next)[1])
                
                res_onset_df = pd.DataFrame({
                    'xcorr_0_onsets' : [xcorr_0_onsets],
                    'lag_onsets' : [lag_onsets],
                    'xcorr_lag_onsets' : [xcorr_lag_onsets],
                    'async_mm_next_detais' : [async_mm_next_details],#je peux récupérer le temps à partir ce ça
                    'async_m_nextm_details' : [async_m_nextm_details],
                    'si_onsets' : [si_onsets],
                    'psync_onsets' : [psync_onsets]
                })

                columns_to_add = ['trial_n', 'factor1', 'factor2','change']
                for col in columns_to_add:
                    if col in onset_df.columns:
                        res_onset_df[col] = onset_df[col].values[0]

                col = 'musician'
                res_onset_df[col] = musician

                col = 'musician_next'
                res_onset_df[col] = musician_next

                
                # res_df = pd.DataFrame()
                range_sample = gc_sample
                # Musician_next -> Musician
                    # onset_next -> motion
                        #head
                res_mnextm_head = compute_entrainment_features(sig_onset_apodize_next,head_sig,range_s,range_sample,min_len,True)
                
                        #limb
                res_mnextm_limb = compute_entrainment_features(sig_onset_apodize_next,limb_sig,range_s,range_sample,min_len,True)

                # Musician -> Musician_next
                    # onset -> motion_next
                        #head
                res_mmnext_head = compute_entrainment_features(sig_onset_apodize,head_sig_next,range_s,range_sample,min_len,True)
                        #limb
                res_mmnext_limb = compute_entrainment_features(sig_onset_apodize,limb_sig_next,range_s,range_sample,min_len,True)

                # motion_next <-> motion
                        #head
                res_mnextm_motion_head,res_mmnext_motion_head = compute_entrainment_features(head_sig_next,head_sig,range_s,range_sample,min_len)
                        #limb
                res_mnextm_motion_limb,res_mmnext_motion_limb = compute_entrainment_features(limb_sig,limb_sig_next,range_s,range_sample,min_len)

                #ajouter les colonnes musician
                columns_to_add = ['trial_n', 'musician', 'factor1', 'factor2','change']
                for col in columns_to_add:
                    if col in onset_df.columns:
                        res_mnextm_head[col] = onset_df[col].values[0]
                        res_mnextm_limb[col] = onset_df[col].values[0]
                        res_mnextm_motion_head[col] = onset_df[col].values[0]
                        res_mnextm_motion_limb[col] = onset_df[col].values[0]

                col = 'attract_by'
                res_mnextm_head[col] = musician_next
                res_mnextm_limb[col] = musician_next
                res_mnextm_motion_head[col] = musician_next
                res_mnextm_motion_limb[col] = musician_next

                col = 'riot_type'
                res_mnextm_head[col] = riot_head
                res_mnextm_motion_head[col] = riot_head
                res_mnextm_limb[col] = riot_limb
                res_mnextm_motion_limb[col] = riot_limb

                #ajouter les colonnes musician_next
                columns_to_add = ['trial_n', 'musician', 'factor1', 'factor2','change']
                for col in columns_to_add:
                    if col in onset_df_next.columns:
                        res_mmnext_head[col] = onset_df_next[col].values[0]
                        res_mmnext_limb[col] = onset_df_next[col].values[0]
                        res_mmnext_motion_head[col] = onset_df_next[col].values[0]
                        res_mmnext_motion_limb[col] = onset_df_next[col].values[0]

                col = 'attract_by'
                res_mmnext_head[col] = musician
                res_mmnext_limb[col] = musician
                res_mmnext_motion_head[col] = musician
                res_mmnext_motion_limb[col] = musician

                col = 'riot_type'
                res_mmnext_head[col] = riot_head
                res_mmnext_motion_head[col] = riot_head
                res_mmnext_limb[col] = riot_limb
                res_mmnext_motion_limb[col] = riot_limb
                        
                res_onset_motion = pd.concat([res_onset_motion, res_mnextm_head,res_mnextm_limb,res_mmnext_head,res_mmnext_limb], ignore_index=True)
                res_motion = pd.concat([res_motion, res_mnextm_motion_head,res_mnextm_motion_limb,res_mmnext_motion_head,res_mmnext_motion_limb], ignore_index=True)
                res_onset = pd.concat([res_onset,res_onset_df], ignore_index=True)
    return{'entrainment_onset':res_onset,'entrainment_onset_motion':res_onset_motion,'entrainment_motion':res_motion}






#                                           *                         
#                                         (  `                        
#                                         )\))(       )   (           
#                                        ((_)()\   ( /(   )\    (     
#                                        (_()((_)  )(_)) ((_)   )\ )  
#                                        |  \/  | ((_)_   (_)  _(_/(  
#                                        | |\/| | / _` |  | | | ' \)) 
#                                        |_|  |_| \__,_|  |_| |_||_|  
                             




# main funcs






def run_build():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    #Click Tempo

    folder = CLICK_TEMPO_FOLDER

    print(folder,'building')

    click_tempo_dataset = Dataset(trunc_logic=trunc_dfs,exp_name='click_tempo')
    click_tempo_dataset.build_add_df('sig',compiling_exp_to_df,
                                    os.path.join(MAIN_DATA_FOLDER,folder,"trials"),
                                    os.path.join(MAIN_DATA_FOLDER,folder,"infos"),
                                    trials_folder_to_sig_df,
                                    infos_folder_to_complete_sig_df)
    click_tempo_dataset.build_add_df('onset',compiling_exp_to_df,
                                    os.path.join(MAIN_DATA_FOLDER,folder,"onsets"),
                                    os.path.join(MAIN_DATA_FOLDER,folder,"infos"),
                                    onsets_folder_to_onset_df,
                                    infos_folder_to_complete_onset_df)

    trunc_before_first_onset(click_tempo_dataset)
    click_tempo_dataset.save(os.path.join(MAIN_DATA_FOLDER,'cache/click_tempo_dataset.pkl'))

    # print(click_tempo_dataset.truncs)
    print('done')

    t_dataset = click_tempo_dataset.get_truncate_dfs()
    truncated_dataset = Dataset(t_dataset,trunc_dfs,click_tempo_dataset.exp_name)

    truncated_dataset.get_df('sig').to_csv(os.path.join(MAIN_DATA_FOLDER,'csv/click_tempo_all.csv'))

    drop_column_sig_dataset(truncated_dataset,columns_to_drop)
    truncated_dataset.save(os.path.join(MAIN_DATA_FOLDER,'cache/click_tempo_intensity_dataset.pkl'))

    truncated_dataset.get_df('sig').to_csv(os.path.join(MAIN_DATA_FOLDER,'csv/click_tempo_intensity.csv'))
    truncated_dataset.get_df('onset').to_csv(os.path.join(MAIN_DATA_FOLDER,'csv/click_tempo_onset.csv'))

    #Mask attack
    warnings.simplefilter(action='ignore', category=FutureWarning)

    folder = MASK_ATTACK_FOLDER

    print(folder,'building')

    mask_attack_dataset = Dataset(trunc_logic=trunc_dfs,exp_name='mask_attack')
    mask_attack_dataset.build_add_df('sig',compiling_exp_to_df,
                                    os.path.join(MAIN_DATA_FOLDER,folder,"trials"),
                                    os.path.join(MAIN_DATA_FOLDER,folder,"infos"),
                                    trials_folder_to_sig_df,
                                    infos_folder_to_complete_sig_df)
    mask_attack_dataset.build_add_df('onset',compiling_exp_to_df,
                                    os.path.join(MAIN_DATA_FOLDER,folder,"onsets"),
                                    os.path.join(MAIN_DATA_FOLDER,folder,"infos"),
                                    onsets_folder_to_onset_df,
                                    infos_folder_to_complete_onset_df)

    trunc_before_first_onset(mask_attack_dataset)
    mask_attack_dataset.save(os.path.join(MAIN_DATA_FOLDER,'cache/mask_attack_dataset.pkl'))

    truncated_dataset = Dataset(mask_attack_dataset.get_truncate_dfs(),trunc_dfs,mask_attack_dataset.exp_name)
    truncated_dataset.get_df('sig').to_csv(os.path.join(MAIN_DATA_FOLDER,'csv/mask_attack_all.csv'))

    drop_column_sig_dataset(truncated_dataset,columns_to_drop)
    truncated_dataset.save(os.path.join(MAIN_DATA_FOLDER,'cache/mask_attack_intensity_dataset.pkl'))

    truncated_dataset.get_df('sig').to_csv(os.path.join(MAIN_DATA_FOLDER,'csv/mask_attack_intensity.csv'))
    truncated_dataset.get_df('onset').to_csv(os.path.join(MAIN_DATA_FOLDER,'csv/mask_attack_onset.csv'))

    print('done')

    #Change
    warnings.simplefilter(action='ignore', category=FutureWarning)

    folder = CHANGE_FOLDER

    print(folder,'building')

    change_dataset = Dataset(trunc_logic=trunc_dfs,exp_name='change')
    change_dataset.build_add_df('sig',compiling_exp_to_df,
                                    os.path.join(MAIN_DATA_FOLDER,folder,"trials"),
                                    os.path.join(MAIN_DATA_FOLDER,folder,"infos"),
                                    trials_folder_to_sig_df,
                                    infos_folder_to_complete_sig_df)
    change_dataset.build_add_df('onset',compiling_exp_to_df,
                                    os.path.join(MAIN_DATA_FOLDER,folder,"onsets"),
                                    os.path.join(MAIN_DATA_FOLDER,folder,"infos"),
                                    onsets_folder_to_onset_df,
                                    infos_folder_to_complete_onset_df)

    trunc_before_first_onset(change_dataset)
    change_dataset.save(os.path.join(MAIN_DATA_FOLDER,'cache/change_dataset.pkl'))

    truncated_dataset = Dataset(change_dataset.get_truncate_dfs(),trunc_dfs,change_dataset.exp_name)
    truncated_dataset.get_df('sig').to_csv(os.path.join(MAIN_DATA_FOLDER,'csv/change_all.csv'))
    drop_column_sig_dataset(truncated_dataset,columns_to_drop)
    truncated_dataset.save(os.path.join(MAIN_DATA_FOLDER,'cache/change_intensity_dataset.pkl'))

    truncated_dataset.get_df('sig').to_csv(os.path.join(MAIN_DATA_FOLDER,'csv/change_intensity.csv'))
    truncated_dataset.get_df('onset').to_csv(os.path.join(MAIN_DATA_FOLDER,'csv/change_onset.csv'))

    print('done building all exp')

def run_personnal():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    #click tempo
    exp = 'click_tempo'
    print(exp,'personal synchrony')

    data = pd.read_csv(os.path.join(MAIN_DATA_FOLDER,'csv',exp+'_intensity.csv'))
    onset = pd.read_csv(os.path.join(MAIN_DATA_FOLDER,'csv',exp+'_onset.csv'))

    dataset = Dataset()
    dataset.add_df('sig',data)
    dataset.add_df('onset',onset)

    processed_motion = process_motion(dataset)
    processed_onset = process_onset(dataset)

    onset_motion_dataset = Dataset({'processed_sig' : processed_motion['processed_sig']} | 
                                {'onset': onset} |
                                {'dev_onset': processed_onset['dev_onset']} |
                                {'info_onset': processed_onset['info_onset']} |
                                {'f_main_sig': processed_motion['f_main_sig']})

    onset_motion_features = compute_onset_motion_features(onset_motion_dataset,motion_dfname='processed_sig')

    static_features = Dataset(trunc_logic=None,exp_name=exp)
    static_features.add_dfs(processed_motion)
    static_features.add_dfs(processed_onset)
    static_features.add_df('onset_motion_features',onset_motion_features)

    static_features.save(os.path.join(MAIN_DATA_FOLDER,'cache',exp+'_features.pkl'))
    static_features.export_dataframes_to_csv(os.path.join(MAIN_DATA_FOLDER,'csv/'))

    #mask attack
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    exp = 'mask_attack'
    print(exp,'personal synchrony')

    data = pd.read_csv(os.path.join(MAIN_DATA_FOLDER,'csv',exp+'_intensity.csv'))
    onset = pd.read_csv(os.path.join(MAIN_DATA_FOLDER,'csv',exp+'_onset.csv'))

    dataset = Dataset()
    dataset.add_df('sig',data)
    dataset.add_df('onset',onset)

    processed_motion = process_motion(dataset)
    processed_onset = process_onset(dataset)

    onset_motion_dataset = Dataset({'processed_sig' : processed_motion['processed_sig']} | 
                                {'onset': onset} |
                                {'dev_onset': processed_onset['dev_onset']} |
                                {'info_onset': processed_onset['info_onset']} |
                                {'f_main_sig': processed_motion['f_main_sig']})

    onset_motion_features = compute_onset_motion_features(onset_motion_dataset,motion_dfname='processed_sig')

    static_features = Dataset(trunc_logic=None,exp_name=exp)
    static_features.add_dfs(processed_motion)
    static_features.add_dfs(processed_onset)
    static_features.add_df('onset_motion_features',onset_motion_features)

    static_features.save(os.path.join(MAIN_DATA_FOLDER,'cache',exp+'_features.pkl'))
    static_features.export_dataframes_to_csv(os.path.join(MAIN_DATA_FOLDER,'csv/'))

    #change
    warnings.simplefilter(action='ignore', category=FutureWarning)
    exp = 'change'
    print(exp,'personal synchrony')

    data = pd.read_csv(os.path.join(MAIN_DATA_FOLDER,'csv',exp+'_intensity.csv'))
    onset = pd.read_csv(os.path.join(MAIN_DATA_FOLDER,'csv',exp+'_onset.csv'))

    dataset = Dataset()
    dataset.add_df('sig',data)
    dataset.add_df('onset',onset)

    processed_motion = process_motion(dataset)
    processed_onset = process_onset(dataset)

    onset_motion_dataset = Dataset({'processed_sig' : processed_motion['processed_sig']} | 
                                {'onset': onset} |
                                {'dev_onset': processed_onset['dev_onset']} |
                                {'info_onset': processed_onset['info_onset']} |
                                {'f_main_sig': processed_motion['f_main_sig']})

    onset_motion_features = compute_onset_motion_features(onset_motion_dataset,motion_dfname='processed_sig')

    static_features = Dataset(trunc_logic=None,exp_name=exp)
    static_features.add_dfs(processed_motion)
    static_features.add_dfs(processed_onset)
    static_features.add_df('onset_motion_features',onset_motion_features)

    static_features.save(os.path.join(MAIN_DATA_FOLDER,'cache',exp+'_features.pkl'))
    static_features.export_dataframes_to_csv(os.path.join(MAIN_DATA_FOLDER,'csv/'))

def run_interpersonnal():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    #click tempo
    exp = 'click_tempo'
    print(exp,'interpersonal synchrony')

    data = pd.read_csv(os.path.join(MAIN_DATA_FOLDER,'csv',exp+'_processed_sig.csv'))
    onset = pd.read_csv(os.path.join(MAIN_DATA_FOLDER,'csv',exp+'_onset.csv'))
    frequency = pd.read_csv(os.path.join(MAIN_DATA_FOLDER,'csv',exp+'_f_main_sig.csv'))

    dataset = Dataset({'sig_df':data,'onset_df' : onset,'frequency' : frequency})

    entrainment_static_features = compute_static_entrainment(dataset)
    dataset_res = Dataset(trunc_logic=None,exp_name=exp)
    dataset_res.add_dfs(entrainment_static_features)
    dataset_res.save(os.path.join(MAIN_DATA_FOLDER,'cache',exp+'_entrainment_features.pkl'))
    dataset_res.export_dataframes_to_csv(os.path.join(MAIN_DATA_FOLDER,'csv'))

    warnings.simplefilter(action='ignore', category=FutureWarning)
    #mask attack
    exp = 'mask_attack'
    print(exp,'interpersonal synchrony')

    data = pd.read_csv(os.path.join(MAIN_DATA_FOLDER,'csv',exp+'_processed_sig.csv'))
    onset = pd.read_csv(os.path.join(MAIN_DATA_FOLDER,'csv',exp+'_onset.csv'))
    frequency = pd.read_csv(os.path.join(MAIN_DATA_FOLDER,'csv',exp+'_f_main_sig.csv'))

    dataset = Dataset({'sig_df':data,'onset_df' : onset,'frequency' : frequency})

    entrainment_static_features = compute_static_entrainment(dataset)
    dataset_res = Dataset(trunc_logic=None,exp_name=exp)
    dataset_res.add_dfs(entrainment_static_features)
    dataset_res.save(os.path.join(MAIN_DATA_FOLDER,'cache',exp+'_entrainment_features.pkl'))
    dataset_res.export_dataframes_to_csv(os.path.join(MAIN_DATA_FOLDER,'csv'))

    warnings.simplefilter(action='ignore', category=FutureWarning)
    #change
    exp = 'change'
    print(exp,'interpersonal synchrony')

    data = pd.read_csv(os.path.join(MAIN_DATA_FOLDER,'csv',exp+'_processed_sig.csv'))
    onset = pd.read_csv(os.path.join(MAIN_DATA_FOLDER,'csv',exp+'_onset.csv'))
    frequency = pd.read_csv(os.path.join(MAIN_DATA_FOLDER,'csv',exp+'_f_main_sig.csv'))

    dataset = Dataset({'sig_df':data,'onset_df' : onset,'frequency' : frequency})

    entrainment_static_features = compute_static_entrainment(dataset)
    dataset_res = Dataset(trunc_logic=None,exp_name=exp)
    dataset_res.add_dfs(entrainment_static_features)
    dataset_res.save(os.path.join(MAIN_DATA_FOLDER,'cache',exp+'_entrainment_features.pkl'))
    dataset_res.export_dataframes_to_csv(os.path.join(MAIN_DATA_FOLDER,'csv'))

def run_main():
    run_build()
    run_personnal()
    run_interpersonnal()

if __name__ ==  '__main__':
    run_main()
