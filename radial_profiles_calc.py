import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic

def const_width_bins(r : np.ndarray, nbins : int = 20, 
                method :str ='line', verbous : bool = False) -> np.ndarray:

    assert (method == 'line') or (method == 'log'), 'No such method "%s". Use "line" or "log". Exit...' %method
    assert len(r) >= nbins, 'Too few dots for %s bins. Exit...' %nbins

        
    xmin = r.min()
    xmax = r.max()

    if method == 'line':
        bins_fixed_width = np.linspace(xmin, xmax,nbins+1)

    if method == 'log':
        bins_fixed_width = np.logspace(np.log10(xmin), np.log10(xmax),nbins+1)

    if verbous: 
        print(bins_fixed_width, bins_fixed_width[1] - bins_fixed_width[0])
    return bins_fixed_width

def const_numb_bins(r : np.ndarray, nbins : int = 20, ndots : int = None,
                     verbous : bool = False) -> np.ndarray:
    assert len(r) >= nbins, 'Too few dots for %s bins. Exit...' %nbins

    r_sorted = np.sort(r)
    if ndots is None:
        split_indices = np.arange(0,len(r_sorted),step=np.int64(np.floor(len(r_sorted)/nbins)))
    else:
        split_indices = np.arange(0,len(r_sorted),step=ndots)

    bins_fixed_number = r_sorted[split_indices]

    if verbous:
        print(bins_fixed_number, len(np.where((r_sorted >= bins_fixed_number[0]) * (r_sorted < bins_fixed_number[1]))[0]))
    return bins_fixed_number

def shell_density_profile(r :np.ndarray, pvalue : np.ndarray, nbins: int = 20, average='mean',
            bin_method :str ='width', weights=None, step='line', 
            mass: bool = True, verbous: bool  ='False', dim: int=3):
    '''
        Plot phisycal value 1d density plot with conts with or particles number. 
    '''

    assert (bin_method == 'width') or (bin_method == 'numb'), 'No such bin_method "%s". Use "width" or "numb". Exit...' %bin_method    
    assert len(r) == len(pvalue), 'r and pvalue mast have the same lenght'
    assert (average == 'mean') or (average == 'median'), 'No such average method "%s". Use "mean", "median" or "weighted"' %average 

    if bin_method == 'width':
        bins = const_width_bins(r, nbins=nbins, method=step)
    if bin_method == 'numb':
        bins = const_numb_bins(r, nbins=nbins)
    
    if mass:
        mass_binned,_,_ = binned_statistic(r, pvalue,bins=bins,statistic='sum')
        if (bin_method == 'width') and (step == 'log'):
            pass
        else:
            if dim == 3:
                shell_volumes = (4./3.) * np.pi * (bins[1:]**3-bins[:-1]**3)
            elif dim == 2:
                shell_volumes = np.pi * (bins[1:]**2-bins[:-1]**2)

        profile = (mass_binned/shell_volumes)
        

    else:    
        profile,_,_ = binned_statistic(r, pvalue,bins=bins,statistic=average)
    
    centres_fixed = (bins[:-1]+ bins[1:])/2.

    if verbous:
        plt.figure()
        plt.plot(centres_fixed, np.log10(profile), 'o-')
        plt.show()

    return centres_fixed, profile









if __name__ == '__main__':
    r = np.arange(1, 1000)
    a = const_width_bins(r, method='line', verbous=True)
    b = const_numb_bins(r, verbous=True)

    plt.figure()
    plt.plot(a, b, 'o')
    plt.show()