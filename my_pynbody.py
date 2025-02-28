import pynbody.test_utils
import pylab
#from plot_2d import Datas, read_matrx
import astropy.units as u
import numpy as np
from ellipse_fit_3d import itter_tensor, shell_fit
from astropy.visualization import (MinMaxInterval, LogStretch,ImageNormalize)
import pandas as pd
#pynbody.test_utils.precache_test_data()
#snapfile = '/media/android/KESU1/EAGLE/RecalL0025N0752_snap_028.tar'
import time

class my_pynbody():
    def __init__(self, data: classmethod, eps: float = 9.01197E-4):
        
        self.mh = pynbody.snapshot.new(dm = len(data.dm['mass']), star = len(data.star['mass']), gas = len(data.gas['mass']),
                                        order='star,gas,dm')
        
        self.mh.s['eps']    = pynbody.array.SimArray(eps*np.ones(len(data.star['mass'])), units='Mpc')
        self.mh.s['pos']    = pynbody.array.SimArray(data.star['coords'], units='Mpc')
        self.mh.s['smooth'] = pynbody.array.SimArray(data.star['sml'], units='Mpc')/2.0
        self.mh.s['mass']   = pynbody.array.SimArray(data.star['mass'], units='Msol')
        self.mh.s['vel']    = pynbody.array.SimArray(data.star['velocity'], units='km s^-1')

        self.mh.g['pos']    = pynbody.array.SimArray(data.gas['coords'], units='Mpc')
        self.mh.g['smooth'] = pynbody.array.SimArray(data.gas['sml'], units='Mpc')
        self.mh.g['mass']   = pynbody.array.SimArray(data.gas['mass'], units='Msol')
        self.mh.g['vel']    = pynbody.array.SimArray(data.gas['velocity'], units='km s^-1')
        self.mh.g['eps']    = pynbody.array.SimArray(eps*np.ones(len(data.gas['mass'])), units='Mpc')

        self.mh.dm['pos']    = pynbody.array.SimArray(data.dm['coords'], units='Mpc')
        self.mh.dm['mass']   = pynbody.array.SimArray(data.dm['mass'], units='Msol')
        self.mh.dm['vel']    = pynbody.array.SimArray(data.dm['velocity'], units='km s^-1')
        self.mh.dm['eps']    = pynbody.array.SimArray(eps*np.ones(len(data.dm['mass'])), units='Mpc')
        
        pynbody.analysis.center(self.mh)

    def plot_ima(self, key : str, orient:str='z', save:str=None, **kwargs):
        save_data = self.mh

        if orient == 'x':
            save_data.rotate_y(90)
        if orient == 'y':
            save_data.rotate_x(90)
        save_data.physical_units()

        if key == 's':
            d = save_data.s
        elif key == 'g':
            d = save_data.mh.g
        else:
            d = save_data.mh.dm
        image_values = pynbody.plot.sph.image(d, **kwargs)
        #pylab.show()
        #self.mh = save_data
        if orient == 'x':
            save_data.rotate_y(-90)
        if orient == 'y':
            save_data.rotate_x(-90)
        return image_values

    def plot_profile(self, key : str = 's', type:str ='equaln', 
                     nbins:int = 100, mmax:float = 50.0, mmin:float = 0.002):

        if key == 's':
            d = self.mh.s
        elif key == 'g':
            d = self.mh.g
        else:
            d = self.mh.dm
    
        profile = pynbody.analysis.Profile(d, min=mmin, max=mmax,type=type, nbins=nbins, ndim=2)
        
        rr = profile['rbins']
        p  = profile['density'].in_units('Msol pc^-2')
        n = profile['n']

        pe = np.sqrt(p/n) #profile['mass_disp'].in_units('Msol ')/profile['mass'].in_units('Msol ') * profile['density'].in_units('Msol pc^-2')
        
        ind = np.where(np.log10(p) >= 0)
        
        return rr[ind], p[ind], pe[ind]
    
if __name__ == '__main__':
    snapfile = '/media/android/KESU1/EAGLE/RecalL0025N0752_snap_028/RecalL0025N0752/snapshot_028_z000p000/snap_028_z000p000.0.hdf5'
    tab = pd.read_csv('RecalL0025N0752.csv')

    #snapfile = 'RefL0025N0376/snapshot_028_z000p000/snap_028_z000p000.0.hdf5'
    #tab = pd.read_csv('RefL0025N0376_cat1.csv')


    # cMpc
    IDs = tab['GalaxyID'].to_numpy()
    xcs = tab['CentreOfPotential_x'].to_numpy() #cMpc
    ycs = tab['CentreOfPotential_y'].to_numpy() #cMpc
    zcs = tab['CentreOfPotential_z'].to_numpy() #cMpc
    kcos = tab['KappaCoRot'].to_numpy()
    Ms  = tab['Mass_Star'].to_numpy()
    #Ms  = tab['MassType_Star'].to_numpy()
    
    tt  = 0
    for id, xc, yc, zc, M, kco in sorted(zip(IDs[tt:], xcs[tt:], ycs[tt:], zcs[tt:], Ms[tt:], kcos[tt:])):

        
        data = Datas(snapfile, xc, yc, zc, 0.05)
        data.main_only()
        X = data.star['coords'][:,0]
        Y = data.star['coords'][:,1]
        Z = data.star['coords'][:,2]
        mass = data.star['mass']  

        #A, B, C, Q1, eps, T = itter_tensor(X, Y, Z, mass)
        #data.coord_rotate(Q1)
        #Q1 = read_matrx('matrxs/%s.txt' %id)

        
        Q = shell_fit(data.star['coords'], mass, data.star['velocity'], ndots=len(mass)//100, method='numb', step='line', rmin=0.0, rmax=0.03)
        data.coord_rotate(Q)
        
        
        main_halo = pynbody.snapshot.new(dm = len(data.dm['mass']), star = len(data.star['mass']), gas = len(data.gas['mass']),
                                        order='star,gas,dm')
        #for key in data.star:
        main_halo.s['eps']    = pynbody.array.SimArray(9.01197E-4*np.ones(len(data.star['mass'])), units='Mpc')
        main_halo.s['pos']    = pynbody.array.SimArray(data.star['coords'], units='Mpc')
        main_halo.s['smooth'] = pynbody.array.SimArray(data.star['sml'], units='Mpc')
        main_halo.s['mass']   = pynbody.array.SimArray(data.star['mass'], units='Msol')
        main_halo.s['vel']    = pynbody.array.SimArray(data.star['velocity'], units='km s^-1')

        main_halo.g['eps'] = pynbody.array.SimArray(9.01197E-4*np.ones(len(data.gas['mass'])), units='Mpc')

        main_halo.g['pos'] = pynbody.array.SimArray(data.gas['coords'], units='Mpc')
        main_halo.g['smooth'] = pynbody.array.SimArray(data.gas['sml'], units='Mpc')/2.
        main_halo.g['mass']   = pynbody.array.SimArray(data.gas['mass'], units='Msol')
        main_halo.g['vel']    = pynbody.array.SimArray(data.gas['velocity'], units='km s^-1')

        main_halo.dm['pos'] = pynbody.array.SimArray(data.dm['coords'], units='Mpc')
        main_halo.dm['mass']   = pynbody.array.SimArray(data.dm['mass'], units='Msol')
        main_halo.dm['vel']    = pynbody.array.SimArray(data.dm['velocity'], units='km s^-1')
        main_halo.dm['eps'] = pynbody.array.SimArray(9.01197E-4*np.ones(len(data.dm['mass'])), units='Mpc')

        #with pynbody.analysis.center(main_halo):
        if True:

        #print(main_halo.s['pos'])
        #pynbody.analysis.center(main_halo)
            main_halo.physical_units()
        #
        #pynbody.analysis.sideon(main_halo)
            #main_halo.rotate_x(90)
            image_values = pynbody.plot.sph.image(main_halo.g, width=100, kernel='WendlandC2Kernel', units="Msol pc^-2", cmap="bone"
                                              , resolution=1000, return_array=True, noplot=False)
            pylab.show()
        #from matplotlib import pyplot as plt
        #plt.figure()

        #norm = ImageNormalize(image_values, interval=MinMaxInterval(), stretch=LogStretch(100_000))

        #plt.imshow(image_values, norm=norm, cmap="twilight")
        #plt.show()
        #image_values = pynbody.plot.image(main_halo.dm, width=100,  units="Msol kpc^-2", cmap="twilight")
            #from astropy.io import fits
            #fits.writeto( 'pynbody_test.fits',image_values, overwrite=True)
            #import subprocess as sp
            #sp.call('ds9 pynbody_test.fits', shell=True)
        #image_values = pynbody.plot.image(main_halo.g, width=100 , units="Msol kpc^-2", cmap="twilight")
        #pylab.show()
        #print('ngas = %e, ndark = %e, nstar = %e\n'%(len(main_halo.gas),len(main_halo.dark),len(main_halo.star)))


            star_profile = pynbody.analysis.Profile(main_halo.s, min=0.2, max=50,type='equaln', nbins=100, ndim=2)
            star_profile1 = pynbody.analysis.Profile(main_halo.s, min=0.2, max=50,type='lin', nbins=100, ndim=2)

            from photutils.profiles import RadialProfile

            edge_radii = np.logspace(0, np.log10(500), 100)

            rp = RadialProfile(image_values, [1000//2, 500], edge_radii, mask=None)

        
        #pynbody.plot.profile.rotation_curve(main_halo, parts=True, rmin=0.1, rmax=50.0)
        #pylab.legend()
        #pylab.show()
            yerr = star_profile['mass_disp'].in_units('Msol ')/star_profile['mass'].in_units('Msol ') * star_profile['density'].in_units('Msol pc^-2')
            #print(yerr) 
            pylab.errorbar(star_profile['rbins'], star_profile['density'].in_units('Msol pc^-2'), yerr=yerr, fmt='-o', color='r', linewidth=1, label='const num')
            #pylab.plot(star_profile1['rbins'], np.log10(star_profile1['density'].in_units('Msol pc^-2')), '-b', linewidth=1, label='linear')
            #pylab.plot(rp.radius/10, np.log10(rp.profile), '-g', linewidth=1, label='Azimutal profile')
            pylab.semilogy()
            pylab.grid()
            pylab.ylabel(r'$\Sigma \quad M_{\odot}/pc^2$', fontsize=14)
            pylab.xlabel('r, kpc', fontsize=14)
            pylab.legend(fontsize=14)
            pylab.show()
            continue
            p_dm = pynbody.analysis.profile.Profile(main_halo.dm, min=.05, max=50, type = 'log')
            p = pynbody.analysis.profile.Profile(main_halo.s, min=.05, max=50, type = 'log')

            p_gas = pynbody.analysis.profile.Profile(main_halo.gas, min=.05, max=50, type = 'log')

            p_all = pynbody.analysis.profile.Profile(main_halo, min=.05, max=50, type = 'log')
        
            for prof, name in zip([p_all, p_dm, p, p_gas],['total', 'dm', 'stars', 'gas']):
                #print(name)
                pylab.plot(prof['rbins'], prof['v_circ'], label=name)
            pylab.xlabel(r'$r, kpc$')
            pylab.ylabel(r'$v \, km/s$')
            pylab.legend()
            pylab.show()