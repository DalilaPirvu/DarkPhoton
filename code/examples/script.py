# To run this script, in a separate terminal type:
#### conda activate conda_env
#### python3 compute_CMB_conv.py >> output.txt

import os,sys
import time
sys.path.append('../')
import hmvec as hm # Git clone and pip install as in readme from github.com/msyriac/hmvec
import numpy as np
from compute_power_spectra import *
from os import cpu_count
from concurrent.futures import ProcessPoolExecutor
import functools


num_workers = 40*8 # 40 cores per 1 symmetry node

ellMax = 10000 #

# Halo model parameters
nMasses = 500
nZs = 20

MA = 10.**(-12.)        # Mass of dark photon
omega0 = 30*4.1e-6      # photon frequency today in eV
aa = lambda z: 1./(1.+z)

# Storage
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
dirdata = lambda mDP, om, nZs, nMs, lMax: '/home/dpirvu/dark-photon/code/examples/data/MA'+str(mDP)+'_omega0'+str(om)+'_nZs'+str(nZs)+'_nMs'+str(nMs)+'_ellMax'+str(lMax)

# Create HALO MODEL
zs = np.linspace(0.01,4.,nZs)             # redshifts
ms = np.geomspace(1e7,1e17,nMasses)       # masses
ks = np.geomspace(1e-4,100,1001)          # wavenumbers
rs = np.linspace(1e-5,1e3,100000)         # halo radius
print('Creating Halo Model.')
hcos = hm.HaloModel(zs, ks, ms=ms, mass_function='tinker', mdef='vir')
chis = hcos.comoving_radial_distance(zs)


uell_path = dirdata(MA, omega0, nZs, nMasses, ellMax)+'_uell.npy'
if not os.path.isfile(uell_path):
    print('Computing conversion probability for halo model.')
    cs = hcos.concentration()
    Hz = hcos.h_of_z(zs)
    rvir = np.asarray([hcos.rvir(ms,zz) for zz in zs])
    rcross, prob, harea = dark_photon_conv_prob(zs, ms, rs, rvir, cs, Hz, MA, omega0)

    print('Computing new angular profile of halos.')
    # proper distance from observer to redshift bin in Mpc
    rchis = chis*aa(zs)
    partial_get_u = functools.partial(get_angular_u_ell, zs, rchis, rcross)
    chunksize = max(1, ellMax//num_workers)
    with ProcessPoolExecutor(num_workers) as executor:
        u_ell_list = list(executor.map(partial_get_u, np.arange(ellMax), chunksize=chunksize))
    np.save(dirdata(MA, omega0, nZs, nMasses, ellMax)+'_uell', u_ell_list)
else:
    print('Importing existing angular profile of halos.')
    u_ell_list = np.load(uell_path)




ang_scr_powspec_1hpath = dirdata(MA, omega0, nZs, nMasses, ellMax)+'_1hdata.npy'
if not os.path.isfile(ang_scr_powspec_1hpath):
    print('Computing new 1-halo angular PS.')
    nzm = hcos.get_nzm()
    biases = hcos.get_bh()
    data = get_scr_powspec_1h(zs, ks, ms, nzm, biases, harea, prob, u_ell_list)
    avtau, num2h, CLdTau1h = data
    np.save(dirdata(MA, omega0, nZs, nMasses, ellMax)+'_1hdata', data)
else:
    print('Importing existing 1-halo angular PS.')
    avtau, num2h, CLdTau1h = np.load(ang_scr_powspec_1hpath)




ang_scr_powspec_2hpath = dirdata(MA, omega0, nZs, nMasses, ellMax)+'_2hdata.npy'
if not os.path.isfile(ang_scr_powspec_2hpath):
    print('Computing new 2-halo angular PS.')
    PzkLin = hcos._get_matter_power(zs,ks,nonlinear=False)
    partial_2h_powspec = functools.partial(get_scr_powspec_2h, zs, ks, chis, avtau, PzkLin, num2h)
    chunksize = max(1, ellMax//num_workers)
    with ProcessPoolExecutor(num_workers) as executor:
        data = list(executor.map(partial_2h_powspec, np.arange(ellMax), chunksize=chunksize))
    CLhh, CLdTau2h = np.asarray(data)[:,0], np.asarray(data)[:,1]
    np.save(dirdata(MA, omega0, nZs, nMasses, ellMax)+'_2hdata', [CLhh, CLdTau2h])
else:
    print('Importing existing 2-halo angular PS.')
    CLhh, CLdTau2h = np.load(ang_scr_powspec_2hpath)

print('Importing CMB power spectra.')
powers = hcos.CMB_power_spectra()
unlensedCL = powers['unlensed_scalar']
cellDeltaTau = CLdTau1h + CLdTau2h

dell = 14
ellstemp = np.arange(10, ellMax, dell) # multipole moments
ells = np.asarray(np.arange(10).tolist()+ellstemp.tolist())

# Compute Screened CMB PS
for ell2max in [6000]:
    if ell2max >= ellMax:
        print('Must have ell2 >= ellMax. Recalculate screening PS to higher multipole.')
        continue
    else:
        sum_scr_powspec_path = dirdata(MA, omega0, nZs, nMasses, ellMax)+'_CMBDP_dell'+str(dell)+'_ell2max'+str(ell2max)+'.npy'
        
        if not os.path.isfile(sum_scr_powspec_path):
            print('Computing new screened CMB PS using parallel cores. Ell2max in sum=', ell2max)
            
            partial_get_CLs = functools.partial(parallelly_get_scrCLs, cellDeltaTau, unlensedCL, ell2max)
            chunksize = max(1, len(ells)//num_workers)
            
            with ProcessPoolExecutor(num_workers) as executor:
                CMBDP_powspec = list(executor.map(partial_get_CLs, ells, chunksize=chunksize))
            
            np.save(dirdata(MA, omega0, nZs, nMasses, ellMax)+'_CMBDP_dell'+str(dell)+'_ell2max'+str(ell2max), CMBDP_powspec)
        
        else:
            print('Ell2max in sum = ', ell2max, ' already exists.')
print('All Done.')
