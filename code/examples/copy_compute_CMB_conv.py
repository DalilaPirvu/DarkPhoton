# To run this script, in a separate terminal type:
#### conda activate conda_env
#### python3 compute_CMB_conv.py >> output.txt

import os,sys
import itertools
import time
sys.path.append('../')
import hmvec as hm # Git clone and pip install as in readme from github.com/msyriac/hmvec
import numpy as np

# Halo model parameters
nMasses = 200#500
nZs = 10#20
nElls = 500#2401
dell = 10#75

MA = 10.**(-12.)                         # Mass of dark photon
omega0 = 30*4.1e-6                       # photon frequency today in eV


# Storage
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
dirdata = lambda mDP, om, nZs, nMs, nElls, dell: '/home/dpirvu/dark-photon/code/examples/data/MA'+str(mDP)+'_omega0'+str(om)+'_nZs'+str(nZs)+'_nMs'+str(nMs)+'_nElls'+str(nElls)+'_dell'+str(dell)+'_'

# Create HALO MODEL
zs = np.linspace(0.01,4.,nZs)             # redshifts
ms = np.geomspace(1e7,1e17,nMasses)       # masses
ks = np.geomspace(1e-4,100,1001)          # wavenumbers
rs = np.linspace(1e-5,1e3,100000)         # halo radius
ells = np.arange(0,nElls,dell)            # multipole moments
hcos = hm.HaloModel(zs, ks, ms=ms, mass_function='tinker', mdef='vir')


# Import or compute DARK PHOTON signal PS
path = dirdata(MA, omega0, nZs, nMasses, nElls, dell)+'data.npy'
isFile = os.path.isfile(path)
if not isFile:
    print('Computing new optical depth PS.')
    rcross, chicross, prob = dark_photon_conv(zs, ms)
    uL0_List = get_angular_u(ells, zs, rcross)
    data = spectrum_conv(ells, zs, ks, ms, chicross, prob, uL0_List)
    np.save(dirdata(MA, omega0, nZs, nMasses, nElls, dell)+'data', data)

else:
    print('Importing existing file.')
    powers = hcos.CMB_power_spectra()
    unlensedCL = powers['unlensed_scalar']
    cellDeltaTau = np.load(dirdata(MA, omega0, nZs, nMasses, nElls, dell)+'data.npy')[-1]

# Compute Screened CMB PS
for ellmaxout in [100, 200, 500, 1000, 1500, 2000]:
    print('Computing new screened CMB PS.')
    start = time.time()
    cells_CMBDP = get_scrCLs(ells, cellDeltaTau, unlensedCL, ellmaxout)
    np.save(dirdata(MA, omega0, nZs, nMasses, nElls, dell)+'CMBDP_ellmaxout'+str(ellmaxout), cells_CMBDP)
    end = time.time()
    print('time taken:', end-start, 'ellmaxout = ', ellmaxout)

    

