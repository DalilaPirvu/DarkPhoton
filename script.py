# To run this script, in a separate terminal type:
#### conda activate conda_env
#### python3 script.py >> ./data/output.txt

import os,sys
sys.path.append('../')
sys.path.append('./hmvec-master/')
import hmvec as hm
from compute_power_spectra import *
from params import *
from plotting import *
import random

import functools
from concurrent.futures import ProcessPoolExecutor

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# Select electron profile
conv_gas = True
conv_NFW = False

# Select which optical depth PS to compute
compute_dark = False
compute_thomson = False
compute_cross = False

# Select which CMB isotropic screening to compute
compute_screening = False
screening_compute_dark = False
screening_compute_thomson = False
screening_compute_cross = False

# Select DP mass
maind=0

# If parallelized 
num_workers = 4

zthr  = 6.
zreio = 6.

if conv_gas:
    MA = dictKey_gas[maind]
    zMin, zMax, rMin, rMax = chooseModel(MA, modelParams_gas)
    name = 'battagliaAGN'
    rscale = False

elif conv_NFW:
    MA = dictKey_NFW[maind]
    zMin, zMax, rMin, rMax = chooseModel(MA, modelParams_NFW)
    rscale = True

print('DARK PHOTON MASS:', MA)
print('HALO MODEL:', zMin, zMax, rMin, rMax)

zMax = min(zthr, zMax)
nZs  = 50
ms  = np.geomspace(1e11,1e17,100)                  # masses
zs  = np.linspace(zMin, zMax,nZs)                  # redshifts
rs  = np.linspace(rMin, rMax,1000000)              # halo radius
ks  = np.geomspace(1e-4,1e3, 10001)                # wavenumbers

ellMax = 9600
ells      = np.arange(ellMax)
chunksize = max(1, len(ells)//num_workers)

# Halo Model
hcos = hm.HaloModel(zs, ks, ms=ms, mass_function='tinker', mdef='vir')
#gas = hcos.add_battaglia_profile("y", family="AGN", xmax=2, nxs=30000)

chis   = hcos.comoving_radial_distance(zs)
rvirs   = np.asarray([hcos.rvir(ms,zz) for zz in zs])
cs     = hcos.concentration()
Hz     = hcos.h_of_z(zs)
nzm    = hcos.get_nzm()
biases = hcos.get_bh()
deltav = hcos.deltav(zs)
rhocritz = hcos.rho_critical_z(zs)

path_params = np.asarray([MA, nZs, zMin, zMax, ellMax, rscale])
path_params_thom = np.asarray([nZs, zMin, zreio, ellMax, rscale])

# Milky Way stuff: zMW=0.
HMW  = hcos.h_of_z(0.)
rsMW = np.geomspace(1e-10,10,1000000)  # halo radius
deltavMW = hcos.deltav(0.)[0]
rhocritzMW = hcos.rho_critical_z(0.)

if compute_dark:
    print('Computing Milky Way probability.')
    if conv_gas:
        rcrossMW, probMW = dark_photon_conv_prob_MilkyWay_gas(mMWvir, rMWvir, rhocritzMW, deltavMW, csMW, HMW, rEarth, rsMW, MA, name=name)
        np.save(MW_path_gas(*path_params), [rcrossMW, probMW])
    elif conv_NFW:
        rcrossMW, probMW = dark_photon_conv_prob_MilkyWay_NFW(mMWvir, rMWvir, csMW, HMW, rEarth, rsMW, MA)
        np.save(MW_path(*path_params), [rcrossMW, probMW])

    print('Computing crossing radii.')
    dvols = get_volume_conv(chis, Hz)
    if conv_gas:
        rcross = get_rcross_per_halo_gas(zs, ms, rs, rvirs, rhocritz, deltav, cs, MA, name=name)
        np.save(rcross_path_gas(*path_params), rcross)
    elif conv_NFW:
        rcross = get_rcross_per_halo_NFW(zs, ms, rs, rvirs, cs, MA)
        np.save(rcross_path(*path_params), rcross)

    print('Computing probability to convert.')
    ucosth, angs = get_halo_skyprofile(zs, chis, rcross)
    if conv_gas:
        prob = dark_photon_conv_prob_gas(zs, ms, rs, rvirs, rhocritz, deltav, cs, MA, rcross, name=name)
        np.save(prob_path_gas(*path_params), prob)
    elif conv_NFW:
        prob = dark_photon_conv_prob_NFW(zs, ms, rs, rvirs, cs, MA, rcross, rscale=rscale)
        np.save(prob_path(*path_params), prob)

    print('Computing optical depth.')
    u00 = get_u00(angs, ucosth)
    avtau, dtaudz = get_avtau(zs, ms, nzm, prob, dvols, u00)
    if conv_gas:
        np.save(tau_path_gas(*path_params), [avtau, dtaudz])
    elif conv_NFW:
        np.save(tau_path(*path_params), [avtau, dtaudz])

    print('Computing multipole expansion of angular probability u.')
    partial_u = functools.partial(get_uell0, angs, ucosth)
    with ProcessPoolExecutor(num_workers) as executor:
        uell0 = list(executor.map(partial_u, ells, chunksize=chunksize))
    if conv_gas:
        np.save(uell_path_gas(*path_params), uell0)
    elif conv_NFW:
        np.save(uell_path(*path_params), uell0)

    print('Computing 1-halo angular PS.')
    Cell1Halo  = get_Celldtaudtau_1h(zs, ms, ks, nzm, dvols, prob, uell0, ells)
    Cell1Halo[0] = 0.
    if conv_gas:
        np.save(data1h_path_gas(*path_params), Cell1Halo)
    elif conv_NFW:
        np.save(data1h_path(*path_params), Cell1Halo)

    PzkLin     = hcos._get_matter_power(zs, ks, nonlinear=False)
    PzkLinz1z2 = np.asarray([[(PzkLin[z1,:]*PzkLin[z2,:])**0.5 for z2 in range(nZs)] for z1 in range(nZs)])
    dtaudz_ell = get_dtauell(ms, nzm, prob, dvols, biases, uell0)

    print('Computing 2-halo angular PS.')
    partial_get_2h = functools.partial(get_Celldtaudtau_2h, zs, ms, ks, chis, PzkLinz1z2, dtaudz_ell)
    with ProcessPoolExecutor(num_workers) as executor:
        Cell2Halo  = list(executor.map(partial_get_2h, ells, chunksize=chunksize))
    Cell2Halo[0] = 0.
    print('Done Cell2Halo:', Cell2Halo)
    if conv_gas:
        np.save(data2h_path_gas(*path_params), Cell2Halo)
    elif conv_NFW:
        np.save(data2h_path(*path_params), Cell2Halo)

elif compute_thomson:
    print('Compute probability to Thomson scatter and angular dependence.')
    thom_probell = get_thomsontau_per_halo(zs, ms, ellMax, chis, rvirs, rhocritz, deltav, cs, MA, name=name)
    np.save(thmonopell_path_gas(*path_params_thom), thom_probell)

    dvols = get_volume_conv(chis, Hz)
    dthomdz_ell = get_dthomell(ms, nzm, thom_probell, biases, dvols)

    print('Computing Thomson x Thomson 1-halo angular PS.')
    thomthom_Cell1Halo = get_Celldthomdthom_1h(zs, ms, ks, nzm, dvols, thom_probell, ells)
    thomthom_Cell1Halo[0] = 0.
    np.save(data1h_thomthom_path_gas(*path_params_thom), thomthom_Cell1Halo)

    PzkLin     = hcos._get_matter_power(zs, ks, nonlinear=False)
    PzkLinz1z2 = np.asarray([[(PzkLin[z1,:]*PzkLin[z2,:])**0.5 for z2 in range(nZs)] for z1 in range(nZs)])

    print('Computing Thomson x Thomson 2-halo angular PS.')
    partial_get_2h = functools.partial(get_Celldthomdthom_2h, zs, ms, ks, chis, PzkLinz1z2, dthomdz_ell)
    with ProcessPoolExecutor(num_workers) as executor:
        thomthom_Cell2Halo  = list(executor.map(partial_get_2h, ells, chunksize=chunksize))
    thomthom_Cell2Halo[0] = 0.
    print('Done Cell2Halo:', thomthom_Cell2Halo)
    np.save(data2h_thomthom_path_gas(*path_params_thom), thomthom_Cell2Halo)

elif compute_cross:
    print('Importing data.')
    if conv_gas:
        prob = np.load(prob_path_gas(*path_params)+'.npy')
        uell0 = np.load(uell_path_gas(*path_params)+'.npy')
    elif conv_NFW:
        prob = np.load(prob_path(*path_params)+'.npy')
        uell0 = np.load(uell_path(*path_params)+'.npy')

    dvols = get_volume_conv(chis, Hz)
    thom_probell = np.load(thmonopell_path_gas(*path_params_thom)+'.npy')

    dthomdz_ell = get_dthomell(ms, nzm, thom_probell, biases, dvols)
    dtaudz_ell = get_dtauell(ms, nzm, prob, dvols, biases, uell0)

    print('Computing Thomson x Dark Photon 1-halo angular PS.')
    thom_Cell1Halo = get_Celldtaudthom_1h(zs, ms, ks, nzm, dvols, thom_probell, prob, uell0, ells)
    thom_Cell1Halo[0] = 0.
    np.save(data1h_thom_path_gas(*path_params), thom_Cell1Halo)

    PzkLin     = hcos._get_matter_power(zs, ks, nonlinear=False)
    PzkLinz1z2 = np.asarray([[(PzkLin[z1,:]*PzkLin[z2,:])**0.5 for z2 in range(nZs)] for z1 in range(nZs)])

    print('Computing Thomson x Dark Photon 2-halo angular PS.')
    partial_get_2h = functools.partial(get_Celldtaudthom_2h, zs, ms, ks, chis, PzkLinz1z2, dthomdz_ell, dtaudz_ell)
    with ProcessPoolExecutor(num_workers) as executor:
        thom_Cell2Halo  = list(executor.map(partial_get_2h, ells, chunksize=chunksize))
    thom_Cell2Halo[0] = 0.
    print('Done Cell2Halo:', thom_Cell2Halo)
    np.save(data2h_thom_path_gas(*path_params), thom_Cell2Halo)

else:
    print('Importing data.')
    if conv_gas:
        Cell1Halo = np.load(data1h_path_gas(*path_params)+'.npy')
        Cell2Halo = np.load(data2h_path_gas(*path_params)+'.npy')
        Celltautau = Cell1Halo + Cell2Halo

    elif conv_NFW:
        Cell1Halo = np.load(data1h_path(*path_params)+'.npy')
        Cell2Halo = np.load(data2h_path(*path_params)+'.npy')
        Celltautau = Cell1Halo + Cell2Halo

    thom_Cell1Halo = np.load(data1h_thom_path_gas(*path_params)+'.npy')
    thom_Cell2Halo = np.load(data2h_thom_path_gas(*path_params)+'.npy')
    Cellthomtau = thom_Cell1Halo + thom_Cell2Halo

    thomthom_Cell1Halo = np.load(data1h_thomthom_path_gas(*path_params_thom)+'.npy')
    thomthom_Cell2Halo = np.load(data2h_thomthom_path_gas(*path_params_thom)+'.npy')
    Cellthomthom = thomthom_Cell1Halo + thomthom_Cell2Halo


print('Importing CMB power spectra and adding temperature monopole.')
CMB_ps        = hcos.CMB_power_spectra()
unlenCMB      = CMB_ps['unlensed_scalar']
unlenCMB      = unlenCMB[:ellMax, :]
unlenCMB[0,0] = TCMB**2.
lensedCMB     = CMB_ps['lensed_scalar']
lensedCMB     = lensedCMB[:ellMax, :]
lensedCMB[0,0]= TCMB**2.

if compute_screening:
    l0Max, l1Max, l2Max = ellMax, ellMax, ellMax

    if screening_compute_dark:
        print('Computing new CMB dark screening PS from monopole.')
        llist, scrTT0 = get_Tmonopole_screeningPS(l0Max, l2Max, CMB=unlenCMB, DPCl=Celltautau)
        if conv_gas:
            np.save(monoplscrPS_path_gas(*path_params), [llist, scrTT0])
        elif conv_NFW:
            np.save(monoplscrPS_path(*path_params), [llist, scrTT0])

        print('Computing new CMB dark screening PS.')
        llist, scrTT, scrEE, scrBB, scrTE = get_scrCLs(l0Max, l1Max, l2Max, CMB=unlenCMB, DPCl=Celltautau)
        if conv_gas:
            np.save(screeningPS_path_gas(*path_params), [llist, scrTT, scrEE, scrBB, scrTE])
        elif conv_NFW:
            np.save(screeningPS_path(*path_params), [llist, scrTT, scrEE, scrBB, scrTE])

    elif screening_compute_thomson:
        print('Computing new CMB thomson screening PS.')
        llist, scrTT, scrEE, scrBB, scrTE = get_scrCLs(l0Max, l1Max, l2Max, CMB=unlenCMB, DPCl=Cellthomthom)
        np.save(screeningPS_thomthom_path_gas(*path_params_thom), [llist, scrTT, scrEE, scrBB, scrTE])

    elif screening_compute_cross:
        print('Computing new CMB thomson cross dark screening PS.')
        llist, scrTT, scrEE, scrBB, scrTE = get_scrCLs(l0Max, l1Max, l2Max, CMB=unlenCMB, DPCl=Cellthomtau)
        np.save(screeningPS_thom_path_gas(*path_params), [llist, scrTT, scrEE, scrBB, scrTE])

for eind, (expname, experiment) in enumerate(zip(['Planck', 'CMBS4', 'CMBHD'], [Planck, CMBS4, CMBHD])):
    if expname == 'Planck':
        l0Max, l1Max, l2Max = 5000, 5000, 5000
        ellss = np.arange(5000)
    else:
        l0Max, l1Max, l2Max = 9000, 9000, 9000
        ellss = np.arange(9000)

#    Cellthomtau = Cellthomtau / ((4.*np.pi) / (2.*ells+1.))**0.5

    print('Computing bispectrum for ', expname)
    ILCnoise = np.load(ILCnoisePS_path_gas(expname, zreio))
    NTTdscdsc = ILCnoise[maind, 0, ellss]

    BB_ILCnoise = np.load(BB_ILCnoisePS_path_gas(expname, zreio))
    NTTscsc = BB_ILCnoise[0, ellss]
    NEEscsc = BB_ILCnoise[1, ellss]
    NBBscsc = BB_ILCnoise[2, ellss]

    ClTT = lensedCMB[ellss, 0]
    ClEE = lensedCMB[ellss, 1]

    bisp = bispectrum_Tdsc_Tsc_Tsc(l0Max, l1Max, l2Max, TCMB, ClTT, Cellthomtau, NTTdscdsc, NTTscsc)
    np.save(bispec_Tdsc_Tsc_Tsc(*path_params, expname), bisp)

    bisp = bispectrum_Tdsc_Esc_Bsc(l0Max, l1Max, l2Max, TCMB, ClEE, Cellthomtau, NTTdscdsc, NEEscsc, NBBscsc)
    np.save(bispec_Tdsc_Esc_Bsc(*path_params, expname), bisp)

    print(expname, 'Done!')

print('All Done.')

