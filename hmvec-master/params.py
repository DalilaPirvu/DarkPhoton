import numpy as np

############# Define constants ################
# From hmvec:
# proper radius r is always in Mpc
# comoving momentum k is always in Mpc-1
# All masses m are in Msolar
# rho densities are in Msolar/Mpc^3
# No h units anywhere

TFIRAS   = 2.725
TCMB     = 2.725*1e6
cmMpc    = 3.2407792896e-25    # Mpc/cm            # how many Mpc in a cm
eVinvCm  = 1.97*1e-5  #1.2398419e-4        # cm/eV^-1          # how many cm in a eV^-1
mpcEVinv = 1./(cmMpc*eVinvCm)  # eV^-1/Mpc         # how many eV^-1 in a Mpc

mMWvir  = 1.3*1e12
rMWvir  = 287*1e-3
rEarth  = 8*1e-3
csMW = 10.72 # log10 c = 1.025 - 0.097 log10 (M / (10^12 /h solar masses)) # h = 0.68

msun   = 1.9891e30     # kg               # Sun mass
mprot  = 1.67262e-27   # kg               # Proton mass
m2eV   = 1.4e-21       # eV^2             # conversion factor for plasma mass (eq. (2) in Caputo et al; PRL)
ombh2  = 0.02225                 # Physical baryon density parameter Ωb h2
omch2  = 0.1198                  # Physical dark matter density parameter Ωc h2
conv   = m2eV*(ombh2/omch2)*(msun/mprot)*(cmMpc)**3.

thomson = 0.6652*1e-24
conv2 = thomson*(ombh2/omch2)*(msun/mprot)*(cmMpc)**2.

ev2Joule = lambda ev: 1.602176565*1e-19*ev
K2eV = lambda K: 8.61732814974493*1e-5*K
cligth = 299792458.0 #m/s
hplanck = 6.62607015*1e-34 # m2 kg / s
kboltzmann = 1.380649*1e-23 # m2 kg s-2 K-1

frq = lambda nu: 100. * nu * cligth

BBf = lambda frq: 1e26/1e6*(2.*frq**3.*hplanck)/cligth**2. / (np.exp(hplanck*frq/kboltzmann/TFIRAS) - 1.)
BBω = lambda omg: (omg**3.)/(2.*np.pi**2.) / (np.exp(omg/K2eV(TFIRAS)) - 1.)

aa = lambda z: 1./(1.+z)

arcmin2rad = lambda arcm: arcm/60. * np.pi/180.
ghztoev    = lambda GHz: 4.13566553853809E-06 * GHz

############# Halo models ############# 

dictKey_NFW = np.asarray([1.60e-13, 1.80e-13, 2.00e-13, 2.50e-13, 3.00e-13, 3.50e-13,
                          4.00e-13, 4.50e-13, 5.00e-13, 5.50e-13, 6.00e-13, 7.00e-13,
                          8.00e-13, 9.00e-13, 1.00e-12, 1.50e-12, 2.00e-12, 2.86e-12,
                          3.50e-12, 4.00e-12, 5.00e-12, 6.00e-12, 7.00e-12, 7.50e-12])

modelParams_NFW = {1.6e-13: np.asarray([0.01,  0.1,  1e-2,  5e1]),\
                   1.8e-13: np.asarray([0.01,  0.25, 1e-2,  5e1]),\
                   2e-13:   np.asarray([0.01,  0.35, 1e-2,  5e1]),\
                   2.5e-13: np.asarray([0.01,  0.7,  1e-2,  5e1]),\
                   3e-13:   np.asarray([0.01,  0.9,  1e-2,  5e1]),\
                   3.5e-13: np.asarray([0.01,  1.25, 1e-2,  1e1]),\
                   4e-13:   np.asarray([0.01,  1.25, 1e-2,  1e1]),\
                   4.5e-13: np.asarray([0.01,  1.6,  5e-3,  1e1]),\
                   5e-13:   np.asarray([0.01,  1.6,  5e-3,  1e1]),\
                   5.5e-13: np.asarray([0.01,  1.9,  5e-3,  1e1]),\

                   6e-13:   np.asarray([0.01,  1.9,  5e-3,  1e1]),\
                   7e-13:   np.asarray([0.01,  2.2,  1e-3,  1e1]),\
                   8e-13:   np.asarray([0.01,  2.5,  1e-3,  1e1]),\
                   9e-13:   np.asarray([0.01,  2.8,  1e-3,  1e1]),\
                   1e-12:   np.asarray([0.01,  3.,   1e-3,  1e1]),\
                   1.5e-12: np.asarray([0.01,  5.,   1e-3,  1e1]),\
                   2e-12:   np.asarray([0.01,  5.,   1e-3,  1e1]),\
                   2.86e-12:np.asarray([0.01,  7.,   1e-3,  10.]),\
                   3.5e-12: np.asarray([0.01,  8.5,  1e-7,  1e1]),\
                   4e-12:   np.asarray([0.01,  8.5,  1e-7,  1e1]),\

                   5e-12:   np.asarray([0.01,  10.,  1e-7,  1e1]),\
                   6e-12:   np.asarray([0.01,  10.,  1e-7,  1e1]),\
                   7e-12:   np.asarray([0.01,  10.,  1e-7,  1e1]),\
                   7.5e-12: np.asarray([0.01,  10.,  1e-7,  1e1])}
#                   8.15e-12:np.asarray([0.01,  10.,  1e-7,  1e1]),\ # these give nans
#                   9e-12:   np.asarray([0.01,  10.,  1e-7,  1e1]),\
#                   1e-11:   np.asarray([0.01,  10.,  1e-7,  1e1]),\
#                   2e-11:   np.asarray([0.01,  10.,  1e-7,  1.]),\
#                   3e-11:   np.asarray([0.01,  10.,  1e-16,  1.]),\
#                   4e-11:   np.asarray([0.01,  10.,  1e-16,  1.]),\

#                   5e-11:   np.asarray([0.01,  10.,  1e-16,  1.]),\
#                   6e-11:   np.asarray([0.01,  10.,  1e-16,  1.]),\
#                   6.5e-11: np.asarray([0.01,  10.,  1e-16,  1.]),\
#                   7e-11:   np.asarray([0.01,  10.,  1e-16,  1.]),\
#                   7.5e-11: np.asarray([0.01,  10.,  1e-16,  1.]),\
#                   8e-11:   np.asarray([0.01,  10.,  1e-16,  1.]),\
#                   9e-11:   np.asarray([0.01,  10.,  1e-16,  1.]),\
#                   1e-10:   np.asarray([0.01,  10.,  1e-16,  1.])}


dictKey_gas = np.asarray([1.2e-13, 1.4e-13, 1.6e-13, 1.7e-13, 1.8e-13, 1.9e-13, 2e-13, 2.3e-13, 2.7e-13, 3e-13, \
                          3.3e-13, 3.7e-13, 4e-13, 4.3e-13, 4.7e-13, 5e-13, 5.3e-13, 5.7e-13, 6e-13, 6.3e-13, \
                          6.7e-13, 7e-13, 7.3e-13, 7.7e-13, 8e-13, 8.3e-13, 8.7e-13, 9e-13, 9.3e-13, 9.7e-13, \
                          1e-12, 1.3e-12, 1.7e-12, 2e-12, 3e-12, 4e-12, 5e-12, 6e-12, 7e-12, 8e-12, \
                          9e-12, 1e-11, 1.5e-11, 2e-11])#, 2.5e-11, 3e-11, 3.5e-11, 4e-11, 4.5e-11, 5e-11])

modelParams_gas = {1.2e-13: np.asarray([0.01, 1.,  1e-16, 1e2]),\
                   1.4e-13: np.asarray([0.01, 1.,  1e-16, 1e2]),\
                   1.6e-13: np.asarray([0.01, 1.,  1e-16, 1e2]),\
                   1.7e-13: np.asarray([0.01, 2.,  1e-16, 1e2]),\
                   1.8e-13: np.asarray([0.01, 2.,  1e-16, 1e2]),\
                   1.9e-13: np.asarray([0.01, 2.,  1e-16, 1e2]),\
                   2e-13:   np.asarray([0.01, 2.,  1e-16, 1e2]),\
                   2.3e-13: np.asarray([0.01, 2.,  1e-16, 1e2]),\
                   2.7e-13: np.asarray([0.01, 3.,  1e-16, 1e2]),\
                   3e-13:   np.asarray([0.01, 3.,  1e-16, 1e2]),\

                   3.3e-13: np.asarray([0.01, 3.,  1e-16, 1e2]),\
                   3.7e-13: np.asarray([0.01, 3.,  1e-16, 1e2]),\
                   4e-13:   np.asarray([0.01, 4.,  1e-16, 5e1]),\
                   4.3e-13: np.asarray([0.01, 4.,  1e-16, 5e1]),\
                   4.7e-13: np.asarray([0.01, 5.,  1e-16, 5e1]),\
                   5e-13:   np.asarray([0.01, 5.,  1e-16, 5e1]),\
                   5.3e-13: np.asarray([0.01, 5.,  1e-16, 5e1]),\
                   5.7e-13: np.asarray([0.01, 6.,  1e-16, 5e1]),\
                   6e-13:   np.asarray([0.01, 6.,  1e-16, 5e1]),\
                   6.3e-13: np.asarray([0.01, 6.,  1e-16, 5e1]),\

                   6.7e-13: np.asarray([0.01, 6.,  1e-16, 5e1]),\
                   7e-13:   np.asarray([0.01, 6.,  1e-16, 5e1]),\
                   7.3e-13: np.asarray([0.01, 6.,  1e-16, 5e1]),\
                   7.7e-13: np.asarray([0.01, 10., 1e-16, 5e1]),\
                   8e-13:   np.asarray([0.01, 10., 1e-16, 5e1]),\
                   8.3e-13: np.asarray([0.01, 10., 1e-16, 5e1]),\
                   8.7e-13: np.asarray([0.01, 10., 1e-16, 5e1]),\
                   9e-13:   np.asarray([0.01, 10., 1e-16, 5e1]),\
                   9.3e-13: np.asarray([0.01, 10., 1e-16, 5e1]),\
                   9.7e-13: np.asarray([0.01, 10., 1e-16, 5e1]),\

                   1e-12:   np.asarray([0.01, 10., 1e-16, 5e1]),\
                   1.3e-12: np.asarray([0.01, 10., 1e-16, 5e1]),\
                   1.7e-12: np.asarray([0.01, 10., 1e-16, 5e1]),\
                   2e-12:   np.asarray([0.01, 10., 1e-16, 5e1]),\
                   3e-12:   np.asarray([0.01, 10., 1e-16, 1e1]),\
                   4e-12:   np.asarray([0.01, 10., 1e-16, 1e1]),\
                   5e-12:   np.asarray([0.01, 10., 1e-16, 1e1]),\
                   6e-12:   np.asarray([0.01, 10., 1e-16, 1e1]),\
                   7e-12:   np.asarray([0.01, 10., 1e-16, 1e1]),\
                   8e-12:   np.asarray([0.01, 10., 1e-16, 1e1]),\

                   9e-12:   np.asarray([0.01, 10., 1e-16, 1e1]),\
                   1e-11:   np.asarray([0.01, 10., 1e-16, 1e1]),\
                   1.5e-11: np.asarray([0.01, 10., 1e-16, 1e1]),\
                   2e-11:   np.asarray([0.01, 10., 1e-16, 1e1])}
#                   2.5e-11: np.asarray([0.01, 10., 1e-16, 1e1]),\ # these give nans
#                   3e-11:   np.asarray([0.01, 10., 1e-16, 1e1]),\
#                   3.5e-11: np.asarray([0.01, 10., 1e-16, 1e1]),\
#                   4e-11:   np.asarray([0.01, 10., 1e-16, 1e1]),\
#                   4.5e-11: np.asarray([0.01, 10., 1e-16, 1e1]),\
#                  5e-11:   np.asarray([0.01, 10., 1e-16, 1e1])}

def chooseModel(chosenMass, models):
    try:
        return models[chosenMass]
    except:
        print('Mass not implemented.')

def import_data_thomsoncrosstau(MA, nZs, zMin, zMax, ellMax, getgas=True, rscale=True):
    baseline = dirdata(MA, nZs, zMin, zMax, ellMax)
    data1h_thom_path = baseline+'_thom_1hdata'+('rscale' if rscale else 'r0')
    data2h_thom_path = baseline+'_thom_2hdata'+('rscale' if rscale else 'r0')

    thomCell1Halo = np.load(data1h_thom_path+'.npy')
    thomCell2Halo = np.load(data2h_thom_path+'.npy')

    try:
        baseline = dirdata_gas(MA, nZs, zMin, zMax, ellMax)
        screeningPS_thom_path_gas = baseline+'_thom_CMBDP'+('rscale' if rscale else 'r0')
        l0List, scrTT, scrEE, scrBB, scrTE = np.load(screeningPS_thom_path_gas+'.npy')
        CMBDPtauthom = np.array([scrTT, scrEE, scrBB, scrTE])
    except:
        baseline = dirdata(MA, nZs, zMin, zMax, ellMax)
        screeningPS_thom_path_gas = baseline+'_thom_CMBDP'+('rscale' if rscale else 'r0')
        l0List, scrTT, scrEE, scrBB, scrTE = np.load(screeningPS_thom_path_gas+'.npy')
        CMBDPtauthom = np.array([scrTT, scrEE, scrBB, scrTE])
    return thomCell1Halo, thomCell2Halo, CMBDPtauthom

def import_data(MA, nZs, zMin, zMax, ellMax, getgas=False, rscale=False):
    if getgas:
        baseline = dirdata_gas(MA, nZs, zMin, zMax, ellMax)
    else:
        baseline = dirdata(MA, nZs, zMin, zMax, ellMax)

    rcross_path      = baseline+'_rcross'+('rscale' if rscale else 'r0')
    prob_path        = baseline+'_prob'  +('rscale' if rscale else 'r0')
    tau_path         = baseline+'_tau'   +('rscale' if rscale else 'r0')
    uell_path        = baseline+'_uell'  +('rscale' if rscale else 'r0')
    MW_path          = baseline+'_MWprob'+('rscale' if rscale else 'r0')
    data1h_path      = baseline+'_1hdata'+('rscale' if rscale else 'r0')
    data2h_path      = baseline+'_2hdata'+('rscale' if rscale else 'r0')
    screeningPS_path = baseline+'_CMBDP' +('rscale' if rscale else 'r0')
    monoplscrPS_path = baseline+'_CMBDP_monopole'+('rscale' if rscale else 'r0')

    rcrossMW, probMW = np.load(MW_path+'.npy')
    rcross           = np.load(rcross_path+'.npy')
    prob             = np.load(prob_path+'.npy')
    avtau, dtaudz    = np.load(tau_path+'.npy')
    uell0            = np.load(uell_path+'.npy')
    Cell1Halo        = np.load(data1h_path+'.npy')
    Cell2Halo        = np.load(data2h_path+'.npy')

    l0List, scrTT, scrEE, scrBB, scrTE = np.load(screeningPS_path+'.npy')
    l0List, scrTT0 = np.load(monoplscrPS_path+'.npy')
    CMBDP          = np.array([scrTT+scrTT0, scrEE, scrBB, scrTE])
    CMBDP2         = np.array([scrTT, scrEE, scrBB, scrTE])
    return rcross, prob, avtau, dtaudz, rcrossMW, probMW, uell0, Cell1Halo, Cell2Halo, CMBDP, CMBDP2

def import_data_thomson(nZs, zMin, zMax, ellMax, getgas=True, rscale=False):
    baseline = dirdata_thom(nZs, zMin, zMax, ellMax)
    thmonopell_path = baseline+'_thmonopell'+('rscale' if rscale else 'r0')

    data1h_thomthom_path = baseline+'_thomthom_1hdata'+('rscale' if rscale else 'r0')
    data2h_thomthom_path = baseline+'_thomthom_2hdata'+('rscale' if rscale else 'r0')
    screeningPS_thomthom_path_gas = baseline+'_thomthom_CMBDP'+('rscale' if rscale else 'r0')

    thom_probell   = np.load(thmonopell_path+'.npy')
    thom_Cell1Halo = np.load(data1h_thomthom_path+'.npy')
    thom_Cell2Halo = np.load(data2h_thomthom_path+'.npy')

    l0List, scrTT, scrEE, scrBB, scrTE = np.load(screeningPS_thomthom_path_gas+'.npy')
    thom_CMBDP = np.array([scrTT, scrEE, scrBB, scrTE])
    return thom_probell, thom_Cell1Halo, thom_Cell2Halo, thom_CMBDP

def import_data_short(MA, nZs, zMin, zMax, ellMax, getgas=False, rscale=False):
    if getgas:
        baseline = dirdata_gas(MA, nZs, zMin, zMax, ellMax)
    else:
        baseline = dirdata(MA, nZs, zMin, zMax, ellMax)

    rcross_path  = baseline+'_rcross' +('rscale' if rscale else 'r0')
    prob_path    = baseline+'_prob'   +('rscale' if rscale else 'r0')
    tau_path     = baseline+'_tau'    +('rscale' if rscale else 'r0')
    uell_path    = baseline+'_uell'   +('rscale' if rscale else 'r0')
    MW_path      = baseline+'_MWprob' +('rscale' if rscale else 'r0')
    data1h_path  = baseline+'_1hdata' +('rscale' if rscale else 'r0')
    data2h_path  = baseline+'_2hdata' +('rscale' if rscale else 'r0')

    rcrossMW, probMW = np.load(MW_path+'.npy')
    rcross           = np.load(rcross_path+'.npy')
    prob             = np.load(prob_path+'.npy')
    avtau, dtaudz    = np.load(tau_path+'.npy')
    uell0            = np.load(uell_path+'.npy')
    Cell1Halo        = np.load(data1h_path+'.npy')
    Cell2Halo        = np.load(data2h_path+'.npy')
    return rcross, prob, avtau, dtaudz, rcrossMW, probMW, uell0, Cell1Halo, Cell2Halo

def import_data_shortest(MA, nZs, zMin, zMax, ellMax, getgas=False, rscale=False):
    if getgas:
        baseline = dirdata_gas(MA, nZs, zMin, zMax, ellMax)
    else:
        baseline = dirdata(MA, nZs, zMin, zMax, ellMax)

    rcross_path = baseline+'_rcross'+('rscale' if rscale else 'r0')
    prob_path   = baseline+'_prob'  +('rscale' if rscale else 'r0')
    tau_path    = baseline+'_tau'   +('rscale' if rscale else 'r0')
    MW_path     = baseline+'_MWprob'+('rscale' if rscale else 'r0')

    rcrossMW, probMW = np.load(MW_path+'.npy')
    rcross           = np.load(rcross_path+'.npy')
    prob             = np.load(prob_path+'.npy')
    avtau, dtaudz    = np.load(tau_path+'.npy')
    return rcross, prob, avtau, dtaudz, rcrossMW, probMW


############# Noise modelling ############# 

Planck = {'freqsGHz':              np.asarray([30,     44,   70,     100,  143, 217,  353,  545,  857  ]) ,\
          'freqseV':       ghztoev(np.asarray([30,     44,   70,     100,  143, 217,  353,  545,  857  ])),\
          'FWHMrad':    arcmin2rad(np.asarray([32.408, 27.1, 13.315, 9.69, 7.3, 5.02, 4.94, 4.83, 4.64 ])),\
          'SensitivityμK': arcmin2rad(1.)*np.asarray([195.1, 226.1, 199.1, 77.4, 33., 46.8, 153.6, 818.2, 40090.7]),\
          'Knee ell': np.asarray([0., 0., 0., 0., 0., 0., 0., 0., 0.]),\
          'Exponent': np.asarray([0., 0., 0., 0., 0., 0., 0., 0., 0.])}

CMBS4 = {'freqsGHz':                     np.asarray([20,    27,   39,   93,   145,  225,  278 ]) ,\
         'freqseV':              ghztoev(np.asarray([20,    27,   39,   93,   145,  225,  278 ])),\
         'FWHMrad':           arcmin2rad(np.asarray([11.0,  8.4,  5.8,  2.5,  1.6,  1.1,  1.0 ])),\
         'SensitivityμK': arcmin2rad(1.)*np.asarray([10.41, 5.14, 3.28, 0.50, 0.46, 1.45, 3.43]) ,\
         'Knee ell': np.asarray([100., 100., 100., 100., 100., 100., 100.]),\
         'Exponent': np.asarray([-3.,  -3.,  -3.,  -3.,  -3.,  -3.,  -3.])}

CMBHD = {'freqsGHz':                     np.asarray([30,   40,   90,   150,  220,  280,  350])  ,\
         'freqseV':              ghztoev(np.asarray([30,   40,   90,   150,  220,  280,  350])) ,\
         'FWHMrad':           arcmin2rad(np.asarray([1.25, 0.94, 0.42, 0.25, 0.17, 0.13, 0.11])),\
         'SensitivityμK': arcmin2rad(1.)*np.asarray([6.5,  3.4,  0.7,  0.8,  2.0,  2.7,  100.0]),\
         'Knee ell': np.asarray([100., 100., 100., 100., 100., 100., 100.]),\
         'Exponent': np.asarray([-3.,  -3.,  -3.,  -3.,  -3.,  -3.,  -3.])}


#############  Storage #############

dirplots    = './plots/'
dirsomedata = './data/'
dirplots_gas    = './plots/gas_'
dirsomedata_gas = './data/gas_'

dirdata      = lambda MA, nZs, zMin, zMax, lMax: '/gpfs/dpirvu/darkphotondata/MA%5.4e'%(MA)+'_nZs'+str(int(nZs))+'_zmin'+str(zMin)+'_zmax'+str(zMax)+'_ellMax'+str(int(lMax))
dirdata_gas  = lambda MA, nZs, zMin, zMax, lMax: '/gpfs/dpirvu/darkphotondata/gas_MA%5.4e'%(MA)+'_nZs'+str(int(nZs))+'_zmin'+str(zMin)+'_zmax'+str(zMax)+'_ellMax'+str(int(lMax))
dirdata_thom = lambda nZs, zMin, zMax, lMax: '/gpfs/dpirvu/darkphotondata/nZs'+str(int(nZs))+'_zmin'+str(zMin)+'_zmax'+str(zMax)+'_ellMax'+str(int(lMax))

rcross_path = lambda MA, nZs, zMin, zMax, lMax, rscale: dirdata(MA, nZs, zMin, zMax, lMax) + '_rcross' + ('rscale' if rscale else 'r0')
prob_path   = lambda MA, nZs, zMin, zMax, lMax, rscale: dirdata(MA, nZs, zMin, zMax, lMax) + '_prob'   + ('rscale' if rscale else 'r0')
tau_path    = lambda MA, nZs, zMin, zMax, lMax, rscale: dirdata(MA, nZs, zMin, zMax, lMax) + '_tau'    + ('rscale' if rscale else 'r0')
MW_path     = lambda MA, nZs, zMin, zMax, lMax, rscale: dirdata(MA, nZs, zMin, zMax, lMax) + '_MWprob' + ('rscale' if rscale else 'r0')
uell_path   = lambda MA, nZs, zMin, zMax, lMax, rscale: dirdata(MA, nZs, zMin, zMax, lMax) + '_uell'   + ('rscale' if rscale else 'r0')

rcross_path_gas  = lambda MA, nZs, zMin, zMax, lMax, rscale: dirdata_gas(MA, nZs, zMin, zMax, lMax) + '_rcross'  + ('rscale' if rscale else 'r0')
prob_path_gas    = lambda MA, nZs, zMin, zMax, lMax, rscale: dirdata_gas(MA, nZs, zMin, zMax, lMax) + '_prob'    + ('rscale' if rscale else 'r0')
tau_path_gas     = lambda MA, nZs, zMin, zMax, lMax, rscale: dirdata_gas(MA, nZs, zMin, zMax, lMax) + '_tau'     + ('rscale' if rscale else 'r0')
MW_path_gas      = lambda MA, nZs, zMin, zMax, lMax, rscale: dirdata_gas(MA, nZs, zMin, zMax, lMax) + '_MWprob'  + ('rscale' if rscale else 'r0')
uell_path_gas    = lambda MA, nZs, zMin, zMax, lMax, rscale: dirdata_gas(MA, nZs, zMin, zMax, lMax) + '_uell'    + ('rscale' if rscale else 'r0')

thmonopell_path_gas  = lambda nZs, zMin, zMax, lMax, rscale: dirdata_thom(nZs, zMin, zMax, lMax) + '_thmonopell' + ('rscale' if rscale else 'r0')
thmonop_path_gas     = lambda nZs, zMin, zMax, lMax, rscale: dirdata_thom(nZs, zMin, zMax, lMax) + '_thmonop' + ('rscale' if rscale else 'r0')
thangls_path_gas     = lambda nZs, zMin, zMax, lMax, rscale: dirdata_thom(nZs, zMin, zMax, lMax) + '_thangs' + ('rscale' if rscale else 'r0')
thomtau_path_gas     = lambda nZs, zMin, zMax, lMax, rscale: dirdata_thom(nZs, zMin, zMax, lMax) + '_thtau' + ('rscale' if rscale else 'r0')
thuell_path_gas      = lambda nZs, zMin, zMax, lMax, rscale: dirdata_thom(nZs, zMin, zMax, lMax) + '_thuell'  + ('rscale' if rscale else 'r0')


data1h_path = lambda MA, nZs, zMin, zMax, lMax, rscale: dirdata(MA, nZs, zMin, zMax, lMax) + '_1hdata' + ('rscale' if rscale else 'r0')
data2h_path = lambda MA, nZs, zMin, zMax, lMax, rscale: dirdata(MA, nZs, zMin, zMax, lMax) + '_2hdata' + ('rscale' if rscale else 'r0')

data1h_path_gas = lambda MA, nZs, zMin, zMax, lMax, rscale: dirdata_gas(MA, nZs, zMin, zMax, lMax) + '_1hdata' + ('rscale' if rscale else 'r0')
data2h_path_gas = lambda MA, nZs, zMin, zMax, lMax, rscale: dirdata_gas(MA, nZs, zMin, zMax, lMax) + '_2hdata' + ('rscale' if rscale else 'r0')

data1h_thom_path_gas = lambda MA, nZs, zMin, zMax, lMax, rscale: dirdata(MA, nZs, zMin, zMax, lMax) + '_thom_1hdata'  + ('rscale' if rscale else 'r0')
data2h_thom_path_gas = lambda MA, nZs, zMin, zMax, lMax, rscale: dirdata(MA, nZs, zMin, zMax, lMax) + '_thom_2hdata'  + ('rscale' if rscale else 'r0')

data1h_thomthom_path_gas = lambda nZs, zMin, zMax, lMax, rscale: dirdata_thom(nZs, zMin, zMax, lMax) + '_thomthom_1hdata'  + ('rscale' if rscale else 'r0')
data2h_thomthom_path_gas = lambda nZs, zMin, zMax, lMax, rscale: dirdata_thom(nZs, zMin, zMax, lMax) + '_thomthom_2hdata'  + ('rscale' if rscale else 'r0')

screeningPS_path = lambda MA, nZs, zMin, zMax, lMax, rscale: dirdata(MA, nZs, zMin, zMax, lMax) + '_CMBDP' + ('rscale' if rscale else 'r0')
monoplscrPS_path = lambda MA, nZs, zMin, zMax, lMax, rscale: dirdata(MA, nZs, zMin, zMax, lMax) + '_CMBDP_monopole' + ('rscale' if rscale else 'r0')

screeningPS_path_gas = lambda MA, nZs, zMin, zMax, lMax, rscale: dirdata_gas(MA, nZs, zMin, zMax, lMax) + '_CMBDP' + ('rscale' if rscale else 'r0')
monoplscrPS_path_gas = lambda MA, nZs, zMin, zMax, lMax, rscale: dirdata_gas(MA, nZs, zMin, zMax, lMax) + '_CMBDP_monopole' + ('rscale' if rscale else 'r0')

screeningPS_thom_path_gas = lambda MA, nZs, zMin, zMax, lMax, rscale: dirdata_gas(MA, nZs, zMin, zMax, lMax) + '_thom_CMBDP' + ('rscale' if rscale else 'r0')

screeningPS_thomthom_path_gas = lambda nZs, zMin, zMax, lMax, rscale: dirdata_thom(nZs, zMin, zMax, lMax) + '_thomthom_CMBDP' + ('rscale' if rscale else 'r0')

ILCnoisePS_path  = lambda expname: dirsomedata + expname + '_ILC_remainder.npy'
weights_path     = lambda expname: dirsomedata + expname + '_ILC_weights.npy'
reconoisePS_path = lambda expname, zreio: dirsomedata + expname + '_recoNoise_zreio' + str(int(zreio)) + '.npy'

#ILCnoisePS_path_gas  = lambda expname: dirsomedata_gas + expname + '_ILC_remainder.npy'
#weights_path_gas     = lambda expname: dirsomedata_gas + expname + '_ILC_weights.npy'
ILCnoisePS_path_gas  = lambda expname, zreio: dirsomedata_gas + expname + '_ILC_remainder_zreio' + str(int(zreio)) + '.npy'
weights_path_gas     = lambda expname, zreio: dirsomedata_gas + expname + '_ILC_weights_zreio' + str(int(zreio)) + '.npy'
reconoisePS_path_gas = lambda expname, zreio: dirsomedata_gas + expname + '_recoNoise_zreio' + str(int(zreio)) + '.npy'

ILCnoisePS_path_NFW  = lambda expname, zreio: dirsomedata_gas + expname + '_ILC_NFW_remainder_zreio' + str(int(zreio)) + '.npy'
weights_path_NFW     = lambda expname, zreio: dirsomedata_gas + expname + '_ILC_NFW_weights_zreio' + str(int(zreio)) + '.npy'

BB_ILCnoisePS_path_gas  = lambda expname, zreio: dirsomedata_gas + expname + '_BB_ILC_remainder_zreio' + str(int(zreio)) + '.npy'
BB_weights_path_gas     = lambda expname, zreio: dirsomedata_gas + expname + '_BB_ILC_weights_zreio' + str(int(zreio)) + '.npy'
BB_reconoisePS_path_gas = lambda expname, zreio: dirsomedata_gas + expname + '_BB_recoNoise_zreio' + str(int(zreio)) + '.npy'

bispec_Tdsc_Tsc_Tsc = lambda MA, nZs, zMin, zMax, lMax, rscale, expname: dirdata(MA, nZs, zMin, zMax, lMax) + '_bispec'+expname+('rscale' if rscale else 'r0')+'.npy'
bispec_Tdsc_Esc_Bsc = lambda MA, nZs, zMin, zMax, lMax, rscale, expname: dirdata(MA, nZs, zMin, zMax, lMax) + '_bispecEB'+expname+('rscale' if rscale else 'r0')+'.npy'
