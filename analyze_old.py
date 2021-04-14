"""
script used to perform an analysis of the results of a simulation.
loads results files
    (.npz, which contains the data, and the .pkl, which contains the parameters),
does certain calculations,
then plots results.
"""

from results import Results
import matplotlib.pyplot as plt
from generate import weight_matrix_from_flat_inds_weights
from plot import weight_matrix
import numpy as np

MAKE_UP_PLOTS = False

loadFolder = 'C:/Users/mikejseay/Documents/BrianResults/'

# shows that under normal Jercog settings an ignited Up state will putter out after 2 seconds
# targetSim = 'jercogFullConn_2020-11-04-16-13'
# targetSim = 'jercogFullConn_2020-11-04-16-38'  # shows that the Buono paramters lead to a never-ending Up state


# targetSim = 'jercogUpCritFullConn500Units_2020-10-26-17-07'  # shows the tuning of the UpCrit
# targetSim = 'jercogUpCritFullConn500Units_2020-11-25-11-28'  # shows the currents

# targetSim = 'destexheEphysBuono_2021-03-29-15-23'  # good one
# targetSim = 'destexheEphysBuono_2021-03-29-16-24'  # also good with half as many sources
# targetSim = 'destexheEphysBuono_2021-03-30-09-01'  # pretty good 60 s one

# targetSim = 'destexheUpCrit_1e3_0p05_2021-04-06-14-36'  # up crit for  ongoing home_rules start points

# targetSim = 'destexheUpCrit_1e3_0p05_2021-04-06-14-36'  # 1e4 units, pConn = 0.1, Buono units, works
# targetSim = 'destexheUpCrit_1e4_2021-04-08-14-46'  # altering pConn of each bundle leads to permanent Up state

# targetSim = 'destexheUpCrit_1e4_2021-04-08-14-50'
# targetSim = 'jercogUpLearn_2021-04-09-11-23'
# targetSim = 'jercog500UpLearn_2021-04-09-11-27'

# loadFolder = 'C:/Users/mikejseay/Documents/BrianResults/perfect_files/'

# targetSim = 'classicJercog_2020-10-12-16-02'
# targetSim = 'classicJercog0p05Conn_2020-10-13-11-45'
# targetSim = 'classicDestexhe0p05Conn_2020-10-13-12-09'
# targetSim = 'classicDestexheFullConn_2020-10-13-11-21'
# targetSim = 'destexheNMDAUpCrit_2020-11-11-16-38'
# targetSim = 'jercogUpCritFullConn500Units_2021-04-09-11-30'
# targetSim = 'jercog500UpLearn_2021-04-09-11-50'
targetSim = 'jercogUpLearn_2021-04-12-21-29'

# loadFolder = 'C:/Users/mikejseay/Documents/BrianResults/long_files/'

# targetSim = 'jercog0p05Conn_2020-10-26-16-21'  # default Jercog with 0.05 pConn... firing rates implausibly high
# targetSim = 'jercog0p05Conn_2020-10-27-11-34'  # decrease excitation by 25%, works...
# targetSim = 'destexhe0p05Conn5e4units_2020-11-02-11-06'  # Destexhe 0.05 pConn, 50k units, kicks... bimodality!
# targetSim = 'destexhe0p05Conn5e4units_2020-11-02-22-08'  # second example of the above...
# targetSim = 'destexhe0p05Conn5e4units_2020-11-03-09-40'  # third example with also weak uncorrelated inputs

# decrease noise amplitude substantially (5x), increase propKicked by ~2x, bimodality is not impressive
# targetSim = 'jercog0p05Conn_2020-11-10-10-12'

# Destexhe 0.05 pConn, 50k units, with weak uncorrelate and strong uncorrelated inputs
# targetSim = 'destexheEphysBuono_2020-11-10-12-59'
# exact same as above with 10k units... no apparent bimodality!!
# targetSim = 'destexheEphysBuono_2020-11-11-12-55'

R = Results()
R.init_from_file(targetSim, loadFolder)

printParameters = ['propConnectFeedforwardProjectionCorr', 'poissonCorrInputRate', 'nPoissonCorrInputUnits',
                   'qExcFeedforwardCorr']
for printParam in printParameters:
    if printParam in R.p:
        print(printParam, ':', R.p[printParam])

R.calculate_spike_rate()
R.calculate_voltage_histogram(removeMode=True)
R.calculate_upstates()
if len(R.ups) > 0:
    R.reshape_upstates()
    R.calculate_FR_in_upstates()
    print('average FR in upstate for Exc: {:.2f}, Inh: {:.2f} '.format(R.upstateFRExc.mean(), R.upstateFRInh.mean()))

fig1, ax1 = plt.subplots(2, 1, num=1, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 1]},
                         sharex=True)
R.plot_spike_raster(ax1[0])
R.plot_firing_rate(ax1[1])

if R.stateMonExcV.shape[0] > 1:
    fig2, ax2 = plt.subplots(3, 1, num=2, figsize=(10, 9), sharex=True)
    R.plot_voltage_detail(ax2[0], unitType='Exc', useStateInd=0)
    R.plot_updur_lines(ax2[0])
    R.plot_voltage_detail(ax2[1], unitType='Exc', useStateInd=1)
    R.plot_updur_lines(ax2[1])
    R.plot_voltage_detail(ax2[2], unitType='Inh', useStateInd=0)
else:
    fig2, ax2 = plt.subplots(2, 1, num=2, figsize=(10, 9), sharex=True)
    R.plot_voltage_detail(ax2[0], unitType='Exc', useStateInd=0)
    R.plot_updur_lines(ax2[0])
    R.plot_voltage_detail(ax2[1], unitType='Inh', useStateInd=0)
    R.plot_updur_lines(ax2[1])

fig2b, ax2b = plt.subplots(1, 1, num=21, figsize=(4, 3))
R.plot_voltage_histogram(ax2b, yScaleLog=True)

if MAKE_UP_PLOTS:
    fig3, ax3 = plt.subplots(num=3, figsize=(4, 3))
    R.plot_state_duration_histogram(ax3)

    fig4, ax4 = plt.subplots(1, 2, num=4, figsize=(8, 4))
    R.plot_consecutive_state_correlation(ax4)

    fig5, ax5 = plt.subplots(2, 1, num=5, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 1]},
                             sharex=True)
    R.plot_upstate_voltage_image(ax5[0])
    R.plot_upstate_voltages(ax5[1])

    fig6, ax6 = plt.subplots(2, 1, num=6, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 1]},
                             sharex=True)
    R.plot_upstate_raster(ax6[0])
    R.plot_upstate_FR(ax6[1])

# plotting weight matrices
if hasattr(R, 'weights_EE'):
    wEE = weight_matrix_from_flat_inds_weights(R.p['nExc'], R.p['nExc'], R.preInds_EE, R.postInds_EE, R.weights_EE)
    wIE = weight_matrix_from_flat_inds_weights(R.p['nExc'], R.p['nInh'], R.preInds_IE, R.postInds_IE, R.weights_IE)
    wEI = weight_matrix_from_flat_inds_weights(R.p['nInh'], R.p['nExc'], R.preInds_EI, R.postInds_EI, R.weights_EI)
    wII = weight_matrix_from_flat_inds_weights(R.p['nInh'], R.p['nInh'], R.preInds_II, R.postInds_II, R.weights_II)

    wFull = np.block([[wEE, wIE], [-wEI, -wII]])

    fig7, ax7 = plt.subplots(num=7, figsize=(10, 9))
    weight_matrix(ax7, wFull, xlabel='Post Index', ylabel='Pre Index', clabel='Normalized Weight',)
