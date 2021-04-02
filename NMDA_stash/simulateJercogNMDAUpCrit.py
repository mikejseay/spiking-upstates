from brian2 import defaultclock, ms, mV, TimedArray
from params import paramsJercog as p
from network import JercogNetwork
from results import Results
import matplotlib.pyplot as plt
import numpy as np

MAKE_UP_PLOTS = False

p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'
p['saveWithDate'] = True
p['dt'] = 0.1 * ms

# NMDA gating parameters
p['vStepSigmoid'] = 12 * mV
p['kSigmoid'] = 0.4
p['vMidSigmoid'] = 20 * mV

p['nUnits'] = 500

p['nIncInh'] = int(p['propInh'] * p['nUnits'])
p['nIncExc'] = p['nUnits'] - p['nIncInh']

# alter AMPA excitation
p['jEE'] = p['jEE'] * 0.6
p['jIE'] = p['jIE'] * 0.6

defaultclock.dt = p['dt']

# determine 'up crit' empirically

p['simName'] = 'jercogNMDAUpCrit'
# p['recordStateVariables'] = ['v', ]
p['recordStateVariables'] = ['v', 'sE', 's_NMDA_tot']

nSamples = int(p['duration'] / defaultclock.dt)
iExtArray = np.zeros((nSamples,))
iKickRecorded = TimedArray(iExtArray, dt=defaultclock.dt)

NMDA_FACTOR = 0.5
p['jEE_NMDA'] = NMDA_FACTOR * p['jEE']
p['jIE_NMDA'] = NMDA_FACTOR * p['jIE']

# monitor an unkciked EXc unit as well
indUnkickedExc = int(p['nUnits'] - (p['propInh'] * p['nUnits']) - 1)
p['indsRecordStateExc'].append(indUnkickedExc)

JN = JercogNetwork(p)
JN.initialize_network()
JN.initialize_units_NMDA()
JN.initialize_recurrent_excitation_NMDA()
JN.initialize_recurrent_inhibition()

JN.prepare_upCrit_experiment(minUnits=15, maxUnits=25, unitSpacing=5, timeSpacing=1000 * ms,
                             startTime=100 * ms)

JN.create_monitors()
JN.run_NMDA()
JN.save_results()
JN.save_params()

R = Results(JN.saveName, JN.p['saveFolder'])

R.calculate_spike_rate()
R.calculate_voltage_histogram(removeMode=True)
R.calculate_upstates()
if len(R.ups) > 0:
    R.reshape_upstates()
    R.calculate_FR_in_upstates()
    print('average FR in upstate for Exc: {:.2f}, Inh: {:.2f} '.format(R.upstateFRExc.mean(), R.upstateFRInh.mean()))

# quit()

fig1, ax1 = plt.subplots(2, 1, num=1, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 1]},
                         sharex=True)
R.plot_spike_raster(ax1[0])
R.plot_firing_rate(ax1[1])

fig2, ax2 = plt.subplots(3, 1, num=2, figsize=(10, 9), sharex=True, sharey=True)
R.plot_voltage_detail(ax2[0], unitType='Exc', useStateInd=0)
R.plot_updur_lines(ax2[0])
R.plot_voltage_detail(ax2[1], unitType='Exc', useStateInd=1)
R.plot_updur_lines(ax2[1])
R.plot_voltage_detail(ax2[2], unitType='Inh', useStateInd=0)

fig2b, ax2b = plt.subplots(1, 1, num=21)
R.plot_updur_lines(ax2b)
R.plot_voltage_histogram(ax2b, yScaleLog=True)

fig2c, ax2c = plt.subplots(3, 1, num=22, figsize=(10, 9), sharex=True)
ax2c[0].plot(JN.stateMonExc.sE[0, :].T)
ax2c[1].plot(JN.stateMonExc.sE[1, :].T)
ax2c[2].plot(JN.stateMonInh.sE[0, :].T)

fig2d, ax2d = plt.subplots(3, 1, num=23, figsize=(10, 9), sharex=True)
ax2c[0].plot(JN.stateMonExc.s_NMDA_tot[0, :].T)
ax2c[1].plot(JN.stateMonExc.s_NMDA_tot[1, :].T)
ax2c[2].plot(JN.stateMonInh.s_NMDA_tot[0, :].T)

if MAKE_UP_PLOTS:
    fig3, ax3 = plt.subplots(num=3, figsize=(10, 9))
    R.plot_state_duration_histogram(ax3)

    fig4, ax4 = plt.subplots(1, 2, num=4, figsize=(10, 9))
    R.plot_consecutive_state_correlation(ax4)

    fig5, ax5 = plt.subplots(2, 1, num=5, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 1]},
                             sharex=True)
    R.plot_upstate_voltage_image(ax5[0])
    R.plot_upstate_voltages(ax5[1])

    fig6, ax6 = plt.subplots(2, 1, num=6, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 1]},
                             sharex=True)
    R.plot_upstate_raster(ax6[0])
    R.plot_upstate_FR(ax6[1])
