from brian2 import defaultclock, ms, uS, TimedArray
from params import paramsDestexhe as p
from network import DestexheNetwork
from results import Results
import matplotlib.pyplot as plt
import numpy as np

MAKE_UP_PLOTS = False

p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'
p['saveWithDate'] = True
p['dt'] = 0.1 * ms

p['qExc'] = 0.35 * uS  # normally 0.6

defaultclock.dt = p['dt']

# determine 'up crit' empirically

p['simName'] = 'destexheNMDAUpCrit'
p['recordStateVariables'] = p['recordStateVariables'] = ['v', 'ge', 's_NMDA_tot']

nSamples = int(p['duration'] / defaultclock.dt)
iExtArray = np.zeros((nSamples,))
iExtRecorded = TimedArray(iExtArray, dt=defaultclock.dt)

p['nInh'] = int(p['propInh'] * p['nUnits'])
p['nExc'] = int(p['nUnits'] - p['nInh'])
nRecurrentExcitatorySynapsesPerUnit = int(p['nExc'] * p['propConnect'])
useQExc = p['qExc'] / nRecurrentExcitatorySynapsesPerUnit
useQNMDA = 0.5 * useQExc
nRecurrentInhibitorySynapsesPerUnit = int(p['nInh'] * p['propConnect'])
useQInh = p['qInh'] / nRecurrentInhibitorySynapsesPerUnit

# monitor an unkciked EXc unit as well
indUnkickedExc = int(p['nUnits'] - (p['propInh'] * p['nUnits']) - 1)
p['indsRecordStateExc'].append(indUnkickedExc)

DN = DestexheNetwork(p)
DN.initialize_network()
DN.initialize_units_NMDA()
DN.initialize_recurrent_excitation_NMDA(useQExc=useQExc)
DN.initialize_recurrent_inhibition()

DN.prepare_upCrit_experiment(minUnits=200, maxUnits=200, unitSpacing=40, timeSpacing=1000 * ms,
                             startTime=100 * ms)

DN.create_monitors()
DN.run_NMDA(iExtRecorded=iExtRecorded, useQNMDA=useQNMDA)
DN.save_results()
DN.save_params()

R = Results(DN.saveName, DN.p['saveFolder'])

R.calculate_PSTH()
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

fig2, ax2 = plt.subplots(3, 1, num=2, figsize=(10, 9), sharex=True)
R.plot_voltage_detail(ax2[0], unitType='Exc', useStateInd=0)
R.plot_updur_lines(ax2[0])
R.plot_voltage_detail(ax2[1], unitType='Exc', useStateInd=1)
R.plot_updur_lines(ax2[1])
R.plot_voltage_detail(ax2[2], unitType='Inh', useStateInd=0)

fig2b, ax2b = plt.subplots(1, 1, num=21)
R.plot_updur_lines(ax2b)
R.plot_voltage_histogram(ax2b, yScaleLog=True)

fig2c, ax2c = plt.subplots(3, 1, num=22, figsize=(10, 9), sharex=True)
ax2c[0].plot(DN.stateMonExc.s_NMDA_tot[0, :].T)
ax2c[1].plot(DN.stateMonExc.s_NMDA_tot[1, :].T)
ax2c[2].plot(DN.stateMonInh.s_NMDA_tot[0, :].T)