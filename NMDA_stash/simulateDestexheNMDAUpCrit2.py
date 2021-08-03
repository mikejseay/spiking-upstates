from brian2 import defaultclock, ms, mV, pA, second
from params import paramsDestexhe as p
from network import DestexheNetwork
from results import Results
import matplotlib.pyplot as plt
import numpy as np

p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'
p['saveWithDate'] = True
p['dt'] = 0.1 * ms

defaultclock.dt = p['dt']

# NMDA parameters
p['tau_NMDA_rise'] = 2. * ms
p['tau_NMDA_decay'] = 50. * ms
p['alpha'] = 0.5 / ms
p['Mg2'] = 1.

# NMDA gating parameters
p['kSigmoid'] = 0.3
p['vMidSigmoid'] = -50 * mV
p['vStepSigmoid'] = -200 * mV  # having this be very low effectively removes any hard gating effect

# alter AMPA excitation
p['qExc'] = 0.6 * p['qExc']

# alter GABA inhibition
p['qInh'] = 1 * p['qInh']

# set the NMDA weight
p['qExcNMDA'] = 0.5 * p['qExc']

# network params
p['propConnect'] = 1
p['simName'] = 'destexheNMDAUpCritFullConn500Units'
p['nUnits'] = 500

p['recordStateVariables'] = p['recordStateVariables'] = ['v', 'ge', 'gi', 'w', 's_NMDA_tot']
indUnkickedExc = int(p['nUnits'] - (p['propInh'] * p['nUnits']) - 1)
p['indsRecordStateExc'].append(indUnkickedExc)

p['nInh'] = int(p['propInh'] * p['nUnits'])
p['nExc'] = int(p['nUnits'] - p['nInh'])

# ultimately this shouldn't be needed
# nRecurrentExcitatorySynapsesPerUnit = int(p['nExc'] * p['propConnect'])
# useQExc = p['qExc'] / nRecurrentExcitatorySynapsesPerUnit
# useQNMDA = 0.5 * useQExc
# nRecurrentInhibitorySynapsesPerUnit = int(p['nInh'] * p['propConnect'])
# useQInh = p['qInh'] / nRecurrentInhibitorySynapsesPerUnit

# nSamples = int(p['duration'] / defaultclock.dt)
# iExtArray = np.zeros((nSamples,))
# iExtRecorded = TimedArray(iExtArray, dt=defaultclock.dt)

DN = DestexheNetwork(p)
DN.initialize_network()
DN.initialize_units_NMDA()
DN.initialize_recurrent_excitation_NMDA()
DN.initialize_recurrent_inhibition()

DN.prepare_upCrit_experiment(minUnits=30, maxUnits=30, unitSpacing=40, timeSpacing=1000 * ms,
                             startTime=100 * ms)

DN.create_monitors()
DN.run_NMDA()
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

AMPA_E0 = DN.stateMonExc.ge[0, :].T * (DN.p['eLeakExc'] - DN.stateMonExc.v[0, :].T) / pA
AMPA_E1 = DN.stateMonExc.ge[1, :].T * (DN.p['eLeakExc'] - DN.stateMonExc.v[1, :].T) / pA
AMPA_I0 = DN.stateMonInh.ge[0, :].T * (DN.p['eLeakExc'] - DN.stateMonInh.v[0, :].T) / pA

GABA_E0 = DN.stateMonExc.gi[0, :].T * (DN.p['eLeakInh'] - DN.stateMonExc.v[0, :].T) / pA
GABA_E1 = DN.stateMonExc.gi[1, :].T * (DN.p['eLeakInh'] - DN.stateMonExc.v[1, :].T) / pA
GABA_I0 = DN.stateMonInh.gi[0, :].T * (DN.p['eLeakInh'] - DN.stateMonInh.v[0, :].T) / pA

NMDA_E0 = convert_sNMDA_to_current_exc(DN, 0) / pA
NMDA_E1 = convert_sNMDA_to_current_exc(DN, 1) / pA
NMDA_I0 = convert_sNMDA_to_current_inh(DN, 0) / pA

fig2c, ax2c = plt.subplots(3, 1, num=3, figsize=(10, 9), sharex=True, sharey=True)

ax2c[0].plot(R.timeArray, AMPA_E0, label='AMPA', lw=.5)
ax2c[1].plot(R.timeArray, AMPA_E1, label='AMPA', lw=.5)
ax2c[2].plot(R.timeArray, AMPA_I0, label='AMPA', lw=.5)
ax2c[0].plot(R.timeArray, NMDA_E0, label='NMDA', lw=.5)
ax2c[1].plot(R.timeArray, NMDA_E1, label='NMDA', lw=.5)
ax2c[2].plot(R.timeArray, NMDA_I0, label='NMDA', lw=.5)
ax2c[0].plot(R.timeArray, AMPA_E0 + NMDA_E0, label='AMPA+NMDA', lw=.5)
ax2c[1].plot(R.timeArray, AMPA_E1 + NMDA_E1, label='AMPA+NMDA', lw=.5)
ax2c[2].plot(R.timeArray, AMPA_I0 + NMDA_I0, label='AMPA+NMDA', lw=.5)
ax2c[0].plot(R.timeArray, GABA_E0, label='GABA', lw=.5)
ax2c[1].plot(R.timeArray, GABA_E1, label='GABA', lw=.5)
ax2c[2].plot(R.timeArray, GABA_I0, label='GABA', lw=.5)

# w is a straight up current so no need to mess with it
ax2c[0].plot(R.timeArray, DN.stateMonExc.w[0, :].T / pA, label='iAdapt', lw=.5)
ax2c[1].plot(R.timeArray, DN.stateMonExc.w[1, :].T / pA, label='iAdapt', lw=.5)
ax2c[2].plot(R.timeArray, DN.stateMonInh.w[0, :].T / pA, label='iAdapt', lw=.5)

ax2c[2].legend()
ax2c[2].set(xlabel='Time (s)', ylabel='Current (pA)', xlim=(0., DN.p['duration'] / second))

# find the average AMPA / GABA current during the first Up state duration

startUpInd = int(R.ups[0] * second / R.p['dt'])
endUpInd = int(R.downs[0] * second / R.p['dt'])

AMPA_E1_avg = AMPA_E1[startUpInd:endUpInd].mean()
NMDA_E1_avg = NMDA_E1[startUpInd:endUpInd].mean()
GABA_E0_avg = GABA_E0[startUpInd:endUpInd].mean()

print('average AMPA/NMDA/GABA current during Up state: {:.1f}/{:.1f}/{:.1f} pA'.format(AMPA_E1_avg, NMDA_E1_avg,
                                                                                       GABA_E0_avg))

# plot voltage histogram

fig2b, ax2b = plt.subplots(1, 1, num=21, figsize=(4, 3))
R.plot_voltage_histogram(ax2b, yScaleLog=True)
