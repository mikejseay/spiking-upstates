from brian2 import *
from params import paramsDestexhe as p
from network import DestexheNetwork

defaultclock.dt = p['dt']

p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'
p['saveWithDate'] = True
p['simName'] = 'classicDestexheNMDATest'
p['nRecordStateExc'] = 2
p['recordStateVariables'] = ['v', 'ge', 's_NMDA_tot']

# define synaptic weights based on default settings...
p['nInh'] = int(p['propInh'] * p['nUnits'])
p['nExc'] = int(p['nUnits'] - p['nInh'])
nRecurrentExcitatorySynapsesPerUnit = int(p['nExc'] * p['propConnect'])
useQExc = p['qExc'] / nRecurrentExcitatorySynapsesPerUnit
useQNMDA = 0.5 * useQExc
nRecurrentInhibitorySynapsesPerUnit = int(p['nInh'] * p['propConnect'])
useQInh = p['qInh'] / nRecurrentInhibitorySynapsesPerUnit

# we make 2 Exc / 1 Inh unit each, make them not adapt, and connect the two exc units to each other
p['nUnits'] = 3
p['propInh'] = float(1/3)
p['bExc'] = 0 * pA
p['bInh'] = 0 * pA
p['propConnect'] = 1

# make a network like this
# 1 external Poisson unit
# 1 E unit / 1 I unit (E connects to the I)
# 1 external Poisson unit makes a very strong excitatory connection to 1 E unit (enough to cause a spike)
# the 1 external Poisson unit spikes once every 250 ms
# increasingly depolarize the units over the course of the sim (10 s?)

DN = DestexheNetwork(p)
DN.initialize_network()
DN.initialize_units_NMDA()
DN.initialize_recurrent_excitation_NMDA(useQExc=useQExc)

# make the poisson unit and spike pattern and its connection to the exc unit
nSpikes = 40
timeSpacing = 250 * ms
totalDuration = (nSpikes + 1) * timeSpacing
indices = [0,] * nSpikes
times = np.arange(1, nSpikes + 1) * timeSpacing
Spiker = SpikeGeneratorGroup(1, array(indices), times)
SpikerSyn = Synapses(
    source=Spiker,
    target=DN.unitsExc[:1],
    on_pre='ge_post += 18.5 * nS',
)
SpikerSyn.connect()
DN.N.add(Spiker, SpikerSyn)

# make the current time series (ramp up to 0.1 nA only for the second unit)
DN.p['duration'] = totalDuration
nSamples = int(DN.p['duration'] / defaultclock.dt)
iExtArray = np.linspace(0, 1, nSamples)
iExtRecorded = TimedArray(iExtArray, dt=defaultclock.dt)
DN.unitsExc[:1].iAmp = 0 * nA
DN.unitsExc[1:].iAmp = 0.1 * nA

# # create custom monitors
# spikeMonExc = SpikeMonitor(DN.unitsExc)
# spikeMonInh = SpikeMonitor(DN.unitsInh)
#
# stateMonExc = StateMonitor(DN.unitsExc, DN.p['recordStateVariables'],
#                            record=list(range(DN.p['nRecordStateExc'])))
# stateMonInh = StateMonitor(DN.unitsInh, DN.p['recordStateVariables'],
#                            record=list(range(DN.p['nRecordStateInh'])))
#
# DN.spikeMonExc = spikeMonExc
# DN.spikeMonInh = spikeMonInh
# DN.stateMonExc = stateMonExc
# DN.stateMonInh = stateMonInh
# DN.N.add(spikeMonExc, spikeMonInh, stateMonExc, stateMonInh)

DN.create_monitors()
DN.run_NMDA(iExtRecorded=iExtRecorded, useQNMDA=useQNMDA)
DN.save_results()
DN.save_params()