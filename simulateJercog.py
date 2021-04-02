# from brian2 import *
from brian2 import second, ms, volt, mV, Hz, defaultclock
from params import paramsJercog as p
from params import paramsJercogEphysBuono
from generate import generate_poisson_kicks_jercog, convert_kicks_to_current_series
from network import JercogNetwork
from results import Results
import matplotlib.pyplot as plt

MAKE_UP_PLOTS = False
USE_NEW_EPHYS_PARAMS = False
KICKS_POISSON = True
APPLY_UNCORRELATED_INPUTS = True
APPLY_KICKS = False
p['onlyKickExc'] = True
KICK_TYPE = 'spike'  # kick or spike

# p['propKicked'] = 0.04

# remove protected keys from the dict whose params are being imported
ephysParams = paramsJercogEphysBuono.copy()
protectedKeys = ('nUnits', 'propInh', 'duration')
for pK in protectedKeys:
    del ephysParams[pK]

if USE_NEW_EPHYS_PARAMS:
    p.update(ephysParams)

# buono params
# p['jEE'] = p['jEE'] * 452 / 560
# p['jEI'] = p['jEI'] * 452 / 560
# p['jIE'] = p['jIE'] * 802 / 541
# p['jII'] = p['jII'] * 802 / 541
# p['propKicked'] = 0.045

# p['critExc'] = 0.784 * volt * 452 / 560  # this fraction is to adjust for the new Buono params
# p['critInh'] = 0.67625 * volt * 802 / 541

p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'

p['saveWithDate'] = True
p['duration'] = 10 * second
p['refractoryPeriod'] = 0 * ms  # 0 by default, 1 works, 2 causes explosion
# on pConn = 0.05, 3 also works but no higher

# p['propConnect'] = 1
# p['simName'] = 'jercogFullConn'

# THESE SETTINGS WORK FOR PCONN = 0.05 WITH REASONABLE FIRING RATE!!!

# p['propConnect'] = 0.05
# p['simName'] = 'jercog0p05Conn'
# p['jEE'] = p['jEE'] * 0.75
# p['jIE'] = p['jIE'] * 0.75
# p['onlyKickExc'] = True
# KICK_TYPE = 'spike'  # kick or spike
# p['propKicked'] = 0.076

# now try with very small noise
p['noiseSigma'] = 0 * mV

# p['propConnect'] = 0.05
# p['simName'] = 'jercog0p05ConnSmallWeights'
# p['scaleWeightsByPConn'] = False

# p['propConnect'] = 1
# p['simName'] = 'jercogFullConn1kUnits'
# p['nUnits'] = 1000

# p['propConnect'] = 1
# p['simName'] = 'jercogFullConn500Units'
# p['nUnits'] = 500
# p['propKicked'] = 0.05
# p['onlyKickExc'] = True

# p['propConnect'] = 1
# p['simName'] = 'jercogFullConn250Units'
# p['nUnits'] = 250
# p['propKicked'] = 0.08
# p['onlyKickExc'] = True

indUnkickedExc = int(p['nUnits'] - (p['propInh'] * p['nUnits']) - 1)
p['indsRecordStateExc'].append(indUnkickedExc)

# these represent the number of incoming excitatory / inhibtory synapses per unit
p['nIncExc'] = int(p['propConnect'] * (1 - p['propInh']) * p['nUnits'])
p['nIncInh'] = int(p['propConnect'] * p['propInh'] * p['nUnits'])

# uncorrelated inputs
p['propConnectFeedforwardProjectionUncorr'] = 0.05  # proportion of feedforward projections that are connected
p['nPoissonUncorrInputUnits'] = p['nUnits']
p['nUncorrFeedforwardSynapsesPerUnit'] = int(p['propConnectFeedforwardProjectionUncorr'] * 1e4 * (1 - p['propInh']))
p['poissonUncorrInputRate'] = 0.315 * p['nUncorrFeedforwardSynapsesPerUnit'] * Hz
# p['jExcFeedforwardUncorr'] = 0.6 * uS / p['nUncorrFeedforwardSynapsesPerUnit']
excWeightMultiplier = 35  # it's right around 35 that it begins to be able to ignite an endless AI Up state

defaultclock.dt = p['dt']
if KICKS_POISSON:
    kickTimes, kickSizes = generate_poisson_kicks_jercog(p['kickLambda'], p['duration'],
                                                         p['kickMinimumISI'], p['kickMaximumISI'])
else:
    kickTimes = [100 * ms, 1100 * ms]
    kickSizes = [1, 0.5]
    p['duration'] = 1200 * ms
    # kickTimes = [100 * ms,]
    # kickTimes = [100 * ms, 1100 * ms, 2100 * ms, 3100 * ms, 4100 * ms,
    #              5100 * ms, 6100 * ms, 7100 * ms, 8100 * ms, 9100 * ms]
    # kickSizes = [1]
    # p['duration'] = 20000 * ms

print(kickTimes)
p['kickTimes'] = kickTimes
p['kickSizes'] = kickSizes
iKickRecorded = convert_kicks_to_current_series(p['kickDur'], p['kickTau'],
                                                p['kickTimes'], p['kickSizes'], p['duration'], p['dt'])

p['iKickRecorded'] = iKickRecorded

JN = JercogNetwork(p)
JN.initialize_network()
JN.initialize_units()

if APPLY_UNCORRELATED_INPUTS:
    JN.initialize_external_input_uncorrelated(excWeightMultiplier=excWeightMultiplier)
if APPLY_KICKS:
    if KICK_TYPE == 'kick':
        JN.set_kicked_units(onlyKickExc=p['onlyKickExc'])
    elif KICK_TYPE == 'spike':
        JN.set_spiked_units(onlySpikeExc=p['onlyKickExc'], critExc=p['critExc'], critInh=p['critInh'])

JN.initialize_recurrent_synapses()
JN.create_monitors()
JN.run()
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

fig2, ax2 = plt.subplots(3, 1, num=2, figsize=(10, 9), sharex=True)
R.plot_voltage_detail(ax2[0], unitType='Exc', useStateInd=0)
R.plot_updur_lines(ax2[0])
R.plot_voltage_detail(ax2[1], unitType='Exc', useStateInd=1)
R.plot_updur_lines(ax2[1])
R.plot_voltage_detail(ax2[2], unitType='Inh', useStateInd=0)

fig2b, ax2b = plt.subplots(1, 1, num=21)
R.plot_updur_lines(ax2b)
R.plot_voltage_histogram(ax2b, yScaleLog=True)

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
