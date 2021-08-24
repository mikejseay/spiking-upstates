from brian2 import defaultclock, ms, pA, nA, Hz, seed, mV
from params import paramsJercog as p
from params import paramsJercogEphysBuono22, paramsJercogEphysBuono4, paramsJercogEphysBuono5, paramsJercogEphysBuono6
import numpy as np
from generate import convert_kicks_to_current_series
from trainer import JercogTrainer
from results import Results
import matplotlib.pyplot as plt

rngSeed = None
defaultclock.dt = p['dt']

p['useNewEphysParams'] = True
ephysParams = paramsJercogEphysBuono6.copy()
p['useSecondPopExc'] = False

if p['useNewEphysParams']:
    # remove protected keys from the dict whose params are being imported
    protectedKeys = ('nUnits', 'propInh', 'duration')
    for pK in protectedKeys:
        del ephysParams[pK]
    p.update(ephysParams)

p['useRule'] = 'upCrit'
p['nameSuffix'] = ''
p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/upCrit'
p['saveWithDate'] = True
p['useOldWeightMagnitude'] = True
p['disableWeightScaling'] = True
p['applyLogToFR'] = False
p['setMinimumBasedOnBalance'] = False
p['recordMovieVariables'] = False
p['downSampleVoltageTo'] = 1 * ms

# simulation params
p['nUnits'] = 2e3
p['propConnect'] = 0.5

p['initWeightMethod'] = 'guessBuono6Weights2e3p05Beta10'
# p['initWeightMethod'] = 'guessBuono4Weights2e3p025LogNormalStart2'
p['kickType'] = 'spike'  # kick or spike
p['nUnitsToSpike'] = int(np.round(0.05 * p['nUnits']))
p['timeToSpike'] = 100 * ms
p['timeAfterSpiked'] = 5000 * ms
p['spikeInputAmplitude'] = 0.95  # in nA
p['allowAutapses'] = False

# params not important unless using "kick" instead of "spike"
p['propKicked'] = 0.1
p['onlyKickExc'] = True
p['kickTimes'] = [100 * ms]
p['kickSizes'] = [1]
iKickRecorded = convert_kicks_to_current_series(p['kickDur'], p['kickTau'],
                                                p['kickTimes'], p['kickSizes'], p['duration'], p['dt'])
p['iKickRecorded'] = iKickRecorded

# boring params
p['nIncInh'] = int(p['propConnect'] * p['propInh'] * p['nUnits'])
p['nIncExc'] = int(p['propConnect'] * (1 - p['propInh']) * p['nUnits'])
indUnkickedExc = int(p['nUnits'] - (p['propInh'] * p['nUnits']) - 1)
p['indsRecordStateExc'].append(indUnkickedExc)

# END OF PARAMS

# set RNG seeds...
p['rngSeed'] = rngSeed
rng = np.random.default_rng(rngSeed)  # for numpy
seed(rngSeed)  # for Brian... will insert code to set the random number generator seed into the generated code
p['rng'] = rng

JT = JercogTrainer(p)
# JT.calculate_unit_thresh_and_gain()
JT.set_up_network()
JT.initialize_weight_matrices()
JT.run_upCrit()
JT.save_params()
JT.save_results_upCrit()

R = Results()
R.init_from_file(JT.saveName, JT.p['saveFolder'])

R.calculate_PSTH()
R.calculate_voltage_histogram(removeMode=True, useAllRecordedUnits=True)
R.calculate_upstates()
if len(R.ups) > 0:
    R.reshape_upstates()
    R.calculate_FR_in_upstates()
    print('average FR in upstate for Exc: {:.2f}, Inh: {:.2f} '.format(R.upstateFRExcHist.mean(), R.upstateFRInhHist.mean()))


R.calculate_voltage_histogram(removeMode=True)
R.reshape_upstates()

fig1, ax1 = plt.subplots(5, 1, num=1, figsize=(5, 9),
                         gridspec_kw={'height_ratios': [3, 2, 1, 1, 1]},
                         sharex=True)
R.plot_spike_raster(ax1[0], downSampleUnits=False)  # uses RNG but with a separate random seed
R.plot_firing_rate(ax1[1])
ax1[1].set_ylim(0, 30)
R.plot_voltage_detail(ax1[2], unitType='Exc', useStateInd=0)
R.plot_updur_lines(ax1[2])
R.plot_voltage_detail(ax1[3], unitType='Inh', useStateInd=0)
R.plot_updur_lines(ax1[3])
R.plot_voltage_detail(ax1[4], unitType='Exc', useStateInd=1)
R.plot_updur_lines(ax1[4])
ax1[3].set(xlabel='Time (s)')
R.plot_voltage_histogram_sideways(ax1[2], 'Exc')
R.plot_voltage_histogram_sideways(ax1[3], 'Inh')
fig1.suptitle(R.p['simName'])
uniqueSpikers = np.unique(R.spikeMonExcI).size
totalSpikes = R.spikeMonExcI.size
print(uniqueSpikers, 'neurons fired an average of', totalSpikes / uniqueSpikers, 'spikes')

'''
plt.close('all')
# fig1, ax1 = plt.subplots(2, 1, num=1, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
fig1, ax1 = plt.subplots(2, 1, num=1, figsize=(5, 5), sharex=True)
R.plot_spike_raster(ax1[0], downSampleUnits=True)
R.plot_firing_rate(ax1[1])
fig1.savefig(targetPath + targetFile + '_spikes.tif')
'''
