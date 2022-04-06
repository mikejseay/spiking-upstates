from params import paramsJercogEphysBuono
from network import JercogEphysNetwork
from results import ResultsEphys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from brian2 import nA, ms
import os
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

p = paramsJercogEphysBuono.copy()

# set params to be appropriate for the ephys experiment
p['propInh'] = 0.5
p['baselineDur'] = 100 * ms
p['iDur'] = 250 * ms
p['afterDur'] = 100 * ms
p['iExtRange'] = np.linspace(-.1, .3, 301) * nA
p['useSecondPopExc'] = True
p['saveFolder'] = os.path.join(os.getcwd(), 'results')

JEN = JercogEphysNetwork(p)
JEN.build_classic()
JEN.run()
JEN.save_results()
JEN.save_params()

R = ResultsEphys()
R.init_from_file(JEN.saveName, JEN.p['saveFolder'])
R.calculate_thresh_and_gain()

print('excThresh:', R.threshExc)
print('excGain:', R.gainExc)
print('excTau:', R.p['membraneCapacitanceExc'] / R.p['gLeakExc'])

print('inhThresh:', R.threshInh)
print('inhGain:', R.gainInh)
print('inhTau:', R.p['membraneCapacitanceInh'] / R.p['gLeakInh'])

if R.p['useSecondPopExc']:
    fig1, ax1 = plt.subplots(2, 3, figsize=(8, 6), num=1)
    R.calculate_and_plot_secondExcPop(fig1, ax1)
    print('excThresh2:', R.threshExc2)
    print('excGain2:', R.gainExc2)
    print('excTau2:', R.p['membraneCapacitanceExc2'] / R.p['gLeakExc2'])
else:
    fig1, ax1 = plt.subplots(2, 3, figsize=(8, 6), num=1)
    R.calculate_and_plot_multiVolt(fig1, ax1)

SAVE_PLOT = True
if SAVE_PLOT:
    fig1.savefig('fig5bc.pdf', transparent=True)
