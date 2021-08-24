from params import (paramsJercog, paramsJercogEphysOrig, paramsJercogEphysBuono, paramsJercogEphysBuono2,
                    paramsJercogEphysBuono3, paramsJercogEphysBuono4, paramsJercogEphysBuono5, paramsJercogEphysBuono6,
                    paramsJercogEphysBuonoBen11, paramsJercogEphysBuonoBen21,
                    paramsJercogEphysBuono22, paramsJercogBen)
from network import JercogEphysNetwork
from results import ResultsEphys
import matplotlib.pyplot as plt
import numpy as np
from brian2 import nA, ms

USE_NEW_EPHYS_PARAMS = True

useParams = paramsJercog.copy()

# remove protected keys from the dict whose params are being imported
# ephysParams = paramsJercogEphysOrig.copy()
# ephysParams = paramsJercogBen.copy()
# ephysParams = paramsJercogEphysBuono.copy()
# ephysParams = paramsJercogEphysBuono2.copy()
# ephysParams = paramsJercogEphysBuono3.copy()
ephysParams = paramsJercogEphysBuono6.copy()
# ephysParams = paramsJercogEphysBuonoBen11.copy()
# ephysParams = paramsJercogEphysBuonoBen21.copy()
# ephysParams = paramsJercogEphysBuono22.copy()
protectedKeys = ('nUnits', 'propInh', 'duration')
for pK in protectedKeys:
    del ephysParams[pK]

if USE_NEW_EPHYS_PARAMS:
    useParams.update(ephysParams)

useParams['propInh'] = 0.5
useParams['duration'] = 250 * ms
useParams['iExtRange'] = np.linspace(0, .3, 301) * nA
useParams['useSecondPopExc'] = True

JEN = JercogEphysNetwork(useParams)
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
    fig1, ax1 = plt.subplots(2, 3, num=1)
    R.calculate_and_plot_secondExcPop(fig1, ax1)
    print('excThresh2:', R.threshExc2)
    print('excGain2:', R.gainExc2)
    print('excTau2:', R.p['membraneCapacitanceExc2'] / R.p['gLeakExc2'])
else:
    fig1, ax1 = plt.subplots(2, 2, num=1)
    R.calculate_and_plot(fig1, ax1)
