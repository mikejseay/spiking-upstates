from params import (paramsJercog, paramsJercogEphysOrig, paramsJercogEphysBuono,
                    paramsJercogEphysBuonoBen1, paramsJercogEphysBuonoBen2)
from network import JercogEphysNetwork
from results import ResultsEphys
import matplotlib.pyplot as plt
import numpy as np
from brian2 import nA, ms

USE_NEW_EPHYS_PARAMS = True

useParams = paramsJercog.copy()

# remove protected keys from the dict whose params are being imported
# ephysParams = paramsJercogEphysOrig.copy()
ephysParams = paramsJercogEphysBuono.copy()
# ephysParams = paramsJercogEphysBuonoBen1.copy()
# ephysParams = paramsJercogEphysBuonoBen2.copy()
protectedKeys = ('nUnits', 'propInh', 'duration')
for pK in protectedKeys:
    del ephysParams[pK]

if USE_NEW_EPHYS_PARAMS:
    useParams.update(ephysParams)

useParams['propInh'] = 0.5
useParams['duration'] = 250 * ms
useParams['iExtRange'] = np.linspace(0, .3, 301) * nA

JEN = JercogEphysNetwork(useParams)
JEN.build_classic()
JEN.run()
JEN.save_results()
JEN.save_params()

R = ResultsEphys()
R.init_from_file(JEN.saveName, JEN.p['saveFolder'])
R.calculate_thresh_and_gain()

print(R.threshExc)
print(R.threshInh)
print(R.gainExc)
print(R.gainInh)

fig1, ax1 = plt.subplots(2, 2, num=1)

R.calculate_and_plot(fig1, ax1)
