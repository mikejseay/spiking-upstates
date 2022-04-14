from params import paramsDestexhe, paramsDestexheEphysBuono
from network import DestexheEphysNetwork
from results import ResultsEphys
import matplotlib.pyplot as plt
from brian2 import ms, nA
import numpy as np

p = paramsDestexhe.copy()

p['duration'] = 250 * ms
p['nUnits'] = 2
p['propInh'] = 0.5
p['baselineDur'] = 100 * ms
p['iDur'] = 250 * ms
p['afterDur'] = 100 * ms
p['iExtRange'] = np.linspace(-.1, .3, 301) * nA
p['useSecondPopExc'] = False


DEN = DestexheEphysNetwork(p)
DEN.build_classic()
DEN.run()
DEN.save_results()
DEN.save_params()

R = ResultsEphys()
R.init_from_file(DEN.saveName, DEN.p['saveFolder'])

fig1, ax1 = plt.subplots(2, 2, num=1)

R.calculate_and_plot(fig1, ax1)
