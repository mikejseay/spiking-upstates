from params import paramsDestexheEphysOrig, paramsDestexheEphysBuono
from network import DestexheEphysNetwork
from results import ResultsEphys
import matplotlib.pyplot as plt

DEN = DestexheEphysNetwork(paramsDestexheEphysBuono)
DEN.build_classic()
DEN.run()
DEN.save_results()
DEN.save_params()

R = ResultsEphys(DEN.saveName, DEN.p['saveFolder'])

fig1, ax1 = plt.subplots(2, 2, num=1)

R.calculate_and_plot(fig1, ax1)
