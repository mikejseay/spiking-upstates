from params import paramsJercog, paramsJercogEphysOrig, paramsJercogEphysBuono
from network import JercogEphysNetwork
from results import ResultsEphys
import matplotlib.pyplot as plt
import numpy as np
from brian2 import nA, ms

useParams = paramsJercog.copy()

if 'iExtRange' not in useParams:
    useParams['propInh'] = 0.5
    useParams['duration'] = 250 * ms
    useParams['iExtRange'] = np.linspace(0, .3, 31) * nA

DEN = JercogEphysNetwork(useParams)
DEN.build_classic()
DEN.run()
DEN.save_results()
DEN.save_params()

R = ResultsEphys(DEN.saveName, DEN.p['saveFolder'])

fig1, ax1 = plt.subplots(2, 2, num=1)

R.calculate_and_plot(fig1, ax1)
