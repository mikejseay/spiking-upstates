import os
import numpy as np
from results import Results
from analysis import detail_trial_plot, FR_weights, weights_matrix_compare, summed_weights_scatter, FR_hist1d

SAVE_PLOTS = True
DETAIL_PLOT_TRIALS = [0, 200, 400, 1500]

resultsPath = os.path.join(os.getcwd(), 'results')
trainingFile = 'buonoEphysBen1_2000_0p25_cross-homeo-pre-outer-homeo_goodCrossHomeoExamp_buonoParams_2022-05-27-01-42-18_results'

R = Results()
R.init_from_file(trainingFile, resultsPath)

saveTrials = R.p['saveTrials'] + 1
for actualTrialInd in DETAIL_PLOT_TRIALS:
    useTrialInd = np.where(saveTrials == actualTrialInd)[0][0]
    f0, ax0 = detail_trial_plot(R, useTrialInd=useTrialInd)
    if SAVE_PLOTS:
        f0.savefig('fig0_trial' + str(actualTrialInd) + '.pdf', transparent=True)
    f0.clf()

f1, ax1 = FR_weights(R, startTrialInd=0, endTrialInd=-1, movAvgWidth=5)
f2, ax2 = weights_matrix_compare(R)
f3, ax3 = summed_weights_scatter(R)
f4, ax4 = FR_hist1d(R)

if SAVE_PLOTS:
    f1.savefig('fig1.pdf', transparent=True)
    f2.savefig('fig2.pdf', transparent=True)
    f3.savefig('fig3.pdf', transparent=True)
    f4.savefig('fig4.pdf', transparent=True)
