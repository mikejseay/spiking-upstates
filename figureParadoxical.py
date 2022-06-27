import os
import numpy as np
from results import Results
import matplotlib.pyplot as plt

SAVE_PLOTS = True

resultsPath = os.path.join(os.getcwd(), 'results')

paradoxicalFile = 'userRunParadox_PSTH.npz'

npzObject = np.load(os.path.join(resultsPath, paradoxicalFile), allow_pickle=True)
R = Results()
for savedObjectName in npzObject.files:
    setattr(R, savedObjectName, npzObject[savedObjectName])

timeVector = np.arange(0, 5.1, 0.01)[:-1]
histCenters = timeVector + (timeVector[1] - timeVector[0]) / 2

f, ax = plt.subplots(1, 2, sharex=True, figsize=(10.5, 4.5))

nCurrentValues = len(R.currentValues)
redColors = plt.cm.Reds(np.linspace(0, 1, nCurrentValues + 3))
blueColors = plt.cm.Blues(np.linspace(0, 1, nCurrentValues + 3))

paraStartTime = 3
paraEndTime = 4
paraStartInd = np.argmin(np.abs(histCenters - paraStartTime))
paraEndInd = np.argmin(np.abs(histCenters - paraEndTime))

paraPeriodTrialMeansExc = R.PSTHExc[:, :, paraStartInd:paraEndInd].mean(2)
paraPeriodTrialMeansInh = R.PSTHInh[:, :, paraStartInd:paraEndInd].mean(2)

outlierPSTHExc = paraPeriodTrialMeansExc < 2
outlierPSTHInh = paraPeriodTrialMeansInh < 4

PSTHExcOutNan = R.PSTHExc.copy()
PSTHExcOutNan[outlierPSTHExc, :] = np.nan

PSTHInhOutNan = R.PSTHInh.copy()
PSTHInhOutNan[outlierPSTHInh, :] = np.nan

for currentValueInd in range(len(R.currentValues)):
    currentValue = R.currentValues[currentValueInd]
    useLabel = str(int(currentValue * 1e12)) + ' pA'
    xVals = timeVector
    yValsExc = np.nanmean(PSTHExcOutNan, 1)[currentValueInd, :]
    yValsInh = np.nanmean(PSTHInhOutNan, 1)[currentValueInd, :]
    ax[0].plot(xVals, yValsExc, label=useLabel, color=blueColors[currentValueInd + 3])
    ax[1].plot(xVals, yValsInh, label=useLabel, color=redColors[currentValueInd + 3])

ax[0].legend(loc='upper right', frameon=False)
ax[1].legend(loc='upper right', frameon=False)

ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('E FR (Hz)')
ax[0].set_ylim(0, 17)

ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('I FR (Hz)')
ax[1].set_ylim(0, 35)

f.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.95)

if SAVE_PLOTS:
    saveFileName = 'fig5_paradox.pdf'
    f.savefig(saveFileName, transparent=True)
