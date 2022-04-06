from brian2 import Hz
from results import Results

targetFile = 'buonoEphysBen1_2000_0p25_upPoisson_resumePrior_lessAdapt_2022-03-07-15-48-20_results'
targetFolder = 'C:/Users/mikejseay/Documents/BrianResults/'
R = Results()
R.init_from_file(targetFile, targetFolder)
R.calculate_PSTH()
R.calculate_voltage_histogram(removeMode=True, removeReset=True, useAllRecordedUnits=False, useExcUnits=1)
R.calculate_upstates()
if len(R.ups) > 0:
    R.reshape_upstates()
    R.calculate_FR_in_upstates()
    infoStr = 'average FR in upstate for Exc: {:.2f}, Inh: {:.2f}, average upDur: {:.2f}, upFreq: {:.2f}'.format(R.upstateFRExcHist.mean(), R.upstateFRInhHist.mean(), R.upDurs.mean(), R.ups.size / R.p['duration'] / Hz)
    print(infoStr)

R.calculate_voltage_histogram(useExcUnits=(2, 4,), useInhUnits=(0,))
R.reshape_upstates()
R.calculate_upFR_units()
R.calculate_upCorr_units()
