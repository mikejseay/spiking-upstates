from results import Results

loadFolder = 'C:/Users/mikejseay/Documents/BrianResults/long_files/'
targetSim = 'destexheEphysBuono_2020-11-10-12-59'

R1 = Results(targetSim, loadFolder)

loadFolder = 'C:/Users/mikejseay/Documents/BrianResults/'
targetSim = 'destexheEphysBuono_2020-11-18-16-31'

R2 = Results(targetSim, loadFolder)

for k1, v1, in R1.p.items():
    try:
        v2 = R2.p[k1]
    except KeyError:
        print(k1, 'was not in R2')
        continue
    comparisonBool = v2 != v1
    try:
        if comparisonBool:
            print('the value for R1 of key', k1, 'was', v1, 'but for R2 it was ', v2)
    except ValueError:
        print('the key', k1, 'wasnt able to be compared properly')
        continue
