"""
a script for comparing the parameters used for two different simulations
"""

from results import Results
from brian2.units.fundamentalunits import DimensionMismatchError

loadFolder1 = 'C:/Users/mikejseay/Documents/BrianResults/'
targetSim1 = 'jercogUpCritFullConn500Units_2020-10-26-17-07'

R1 = Results(targetSim1, loadFolder1)

loadFolder2 = 'C:/Users/mikejseay/Documents/BrianResults/'
targetSim2 = 'jercogUpCritFullConn500Units_2021-04-09-11-30'

R2 = Results(targetSim2, loadFolder2)

for k1, v1, in R1.p.items():
    try:
        v2 = R2.p[k1]
    except KeyError:
        print(k1, 'was not in R2')
        continue

    try:
        comparisonBool = v2 != v1
        if comparisonBool:
            print('the value for R1 of key', k1, 'was', v1, 'but for R2 it was ', v2)
    except ValueError:
        print('the key', k1, 'wasnt able to be compared properly')
        continue
    except DimensionMismatchError:
        print('the key', k1, 'wasnt able to be compared properly')
        print('v1 was', v1, 'and v2 was', v2)
