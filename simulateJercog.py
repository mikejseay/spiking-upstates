from brian2 import *
from params import paramsJercog as p
from generate import generate_poisson_kicks_jercog, convert_kicks_to_current_series
from network import JercogNetwork

p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'
p['simName'] = 'classicJercog'
p['saveWithDate'] = True

defaultclock.dt = p['dt']

kickTimes, kickSizes = generate_poisson_kicks_jercog(p['kickLambda'], p['duration'],
                                                     p['kickMinimumISI'], p['kickMaximumISI'])
print(kickTimes)
p['kickTimes'] = kickTimes
p['kickSizes'] = kickSizes
iKickRecorded = convert_kicks_to_current_series(p['kickDur'], p['kickTau'],
                                                p['kickTimes'], p['kickSizes'], p['duration'], p['dt'])

p['iKickRecorded'] = iKickRecorded

N = JercogNetwork(p)
N.build()
N.run()
N.save_results()
N.save_params()
