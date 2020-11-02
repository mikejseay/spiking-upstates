from brian2 import *
from params import paramsJercog as p
from generate import generate_poisson_kicks_jercog, convert_kicks_to_current_series
from network import JercogNetwork

p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'
# p['simName'] = 'classicJercog'
p['saveWithDate'] = True
p['duration'] = 10 * second
p['refractoryPeriod'] = 0 * ms  # 0 by default, 1 works, 2 causes explosion

# p['propConnect'] = 1
# p['simName'] = 'classicJercogFullConn'

# p['propConnect'] = 0.05
# p['simName'] = 'classicJercog0p05Conn'
# p['scaleWeightsByPConn'] = True

# p['propConnect'] = 0.05
# p['simName'] = 'classicJercog0p05ConnSmallWeights'
# p['scaleWeightsByPConn'] = False

p['propConnect'] = 1
# p['simName'] = 'classicJercogFullConn250Units'
# p['nUnits'] = 250

p['nIncInh'] = int(p['propInh'] * p['nUnits'])
p['nIncExc'] = p['nUnits'] - p['nIncInh']

defaultclock.dt = p['dt']

kickTimes, kickSizes = generate_poisson_kicks_jercog(p['kickLambda'], p['duration'],
                                                     p['kickMinimumISI'], p['kickMaximumISI'])
print(kickTimes)
p['kickTimes'] = kickTimes
p['kickSizes'] = kickSizes
iKickRecorded = convert_kicks_to_current_series(p['kickDur'], p['kickTau'],
                                                p['kickTimes'], p['kickSizes'], p['duration'], p['dt'])

p['iKickRecorded'] = iKickRecorded

JN = JercogNetwork(p)
JN.initialize_network()
JN.initialize_units()
JN.set_kicked_units()
JN.initialize_recurrent_synapses()
JN.create_monitors()
JN.run()
JN.save_results()
JN.save_params()
