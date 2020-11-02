from brian2 import *
from params import paramsDestexhe as p
from network import DestexheNetwork

p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'
p['saveWithDate'] = True

defaultclock.dt = p['dt']

# determine 'fan in' empirically

p['simName'] = 'classicDestexheFanIn'

# p['propConnect'] = 0.1  # recurrent connection probability
# p['simName'] = 'classicDestexheFanInFullConn'

# we set the number of units that will be used to set the weight
# ONLY FOR THE FAN IN EXPERIMENT
p['nInhFan'] = int(p['propInh'] * p['nUnits'])
p['nExcFan'] = int(p['nUnits'] - p['nInhFan'])

# we make one Exc/Inh unit each and make them not adapt
p['nUnits'] = 2
p['propInh'] = 0.5
p['bExc'] = 0 * pA
p['bInh'] = 0 * pA

DN = DestexheNetwork(p)
DN.initialize_network()
DN.initialize_units()

DN.determine_fan_in(minUnits=234, maxUnits=244, unitSpacing=1, timeSpacing=250 * ms)
DN.save_results()
DN.save_params()
