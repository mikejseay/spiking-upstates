from brian2 import *
from params import paramsJercog as p
from network import JercogNetwork

p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'
p['saveWithDate'] = True

defaultclock.dt = p['dt']

# determine 'fan in' empirically

# p['simName'] = 'jercogFan'

# p['propConnect'] = 0.05
# p['simName'] = 'jercogFan0p05Conn'

p['propConnect'] = 1
p['simName'] = 'jercogFanFullConn250Units'
p['nUnits'] = 250

# here we override the # of "incoming" exc/inh synapses per unit...
# because the initialize_units defaultly sets the weights based on this number
# but we want to initialize only 2 units (to test)
# these represent the number of incoming excitatory / inhibtory synapses per unit
p['nIncInh'] = int(p['propConnect'] * p['propInh'] * p['nUnits'])
p['nIncExc'] = int(p['propConnect'] * (1 - p['propInh']) * p['nUnits'])

# we make one Exc/Inh unit each and make them not adapt
# note i do this after the above to not mess it up
p['nUnits'] = 2
p['propInh'] = 0.5
p['adaptStrengthExc'] = 0 * p['adaptStrengthExc']
p['adaptStrengthInh'] = 0 * p['adaptStrengthInh']

JN = JercogNetwork(p)
JN.initialize_network()
JN.initialize_units()

# JN.determine_fan_in(minUnits=540, maxUnits=560, unitSpacing=1, timeSpacing=250 * ms)  # *** perfect
JN.determine_fan_in(minUnits=20, maxUnits=30, unitSpacing=1, timeSpacing=250 * ms)  # *** perfect
JN.save_results()
JN.save_params()
