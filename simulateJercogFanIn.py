from brian2 import *
from params import paramsJercog as p
from network import JercogNetwork

p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'
p['saveWithDate'] = True

defaultclock.dt = p['dt']

# determine 'fan in' empirically

p['simName'] = 'classicJercogFanIn'

JN = JercogNetwork(p)

# here we override the # of "incoming" exc/inh synapses per unit...
# because the initialize_units defaultly sets the weights based on this number
# but we want to initialize only 2 units (to test)
JN.p['nIncInh'] = int(JN.p['propInh'] * JN.p['nUnits'])
JN.p['nIncExc'] = JN.p['nUnits'] - JN.p['nIncInh']

# we make one Exc/Inh unit each and make them not adapt
JN.p['nUnits'] = 2
JN.p['propInh'] = 0.5
JN.p['adaptStrengthExc'] = 0 * JN.p['adaptStrengthExc']
JN.p['adaptStrengthInh'] = 0 * JN.p['adaptStrengthInh']

JN.initialize_network()
JN.initialize_units()

JN.determine_fan_in(minUnits=540, maxUnits=561, unitSpacing=1, timeSpacing=250 * ms)
JN.save_results()
JN.save_params()
