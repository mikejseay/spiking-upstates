from brian2 import *
from params import paramsJercog as p
from network import JercogNetwork

p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'
p['saveWithDate'] = True

defaultclock.dt = p['dt']

# determine 'up crit' empirically

# p['simName'] = 'jercogUpCrit'

# p['propConnect'] = 1
# p['simName'] = 'jercogUpCritFullConn500Units'
# p['nUnits'] = 500

# p['propConnect'] = 1
# p['simName'] = 'jercogUpCritFullConn250Units'
# p['nUnits'] = 250

p['propConnect'] = 0.05
p['simName'] = 'jercogUpCrit0p05'
p['adaptStrengthExc'] = 15 * mV
p['jEE'] = p['jEE'] * 0.75
p['jIE'] = p['jIE'] * 0.75
# p['jEI'] = p['jEI'] * 0.9
# p['jII'] = p['jII'] * 0.9

p['nIncInh'] = int(p['propConnect'] * p['propInh'] * p['nUnits'])
p['nIncExc'] = int(p['propConnect'] * (1 - p['propInh']) * p['nUnits'])

JN = JercogNetwork(p)
JN.initialize_network()
JN.initialize_units()
JN.initialize_recurrent_synapses2()

# JN.prepare_upCrit_experiment(minUnits=200, maxUnits=1000, unitSpacing=200, timeSpacing=3000 * ms)
JN.prepare_upCrit_experiment(minUnits=210, maxUnits=240, unitSpacing=10, timeSpacing=3000 * ms)

JN.create_monitors()
JN.run()
JN.save_results()
JN.save_params()
