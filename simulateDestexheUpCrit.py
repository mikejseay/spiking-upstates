from brian2 import *
from params import paramsDestexhe as p
from network import DestexheNetwork

p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'
p['saveWithDate'] = True

defaultclock.dt = p['dt']

# determine 'up crit' empirically

p['simName'] = 'destexheUpCrit0p05Conn5e4units'
p['nUnits'] = 5e4

DN = DestexheNetwork(p)
DN.initialize_network()
DN.initialize_units()
DN.initialize_recurrent_synapses2()

DN.prepare_upCrit_experiment(minUnits=1000, maxUnits=1400, unitSpacing=200, timeSpacing=3000 * ms)

DN.create_monitors()
DN.run()
DN.save_results()
DN.save_params()
