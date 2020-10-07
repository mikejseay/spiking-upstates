from brian2 import *
from params import paramsJercog as p
from network import JercogNetwork

p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'
p['saveWithDate'] = True

defaultclock.dt = p['dt']

# determine 'up crit' empirically

p['simName'] = 'classicJercogUpCrit'

JN = JercogNetwork(p)
JN.initialize_network()
JN.initialize_units()
JN.initialize_recurrent_synapses()

JN.prepare_upCrit_experiment()

JN.create_monitors()
JN.run()
JN.save_results()
JN.save_params()
