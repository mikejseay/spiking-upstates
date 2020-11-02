from brian2 import *
from params import paramsJercog as p
from generate import generate_poisson_kicks_jercog, convert_kicks_to_current_series
from network import JercogNetwork

p['saveFolder'] = 'C:/Users/mikejseay/Documents/BrianResults/'

p['saveWithDate'] = True
p['duration'] = 10 * second
p['refractoryPeriod'] = 0 * ms  # 0 by default, 1 works, 2 causes explosion
# on pConn = 0.05, 3 also works but no higher

# p['propConnect'] = 1
# p['simName'] = 'jercogFullConn'

p['propConnect'] = 0.05
p['simName'] = 'jercog0p05Conn'
p['propKicked'] = 0.1
p['onlyKickExc'] = False
p['adaptStrengthExc'] = 15 * mV
p['jEE'] = p['jEE'] * 0.75
p['jIE'] = p['jIE'] * 0.75
KICK_TYPE = 'kick'  # kick or spike

# THESE SETTINGS WORK FOR PCONN = 0.05 WITH REASONABLE FIRING RATE!!!

# p['propConnect'] = 0.05
# p['simName'] = 'jercog0p05Conn'
# p['propKicked'] = 0.05
# p['onlyKickExc'] = True
# p['adaptStrengthExc'] = 15 * mV
# p['jEE'] = p['jEE'] * 0.75
# p['jIE'] = p['jIE'] * 0.75
# KICK_TYPE = 'spike'  # kick or spike

# p['propConnect'] = 0.05
# p['simName'] = 'jercog0p05ConnSmallWeights'
# p['scaleWeightsByPConn'] = False

# p['propConnect'] = 1
# p['simName'] = 'jercogFullConn1kUnits'
# p['nUnits'] = 1000

# p['propConnect'] = 1
# p['simName'] = 'jercogFullConn500Units'
# p['nUnits'] = 500
# p['propKicked'] = 0.05
# p['onlyKickExc'] = True

# p['propConnect'] = 1
# p['simName'] = 'jercogFullConn250Units'
# p['nUnits'] = 250
# p['propKicked'] = 0.08
# p['onlyKickExc'] = True

# these represent the number of incoming excitatory / inhibtory synapses per unit
p['nIncExc'] = int(p['propConnect'] * (1 - p['propInh']) * p['nUnits'])
p['nIncInh'] = int(p['propConnect'] * p['propInh'] * p['nUnits'])

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

if KICK_TYPE == 'kick':
    JN.set_kicked_units(onlyKickExc=p['onlyKickExc'])
elif KICK_TYPE == 'spike':
    JN.set_spiked_units(onlySpikeExc=p['onlyKickExc'])

JN.initialize_recurrent_synapses()
JN.create_monitors()
JN.run()
JN.save_results()
JN.save_params()
