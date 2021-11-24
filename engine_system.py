from spearbis import *
import numpy.random as rnd
import matplotlib.pyplot as plt


# Definition of constants:

THRESHOLD = 99.8


MIN = 0
MAX = 100
IOS = 3
STRESS_INCR = 0.02
# If in the last 6 instants the temperature of the engine is > MAX for more than IOS (Instants Of Stress) instants, then
# the level of stress is incremented by STRESS_INCR (reaching at most value 1).


HALF = 0
FULL = 1
SLOW = -1
# possible values of actuator speed

ON = 1
OFF = -1
# possible values of actuator cool


HOT = 1
OK = 0
# possible values of channels ch_warning

NONE = 0
LEFT = 1
RIGHT = 1
BOTH = 2

AC = 1.8
TF = 0.4
# constants used by attackers



# Initial data space:
init_data = {# L stays for "Left Engine", "R" for "Right Engine"
             'comp_steps':0, # counter of number of steps
             #
             'temp_L':95, # temperature sensor; the IDS can access this tamperproof value
             'tempfake_L':0, # noise introduced by malicious activity tampering sensor temp_L
             'ch_temp_L':95, # channel from sensor temp_L to controller Ctrl-L. Its value is temp_L + tempfake_L.
             'speed_L':HALF, # speed actuator
             'cool_L':OFF, # cooling actuator
             'stress_L':0, # variable recording the accumulated stress
             'tot_stress_instants_L':0, # variable recording the number of steps with the stress condition
             'p1_L':95, 'p2_L':95, 'p3_L':95, # earlier values of temperature
             'p4_L':95, 'p5_L':95, 'p6_L':95, # earlier values of temperature
             'warn_L':OK, # warning raised by IDS if temp_L is high and cool is off
             'totwarn_L':0, # variable recording the accumulated number of warnings
             'ch_speed_L':HALF, # channel used by IDS to command Ctrl to regulate the speed
             #
             'temp_R':95,
             'tempfake_R':0,
             'ch_temp_R':95,
             'speed_R':HALF,
             'cool_R':OFF,
             'stress_R':0,
             'p1_R':95, 'p2_R':95, 'p3_R':95,
             'p4_R':95, 'p5_R':95, 'p6_R':95,
             'warn_R':OK,
             'totwarn_R':0,
             'ch_speed_R':HALF,
             #
             'ch_speed_L_to_R':HALF, # channel used by IDS_L to ask to Ctrl_R to work at FULL speed
             'ch_speed_R_to_L':HALF, # channel used by IDS_R to ask to Ctrl_L to work at FULL speed
             'fn_L':0, # value of false negatives based on warn_L and stress_L
             'fn_R':0, # value of false negatives based on warn_L and stress_L
             'fn':0, # value of false negatives based on alarm, stress_L and stress_R
             'fp_L':0, # value of false positives based on warn_L and stress_L
             'fp_R':0, # value of false positives based on warn_R and stress_R
             'fp':0, # value of false positives based on alarm, stress_L and stress_R
             'attack_window':0,
}



def rank_avg_stress_L(ds):
    if(ds['comp_steps'] > 0):
       return ds['tot_stress_instants_L'] / ds['comp_steps']
    else:
       return 0
 
def rank_temp_L(ds):
    return ds['temp_L']

def rank_stress_L(ds):
    return ds['stress_L']

def rank_stress_R(ds):
    return ds['stress_R']
    
def rank_stress(ds):
    return 0.7 * ds['stress_L'] + 0.7 * ds['stress_R']
    
def rank_warn_L(ds):
    return ds['totwarn_L']

def rank_warn_R(ds):
    return ds['totwarn_R']
    
def rank_warn(ds):
    return 0.7 * ds['totwarn_L'] + 0.7 * ds['totwarn_R']

def rank_fnL(ds):
    return ds['fn_L']
    
def rank_fpL(ds):
    return ds['fp_L']
    
def rank_fnR(ds):
    return ds['fn_R']
    
def rank_fpR(ds):
    return ds['fp_R']
    
def rank_fn(ds):
    return ds['fn']
    
def rank_fp(ds):
    return ds['fp']
    






# Evolution function:
def engine_environment(ds):
    
    # Temperature evolution for Left Engine:
    if ds['cool_L'] == ON:
        m_L = rnd.uniform(-1.2,-0.8)
    if ds['cool_L'] == OFF and ds['speed_L'] == SLOW:
        m_L = rnd.uniform(0.1,0.3)
    if ds['cool_L'] == OFF and ds['speed_L'] == HALF:
        m_L = rnd.uniform(0.3,0.7)
    if ds['cool_L'] == OFF and ds['speed_L'] == FULL:
        m_L = rnd.uniform(0.7,1.2)
    newtemp_L = ds['temp_L']+m_L
    newch_temp_L = newtemp_L + ds['tempfake_L']
    # the temperature decreases by m_L \in [0.8,1.2] degrees if the cooling is on and increases by m_L \in [a,b] degrees if the cooling is off, where a and b depends on the speed
    # the value ch_temp_L obtained by controller Ctrl is the value of sensor temp_L possibly corrupted by some malicious activity
   
   
    # Temperature evolution for Right Engine:
    if ds['cool_R'] == ON:
        m_R = rnd.uniform(-1.2,-0.8)
    if ds['cool_R'] == OFF and ds['speed_R'] == SLOW:
        m_R = rnd.uniform(0.1,0.3)
    if ds['cool_R'] == OFF and ds['speed_R'] == HALF:
        m_R = rnd.uniform(0.3,0.7)
    if ds['cool_R'] == OFF and ds['speed_R'] == FULL:
        m_R = rnd.uniform(0.7,1.2)
    newtemp_R = ds['temp_R']+m_R
    newch_temp_R = newtemp_R + ds['tempfake_R']
    
    
    # Stress evolution for Left Engine:
    instants_of_stress_L = 0
    if ds['p1_L'] > MAX:
        instants_of_stress_L +=1
    if ds['p2_L'] > MAX:
        instants_of_stress_L +=1
    if ds['p3_L'] > MAX:
        instants_of_stress_L +=1
    if ds['p4_L'] > MAX:
        instants_of_stress_L +=1
    if ds['p5_L'] > MAX:
        instants_of_stress_L +=1
    if ds['p6_L'] > MAX:
        instants_of_stress_L +=1
    if instants_of_stress_L > IOS:
        newstress_L = min(ds['stress_L'] + STRESS_INCR,1)
        newstress_instantsL = ds['tot_stress_instants_L'] + 1
    else:
        newstress_L = ds['stress_L']
        newstress_instantsL = ds['tot_stress_instants_L']
    # the stress is incremented by STRESS_INCR iff in the past 6 instants of time the temperature has been above MAX for more than IOS instants
    
    
    # Stress evolution for Right Engine:
    instants_of_stress_R = 0
    if ds['p1_R'] > MAX:
        instants_of_stress_R +=1
    if ds['p2_R'] > MAX:
        instants_of_stress_R +=1
    if ds['p3_R'] > MAX:
        instants_of_stress_R +=1
    if ds['p4_R'] > MAX:
        instants_of_stress_R +=1
    if ds['p5_R'] > MAX:
        instants_of_stress_R +=1
    if ds['p6_R'] > MAX:
        instants_of_stress_R +=1
    if instants_of_stress_R > IOS:
        newstress_R = min(ds['stress_R'] + STRESS_INCR,1)
    else:
        newstress_R = ds['stress_R']
    
    
    # Evolution of actuators, channels and alarms:
    newcool_L = ds['cool_L']
    newspeed_L = ds['speed_L']
    newwarn_L = ds['warn_L']
    newch_speed_L = ds['ch_speed_L']
    newtempfake_L = ds['tempfake_L']
    newcool_R = ds['cool_R']
    newspeed_R = ds['speed_R']
    newwarn_R = ds['warn_R']
    newch_speed_R = ds['ch_speed_R']
    newtempfake_R = ds['tempfake_R']
    newch_speed_R_to_L = ds['ch_speed_R_to_L']
    newch_speed_L_to_R = ds['ch_speed_L_to_R']
    new_attack_window = ds['attack_window']
    # actuators, channels, alarms are not modified by environment activity
    
    
    # Evolution of variable memorising earlier temperatures:
    newp1_L = ds['temp_L']
    newp2_L = ds['p1_L']
    newp3_L = ds['p2_L']
    newp4_L = ds['p3_L']
    newp5_L = ds['p4_L']
    newp6_L = ds['p5_L']
    newp1_R = ds['temp_R']
    newp2_R = ds['p1_R']
    newp3_R = ds['p2_R']
    newp4_R = ds['p3_R']
    newp5_R = ds['p4_R']
    newp6_R = ds['p5_R']
    # variables memorising past value of temperature are updated as expected
    
    
    # Evolution of total number of warnings
    if ds['warn_L'] == HOT:
        newtotwarn_L = ds['totwarn_L'] +1
    else:
        newtotwarn_L = ds['totwarn_L']
    if ds['warn_R'] == HOT:
        newtotwarn_R = ds['totwarn_R'] +1
    else:
        newtotwarn_R = ds['totwarn_R']
    # totwarn_X incremented iff warn_X in HOT
    
    
    # Evolution of number of computation steps:
    newcompsteps = ds['comp_steps'] + 1
    
    # Evoution of false negatives / false positivees:
    newfn_L = ((newcompsteps -1 ) * ds['fn_L'] + max(0,ds['stress_L'] - ds['warn_L']))/newcompsteps
    newfn_R = ((newcompsteps -1 ) * ds['fn_R'] + max(0,ds['stress_R'] - ds['warn_R']))/newcompsteps
    newfp_L = ((newcompsteps -1 ) * ds['fp_L'] + max(0,-ds['stress_L'] + ds['warn_L']))/newcompsteps
    newfp_R = ((newcompsteps -1 ) * ds['fp_R'] + max(0,-ds['stress_R'] + ds['warn_R']))/newcompsteps
    if ds['alarm'] == BOTH or ds['alarm'] == LEFT:
       l = 1
    else:
       l = 0
    if ds['alarm'] == BOTH or ds['alarm'] == RIGHT:
       r = 1
    else:
       r = 0
    newfn = ((newcompsteps -1 ) * ds['fn'] + 0.7 * max(0,ds['stress_L'] - l) +
        0.7 * max(0,ds['stress_R'] - r))/newcompsteps
    newfp = ((newcompsteps -1 ) * ds['fp'] + 0.7 * max(0,l-ds['stress_L']) +
        0.7 * max(0,r-ds['stress_R']))/newcompsteps
    
    
    return{
      'temp_L':newtemp_L, 'tempfake_L':newtempfake_L, 'ch_temp_L':newch_temp_L,
      'speed_L':newspeed_L, 'cool_L':newcool_L, 'stress_L':newstress_L,
      'p1_L':newp1_L, 'p2_L':newp2_L, 'p3_L':newp3_L,
      'p4_L':newp4_L, 'p5_L':newp5_L,'p6_L':newp6_L,
      'warn_L':newwarn_L, 'totwarn_L':newtotwarn_L, 'ch_speed_L':newch_speed_L,
      'temp_R':newtemp_R, 'tempfake_R':newtempfake_R, 'ch_temp_R':newch_temp_R,
      'speed_R':newspeed_R, 'cool_R':newcool_R, 'stress_R':newstress_R,
      'p1_R':newp1_R, 'p2_R':newp2_R, 'p3_R':newp3_R,
      'p4_R':newp4_R,'p5_R':newp5_R,'p6_R':newp6_R,
      'warn_R':newwarn_R, 'totwarn_R':newtotwarn_R,'ch_speed_R':newch_speed_R,
      'ch_speed_R_to_L':newch_speed_R_to_L, 'ch_speed_L_to_R':newch_speed_L_to_R,
      'fn_L':newfn_L, 'fn_R':newfn_R, 'fn':newfn,
      'fp_L':newfp_L, 'fp_R':newfn_R, 'fp':newfp,
      'comp_steps':newcompsteps, 'attack_window':new_attack_window, 'tot_stress_instants_L':newstress_instantsL,
     }



# Processes:
processes = {
    # Controller of Left Engine:
    'Ctrl_L': if_then_else_process(lambda d: d['ch_temp_L'] > THRESHOLD,
        act_process({'cool_L': lambda d: ON}, 'Cooling1_L'),
        act_process({}, 'Check_L')),
    'Cooling1_L': act_process({}, 'Cooling2_L'),
    'Cooling2_L': act_process({}, 'Cooling3_L'),
    'Cooling3_L': act_process({}, 'Cooling4_L'),
    'Cooling4_L': act_process({}, 'Check_L'),
    # If the (possibly corrupted) temperature ch_temp_L detected by controller is above THRESHOLD then the actuator cool_L is swtiched on and maintained on for 5 instants. Then, a check phase starts
    
    'Check_L': if_then_else_process(lambda d: d['ch_speed_L'] == SLOW,
        act_process({'speed_L': lambda d: SLOW, 'cool_L': lambda d: OFF}, 'Ctrl_L'),
        act_process({'speed_L': lambda d: d['ch_speed_R_to_L'], 'cool_L': lambda d: OFF}, 'Ctrl_L')),
    # This is the check phase of engine L. The actuator cool is set of OFF.
    # Then, if IDS_L, by using channel ch_speed_L, orders to Ctrl_L to slow down the speed, then the actuator speed_L is set to SLOW. Otherwise, if Ctrl_L receives from the right ids IDS_R and through chaneel ch_speed_R_to_L a speed-up request, then this is satisfied.
    
    
    # Controller of Right Engine:
    'Ctrl_R': if_then_else_process(lambda d: d['ch_temp_R'] > THRESHOLD,
        act_process({'cool_R': lambda d: ON}, 'Cooling1_R'),
        act_process({}, 'Check_R')),
    'Cooling1_R': act_process({}, 'Cooling2_R'),
    'Cooling2_R': act_process({}, 'Cooling3_R'),
    'Cooling3_R': act_process({}, 'Cooling4_R'),
    'Cooling4_R': act_process({}, 'Check_R'),
        
    'Check_R': if_then_else_process(lambda d: d['ch_speed_R'] == SLOW,
        act_process({'speed_R': lambda d: SLOW, 'cool_R': lambda d: OFF}, 'Ctrl_R'),
        act_process({'speed_R': lambda d: d['ch_speed_L_to_R'], 'cool_R': lambda d: OFF}, 'Ctrl_R')),
    
    
    # IDS of Left Engine:
    'IDS_L': if_then_else_process(lambda d: d['temp_L'] > MAX and d['cool_L'] == OFF,
        act_process({'warn_L': lambda d: HOT, 'ch_speed_L': lambda d: SLOW, 'ch_speed_L_to_R': lambda d: FULL}, 'IDS_L'),
        act_process({'warn_L': lambda d: OK, 'ch_speed_L': lambda d: HALF, 'ch_speed_L_to_R': lambda d: HALF}, 'IDS_L')),
     # If IDS_L realises that the temperature temp_L is above MAX and the actuator cool_L is OFF, then it raises an alarm through warn_L, it orders to Ctrl_L to slow down and it asks to Ctrl_R to speed up (in order to compensate the slow down of L).
    
    # IDS of Right Engine:
    'IDS_R': if_then_else_process(lambda d: d['temp_R'] > MAX and d['cool_R'] == OFF,
        act_process({'warn_R': lambda d: HOT, 'ch_speed_R': lambda d: SLOW, 'ch_speed_R_to_L': lambda d: FULL}, 'IDS_R'),
        act_process({'warn_R': lambda d: OK, 'ch_speed_R': lambda d: HALF, 'ch_speed_R_to_L': lambda d: HALF}, 'IDS_R')),
    
    # Supervisor:
    'SV': if_then_else_process(lambda d: d['warn_R'] == HOT and d['warn_L'] == HOT,
             act_process({'alarm': lambda d: BOTH}, 'SV'),
             if_then_else_process(lambda d: d['warn_L'] == HOT,
                act_process({'alarm': lambda d: LEFT}, 'SV'),
                if_then_else_process(lambda d: d['warn_R'] == HOT,
                   act_process({'alarm': lambda d: RIGHT}, 'SV'),
                   act_process({'alarm': lambda d: NONE}, 'SV')
                )
             )
          ),
     # The supervisor SV sets the alarm to: BOTH, if both IDSs raise the alarm; LEFT, if only IDS_L raises the alarm; RIGHT, if only IDS_R raises the alarm; NONE, otherwise.
     
     # Attacker tampering actuator cool_L
    'IntActAtt_L': if_then_else_process(lambda d: d['temp_L'] < MAX - AC,
        act_process({'cool_L': lambda d: OFF}, 'IntActAtt_L'),
        act_process({}, 'IntActAtt_L')),
    # This is an integrity attack on actuator cool_L. The malicious activity interrupts the 5-instants cooling activity when the temperatues goes below MAX - 0.5. The aim is to force the temperature to go again above MAX in order to stress the system. This is stealthy attack, since the IDS is alerted only when the temperature goes above MAX.
    
     # Attacker tampering actuator cool_R
    'IntActAtt_R': if_then_else_process(lambda d: d['temp_R'] < MAX - AC,
        act_process({'cool_R': lambda d: OFF}, 'IntActAtt_R'),
        act_process({}, 'IntActAtt_R')),
    
    
     # Attacker tampering sensor temp_L
     'IntSenAtt_L': act_process({'tempfake_L': lambda d: -TF}, 'IntSenAtt_L'),
      # This is an integrity attack on sensor temp_L. The malicious activity consists in preventing the controller to access the real value of the sensor, by introducing an error of -1.6 degrees. This attack is not stealthy, the IDS accesses the sensor in a tamperproof way, therefore it realises that the temperature is high and the actuator cool_L is off.
      
     # Attacker tampering sensor temp_R
     'IntSenAtt_R': act_process({'tempfake_R': lambda d: -TF}, 'IntSenAtt_R'),
     
     # Attacker tampering sensor temp_L for attack_window steps.
     'IntSenAtt_aw': if_then_else_process(lambda d: d['comp_steps'] < d['attack_window'],
          act_process({'tempfake_L': lambda d: -TF}, 'IntSenAtt_aw'),
          act_process({'tempfake_L': lambda d: 0}, 'IntSenAtt_aw')),

}


# system without attacks:
init_process = synch_parallel_process(synch_parallel_process(processes['Ctrl_L'],synch_parallel_process(processes['IDS_L'],synch_parallel_process(processes['Ctrl_R'],processes['IDS_R']))),processes['SV'])

# system with attack on actuator cool_L
init_process_IntActAtt  = synch_parallel_process(init_process,processes['IntActAtt_L'])

# system with attack on sensor temp_L
init_process_IntSenAtt = synch_parallel_process(init_process,processes['IntSenAtt_L'])

# system with attack on sensor temp_L for attaci_window steps
init_process_IntSenAttAW = synch_parallel_process(init_process,processes['IntSenAtt_aw'])

# system with attack on sensor temp_L and actuator cool_R
init_process_IntMixAtt = synch_parallel_process(init_process,synch_parallel_process(processes['IntSenAtt_L'],processes['IntActAtt_R']))











def set_variation_temp(d, a):
    d2 = d.copy()
    d2['temp_L'] = d['temp_L']+a
    d2['p1_L'] = d['p1_L']+a
    d2['p2_L'] = d['p2_L']+a
    d2['p3_L'] = d['p3_L']+a
    d2['p4_L'] = d['p4_L']+a
    d2['p5_L'] = d['p5_L']+a
    d2['p6_L'] = d['p6_L']+a
    print("valore vari: "+str(d2['temp_L'])+str(d2['p1_L'])+str(d2['p2_L'])+str(d2['p3_L'])+str(d2['p4_L'])+str(d2['p5_L'])+str(d2['p6_L']))
    return d2


def variations_temp(ds,scale,size):
    res = []
    for i in range(0,size):
        a = rnd.uniform(0,scale)
        res.append(set_variation_temp(ds,a))
    return res


def set_variation_tempfake(d, a):
    d2 = d.copy()
    d2['tempfake_L'] = a
    print(d2)
    return d2


def variations_tempfake(ds,scale,size):
    res = []
    for i in range(0,size):
        a = rnd.uniform(-scale,scale)
        print("valore di a: "+str(a))
        res.append(set_variation_tempfake(ds,-a))
        print(res)
    return res


def set_variation_attackwindow(d, a):
    d2 = d.copy()
    d2['attack_window'] = a
    print(d2)
    return d2


def variations_attackwindow(ds,scale,size):
    res = []
    for i in range(0,size):
        a = rnd.uniform(0,scale)
        print("valore di a: "+str(a))
        res.append(set_variation_attackwindow(ds,a))
        #print(res)
    return res


          




if __name__ == "__main__":

    for samples in [ 100 ]:
    
        for steps in [ 1000 ]:
    
            
            
            
            # In the following simulations, we always consider 3 scenarios: no attack, attack on actuator cool of engine L, attack on sensor temp of engine L.
           
           
            """
            
            # First we build an evolution sequence for each scenario:
            
            data1 = simulatekkk(processes, engine_environment, init_process, init_data, steps, samples)
            # evolution sequence, no attack scenario
            
            
            #data1IntActAtt = simulate(processes, engine_environment, init_process_IntActAtt, init_data, steps, samples)
            # evolution sequence, attack on actuator cool scenario
            
            
            #data1IntSenAtt = simulate(processes, engine_environment, init_process_IntSenAtt, init_data, steps, samples)
            # evolution sequence,  attack on sensor temp scenario
            
            
            
            
            
            
           
            # Histograms for evolution in time of AVERAGE TEMPERATURE for Left engine
            
            plot_histogram_avg(data1, lambda d: d['temp_L'], "Average temp_L, "+"N="+str(samples)+", k="+str(steps)+", No attack", "temperature_"+str(samples)+"_")
            # evolution in time of average temperature, no-attack scenario
            
           
            plot_histogram_avg(data1IntActAtt, lambda d: d['temp_L'], "Average temp_L, "+"N="+str(samples)+", k="+str(steps)+", Attack on cool_L", "temperature_"+str(samples)+"_")
            # evolution in time of average temperature, attack on actuator cool scenario
            
            
            plot_histogram_avg(data1IntSenAtt, lambda d: d['temp_L'], "Average temp_L, "+"N="+str(samples)+", k="+str(steps)+", Attack on temp_L", "temperature_"+str(samples)+"_")
            # evolution in time of average temperature, attack on sensor temp scenario
            
            """
            
            
            
            
            """
            
            delta = distance_setrho(processes, init_process, init_data, engine_environment, variations_temp(init_data, 0.2, 10), 20, 2000, 10, rank_temp_L)
            plot_histogram_adapt(delta, "x:time, y:max_stress_distance_L, "+" N="+str(samples)+", variations", "temperature_"+str(samples)+"_")
            
            
            
            delta = distance_setrho(processes, init_process, init_data, engine_environment, variations_temp(init_data, 0.2, 10), 20, 2000, 10, rank_temp_L)
            plot_histogram_adapt(delta, "x:time, y:temp_distance_L, "+" N="+str(samples)+", variations", "temperature_"+str(samples)+"_")
            
            
            
            
            delta = distance_setrho(processes, init_process, init_data, engine_environment, variations_tempfake(init_data, 0.1, 20), 1000, 100, 10, rank_fnL)
            plot_histogram_adapt(delta, "x:time, y:false_negative_L distance with 20 -0.5< random <0.5 sensor's error", "temperature_"+str(samples)+"_")
            
            
            """
            
            
           
           
            """
            
           
            
            
           
            # Histrograms to focus on DISTRIBUTION OF TEMPERATURE of Left engine AT SOME TIME INSTANTS.
           
            for instant in [math.ceil(steps/3) , 2*math.ceil(steps/3), steps-1]:
            
               plot_histogram_bis(data1, instant, lambda d: d['temp_L'], 95, 105 , 50, "prob of temp_L at instant "+str(instant)+ " of "+str(steps)+", N="+str(samples)+", No attack", "temperature_"+str(samples)+"_")
                
               plot_histogram_bis(data1IntActAtt, instant, lambda d: d['temp_L'], 95, 105 , 50, "prob of temp_L at instant "+str(instant)+ " of "+str(steps)+", N="+str(samples)+", Attack on cool_L", "temperature_"+str(samples)+"_")
               
               plot_histogram_bis(data1IntSenAtt, instant, lambda d: d['temp_L'], 95, 105 , 50, "prob of temp_L at instant "+str(instant)+ " of "+str(steps)+", N="+str(samples)+", Attack on temp_L", "temperature_"+str(samples)+"_")
                
            
            
            
            
            
           
            
            # Histograms for evolution in time of AVERAGE STRESS for Left engine
           
            plot_histogram_avg(data1, lambda d: rank_stress_L(d), "Average stress_L, "+" N="+str(samples)+", k="+str(steps)+", No attack", "rank_"+str(samples)+"_")
            # evolution in time of average stress, no-attack scenario
            
            
            plot_histogram_avg(data1IntActAtt, lambda d: rank_stress_L(d), "Average stress_L, "+" N="+str(samples)+", k="+str(steps)+", Attack on cool_L", "rank_"+str(samples)+"_")
            # evolution in time of average stress, attack on actuator cool scenario
            
            
            plot_histogram_avg(data1IntSenAtt, lambda d: rank_stress_L(d), "Average stress_L, "+" N="+str(samples)+", k="+str(steps)+", Attack on temp_L", "rank_"+str(samples)+"_")
            # evolution in time of average stress, attack on sensor temp scenario
            
            
            
            
            
            
            
            
            # Histograms for evolution in time of RATIO OF INSTANTS OF STRESS for Left engine
           
            plot_histogram_avg(data1, lambda d: rank_avg_stress_L(d), "Ratio Instants of stress_L, "+" N="+str(samples)+", k="+str(steps)+", No attack", "rank_"+str(samples)+"_")
            # evolution in time of ratio of instants of stress, no-attack scenario
            
            
            plot_histogram_avg(data1IntActAtt, lambda d: rank_avg_stress_L(d), "Ratio Instants of  stress_L, "+" N="+str(samples)+", k="+str(steps)+", Attack on cool_L", "rank_"+str(samples)+"_")
            # evolution in time of ratio of instants of stress, attack on actuator cool scenario
            
            
            plot_histogram_avg(data1IntSenAtt, lambda d: rank_avg_stress_L(d), "Ration Instants of stress_L, "+" N="+str(samples)+", k="+str(steps)+", Attack on temp_L", "rank_"+str(samples)+"_")
            # evolution in time of ration of instants of stress, attack on sensor temp scenario
            
            
            
            
            
            
            
            
            
            # Histograms for evolution in time of AVERAGE NUMBER OF WARNINGS by Left IDS
            
            plot_histogram_avg(data1, lambda d: rank_warn_L(d), "x:time, y:avg_warn_L, "+" N="+str(samples)+" No attack", "rank_"+str(samples)+"_")
            # evolution in time of average warning, no-attack scenario
            
            plot_histogram_avg(data1IntActAtt, lambda d: rank_warn_L(d), "x:time, y:avg_warn_L, "+" N="+str(samples)+" Attack on cool_L", "rank_"+str(samples)+"_")
            # evolution in time of average warning, integrity attack on actuator cool scenario
            
            plot_histogram_avg(data1IntSenAtt, lambda d: rank_warn_L(d), "x:time, y:avg_warn_L, "+" N="+str(samples)+" Attack  on temp_L", "rank_"+str(samples)+"_")
            # evolution in time of average warning, integrity attack on sensor temp scenario
            
            
            
            
            
            
            
            # Histograms for DISTANCE BASED ON STRESS for Left engine / whole systems
            
            dist = distancerho(processes, init_process, init_data, engine_environment, processes, init_process_IntActAtt, init_data, engine_environment, steps, samples, 1, rank_stress_L)
            
            plot_histogram_distance(dist, "x:time, y:stress_distance_L, "+" N="+str(samples)+", No attack vs Attack on cool_L", "temperature_"+str(samples)+"_")
            # stress of engine L: no attack VS attack on cool
          
          
          
            dist = distancerho(processes, init_process, init_data, engine_environment, processes, init_process_IntActAtt, init_data, engine_environment, steps, samples, 1, rank_stress)
            
            plot_histogram_distance(dist, "x:time, y:stress_distance_TOT"+", N="+str(samples)+", No attack vs Attack on cool_L", "temperature_"+str(samples)+"_")
            # total stress : no attack VS attack on cool
            
            
            
            dist = distancerho(processes, init_process, init_data, engine_environment, processes, init_process_IntSenAtt, init_data, engine_environment, steps, samples, 1, rank_stress_L)
            
            plot_histogram_distance(dist, "x:time, y:stress_distance_L,"+" N="+str(samples)+", No attack vs Attack on temp_L", "temperature_"+str(samples)+"_")
            # stress of engine L: no attack VS attack on temp
            
            
            
            dist = distancerho(processes, init_process, init_data, engine_environment, processes, init_process_IntSenAtt, init_data, engine_environment, steps, samples, 1, rank_stress)
            
            plot_histogram_distance(dist, "x:time, y:stress_distance_TOT, "+" N="+str(samples)+", No attack vs Attack on temp_L", "temperature_"+str(samples)+"_")
            # total stress : no attack VS attack on temp
            
            
            
            
            
            
            
            # Histograms for DISTANCE BASED ON NUMBER OF WARNINGS / ALARMS
            
            dist = distancerho(processes, init_process, init_data, engine_environment, processes, init_process_IntActAtt, init_data, engine_environment, steps, samples, 1, rank_warn_L)
            
            plot_histogram_distance(dist,  "x:time, y:warn_distance_L, "+" N="+str(samples)+ ", No attack vs Attack on cool_L", "temperature_"+str(samples)+"_")
            # number of warnings on engine L: no attack VS attack on cool
            
            
            dist = distancerho(processes, init_process, init_data, engine_environment, processes, init_process_IntActAtt, init_data, engine_environment, steps, samples, 1, rank_warn)
            
            plot_histogram_distance(dist,  "x:time, y:warn_distance_TOT, "+" N="+str(samples)+ ", No attack vs Attack on cool_L", "temperature_"+str(samples)+"_")
            # total number of warnings: no attack VS attack on cool
            
            
            dist = distancerho(processes, init_process, init_data, engine_environment, processes, init_process_IntSenAtt, init_data, engine_environment, steps, samples, 1, rank_warn_L)
            
            plot_histogram_distance(dist,  "x:time, y:warn_distance_L, "+", N="+str(samples)+ ", No attack vs Attack on temp_L", "temperature_"+str(samples)+"_")
            # number of warnings on engine L: no attack VS attack on temp
                        
                        
            dist = distancerho(processes, init_process, init_data, engine_environment, processes, init_process_IntSenAtt, init_data, engine_environment, steps, samples, 1, rank_warn)
            
            plot_histogram_distance(dist,  "x:time, y:warn_distance_TOT, "+", N="+str(samples)+ ", No attack vs Attack on temp_L", "temperature_"+str(samples)+"_")
            # total number of warnings: no attack VS attack on temp
            
            
            
            
        
            
            
            
            # Histograms for evolution in time of FALSE NEGATIVES, attacks on sensor temp_L or actuator cool_L of LEFT Engine
            
            plot_histogram_avg(data1, lambda d: rank_fnL(d), "Average fN_L, "+"N="+str(samples)+", k="+str(steps)+", No attack", "rank_"+str(samples)+"_")
            # evolution in time of false negatives L, no-attack scenario
            
            plot_histogram_avg(data1, lambda d: rank_fn(d), "Average fN, "+"N="+str(samples)+", k="+str(steps)+", No attack", "rank_"+str(samples)+"_")
            # evolution in time of false negatives, no-attack scenario
            
            plot_histogram_avg(data1IntActAtt, lambda d: rank_fnL(d), "Average fN_L, "+"N="+str(samples)+", k="+str(steps)+", Attack on cool_L", "rank_"+str(samples)+"_")
            # evolution in time of false negatives L, integrity attack on actuator cool scenario
            
            plot_histogram_avg(data1IntActAtt, lambda d: rank_fn(d), "Average fN, "+"N="+str(samples)+", k="+str(steps)+", Attack on cool_L", "rank_"+str(samples)+"_")
            # evolution in time of false negatives, integrity attack on actuator cool scenario
            
            plot_histogram_avg(data1IntSenAtt, lambda d: rank_fnL(d), "Average fN_L, "+"N="+str(samples)+", k="+str(steps)+", Attack  on temp_L", "rank_"+str(samples)+"_")
            # evolution in time of false negatives L, integrity attack on sensor temp scenario
            
            plot_histogram_avg(data1IntSenAtt, lambda d: rank_fn(d), "Average fN, "+"N="+str(samples)+", k="+str(steps)+", Attack  on temp_L", "rank_"+str(samples)+"_")
            # evolution in time of false negatives, integrity attack on sensor temp scenario
            
            
            
            
            
            
            
            
            # Histograms for evolution in time of DISTANCE BASED ON FALSE NEGATIVES
            
            dist = distancerho(processes, init_process, init_data, engine_environment, processes, init_process_IntActAtt, init_data, engine_environment, steps, samples, 1, rank_fnL)
            
            plot_histogram_distance(dist, "fn_L distance"+" N="+str(samples)+", k="+str(steps)+", No attack vs Attack on cool_L", "temperature_"+str(samples)+"_")
            # evolution in time of false negatives engine L, no attack VS attack on cool
            
            
            dist = distancerho(processes, init_process, init_data, engine_environment, processes, init_process_IntActAtt, init_data, engine_environment, steps, samples, 1, rank_fn)
            
            plot_histogram_distance(dist,  "fn distance, "+" N="+str(samples)+", k="+str(steps)+", No attack vs Attack on cool_L", "temperature_"+str(samples)+"_")
            # evolution in time of false negatives, no attack VS attack on cool_L
            
            dist = distancerho(processes, init_process, init_data, engine_environment, processes, init_process_IntSenAtt, init_data, engine_environment, steps, samples, 1, rank_fnL)
            
            plot_histogram_distance(dist,  "fn_L distance, "+" N="+str(samples)+", k="+str(steps)+", No attack vs Attack on temp_L", "temperature_"+str(samples)+"_")
            # evolution in time of false negatives engine L, no attack VS attack on temp
            
            dist = distancerho(processes, init_process, init_data, engine_environment, processes, init_process_IntSenAtt, init_data, engine_environment, steps, samples, 1, rank_fn)
            
            plot_histogram_distance(dist,  "fn distance, "+" N="+str(samples)+", k="+str(steps)+", No attack vs Attack on temp_L", "temperature_"+str(samples)+"_")
            # evolution in time of false negatives, no attack VS attack on temp_L
            
            
            
            
            
            
            
            
            # Histograms for evolution in time of FALSE POSITIVES, attacks on LEFT Engine
            
            plot_histogram_avg(data1, lambda d: rank_fpL(d), "Average fP_L, "+" N="+str(samples)+", k="+str(steps)+", No attack", "rank_"+str(samples)+"_")
            # evolution in time of false positives L, no-attack scenario
            
            plot_histogram_avg(data1, lambda d: rank_fp(d), "Average fP, "+" N="+str(samples)+", k="+str(steps)+", No attack", "rank_"+str(samples)+"_")
            # evolution in time of false positives, no-attack scenario
            
            plot_histogram_avg(data1IntActAtt, lambda d: rank_fpL(d), "Average fP_L, "+" N="+str(samples)+", k="+str(steps)+", Attack on cool_L", "rank_"+str(samples)+"_")
            # evolution in time of false positives L, integrity attack on actuator cool scenario
            
            plot_histogram_avg(data1IntActAtt, lambda d: rank_fp(d), "Average fP, "+" N="+str(samples)+", k="+str(steps)+", Attack on cool_L", "rank_"+str(samples)+"_")
            # evolution in time of false positives, integrity attack on actuator cool scenario
            
            plot_histogram_avg(data1IntSenAtt, lambda d: rank_fpL(d), "Average fP_L, "+" N="+str(samples)+", k="+str(steps)+", Attack  on temp_L", "rank_"+str(samples)+"_")
            # evolution in time of false positives L, integrity attack on sensor temp scenario
            
            plot_histogram_avg(data1IntSenAtt, lambda d: rank_fp(d), "Average fP, "+" N="+str(samples)+", k="+str(steps)+", Attack  on temp_L", "rank_"+str(samples)+"_")
            # evolution in time of false positives, integrity attack on sensor temp scenario
            
            
            
            
            
            
            
            
            # Histograms for evolution in time of DISTANCE BASED ON FALSE POSITIVES for warn_L
            
            dist = distancerho(processes, init_process, init_data, engine_environment, processes, init_process_IntActAtt, init_data, engine_environment, steps, samples, 1, rank_fpL)
            
            plot_histogram_distance(dist, "fp_L distance, "+"N="+str(samples)+", k="+str(steps)+ ", No attack vs Attack on cool_L", "temperature_"+str(samples)+"_")
            # evolution in time of false positives engine L, no attack VS attack on cool
            
            
            dist = distancerho(processes, init_process, init_data, engine_environment, processes, init_process_IntActAtt, init_data, engine_environment, steps, samples, 1, rank_fp)
            
            plot_histogram_distance(dist, "fp distance, "+"N="+str(samples)+", k="+str(steps)+ ", No attack vs Attack on cool_L", "temperature_"+str(samples)+"_")
            # evolution in time of false positives engine L, no attack VS attack on cool
            
            
            dist = distancerho(processes, init_process, init_data, engine_environment, processes, init_process_IntSenAtt, init_data, engine_environment, steps, samples, 1, rank_fpL)
            
            plot_histogram_distance(dist, "fp_L distance, "+"N="+str(samples)+", k="+str(steps)+ ", No attack vs Attack on temp_L", "temperature_"+str(samples)+"_")
            # evolution in time of false positivess engine L, no attack VS attack on temp
            
            
            dist = distancerho(processes, init_process, init_data, engine_environment, processes, init_process_IntSenAtt, init_data, engine_environment, steps, samples, 1, rank_fp)
            
            plot_histogram_distance(dist, "fp distance, "+"N="+str(samples)+", k="+str(steps)+ ", No attack vs Attack on temp_L", "temperature_"+str(samples)+"_")
            # evolution in time of false positivess engine L, no attack VS attack on temp
            
            
            
           
           
            # Histograms for evolution in time of DISTANCE with MIXED ATTACKS
            
            dist = distancerho(processes, init_process, init_data, engine_environment, processes, init_process_IntMixAtt, init_data, engine_environment, steps, samples, 1, rank_fnL)
            
            plot_histogram_distance(dist, "fN_L distance, "+"N="+str(samples)+", k="+str(steps)+", No attack vs Attack on temp_L and cool_R", "temperature_"+str(samples)+"_")
        
            
            
            
            dist = distancerho(processes, init_process, init_data, engine_environment, processes, init_process_IntMixAtt, init_data, engine_environment, steps, samples, 1, rank_fnR)
            
            plot_histogram_distance(dist, "fN_R distance, "+"N="+str(samples)+", k="+str(steps)+", No attack vs Attack on temp_L and cool_R", "temperature_"+str(samples)+"_")
            
            
            
            
            dist = distancerho(processes, init_process, init_data, engine_environment, processes, init_process_IntMixAtt, init_data, engine_environment, steps, samples, 1, rank_fn)
            
            plot_histogram_distance(dist, "fn distance, "+"N="+str(samples)+", k="+str(steps)+", No attack vs Attack on temp_L and cool_R", "temperature_"+str(samples)+"_")
            
            
            
            
            dist = distancerho(processes, init_process, init_data, engine_environment, processes, init_process_IntMixAtt, init_data, engine_environment, steps, samples, 1, rank_fpL)
            
            plot_histogram_distance(dist, "fp_L distance, "+"N="+str(samples)+", k="+str(steps)+", No attack vs Attack on temp_L and cool_R", "temperature_"+str(samples)+"_")
            
            
            
            
            dist = distancerho(processes, init_process, init_data, engine_environment, processes, init_process_IntMixAtt, init_data, engine_environment, steps, samples, 1, rank_fpR)
            
            plot_histogram_distance(dist, "fp_R distance, "+"N="+str(samples)+", k="+str(steps)+", No attack vs Attack on temp_L and cool_R", "temperature_"+str(samples)+"_")
            
            
            
            dist = distancerho(processes, init_process, init_data, engine_environment, processes, init_process_IntMixAtt, init_data, engine_environment, steps, samples, 1, rank_fp)
            
            plot_histogram_distance(dist, "fp distance, "+"N="+str(samples)+", k="+str(steps)+ ", No attack vs Attack on temp_L and cool_R", "temperature_"+str(samples)+"_")
            
            
            
            """
           
# Histograms for evolution in time of adaptability of avergage stress L with respect to attack in firts <100 steps selected randomly
delta = distance_setrho(processes, init_process_IntSenAttAW, init_data, engine_environment, variations_attackwindow(init_data, 100, 20), 10000, 100, 10, rank_avg_stress_L)
plot_histogram_adapt(delta, "x:time, y:avg_stress__L distance with 20 random <100 attack steps, offset "+str(TF), "temperature_"+str(samples)+"_")
            
"""
delta = distance_setrho(processes, init_process_IntSenAttAW, init_data, engine_environment, variations_attackwindow(init_data, 100, 20), 10000, 100, 10, rank_fnL)
plot_histogram_adapt(delta, "x:time, y:fn_L distance with 20 random < 100 attack steps, offset "+str(TF), "temperature_"+str(samples)+"_")
"""
