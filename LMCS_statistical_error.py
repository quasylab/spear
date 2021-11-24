from spear import *
import numpy.random as rnd
import matplotlib.pyplot as plt
import numpy
from statistics import mean


a = 0.5
az0 = 0.75
az12 = 0.75
az23 = 0.75
g = 9.81
t_step = 0.1 #duration of a tick in seconds
q_max = 6.0
q_step = q_max/5.0
q_med = q_max/2.0
l_max = 20.0
l_min = 0.0

l_goal = 10.0
delta_l = 0.5
epsilon = 0.3

q2_dev = 0.5


### ENVIRONMENT EVOLUTION

def compute_flow_rate(x1, x2, a, a12):
    if x1 > x2:
        return a12*a*numpy.sqrt(2*g)*numpy.sqrt(x1-x2)
    else:
        return -a12*a*numpy.sqrt(2*g)*numpy.sqrt(x2-x1)


def compute_q12(ds):
    l1 = ds['l1']
    l2 = ds['l2']
    return compute_flow_rate(l1, l2, a, az12)


def compute_q23(ds):
    l2 = ds['l2']
    l3 = ds['l3']
    return compute_flow_rate(l2, l3, a, az23)


def Env_scenario1(ds):
    newds = ds.copy()
    q1 = ds['q1']
    q2 = ds['q2']
    q3 = ds['q3']
    newds['q2'] = max(0.0, rnd.normal(q_med, q2_dev))
    q12 = compute_q12(ds)
    q23 = compute_q23(ds)
    newds['l1'] = max(0.0 , ds['l1'] + q1*t_step - q12*t_step)
    newds['l2'] = max(0.0 , ds['l2'] + q12*t_step - q23*t_step)
    newds['l3'] = max(0.0 , ds['l3'] + q2*t_step + q23*t_step - q3*t_step)
    return newds


def Env_scenario2(ds):
    newds = ds.copy()
    q1 = ds['q1']
    q2 = ds['q2']
    q3 = ds['q3']
    newds['q2'] =  min( max(0.0, q2 + rnd.normal(0,1)), q_max)
    q12 = compute_q12(ds)
    q23 = compute_q23(ds)
    newds['l1'] = max(0.0 , ds['l1'] + q1*t_step - q12*t_step)
    newds['l2'] = max(0.0 , ds['l2'] + q12*t_step - q23*t_step)
    newds['l3'] = max(0.0 , ds['l3'] + q2*t_step + q23*t_step - q3*t_step)
    return newds


### PENALTY FUNCTIONS

def rho_fun(x):
    v = abs(x-l_goal)/max(l_max-l_goal,l_goal-l_min)
    return v

def ranking_function_1(i, ds):
    return rho_fun(ds['l1'])

def ranking_function_2(i, ds):
    return rho_fun(ds['l2'])

def ranking_function_3(i, ds):
    return rho_fun(ds['l3'])

def ranking_function_max(i, ds):
    return max(rho_fun(ds['l1']),rho_fun(ds['l2']),rho_fun(ds['l3']))



### PROCESSES


processes = {
    'Pin': if_then_else_process(lambda d: d['l1'] > l_goal + d['delta_l'], 
                                act_process({'q1': lambda d: max(0.0, d['q1'] - q_step)}, 'Pin'), 
                                if_then_else_process(lambda d: d['l1'] < l_goal - d['delta_l'], 
                                                             act_process({'q1': lambda d: min(q_max, d['q1'] + q_step)}, 'Pin'), 
                                                             act_process({}, 'Pin'))),
    'Pout': if_then_else_process(lambda d: d['l3'] > l_goal + d['delta_l'], 
                                act_process({'q3': lambda d: min(q_max, d['q3'] + q_step)}, 'Pout'),
                                if_then_else_process(lambda d: d['l3'] < l_goal - d['delta_l'], 
                                                             act_process({'q3': lambda d: max(0.0, d['q3'] - q_step)}, 'Pout'),
                                                             act_process({},'Pout')))
}

PTanks = synch_parallel_process(processes['Pin'], processes['Pout'])

def init_ds(q1, q2, q3, l1, l2, l3, delta_l):
    return {'q1': q1, 'q2': q2, 'q3': q3, 'l1': l1, 'l2': l2, 'l3': l3, 'delta_l': delta_l}



### EVALUATION OF STATISTICAL ERROR

ds_basic = init_ds(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, delta_l)

k = 151
n = 1000
l = 5

M = 100

ds_start = init_ds(0.0, 0.0, 0.0, 5.0, 5.0, 5.0, delta_l)


err1000 = simulate(processes, Env_scenario1, PTanks, ds_basic, k, 1000)
err5000 = simulate(processes, Env_scenario1, PTanks, ds_basic, k, 5000)
err10000 = simulate(processes, Env_scenario1, PTanks, ds_basic, k, 10000)


lstl1_1000 = [ [] for i in range(k) ]
lstl2_1000 = [ [] for i in range(k) ]
lstl3_1000 = [ [] for i in range(k) ]
standev_1000 = [ [] for i in range(k) ]
standerr_1000 = [ [] for i in range(k) ]
zscorel1_1000 = [ [] for i in range(k) ]
zscorel2_1000 = [ [] for i in range(k) ]
zscorel3_1000 = [ [] for i in range(k) ]

lstl3_5000 = [ [] for i in range(k) ]
standev_5000 = [ [] for i in range(k) ]
standerr_5000 = [ [] for i in range(k) ]
zscorel3_5000 = [ [] for i in range(k) ]

lstl3_10000 = [ [] for i in range(k) ]
standev_10000 = [ [] for i in range(k) ]
standerr_10000 = [ [] for i in range(k) ]
zscorel3_10000 = [ [] for i in range(k) ]


for i in range(k):
    lstl3_1000[i] = list(map(lambda ds: ds['l3'], err1000[i]))
    standev_1000[i] = numpy.std(lstl3_1000[i])
    standerr_1000[i] = standev_1000[i]/ numpy.sqrt(1000)
    lstl3_5000[i] = list(map(lambda ds: ds['l3'], err5000[i]))
    standev_5000[i] = numpy.std(lstl3_5000[i])
    standerr_5000[i] = standev_5000[i]/ numpy.sqrt(5000)
    lstl3_10000[i] = list(map(lambda ds: ds['l3'], err10000[i]))
    standev_10000[i] = numpy.std(lstl3_10000[i])
    standerr_10000[i] = standev_10000[i]/ numpy.sqrt(10000)
     

fix, ax = plt.subplots()
ax.plot(range(0,k),standev_1000,label="N = 1000")
ax.plot(range(0,k),standev_5000,label="N = 5000")
ax.plot(range(0,k),standev_10000,label="N = 10000")
legend = ax.legend()
plt.title("Standard deviation")
plt.savefig("tanks_SD.png")
plt.show()

fix, ax = plt.subplots()
ax.plot(range(0,k),standerr_1000,label="N = 1000")
ax.plot(range(0,k),standerr_5000,label="N = 5000")
ax.plot(range(0,k),standerr_10000,label="N = 10000")
legend = ax.legend()
plt.title("Standard error of the mean")
plt.savefig("tanks_SEM.png")
plt.show()


print("I will now proceed to compute several simulations to obtain the analysis of the error, please wait")

expected1 = simulate(processes, Env_scenario1, PTanks, ds_basic, k, 10000)
expected2 = simulate(processes, Env_scenario1, PTanks, ds_basic, k, 10000)
expected3 = simulate(processes, Env_scenario1, PTanks, ds_basic, k, 10000)
expected4 = simulate(processes, Env_scenario1, PTanks, ds_basic, k, 10000)
expected5 = simulate(processes, Env_scenario1, PTanks, ds_basic, k, 10000)
expected6 = simulate(processes, Env_scenario1, PTanks, ds_basic, k, 10000)
expected7 = simulate(processes, Env_scenario1, PTanks, ds_basic, k, 10000)
expected8 = simulate(processes, Env_scenario1, PTanks, ds_basic, k, 10000)
expected9 = simulate(processes, Env_scenario1, PTanks, ds_basic, k, 10000)
expected10 = simulate(processes, Env_scenario1, PTanks, ds_basic, k, 10000)


print("Simulations completed. Next we compute the expected value of the (real) distribution")

expectedl1_1 = [ [] for i in range(k) ]
expectedl2_1 = [ [] for i in range(k) ]
expectedl3_1 = [ [] for i in range(k) ]

expectedl1_2 = [ [] for i in range(k) ]
expectedl2_2 = [ [] for i in range(k) ]
expectedl3_2 = [ [] for i in range(k) ]

expectedl1_3 = [ [] for i in range(k) ]
expectedl2_3 = [ [] for i in range(k) ]
expectedl3_3 = [ [] for i in range(k) ]

expectedl1_4 = [ [] for i in range(k) ]
expectedl2_4 = [ [] for i in range(k) ]
expectedl3_4 = [ [] for i in range(k) ]

expectedl1_5 = [ [] for i in range(k) ]
expectedl2_5 = [ [] for i in range(k) ]
expectedl3_5 = [ [] for i in range(k) ]

expectedl1_6 = [ [] for i in range(k) ]
expectedl2_6 = [ [] for i in range(k) ]
expectedl3_6 = [ [] for i in range(k) ]

expectedl1_7 = [ [] for i in range(k) ]
expectedl2_7 = [ [] for i in range(k) ]
expectedl3_7 = [ [] for i in range(k) ]

expectedl1_8 = [ [] for i in range(k) ]
expectedl2_8 = [ [] for i in range(k) ]
expectedl3_8 = [ [] for i in range(k) ]

expectedl1_9 = [ [] for i in range(k) ]
expectedl2_9 = [ [] for i in range(k) ]
expectedl3_9 = [ [] for i in range(k) ]

expectedl1_10 = [ [] for i in range(k) ]
expectedl2_10 = [ [] for i in range(k) ]
expectedl3_10 = [ [] for i in range(k) ]


for i in range (0,k):
    expectedl1_1[i] = list(map(lambda ds: ds['l1'], expected1[i]))
    expectedl1_2[i] = list(map(lambda ds: ds['l1'], expected2[i]))
    expectedl1_3[i] = list(map(lambda ds: ds['l1'], expected3[i]))
    expectedl1_4[i] = list(map(lambda ds: ds['l1'], expected4[i]))
    expectedl1_5[i] = list(map(lambda ds: ds['l1'], expected5[i]))
    expectedl1_6[i] = list(map(lambda ds: ds['l1'], expected6[i]))
    expectedl1_7[i] = list(map(lambda ds: ds['l1'], expected7[i]))
    expectedl1_8[i] = list(map(lambda ds: ds['l1'], expected8[i]))
    expectedl1_9[i] = list(map(lambda ds: ds['l1'], expected9[i]))
    expectedl1_10[i] = list(map(lambda ds: ds['l1'], expected10[i]))
 
    expectedl2_1[i] = list(map(lambda ds: ds['l2'], expected1[i]))
    expectedl2_2[i] = list(map(lambda ds: ds['l2'], expected2[i]))
    expectedl2_3[i] = list(map(lambda ds: ds['l2'], expected3[i]))
    expectedl2_4[i] = list(map(lambda ds: ds['l2'], expected4[i]))
    expectedl2_5[i] = list(map(lambda ds: ds['l2'], expected5[i])) 
    expectedl2_6[i] = list(map(lambda ds: ds['l2'], expected6[i]))
    expectedl2_7[i] = list(map(lambda ds: ds['l2'], expected7[i]))
    expectedl2_8[i] = list(map(lambda ds: ds['l2'], expected8[i]))
    expectedl2_9[i] = list(map(lambda ds: ds['l2'], expected9[i]))
    expectedl2_10[i] = list(map(lambda ds: ds['l2'], expected10[i])) 
    
    expectedl3_1[i] = list(map(lambda ds: ds['l3'], expected1[i]))
    expectedl3_2[i] = list(map(lambda ds: ds['l3'], expected2[i]))
    expectedl3_3[i] = list(map(lambda ds: ds['l3'], expected3[i]))
    expectedl3_4[i] = list(map(lambda ds: ds['l3'], expected4[i]))
    expectedl3_5[i] = list(map(lambda ds: ds['l3'], expected5[i]))  
    expectedl3_6[i] = list(map(lambda ds: ds['l3'], expected6[i]))
    expectedl3_7[i] = list(map(lambda ds: ds['l3'], expected7[i]))
    expectedl3_8[i] = list(map(lambda ds: ds['l3'], expected8[i]))
    expectedl3_9[i] = list(map(lambda ds: ds['l3'], expected9[i]))
    expectedl3_10[i] = list(map(lambda ds: ds['l3'], expected10[i]))
  

expected_mean1 = [ [] for i in range(k) ]
expected_mean2 = [ [] for i in range(k) ]
expected_mean3 = [ [] for i in range(k) ]

for j in range (k):
    expected_mean1[j] = mean([mean(expectedl1_1[j]),mean(expectedl1_2[j]),
                              mean(expectedl1_3[j]),mean(expectedl1_4[j]),
                              mean(expectedl1_5[j]),mean(expectedl1_6[j]),
                              mean(expectedl1_7[j]),mean(expectedl1_8[j]),
                              mean(expectedl1_9[j]),mean(expectedl1_10[j])])
    expected_mean2[j] = mean([mean(expectedl2_1[j]),mean(expectedl2_2[j]),
                              mean(expectedl2_3[j]),mean(expectedl2_4[j]),
                              mean(expectedl2_5[j]),mean(expectedl2_6[j]),
                              mean(expectedl2_7[j]),mean(expectedl2_8[j]),
                              mean(expectedl2_9[j]),mean(expectedl2_10[j])])
    expected_mean3[j] = mean([mean(expectedl3_1[j]),mean(expectedl3_2[j]),
                              mean(expectedl3_3[j]),mean(expectedl3_4[j]),
                              mean(expectedl3_5[j]),mean(expectedl3_6[j]),
                              mean(expectedl3_7[j]),mean(expectedl3_8[j]),
                              mean(expectedl3_9[j]),mean(expectedl3_10[j])])

print("Mean computed. Finally we evaluate the z-scores")

for i in range (0,10):
    zscorel3_1000[i] = 0
    zscorel3_5000[i] = 0
    zscorel3_10000[i] = 0
    
for i in range (10,k):
    zscorel3_1000[i] = (mean(lstl3_1000[i]) - expected_mean3[i]) / standerr_1000[i]
    zscorel3_5000[i] = (mean(lstl3_5000[i]) - expected_mean3[i]) / standerr_5000[i]
    zscorel3_10000[i] = (mean(lstl3_10000[i]) - expected_mean3[i]) / standerr_10000[i]

limit1 = [ [1.96] for i in range(k) ]
limit2 = [ [-1.96] for i in range(k) ]

fix, ax = plt.subplots()
ax.plot(range(0,k),zscorel3_1000,label="N = 1000")
ax.plot(range(0,k),zscorel3_5000,label="N = 5000")
ax.plot(range(0,k),zscorel3_10000,label="N = 10000")
ax.plot(range(0,k),limit1, 'r--')
ax.plot(range(0,k),limit2, 'r--')
legend = ax.legend()
plt.xlim([10, k])
plt.title("Value of z-score in time")
plt.savefig("tanks_zScore.png")
plt.show()



for i in range(k): 
    lstl1_1000[i] = list(map(lambda ds: ds['l1'], err1000[i]))
    lstl2_1000[i] = list(map(lambda ds: ds['l2'], err1000[i]))   

for i in range (0,10):
    zscorel1_1000[i] = 0
    zscorel2_1000[i] = 0
    
for i in range (10,k):
    zscorel1_1000[i] = (mean(lstl1_1000[i]) - expected_mean1[i])*numpy.sqrt(1000)/numpy.std(lstl1_1000[i])
    zscorel2_1000[i] = (mean(lstl2_1000[i]) - expected_mean2[i])*numpy.sqrt(1000)/numpy.std(lstl2_1000[i])
    
fix, ax = plt.subplots()
ax.plot(range(0,k),zscorel1_1000,label="z-score on l1")
ax.plot(range(0,k),zscorel2_1000,label="z-score on l2")
ax.plot(range(0,k),zscorel3_1000,label="z-score on l3")
legend = ax.legend()
plt.xlim([10, k])
plt.title("Comparison of the z-scores for the levels of water (N = 1000)")
plt.savefig("tanks_total.png")
plt.show()
