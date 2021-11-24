from spear import *
import numpy.random as rnd
import matplotlib.pyplot as plt
import numpy



### CONSTANTS

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



### ENVIROMENT EVOLUTION

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



### ADDITIONAL FUNCTIONS

def plot_tanks_trajectory(k, trj, title, file):
    fix, ax = plt.subplots()

    ax.plot(range(0, k), [ds['l1'] for ds in trj], label='Tank 1')
    ax.plot(range(0, k), [ds['l2'] for ds in trj], label='Tank 2')
    ax.plot(range(0, k), [ds['l3'] for ds in trj], label='Tank 3')
    ax.plot(range(0,k),[[10] for i in range(k)], '--')
    legend = ax.legend()
    plt.title(title)
    plt.savefig(file)
    plt.show()

def plot_tanks_traj_l1(k, trj1, trj2, title, file):
    fix, ax = plt.subplots()

    ax.plot(range(0, k), [ds['l1'] for ds in trj1], label='Scen 1')
    ax.plot(range(0, k), [ds['l1'] for ds in trj2], label='Scen 2')
    ax.plot(range(0,k),[[10] for i in range(k)], '--')
    legend = ax.legend()
    plt.title(title)
    plt.savefig(file)
    plt.show()

def plot_tanks_traj_l2(k, trj1, trj2, title, file):
    fix, ax = plt.subplots()

    ax.plot(range(0, k), [ds['l2'] for ds in trj1], label='Scen 1')
    ax.plot(range(0, k), [ds['l2'] for ds in trj2], label='Scen 2')
    ax.plot(range(0,k),[[10] for i in range(k)], '--')
    legend = ax.legend()
    plt.title(title)
    plt.savefig(file)
    plt.show()

def plot_tanks_traj_l3(k, trj1, trj2, title, file):
    fix, ax = plt.subplots()

    ax.plot(range(0, k), [ds['l3'] for ds in trj1], label='Scen 1')
    ax.plot(range(0, k), [ds['l3'] for ds in trj2], label='Scen 2')
    ax.plot(range(0,k),[[10] for i in range(k)], '--')
    legend = ax.legend()
    plt.title(title)
    plt.savefig(file)
    plt.show()


def plot_tanks_3runs(k, trj1, trj2, trj3, title, file):
    fix, ax = plt.subplots()

    ax.plot(range(0, k), [ds['l3'] for ds in trj1], label='0.5')
    ax.plot(range(0, k), [ds['l3'] for ds in trj2], label='0.3')  
    ax.plot(range(0, k), [ds['l3'] for ds in trj3], label='0.7')
    ax.plot(range(0,k),[[10] for i in range(k)], '--')
    legend = ax.legend()
    plt.title(title)
    plt.savefig(file)
    plt.show()



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



### SIMULATIONS

ds_basic = init_ds(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, delta_l)

k = 151
n = 1000
l = 5

trj1 = run(processes, Env_scenario1, PTanks, ds_basic, k)
trj2 = run(processes, Env_scenario2, PTanks, ds_basic, k)


plot_tanks_trajectory(k,trj1,"Variation of the level of water in time (Scenario 1)","tank_level_sim_scen1.png")

plot_tanks_trajectory(k,trj2,"Variation of the level of water in time (Scenario 2)","tank_level_sim_scen2.png")

plot_tanks_traj_l1(k,trj1,trj2,"Comparison of the variation in time of l1","tank_sim_1.png")

plot_tanks_traj_l2(k,trj1,trj2,"Comparison of the variation in time of l2","tank_sim_2.png")

plot_tanks_traj_l3(k,trj1,trj2,"Comparison of the variation in time of l3","tank_sim_3.png")
   

for samples in [ 100, 1000, 10000 ]:
    simdata1 = simulate(processes, Env_scenario1, PTanks, ds_basic, k, samples)
    simdata2 = simulate(processes, Env_scenario2, PTanks, ds_basic, k, samples)
    
    plot_histogram_double(simdata1, simdata2, [50], lambda d: d['l1'], 7.0, 13.0, 100, "l1, N="+str(samples)+", ", "comp_l1_"+str(samples)+"_")
    plot_histogram_double(simdata1, simdata2, [50], lambda d: d['l2'], 7.0, 13.0, 100, "l2, N="+str(samples)+", ", "comp_l2_"+str(samples)+"_")
    plot_histogram_double(simdata1, simdata2, [50], lambda d: d['l3'], 7.0, 13.0, 100, "l3, N="+str(samples)+", ", "comp_l3_"+str(samples)+"_")
    
    plot_histogram_double(simdata1, simdata2, [100], lambda d: d['l1'], 8.0, 12.0, 100, "l1, N="+str(samples)+", ", "comp_l1_"+str(samples)+"_")
    plot_histogram_double(simdata1, simdata2, [100], lambda d: d['l2'], 8.0, 12.0, 100, "l2, N="+str(samples)+", ", "comp_l2_"+str(samples)+"_")
    plot_histogram_double(simdata1, simdata2, [100], lambda d: d['l3'], 8.0, 12.0, 100, "l3, N="+str(samples)+", ", "comp_l3_"+str(samples)+"_")

    plot_histogram_double(simdata1, simdata2, [150], lambda d: d['l1'], 8.0, 12.0, 100, "l1, N="+str(samples)+", ", "comp_l1_"+str(samples)+"_")
    plot_histogram_double(simdata1, simdata2, [150], lambda d: d['l2'], 8.0, 12.0, 100, "l2, N="+str(samples)+", ", "comp_l2_"+str(samples)+"_")
    plot_histogram_double(simdata1, simdata2, [150], lambda d: d['l3'], 8.0, 12.0, 100, "l3, N="+str(samples)+", ", "comp_l3_"+str(samples)+"_")


estdata1_n = simulate(processes, Env_scenario1, PTanks, ds_basic, k, n)
estdata1_nl = simulate(processes, Env_scenario1, PTanks, ds_basic, k, n*l)

estdata2_n = simulate(processes, Env_scenario2, PTanks, ds_basic, k, n)
estdata2_nl = simulate(processes, Env_scenario2, PTanks, ds_basic, k, n*l)



### EVALUATION OF DISTANCES DIFFERENT ENVIRONMENTS

(evmet_12_rho3, pointdist_12_rho3) = distance(processes,PTanks,ds_basic,Env_scenario1,processes,PTanks,ds_basic,Env_scenario2,k,n,l,ranking_function_3)

print("Distance scen1-scen2: "+str(evmet_12_rho3[0]))

fix, ax = plt.subplots()
ax.plot(range(k),evmet_12_rho3,'r.')
ax.plot(range(k),pointdist_12_rho3,'b-')
plt.title("Distance modulo rho_3 scenarios 1-2 N="+str(n)+", l="+str(l))
plt.savefig("distance_scen1-scen2_newest.png")
plt.show()

(evmet_21_rho3, pointdist_21_rho3) = distance(processes,PTanks,ds_basic,Env_scenario2,processes,PTanks,ds_basic,Env_scenario1,k,n,l,ranking_function_3)

print("Distance scen2-scen1: "+str(evmet_21_rho3[0]))

fix, ax = plt.subplots()
ax.plot(range(k),evmet_21_rho3,'r.')
ax.plot(range(k),pointdist_21_rho3,'b-')
plt.title("Distance modulo rho_3 scenarios 2-1 N="+str(n)+", l="+str(l))
plt.savefig("distance_scen2-scen1_newest.png")
plt.show()

(evmet_12_rho1, pointdist_12_rho1) = distance(processes,PTanks,ds_basic,Env_scenario1,processes,PTanks,ds_basic,Env_scenario2,k,n,l,ranking_function_1)
(evmet_12_rho2, pointdist_12_rho2) = distance(processes,PTanks,ds_basic,Env_scenario1,processes,PTanks,ds_basic,Env_scenario2,k,n,l,ranking_function_2)
(evmet_12_rhoM, pointdist_12_rhoM) = distance(processes,PTanks,ds_basic,Env_scenario1,processes,PTanks,ds_basic,Env_scenario2,k,n,l,ranking_function_max)


fix, ax = plt.subplots()
ax.plot(range(k),evmet_12_rho1,label="rho^l1")
ax.plot(range(k),evmet_12_rho2,label="rho^l2")
ax.plot(range(k),evmet_12_rho3,label="rho^l3")
ax.plot(range(k),evmet_12_rhoM,label="rho^max")
legend = ax.legend()
plt.title("Evolution metric wrt different penalty functions N="+str(n)+", l="+str(l))
plt.savefig("ev_distance_rho_scen1-scen2_basic.png")
plt.show()

fix, ax = plt.subplots()
ax.plot(range(k),pointdist_12_rho1,label="rho^l1")
ax.plot(range(k),pointdist_12_rho2,label="rho^l2")
ax.plot(range(k),pointdist_12_rho3,label="rho^l3")
ax.plot(range(k),pointdist_12_rhoM,label="rho^max")
legend = ax.legend()
plt.title("Pointiwise distance wrt different penalty functions N="+str(n)+",l="+str(l))
plt.savefig("pt_distance_rho_scen1-scen2_basic.png")
plt.show()

(evmet_21_rho1, pointdist_21_rho1) = distance(processes,PTanks,ds_basic,Env_scenario2,processes,PTanks,ds_basic,Env_scenario1,k,n,l,ranking_function_1)
(evmet_21_rho2, pointdist_21_rho2) = distance(processes,PTanks,ds_basic,Env_scenario2,processes,PTanks,ds_basic,Env_scenario1,k,n,l,ranking_function_2)
(evmet_21_rhoM, pointdist_21_rhoM) = distance(processes,PTanks,ds_basic,Env_scenario2,processes,PTanks,ds_basic,Env_scenario1,k,n,l,ranking_function_max)

fix, ax = plt.subplots()
ax.plot(range(k),evmet_21_rho1,label="rho^l1")
ax.plot(range(k),evmet_21_rho2,label="rho^l2")
ax.plot(range(k),evmet_21_rho3,label="rho^l3")
ax.plot(range(k),evmet_21_rhoM,label="rho^max")
legend = ax.legend()
plt.title("Evolution metric wrt different penalty functions N="+str(n)+",l="+str(l))
plt.savefig("ev_distance_rho_scen2-scen1_basic.png")
plt.show()

fix, ax = plt.subplots()
ax.plot(range(k),pointdist_21_rho1,label="rho^l1")
ax.plot(range(k),pointdist_21_rho2,label="rho^l2")
ax.plot(range(k),pointdist_21_rho3,label="rho^l3")
ax.plot(range(k),pointdist_21_rhoM,label="rho^max")
legend = ax.legend()
plt.title("Pointiwise distance wrt different penalty functions N="+str(n)+",l="+str(l))
plt.savefig("pt_distance_rho_scen2-scen1_basic.png")
plt.show()



### EVALUATION OF DISTANCES DIFFERENT DELTAS

delta_l_less = 0.3
delta_l_more = 0.7

ds_start_less = init_ds(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, delta_l_less)
ds_start_more = init_ds(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, delta_l_more)


run_less = run(processes, Env_scenario1, PTanks, ds_start_less, k)
run_more = run(processes, Env_scenario1, PTanks, ds_start_more, k)
run_normal = run(processes, Env_scenario1, PTanks, ds_basic, k)

plot_tanks_3runs(k,run_normal,run_less,run_more,"Variation of l3 modulo different delta_l","deltas_scen1.png")

estless_n = simulate(processes, Env_scenario1, PTanks, ds_start_less, k, n)
estmore_nl = simulate(processes, Env_scenario1, PTanks, ds_start_more, k, n*l)


(evmet_32_rho3, pointdist_32_rho3) = compute_distance(estless_n,estdata1_nl,k,n,l,ranking_function_3)

print("Distance 0.3-0.5: "+str(evmet_32_rho3[0]))


(evmet_24_rho3, pointdist_24_rho3) = compute_distance(estdata1_n,estmore_nl,k,n,l,ranking_function_3)

print("Distance 0.5-0.7: "+str(evmet_24_rho3[0]))


(evmet_34_rho3, pointdist_34_rho3) = compute_distance(estless_n,estmore_nl,k,n,l,ranking_function_3)

print("Distance 0.3-0.7: "+str(evmet_34_rho3[0]))

fix, ax = plt.subplots()
ax.plot(range(k),evmet_32_rho3, label='0.3,0.5')
ax.plot(range(k),evmet_24_rho3, label='0.5,0.7')
ax.plot(range(k),evmet_34_rho3, label='0.3,0.7')
legend=ax.legend()
plt.title("Evolution metric with delta_l = 0.3,0.5,0.7, N="+str(n)+",l="+str(l))
plt.savefig("ev_distance_scen1_deltas.png")
plt.show()


fix, ax = plt.subplots()
ax.plot(range(k),pointdist_32_rho3, label='0.3,0.5')
ax.plot(range(k),pointdist_24_rho3, label='0.5,0.7')
ax.plot(range(k),pointdist_34_rho3, label='0.3,0.7')
legend=ax.legend()
plt.title("Pointwise distance with delta_l = 0.3,0.5,0.7, N="+str(n)+",l="+str(l))
plt.savefig("pt_distance_scen1_deltas.png")
plt.show()



### EVALUATION OF ROBUSTNESS

def robust_variation(simdata,ds,scale,size,t1,t2,pdef,p,e,k,n,l,rho):
    v = scale*max(l_max-l_goal,l_goal-l_min)
    res = []
    c=0
    sdata = simdata[t1:t2+1]
    t = t2-t1+1
    for j in range(0,size):
        d = ds.copy()
        d['l1'] = rnd.uniform(l_min,l_max)
        d['l2'] = rnd.uniform(l_min,l_max)
        d['l3'] = max(0.0, min(ds['l3'] + rnd.uniform(-v,v),l_max))
        d['q1'] = rnd.uniform(0.0,q_max)
        d['q2'] = rnd.uniform(0.0,q_max)
        d['q3'] = rnd.uniform(0.0,q_max)
        simdata2 = simulate(pdef,e,p,d,k,n*l)
        sdata2 = simdata2[t1:t2+1]
        evdist,ptdist = compute_distance(sdata,sdata2,t,n,l,rho)
        c = c+1
        print(c)
        if evdist[0] <=scale:
            res.append(d)
    return res


def compute_robust(simuldata,dlist,k,n,l,rho):
    print("Starting the simulations of the variations for M="+str(M))
    delta = [ 0 for i in range(k) ]
    c = 0
    for data2 in dlist:
        simuldata2 = simulate(processes,Env_scenario1,PTanks,data2,k,n*l)
        (ev,dist) = compute_distance(simuldata,simuldata2,k,n,l,rho)
        c = c+1
        print (c)
        for i in range(k):
            delta[i] = max(delta[i],ev[i])
    return delta


estrob = simulate(processes, Env_scenario1, PTanks, ds_start, k, n)

eta1= 0.3
i1 = 0
i2 = 50

def new_rho(i,ds):
    return 0.5*rho_fun(ds['l1']) + 0.5*rho_fun(ds['l2'])


print("Computing variations")
estrob2 = robust_variation(estrob, ds_start, eta1, M, i1, i2, processes, PTanks, Env_scenario1, k, n, l, ranking_function_3)

print("Computing robustness")
robustness = compute_robust(estrob, estrob2, k, n, l, new_rho)

plt.plot(range(i2,k),robustness[i2:])
plt.title("Robustness, M="+str(M)+", eta_1=0.3, I = [0,50]")
plt.savefig("tanks_robust_scen1.png")
plt.show()



### EVALUATION OF ADAPTABILITY


def set_variation(l1, l2, l3):
    d = init_ds(0,0,0,0,0,0,delta_l)
    d['l1'] = l1
    d['l2'] = l2
    d['l3'] = l3
    d['q1'] = rnd.uniform(0,q_max)
    d['q2'] = rnd.uniform(0,q_max)
    d['q3'] = rnd.uniform(0,q_max)
    return d

def variations_1(ds,scale,size):
    v = rho_fun(ds['l1'])
    u = scale*max(l_max-l_goal,l_goal-l_min)
    res = []
    for j in range(0,size):
        l1 = max(0.0, min(ds['l1'] + rnd.uniform(-u,u),l_max))
        l2 = rnd.uniform(l_min,l_max)
        l3 = rnd.uniform(l_min,l_max)
        if max(rho_fun(l1)-v, 0.0)<=scale:
            res.append(set_variation(l1,l2,l3))
    return res

def variations_2(ds,scale,size):
    v = rho_fun(ds['l2'])
    u = scale*max(l_max-l_goal,l_goal-l_min)
    res = []
    for j in range(0,size):
        l1 = rnd.uniform(l_min,l_max)
        l2 = max(0.0, min(ds['l2'] + rnd.uniform(-u,u),l_max))
        l3 = rnd.uniform(l_min,l_max)
        if max(rho_fun(l2)-v , 0.0)<=scale:
            res.append(set_variation(l1,l2,l3))
    return res

def variations_3(ds,scale,size):
    v = rho_fun(ds['l3'])
    u = scale*max(l_max-l_goal,l_goal-l_min)
    res = []
    for j in range(0,size):
        l1 = rnd.uniform(l_min,l_max)
        l2 = rnd.uniform(l_min,l_max)
        l3 = max(0.0, min(ds['l3'] + rnd.uniform(-u,u),l_max))
        if max(rho_fun(l3) - v, 0.0)<=scale:
            res.append(set_variation(l1,l2,l3))
    return res

def variations_max(ds,scale,size):
    v = max(rho_fun(ds['l1']),rho_fun(ds['l2']),rho_fun(ds['l3']))
    u = scale*max(l_max-l_goal,l_goal-l_min)
    res = []
    for j in range(0,size):
        l1 = max(0.0, min(ds['l1'] + rnd.uniform(-u,u),l_max))
        l2 = max(0.0, min(ds['l2'] + rnd.uniform(-u,u),l_max))
        l3 = max(0.0, min(ds['l3'] + rnd.uniform(-u,u),l_max))
        m = max(rho_fun(l1),rho_fun(l2),rho_fun(l3))
        if max(m - v, 0.0)<=scale:
            res.append(set_variation(l1,l2,l3))
    return res

M = 100

def compute_adapt(simuldata,dlist,k,n,l,rho):
    print("Starting the simulations of the variations for M="+str(M))
    delta = [ 0 for i in range(k) ]
    c = 0
    for data2 in dlist:
        simuldata2 = simulate(processes,Env_scenario1,PTanks,data2,k,n*l)
        (ev,dist) = compute_distance(simuldata,simuldata2,k,n,l,rho)
        c = c+1
        print (c)
        for i in range(k):
            delta[i] = max(delta[i],ev[i])
    return delta

ds_start = init_ds(0.0, 0.0, 0.0, 5.0, 5.0, 5.0, delta_l)

estadapt = simulate(processes, Env_scenario1, PTanks, ds_start, k, n)

print("Computing adaptability wrt rho^l1")
adapt_rho1 = compute_adapt(estadapt, variations_1(ds_start, 0.3, M), k, n, l, ranking_function_1)

plt.plot(range(k),adapt_rho1)
plt.title("Adaptability modulo rho^l1, M="+str(M)+", eta_1=0.3")
plt.savefig("tanks_adapt_rho1_scen1.png")
plt.show()


print("Computing adaptability wrt rho^l2")
adapt_rho2 = compute_adapt(estadapt, variations_2(ds_start, 0.3, M), k, n, l, ranking_function_2)

plt.plot(range(k),adapt_rho2)
plt.title("Adaptability modulo rho^l2, M="+str(M)+", eta_1=0.3")
plt.savefig("tanks_adapt_rho2_scen1.png")
plt.show()


print("Computing adaptability wrt rho^l3")
adapt_rho3 = compute_adapt(estadapt, variations_3(ds_start, 0.3, M), k, n, l, ranking_function_3)

plt.plot(range(k),adapt_rho3)
plt.title("Adaptability modulo rho^l3, M="+str(M)+", eta_1=0.3")
plt.savefig("tanks_adapt_rho3_scen1.png")
plt.show()


print("Computing adaptability wrt rho^max")
adapt_rhoM = compute_adapt(estadapt, variations_max(ds_start, 0.3, M), k, n, l, ranking_function_max)

plt.plot(range(k),adapt_rhoM)
plt.title("Adaptability modulo rho^max, M="+str(M)+", eta_1=0.3")
plt.savefig("tanks_adapt_rhoM_scen1.png")
plt.show()


fix, ax = plt.subplots()
ax.plot(range(k),adapt_rho1,label='rho^l1')
ax.plot(range(k),adapt_rho2,label='rho^l2')
ax.plot(range(k),adapt_rho3,label='rho^l3')
ax.plot(range(k),adapt_rhoM,label='rho^max')
legend=ax.legend()
plt.title("Adaptability, M="+str(M)+", eta_1=0.3")
plt.savefig("tanks_adaptability_scen1.png")
plt.show()


