from spear import *
import numpy.random as rnd
import matplotlib.pyplot as plt
import numpy


a = 0.5
az0 = 0.75
az12 = 0.75
az23 = 0.75
g = 9.81
t_step = 0.1 #duration of a tick in seconds
q_max = 6.0
q_step = q_max/5
q_med = q_max/2.0
l_max = 20
l_min = 0

def compute_flow_rate(x1, x2, a, a12):
    if x1 > x2:
        return a12*a*numpy.sqrt(2*g)*numpy.sqrt(x1-x2)
    else:
        return -az12*a*numpy.sqrt(2*g)*numpy.sqrt(x2-x1)


def compute_q12(ds):
    l1 = ds['l1']
    l2 = ds['l2']
    return compute_flow_rate(l1, l2, a, az12)


def compute_q23(ds):
    l2 = ds['l2']
    l3 = ds['l3']
    return compute_flow_rate(l2, l3, a, az23)


def compute_q0(ds):
    return az0*a*numpy.sqrt(2*g)*numpy.sqrt(ds['l3'])


def step(fun_q1, fun_q2, fun_q3, ds):
    newds = {}
    q1 = ds['q1']
    q2 = ds['q2']
    q3 = ds['q3']
    q12 = compute_q12(ds)
    q23 = compute_q23(ds)
    #q0 = compute_q0(ds)
    newds['l1'] = max(0 , ds['l1']+q1*t_step-q12*t_step)
    newds['l2'] = max(0 , ds['l2']+q12*t_step-q23*t_step)
    newds['l3'] = max(0 , ds['l3']+q2*t_step+q23*t_step-q3*t_step)
    newds['q1'] = fun_q1(ds)
    newds['q2'] = fun_q2(ds)
    newds['q3'] = fun_q3(ds)
    return newds


def run(fun_q1, fun_q2, fun_q3, ds, k):
    res = []
    ds2 = ds
    for i in range(k):
        res.append(ds2)
        ds2 = step(fun_q1, fun_q2, fun_q3, ds2)
    return res


def simulate(fun_q1, fun_q2, fun_q3, ds, n, l, k):
    data = [ [] for i in range(k) ]
    for i in range(n*l):
        sample = run(fun_q1, fun_q2, fun_q3, ds, k)
        for j in range(k):
            data[j].append(sample[j])
    return data


def q1_scenario_1(q, delta):
    return lambda ds: max(0, rnd.normal(q, delta))


def q1_scenario_2(q, delta):
    return lambda ds: min(max(0.0, ds['q2']+rnd.normal(q, delta)), q_max)



def controller_q1(ds, l, q, delta):
    if ds['l1'] > l+delta:
        return max(0, ds['q1']-q)
    if ds['l1'] < l-delta:
        return min(q_max, ds['q1']+q)
    return ds['q1']


def controller_q3(ds, l, q, delta):
    if ds['l3'] > l+delta:
        return min(q_max, ds['q3'] + q)
    elif ds['l3'] < l-delta:
        return max(0, ds['q3'] - q)
    else:
        return ds['q3']


def init_ds(q1, q2, q3, l1, l2, l3):
    return {'l1': l1, 'l2': l2, 'l3': l3, 'q1': q1, 'q2': q2, 'q3': q3}


def plot_tanks_trajectory(k, trj, title, file):
    fix, ax = plt.subplots()

    ax.plot(range(0, k), [ds['l1'] for ds in trj], label='Tank 1')
    ax.plot(range(0, k), [ds['l2'] for ds in trj], label='Tank 2')
    ax.plot(range(0, k), [ds['l3'] for ds in trj], label='Tank 3')

    legend = ax.legend()

    # Put a nicer background color on the legend.
    # legend.get_frame().set_facecolor('C0')

    plt.title(title)
    plt.savefig(file)
    plt.show()


def rho_fun(x):
    v = abs(x-l_goal)/max(l_max-l_goal,l_goal-l_min)
    return v


def compute_prop1(elist,n,l):
    rho_3 = lambda ds: rho_fun(ds['l3'])

    sample_atomic = lambda n: [rho_fun(rnd.normal(l_goal, 2 * delta_l)) for _ in range(n)]

    vlist = eval_target(n, l, sample_atomic, rho_3, 0.2, elist)
    vlist1 = eval_globally(vlist, 0, 20)
    vlist2 = eval_eventually(vlist1, 0, 30)
    return vlist2


def compute_prop2(elist,n,l):
    rho_1 = lambda ds: rho_fun(ds['l1'])
    rho_2 = lambda ds: rho_fun(ds['l2'])
    rho_3 = lambda ds: rho_fun(ds['l3'])

    sample_atomic1 = lambda n: [rho_fun(rnd.normal(l_max-2*delta_l, delta_l)) for _ in range(n)]
    sample_atomic2 = lambda n: [rho_fun(rnd.normal(l_goal, 2 * delta_l)) for _ in range(n)]

    vlist_b1 = eval_brink(n, l, sample_atomic1, rho_1, 0.2, elist)
    vlist_b2 = eval_brink(n, l, sample_atomic1, rho_2, 0.2, elist)
    vlist_b3 = eval_brink(n, l, sample_atomic1, rho_3, 0.2, elist)

    vlist_t1 = eval_target(n, l, sample_atomic2, rho_1, 0.2, elist)
    vlist_t2 = eval_target(n, l, sample_atomic2, rho_2, 0.2, elist)
    vlist_t3 = eval_target(n, l, sample_atomic2, rho_3, 0.2, elist)

    vlist_or = eval_or(vlist_b3,eval_or(vlist_b2,vlist_b1))
    vlist_and = eval_and(vlist_t3,eval_and(vlist_t2,vlist_t1))

    vlist_ev = eval_eventually(vlist_and,0,50)
    vlist_imp = eval_imply(vlist_or, vlist_ev)

    vlist_glob = eval_globally(vlist_imp,0,50)

    return vlist_glob


q2_start = 3.0
q2_dev = 1.0

ds_start = init_ds(0, 0, 0, 0, 0, 0)
k = 150

l_goal = 10
delta_l = 1


# trj = run(lambda ds: controller_q1(ds, 10, q_step, 1), q1_scenario_1(0,2), lambda ds: controller_q3(ds, 10, q_step, 1), ds_start, k)
# plot_tanks_trajectory(k,trj,"Single Simulation Run (Scenario 1)","tank_level_sim_scenario_1.png")
#
# trj = run(lambda ds: controller_q1(ds, 10, q_step, 1), q1_scenario_2(0,2), lambda ds: controller_q3(ds, 10, q_step, 1), ds_start, k)
# plot_tanks_trajectory(k,trj,"Single Simulation Run (Scenario 2)","tank_level_sim_scenario_2.png")
#
# for samples in [ 1, 10, 100]:
#     elist1 = simulate(lambda ds: controller_q1(ds, 10, q_step, 1), q1_scenario_1(0, 2),
#                       lambda ds: controller_q3(ds, 10, q_step, 1), ds_start, samples, 10, k)
#     plot_histogram(elist1, [20,35,50], lambda d: d['l3'], 0.0,15.0, 50, "Scenario 1: Tank 3 level probability distribution N="+str(samples*10), "tank3_s1_"+str(samples)+"_")
#     elist2 = simulate(lambda ds: controller_q1(ds, 10, q_step, 1), q1_scenario_1(0, 2),
#                       lambda ds: controller_q3(ds, 10, q_step, 1), ds_start, samples, 10, k)
#     plot_histogram(elist2, [20,35,50], lambda d: d['l3'], 0.0,15.0, 50, "Scenario 2: Tank 3 level probability distribution N="+str(samples*10), "tank3_s2_"+str(samples)+"_")


#Prop1

n = 100
l = 10

elist1 = simulate(lambda ds: controller_q1(ds, 10, q_step, 1), q1_scenario_1(0, 2), lambda ds: controller_q3(ds, 10, q_step, 1), ds_start, n, l, k)
elist2 = simulate(lambda ds: controller_q1(ds, 10, q_step, 1), q1_scenario_2(0, 2), lambda ds: controller_q3(ds, 10, q_step, 1), ds_start, n, l, k)

vlist1 = compute_prop1(elist1,n,l)
vlist2 = compute_prop1(elist2,n,l)

fix, ax = plt.subplots()

ax.plot(range(0,k),vlist1,label="Scenario 1")
ax.plot(range(0,k),vlist2,label="Scenario 2")
legend = ax.legend()

plt.title("Estimated satisfaction degree of Prop1")
plt.savefig("sat_prop1.png")
plt.show()


vlist1 = compute_prop2(elist1,n,l)
vlist2 = compute_prop2(elist2,n,l)

fix, ax = plt.subplots()

ax.plot(range(0,k),vlist1,label="Scenario 1")
ax.plot(range(0,k),vlist2,label="Scenario 2")
legend = ax.legend()

plt.title("Estimated satisfaction degree of Prop2")
plt.savefig("sat_prop2.png")
plt.show()


#
# rho_1 = lambda ds: rho_fun(ds['l1'])
# rho_2 = lambda ds: rho_fun(ds['l2'])
# rho_3 = lambda ds: rho_fun(ds['l3'])
#
# sample_atomic = lambda n: [ rho_fun(rnd.normal(l_goal,2*delta_l)) for _ in range(n)]
#
# vlist = eval_true(elist1)
# vlist = eval_target(100,10,sample_atomic,rho_3,0.2,elist1)
#
# plt.plot(range(0,k),vlist)
# plt.show()
#
#
# vlist1 = eval_globally(vlist,0,50)
#
#
# plt.plot(range(0,k),vlist1)
# plt.show()
#
#
# vlist2 = eval_eventually(vlist1,0,50)
#
# plt.plot(range(0,k),vlist2)
# plt.show()
