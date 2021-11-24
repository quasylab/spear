from spear import *
from agents import *
import matplotlib.pyplot as plt


def dist_mf(rho, mf1, mf2):
    dist = [ 0 for i in range(len(mf1)) ]
    for i in range(len(mf1)):
        v1 = rho(mf1[i])
        v2 = rho(mf2[i])
        dist[i] = abs(v1-v2)
    return [max(dist[i:]) for i in range(len(mf1))]


def dist_mf_set(rho, mf1, mflist):
    delta = [ 0 for i in range(len(mf1)) ]
    for mf2 in mflist:
        dist = dist_mf(rho,mf1,mf2)
        for i in range(len(mf1)):
            delta[i] = max(delta[i],dist[i])
    return delta




def switch_red_blue(a):
    if a=="R":
        return "B"
    elif a=="RT":
        return "BT"
    else:
        a


def switch_blue_red(a):
    if a=="B":
        return "R"
    elif a=="BT":
        return "RT"
    else:
        a


def variation_red(state,n):
    new_state = [""]*len(state)
    for i in range(len(state)):
        if (state[i]=="R" or state[i]=="RT") and n>0:
            new_state[i] = switch_red_blue(state[i])
            n = n-1
        else:
            new_state[i] = state[i]
    return new_state


def variation_blue(state,n):
    new_state = [""]*len(state)
    for i in range(len(state)):
        if (state[i]=="R" or state[i]=="RT") and n>0:
            new_state[i] = switch_red_blue(state[i])
            n = n-1
        else:
            new_state[i] = state[i]
    return new_state


def variations(state, number, max):
    red_to_blue = [variation_red(state,rnd.randint(1,max+1)) for _ in range(int(number/2)) ]
    blue_to_red = [variation_blue(state,rnd.randint(1,max+1)) for _ in range(int(number/2)) ]
    return red_to_blue+blue_to_red


def fraction_red(agents,state):
    o = occupancy(agents,state)
    return o["R"]+o["RT"];


def fraction_blue(agents,state):
    o = occupancy(agents,state)
    return o["B"]+o["BT"];


def fraction_red_trajectory(agents,t):
    return [fraction_red(agents, s) for s in t]


def fraction_blue_trajectory(agents,t):
    return [fraction_blue(agents, s) for s in t]


def penalty(o):
    return abs(o["R"]+o["RT"]-o["B"]-o["BT"])


agents = create_agents()
add_agent(agents, "R")
add_agent(agents, "RT")
add_agent(agents, "B")
add_agent(agents, "BT")

add_action(agents, "R", "redSeen", "RT")
add_action(agents, "RT", "blueSeen", "R")
add_action(agents, "RT", "redSeen", "B")
add_action(agents, "B", "redSeen", "BT")
add_action(agents, "BT", "blueSeen", "R")
add_action(agents, "BT", "redSeen", "B")

alpha = 0.5

prob_function = {'redSeen': lambda o: alpha*(o["R"]+o["RT"]), 'blueSeen': lambda o: alpha*(o["B"]+o["BT"])}

steps = 150

state = ['R']*75+['B']*25

# trace = simulation_run(agents,prob_function,state,steps)
#
# plt.plot(range(0,steps+1), fraction_blue_trajectory(agents,trace), fraction_red_trajectory(agents,trace))
# plt.title("Fraction of Red and Blue agents (N=100).")
# plt.legend(["Blue Agents","Red Agents"])
# plt.savefig("red_blue_100.png")
# plt.show()
#
# for samples in [ 10, 100, 1000 ]:
#     data1 = collect(agents,prob_function,state,steps,samples)
#     print(len(data1))
#     plot_histogram(data1, [50], lambda d: fraction_blue(agents,d), 0.0,1.0, 20, "Probability of Fraction of Blue Agents", "prob_blue_"+str(samples)+"_")

state2 = ['R']*100
#
samples = 20
l = 10
data1 = collect(agents, prob_function, state, steps, samples)
# data2 = collect(agents,prob_function,state2,steps,samples*l)
#
# dist = compute_distance(data1,data2,steps,samples,l,lambda i,v: penalty(occupancy(agents,v)))
#
# print("Distance: "+str(dist[0]))
# plt.plot(range(0,30),dist[:30])
#
# plt.title("Distance R=100 l=10")
# plt.savefig("distance_plot.png")
# plt.show()


# dtrace = meanfield(agents,prob_function,state,100)

### ADAPTATION ESTIMATION

# vars = variations(state,20,25)
#
# dlist = [collect(agents, prob_function, s, steps, samples*l) for s in vars ]
#
# delta = compute_distance_set(data1,dlist,steps,samples,l,lambda i,v: penalty(occupancy(agents,v)))
#
# plt.plot(range(0,steps),delta[:steps])
#
# plt.title("Estimation of Adaptability and Reliability (M=20)")
# plt.savefig("adaptation_20_25.png")
# plt.show()


# trace100 = simulation_run(agents,prob_function,state,steps)
# trace1000 = simulation_run(agents,prob_function,state*10,steps)
# trace10000 = simulation_run(agents,prob_function,state*100,steps)
# mf = meanfield(agents,prob_function,state,steps)
#
# mftrace = [ v["R"]+v["RT"] for v in mf ]
# #plt.plot(range(0,steps+1), fraction_red_trajectory(agents,trace100), fraction_red_trajectory(agents,trace1000), fraction_red_trajectory(agents,trace10000))
# plt.plot(range(0,steps+1), fraction_red_trajectory(agents,trace100))
# plt.plot(range(0,steps+1), fraction_red_trajectory(agents,trace1000))
# plt.plot(range(0,steps+1), fraction_red_trajectory(agents,trace10000))
# plt.plot(range(0,steps+1), mftrace)
# plt.title("Simulation vs Mean Field approximation.")
# plt.legend(["N=100","N=1000","N=10000","Mean Field"])
# plt.savefig("sim_vs_meanfield.png")
# plt.show()

steps = 40
vars = variations(state,20,25)

mf = meanfield(agents,prob_function,state,steps)
mf2 = meanfield(agents,prob_function,state2,steps)

mfvars = [ meanfield(agents,prob_function,state,steps) for s in vars]

delta = dist_mf(penalty,mf,mf2)

plt.plot(range(0,steps),delta[:steps])

plt.title("Mean Field Estimation of Adaptability and Reliability (M=20)")
plt.savefig("mfadaptation_20_25.png")
plt.show()
