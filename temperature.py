from spear import *
import numpy.random as rnd
import matplotlib.pyplot as plt

a = 0.1
b = 0.05

tmin = 15.0
tmax = 20.0
eps = 1.0

outside_temperature = 5.0
heater_temperature = 50.0


def rank(t):
    if tmin <= t <= tmax:
        return 0
    if tmax<t:
        return abs(tmax-t)/50
    else:
        return abs(t-tmin)/50


def normalising(ds):
    t = ds['T']
    return rank(t)

def ranking_function(i, ds):
    return normalising(ds)


def temp_environment(ds):
    newT = ds['T']+a*(outside_temperature-ds['T'])+ds['h']*b*(heater_temperature-ds['T'])+rnd.normal(0,1)
    newes = rnd.normal(0,1)
    newTs = newT+newes
    return {'T':newT, 'Ts':newTs, 'h':ds['h'], 'es': newes}

def set_temperature(d,val):
    d2 = d.copy()
    d2['T']=val
    d2['Ts']=val
    return d2

def select( ds, data ):
    i = math.floor(rnd.uniform(0,len(data)-1))
    v1 = data[i]
    v2 = data[i+1]
    return set_temperature(ds, rnd.uniform(v1, v2))

def variations(ds,scale,size):
    v = normalising(ds)
    data = list(filter(lambda v2: abs(v-rank(v2))<scale, list(range(-10,40))))
#    return 10
    return [ select(ds,data) for i in range(size) ]

processes = {
    'Aon': if_then_else_process(lambda d: d['Ts'] > tmax + eps, act_process({'h': lambda d: 0}, 'Aoff'), act_process({}, 'Aon')),
    'Aoff': if_then_else_process(lambda d: d['Ts'] < tmin - eps, act_process({'h': lambda d: 1}, 'Aon'), act_process({}, 'Aoff'))
}


init_data = {'T':5.0, 'Ts':5.0, 'h':0, 'es': 0}
init_data2 = {'T':0.0, 'Ts':0.0, 'h':0, 'es': 0}

init_process = processes['Aoff']


for samples in [ 100, 1000, 10000 ]:
    data1 = simulate(processes, temp_environment, init_process, init_data, 100, samples)
    plot_histogram(data1, [50], lambda d: d['T'], 5.0,30.0, 50, "Temperature probability N="+str(samples)+" T=5", "temperature_"+str(samples)+"_")


dist = distance(processes,init_process,init_data,temp_environment,processes,init_process,init_data2,temp_environment,50,100,10,ranking_function)

print("Distance: "+str(dist[0]))
plt.plot(range(0,30),dist[:30])

plt.title("Distance N=1000 l=10")
plt.savefig("distance_1000_10.png")
plt.show()

delta = distance_set(processes,init_process,init_data,temp_environment,variations(init_data,0.2,50)+[init_data2],30,100,10,ranking_function)

print("Distance: "+str(delta[0]))
plt.plot(range(0,30),delta[:30])

plt.title("Estimation of Adaptability and Reliability (M=100)")
plt.savefig("adaptation_100_02.png")
plt.show()


