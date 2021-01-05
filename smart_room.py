from spear import *
import numpy.random as rnd
import matplotlib.pyplot as plt

a_open = 0.2
a_closed = 0.1
b = 0.05

T_MIN = -10.0
T_MAX = 50.0

tmin = 15.0
tmax = 20.0
t_eps = 1.0
q_plus = 0.1
q_minus = 0.1
a_eps = 0.1
a_threashold = 0.75

t_error = 2.0
a_error = 0.1


outside_temperature = 5.0
heater_temperature = 50.0

def rank_t(t):
    if tmin <= t <= tmax:
        return 0
    if tmax<t:
        return abs(tmax-t)/(T_MAX-T_MIN)
    else:
        return abs(t-tmin)/(T_MAX-T_MIN)


def rank_a(a):
    return max(0, a_threashold-a)


def rank(t,a):
    return max(rank_a(a),rank_t(t))


def normalising(ds):
    t = ds['T']
    a = ds['A']
    return rank(t, a)

def ranking_function(i, ds):
    return normalising(ds)


def room_environment(ds):
    newT = ds['T']+rnd.normal(0,1)
    newA = ds['A']+rnd.normal(0,0.1)
    if ds['e'] == 1:
        newA += q_plus*(1-ds['A'])
        newT += a_open*(outside_temperature-ds['T'])
    else:
        newT += a_closed*(outside_temperature-ds['T'])
        newA -= q_minus*ds['A']
    if ds['h'] == 1:
        newT += b*(heater_temperature-ds['T'])
    newA = max(0, min(1, newA))
    newTs = newT+rnd.normal(t_error, 1.0)
    newAs = newA+rnd.normal(a_error, 0.1)
    return {'T': newT, 'Ts': newTs, 'h': ds['h'], 'A': newA, 'As': newAs, 'e': ds['e']}


def set_temperature(d,val):
    d2 = d.copy()
    d2['T'] = val
    d2['Ts'] = val
    return d2


def set_air_quality(d,val):
    d2 = d.copy()
    d2['A'] = val
    d2['As'] = val
    return d2


def set_variation(d, t, a):
    d2 = d.copy()
    d2['T'] = t
    d2['Ts'] = t
    d2['A'] = a
    d2['As'] = a
    return d2


def select( ds, data ):
    i = math.floor(rnd.uniform(0,len(data)-1))
    v1 = data[i]
    v2 = data[i+1]
    return set_temperature(ds, rnd.uniform(v1, v2))


def variations(ds,scale,size):
    v = normalising(ds)
    res = []
    for i in range(0,size):
        t = rnd.uniform(T_MIN,T_MAX)
        a = rnd.uniform(0.0,1.0)
        if abs(v-rank(t,a))<scale:
            res.append(set_variation(ds,t,a))
    return res


processes = {
    'Ton': if_then_else_process(lambda d: d['Ts'] > tmax + t_eps, act_process({'h': lambda d: 0}, 'Toff'), act_process({}, 'Ton')),
    'Toff': if_then_else_process(lambda d: d['Ts'] < tmin - t_eps, act_process({'h': lambda d: 1}, 'Ton'), act_process({}, 'Toff')),
    'Aon': if_then_else_process(lambda d: d['A'] > a_threashold + a_eps, act_process({'e': lambda d: 0}, 'Aoff'),
                                act_process({}, 'Aon')),
    'Aoff': if_then_else_process(lambda d: d['A'] < a_threashold - a_eps, act_process({'e': lambda d: 1}, 'Aon'),
                                 act_process({}, 'Aoff'))
}


init_data = {'T':5.0, 'Ts':5.0, 'h':0, 'A': 0.5, 'As': 0.5, 'e': 0}
init_data2 = {'T':0.0, 'Ts':0.0, 'h':0, 'A': 0.3, 'As': 0.3, 'e': 0}


init_process = parallel_process(processes['Aoff'],0.5,processes['Toff'])
#init_process = processes['Toff']

if __name__ == "__main__":

    for samples in [ 100, 1000, 10000 ]:
        data1 = simulate(processes, room_environment, init_process, init_data, 100, samples)
        plot_histogram(data1, [50], lambda d: d['T'], 5.0, 30.0, 50, "Temperature probability N="+str(samples)+" T=5", "temperature_"+str(samples)+"_")
        plot_histogram(data1, [50], lambda d: d['A'], 0.0, 1.0, 10, "Air quality probability N="+str(samples)+" T=5", "air_"+str(samples)+"_")
        plot_histogram(data1, [50], lambda d: rank(d['T'], d['A']), 0.0, 1.0, 25, "Ranking probability N="+str(samples)+" T=5", "ranking_"+str(samples)+"_")


    dist = distance(processes, init_process, init_data, room_environment, processes, init_process, init_data2, room_environment, 50, 100, 10, ranking_function)

    print("Distance: "+str(dist[0]))
    plt.plot(range(0,50),dist[:50])

    plt.title("Distance N=1000 l=10")
    plt.savefig("distance_1000_10.png")
    plt.show()

    delta = distance_set(processes, init_process, init_data, room_environment, variations(init_data, 0.2, 50) + [init_data2], 50, 100, 10, ranking_function)

    print("Distance: "+str(delta[0]))
    plt.plot(range(0,50),delta[:50])

    plt.title("Estimation of Adaptability and Reliability (M=100)")
    plt.savefig("adaptation_100_02.png")
    plt.show()


