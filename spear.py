import numpy.random as rnd
from typing import List, Tuple, Any, Dict, Callable
from functools import reduce
import matplotlib.pyplot as plt
import math

probability = List[Tuple[Any, float]]
substitution = Dict[str, float]
data_space = Dict[str, float]
expression = Callable[[data_space],float]


def dirac_of(x: Any) -> probability:
    """
    Build a Dirac distribution of element x

    :param x: element
    :return: probability associating 1 to x.
    """
    return [(x, 1.0)]


def map_prob(prob, f):
    return [ (f(v),p) for (v,p) in prob ]


def flatten_prob(probprob):
    lst = [scale(l,p) for (l,p) in probprob]
    return reduce(lambda x,y: x+y, lst)


def scale(prob: probability, p: float) -> probability:
    """
    Given a discrete probability distribution prob and a scalar value p in [0,1] returns
    the subprobability obtained from prob by multiplying each value by p.

    :param prob: a probability distribution
    :param p: a scalar in [0,1]
    :return: a subprobability distribution
    """
    return [(s, x*p) for (s, x) in prob]


### PROCESSES 


def nil(pdef, d):
    """
    Process function associated with process nil.

    :param pdef: process definitions
    :param d: a data space
    :return: a probability distribution of effects*processes
    """
    return dirac_of(({}, nil))


def parallel_left(step, p, proc1):
    """
    Compute left parallel composition of step with proc1. Used to compute
    interleaving semantics.

    :param step: a pair (effect,process)
    :param p: a float in [0,1]
    :param proc1: a process function
    :return: a pair (effect,process)
    """
    return [((sub, parallel_process(proc1,p,proc2)), prob)  for ((sub, proc2), prob) in step ]


def parallel_right(step, p, proc2):
    """
    Compute right parallel composition of step with proc2. Used to compute
    interleaving semantics.

    :param step: a pair (effect,process)
    :param p: a float in [0,1]
    :param proc1: a process function
    :return: a pair (effect,process)
    """
    return [((sub, parallel_process(proc1,p,proc2)), prob)  for ((sub, proc1), prob) in step ]


def parallel_combine(proc1, next1, p, proc2, next2):
    """
    Compute interleaving semantics of proc1 and proc2

    :param proc1: a process
    :param next1: step function of proc1
    :param p: probability value in [0,1]
    :param proc2: a process
    :param next2: step function of proc2
    :return: step function
    """
    return scale(parallel_right(next1, p, proc2), p)+scale(parallel_left(next2, p, proc1), 1-p)


def apply_from_data(act_fun, d):
    """
    Computes the effect (substitution) of an action.

    :param act_fun: process action (a dictionary mapping variables to expressions)
    :param d: a data space
    :return: a substitution
    """
    return dict([(x, f(d)) for (x,f) in act_fun.items()])


def get_process(pdef, pid):
    """
    Given a process id (that is a string) and a set of process definitions, this fucntion compute
    the process semantics associated with pid.

    :param pid: process identifier
    :param pdef:  process definitions
    :return: process function
    """
    if pid in pdef:
        return pdef[pid]
    else:
        return None


def nil_process( ):
    """
    Constructor of nil process.

    :return: nil process function.
    """
    return nil


def act_process(act_fun, next_id):
    """
    Create a process that first executes act_fun than executes next_id.

    :param act_fun: action
    :param next_id: next process
    :return: process function
    """
    return lambda pdef, d: dirac_of((apply_from_data(act_fun, d), get_process(pdef, next_id)))


def parallel_process(proc1, p, proc2):
    """
    Return the step function of process proc1||_p proc2.

    :param proc1: a process
    :param p: probability value
    :param proc2: a process
    :return: parallel composizion of proc1,p,proc2
    """
    return lambda pdef, d: parallel_combine(proc1, proc1(pdef, d), p, proc2, proc2(pdef, d))

def if_then_else_process(guard,proc1,proc2):
    """
    Return the step function of process if guard then proc1 else proc2.

    :param guard: boolean expression
    :param proc1: process
    :param proc2: process
    :return: process function
    """
    return lambda pdef, d: proc1(pdef,d) if guard(d) else proc2(pdef,d)


def choice_prob(probprocess):
    return lambda pdef,d: flatten_prob(map_prob(lambda p: p(pdef,d),probprocess))


def synch_parallel_process(proc1, proc2):
    """
    Return the step function of process proc1||proc2.

    :param proc1: a process
    :param proc2: a process
    :return: synchronous parallel composition of proc1 and proc2
    """
    return lambda pdef, d: synch_parallel_combine(proc1(pdef, d), proc2(pdef, d))


def synch_parallel_combine(next1, next2):
    """
    Compute synchronous semantics of proc1 and proc2

    :param proc1: a process
    :param next1: step function of proc1
    :param proc2: a process
    :param next2: step function of proc2
    :return: step function
    
    """
    return [((sub1 | sub2 , synch_parallel_process(proc1,proc2)), prob1 * prob2) for ((sub1, proc1), prob1) in next1 for ((sub2, proc2), prob2) in next2 ]


### EVOLUTION SEQUENCES

def sample_element_from_list( l ):
    u = rnd.random()
    count = 0.0
    for i in range(0,len(l)):
        count += l[i][1]
        if u<count:
            return l[i][0]
    return nil


def pstep(pdef, p, d):
    return sample_element_from_list(p(pdef, d))


def apply_substitution(d, sub):
    newd = d.copy()
    for (x,v) in sub.items():
        newd[x]=v
    return newd


def cstep(pdef, env, p, d):
    sub,q = pstep(pdef,p,d)
    newd = apply_substitution(d, sub)
    return q, env(newd)


### SIMULATIONS

def run(pdef, env, p, d, k):
    result = [d]
    for i in range(1,k):
        p, d = cstep(pdef, env, p, d)
        result.append(d)
    return result


def simulate(pdef, env, p, d, k, n):
    data = [ [] for i in range(k) ]
    for i in range(n):
        sample = run(pdef, env, p, d, k)
        for j in range(k):
            data[j].append(sample[j])
    return data


def simulatekkk(pdef, env, p, d, k, n):
    data = [ [] for i in range(k) ]
    for i in range(n):
        sample = run(pdef, env, p, d, k)
        for j in range(k):
            data[j].append(sample[j])
    return data


def average(dlist, f):
    v = reduce(lambda x,y: x+y, list(map(f, dlist)))
    return v/len(dlist)


def estimated_cdf(lst):
    return lambda x: len(filter(lambda y: y<=x, lst))/len(lst)


def count(lst,start,end,counter,density=True):
    gap = (end-start)/counter
    xvalues = [start+gap*i for i in range(counter+1) ]
    yvalues = [0.0 for i in range(counter+1) ]
    for v in lst:
        if (start <= v < end):
            idx = math.ceil((v-start)/gap)
            yvalues[idx] = yvalues[idx]+1.0
    if density:
        return xvalues, list(map(lambda v:v/len(lst),yvalues))
    else:
        return xvalues, yvalues


### PLOT STYLES
    
def plot_histogram(data, indexes, f, start, end, blocks, label, file):
    for i in indexes:
        lst = list(map(f, data[i]))
        #v = plt.histogram(lst, bins=[ start+i*((end-start)/blocks) for i in range(blocks) ], range=(start, end), density=True)
        #print(v)
        xvalues,yvalues = count(lst,start,end,blocks)
        print(reduce(lambda x,y:x+y,yvalues))
        plt.plot(xvalues,yvalues)
        plt.title(label+" (step="+str(i)+")")
        plt.savefig(file+"_"+str(i)+".png")
        plt.show()

        
def plot_histogram_double(data1, data2, indexes, f, start, end, blocks, title, file):
    for i in indexes:
        lst1 = list(map(f, data1[i]))
        lst2 = list(map(f, data2[i]))
        x1,y1 = count(lst1,start,end,blocks)
        x2,y2 = count(lst2,start,end,blocks)
        plt.plot(x1,y1,label='Scen1')
        plt.plot(x2,y2,label='Scen2')
        legend=plt.legend()
        plt.title(title+" step="+str(i)+".")
        plt.savefig(file+"_"+str(i)+".png")
        plt.show()
        
        
  def plot_histogram_bis(data, indexes, f, start, end, blocks, label, file):
        i = indexes
        lst = list(map(f, data[i]))
        xvalues,yvalues = count(lst,start,end,blocks)
        plt.plot(xvalues,yvalues)
        plt.title(label)
        plt.savefig(file+"_"+str(i)+".png")
        plt.show()
        
        
 def plot_histogram_ter(data, f, label, file):
    valori = []
    i=0
    for datasets in data:
        lista_f = list(map(f, datasets))
        totale_temp = reduce(lambda x,y: x+y, lista_f)
        media_temp = totale_temp / len(datasets)
        valori.append([i,media_temp])
        i=i+1
    xvalues = [x for (x,y) in valori ]
    yvalues = [y for (x,y) in valori ]
    plt.plot(xvalues,yvalues)
    plt.title(label)
    plt.savefig(file+"_"+str(i)+".png")
    plt.show()
 

def plot_histogram_avg(data, f, label, file):
    valori = []
    i=0
    for datasets in data:
        lista_f = list(map(f, datasets))
        totale_temp = reduce(lambda x,y: x+y, lista_f)
        media_temp = totale_temp / len(datasets)
        valori.append([i,media_temp])
        i=i+1
    xvalues = [x for (x,y) in valori ]
    yvalues = [y for (x,y) in valori ]
    plt.plot(xvalues,yvalues)
    plt.title(label)
    plt.savefig(file+"_"+str(i)+".png")
    plt.show()


def plot_histogram_distance(data, label, file):
    valori = []
    i=0
    for value in data:
        valori.append([i,value])
        i=i+1
    xvalues = [x for (x,y) in valori ]
    yvalues = [y for (x,y) in valori ]
    plt.plot(xvalues,yvalues)
    plt.title(label)
    plt.savefig(file+"_"+str(i)+".png")
    plt.show()
 

def plot_histogram_adapt(data, label, file):
    valori = []
    i=0
    for value in data:
        valori.append([i,value])
        i=i+1
    xvalues = [x for (x,y) in valori ]
    yvalues = [y for (x,y) in valori ]
    plt.plot(xvalues,yvalues)
    plt.title(label)
    plt.savefig(file+"_"+str(i)+".png")
    plt.show()

 
### EVALUATION OF THE EVOLUTION METRIC

def wasserstein(lst1,lst2,n,l):
    lst1.sort()
    lst2.sort()
    sum = 0.0
    for i in range(n):
        for j in range(l):
            sum += max(lst2[i*l+j]-lst1[i],0.0)
    return sum/(n*l)


def distance(pdef1,p1,d1,env1,pdef2,p2,d2,env2,k,n,l,rho):
    data1 = simulate(pdef1,env1,p1,d1,k,n)
    data2 = simulate(pdef2,env2,p2,d2,k,l*n)
    dist = [ 0 for i in range(k) ]
    for i in range(k):
        lst1 = list(map(lambda d: rho(i,d),data1[i]))
        lst2 = list(map(lambda d: rho(i,d),data2[i]))
        dist[i] = wasserstein(lst1,lst2,n,l)
    return [max(dist[i:]) for i in range(k)], dist


def compute_distance(data1,data2,k,n,l,rho):
    dist = [ 0 for i in range(k) ]
    for i in range(k):
        lst1 = list(map(lambda d: rho(i,d),data1[i]))
        lst2 = list(map(lambda d: rho(i,d),data2[i]))
        dist[i] = wasserstein(lst1,lst2,n,l)
    return [max(dist[i:]) for i in range(k)], dist


def distance_set(pdef1,p1,d1,env1,dlist,k,n,l,rho):
    delta = [ 0 for i in range(k) ]
    for d2 in dlist:
        (M,dist) = distance(pdef1,p1,d1,env1,pdef1,p1,d2,env1,k,n,l,rho)
        for i in range(k):
            delta[i] = max(delta[i],M[i])
    return delta


def compute_distance_set(data1,dlist,k,n,l,rho):
    delta = [ 0 for i in range(k) ]
    for data2 in dlist:
        (M,dist) = compute_distance(data1,data2,k,n,l,rho)
        for i in range(k):
            delta[i] = max(delta[i],M[i])
    return delta


def distancerho(pdef1,p1,d1,env1,pdef2,p2,d2,env2,k,n,l,rho):
    data1 = simulatekkk(pdef1,env1,p1,d1,k,n)
    data2 = simulatekkk(pdef2,env2,p2,d2,k,l*n)
    for u in range(n):
        for v in range(k):
           print("tot_instants_of_stress_L=  "+ str(u) +" "+ str(v)+" "+str(data1[v][u]['tot_stress_instants_L']))
    for u in range(l*n):
        for v in range(k):
           print("tot_instants_of_stress_L=  "+ str(u) +" "+ str(v)+" "+str(data2[v][u]['tot_stress_instants_L']))
    dist = [ 0 for i in range(k) ]
    for i in range(k):
        lst1 = list(map(lambda d: rho(d),data1[i]))
        lst2 = list(map(lambda d: rho(d),data2[i]))
        dist[i] = wasserstein(lst1,lst2,n,l)
        
    return dist


def distance_setrho(pdef1,p1,d1,env1,dlist,k,n,l,rho):
    print("d1_attack_window= "+str(d1['attack_window']))
    delta = [ 0 for i in range(k) ]
    for d2 in dlist:
        print("d2_attack_window= "+str(d2['attack_window']))
        dist = distancerho(pdef1,p1,d1,env1,pdef1,p1,d2,env1,k,n,l,rho)
        for i in range(k):
            print("dist[i]="+str(dist[i]))
            delta[i] = max(delta[i],dist[i])
    return delta


### ROBUSTNESS OF FORMULAE

def eval_target(n, l, sample_function, rho, p, elist):
    return [p-wasserstein(sample_function(n), [rho(ds) for ds in eset],n,l) for eset in elist]


def eval_brink(n, l, sample_function, rho, p, elist):
    return [wasserstein([rho(ds)-p for ds in eset[:n]], sample_function(n*l),n,l) for eset in elist]


def eval_not(vlist):
    return [-v for v in vlist]


def eval_and(vlist1,vlist2):
    l = min(len(vlist1), len(vlist2))
    vlist = [ 0 for _ in range(0,l)]
    for i in range(0,l):
        vlist[i] = min(vlist1[i], vlist2[i])
    return vlist


def eval_or(vlist1,vlist2):
    l = min(len(vlist1), len(vlist2))
    vlist = [ 0 for _ in range(0,l) ]
    for i in range(0,l):
        vlist[i] = max(vlist1[i], vlist2[i])
    return vlist


def eval_true(elist):
    return [1.0 for _ in elist]


def eval_imply(vlist1,vlist2):
    l = min(len(vlist1), len(vlist2))
    vlist = [ 0 for _ in range(0,l) ]
    for i in range(0,l):
        vlist[i] = max(-vlist1[i], vlist2[i])
    return vlist


def compute_eventually(vlist):
    if (len(vlist)==0):
        return -1
    else:
        return max(vlist)


def eval_eventually(vlist,a,b):
    return [ compute_eventually(vlist[i+a:i+b+1]) for i in range(0,len(vlist))]


def compute_globally(vlist):
    if (len(vlist)==0):
        return 1
    else:
        return min(vlist)


def eval_globally(vlist,a,b):
    return [ compute_globally(vlist[i+a:i+b+1]) for i in range(0,len(vlist))]


def compute_until(vlist1,vlist2,a,b):
    k = len(vlist1)
    vlist = [ 0 for i in range(0,k) ]
    w1 = [ 0 for i in range(0,k) ]
    w2 = [ 0 for i in range(0,k) ]
    for i in range(0,k):
        for j in range(a+i,b+i):
            w1[j] = min(vlist1[i:j])
            w2[j] = min(w1[j], vlist2[j])
        vlist[i] = max(w2[a+i:b+i])
    return vlist
                
