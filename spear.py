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


def wasserstein(lst1,lst2,n,l):
    lst1.sort()
    lst2.sort()
    sum = 0.0
    for i in range(n):
        for j in range(l):
            sum += abs(lst1[i]-lst2[i*l+j])
    return sum/(n*l)


def distance(pdef1,p1,d1,env1,pdef2,p2,d2,env2,k,n,l,rho):
    data1 = simulate(pdef1,env1,p1,d1,k,n)
    data2 = simulate(pdef2,env2,p2,d2,k,l*n)
    dist = [ 0 for i in range(k) ]
    for i in range(k):
        lst1 = list(map(lambda d: rho(i,d),data1[i]))
        lst2 = list(map(lambda d: rho(i,d),data2[i]))
        dist[i] = wasserstein(lst1,lst2,n,l)
    return [max(dist[i:]) for i in range(k)]


def compute_distance(data1,data2,k,n,l,rho):
    dist = [ 0 for i in range(k) ]
    for i in range(k):
        lst1 = list(map(lambda d: rho(i,d),data1[i]))
        lst2 = list(map(lambda d: rho(i,d),data2[i]))
        dist[i] = wasserstein(lst1,lst2,n,l)
    return [max(dist[i:]) for i in range(k)]


def distance_set(pdef1,p1,d1,env1,dlist,k,n,l,rho):
    delta = [ 0 for i in range(k) ]
    for d2 in dlist:
        dist = distance(pdef1,p1,d1,env1,pdef1,p1,d2,env1,k,n,l,rho)
        for i in range(k):
            delta[i] = max(delta[i],dist[i])
    return delta


def compute_distance_set(data1,dlist,k,n,l,rho):
    delta = [ 0 for i in range(k) ]
    for data2 in dlist:
        dist = compute_distance(data1,data2,k,n,l,rho)
        for i in range(k):
            delta[i] = max(delta[i],dist[i])
    return delta
