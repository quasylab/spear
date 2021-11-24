import numpy.random as rnd


def create_agents():
    """
    Creates a new agents definition.
    :return: dictionary with agents definitions.
    """
    return {}


def add_agent(agents,a):
    """
    Add an agent to the agents definitions.

    :param agents: agents definition
    :param agent:  agent name
    """
    agents[a]={}


def add_action(agents,agent,act,next_state):
    """
    Agg an action to a given agent.

    :param agents: agents definition
    :param agent: agent name
    :param act: action name
    :param next_state: next state
    """
    agents[agent][act]=next_state


def occupancy(agents, global_state):
    counter = {a: 0.0 for a in agents}
    for v in global_state:
        counter[v] = counter[v]+1.0
    return {a: counter[a]/len(global_state) for a in agents}


def generate_matrix_k(agents, prob_function, o):
    k = {}
    for v in agents:
        k[v] = {}
        row = 0.0
        for a in agents[v]:
            p = prob_function[a](o)
            row += p
            w = agents[v][a]
            if w in k[v]:
                k[v][w] = k[v][w]+p
            else:
                k[v][w] = p
        k[v][v] = 1 - row
    return k


def multiply(k, o):
    o2 = {}
    for v in o:
        row = k[v]
        p = o[v]
        for w in row:
            if w in o2:
                o2[w] = p*row[w]+o2[w]
            else:
                o2[w] = p*row[w]
    return o2


def sample(k, a):
    u = rnd.random()
    counter = 0.0
    for w in k[a]:
        counter += k[a][w]
        if u <= counter:
            return w
    return a


def next_global_state(agents, prob_function, state):
    k = generate_matrix_k(agents,prob_function,occupancy(agents,state))
    return [sample(k, a) for a in state]


def simulation_run(agents, prob_function, state, steps):
    result = [state]
    for i in range(steps):
        state = next_global_state(agents, prob_function, state)
        result.append(state)
    return result


def simulate(agents, prob_function, state, steps, replica):
    return [simulation_run(agents,prob_function,state,steps) for _ in range(replica)]


def collect(agents, prob_function, state, steps, replica):
    data = [ [] for i in range(steps) ]
    for i in range(replica):
        sample = simulation_run(agents, prob_function, state, steps)
        for j in range(steps):
            data[j].append(sample[j])
    return data


def meanfield(agents, prob_function, state, steps):
    o = occupancy(agents,state)
    result = [o]
    for i in range(steps):
        k = generate_matrix_k(agents,prob_function,o)
        o = multiply(k,o)
        result.append(o)
    return result
