import networkx as nx
import numpy as np
import random

nEmployed = 50
nOnlooker = 150
noimp_limit_coeff = 5
p_sq = 0.25
p_better = 0.95
t_k = 5

# ABC algorithm core
def abc(G):
    # Setting
    noimp_limit = noimp_limit_coeff * G.number_of_nodes()

    E = [random_solution(G) for i in range(nEmployed)]
    E = [(Ei, calculate_cost(Ei), 0) for Ei in E]
    best_sol, best_cost, _ = E[np.argmax(np.array([calculate_cost(e) for e in E]))]
    global_noimp = 0
    while no_improvement < 20 * G.number_of_nodes():
        # If the best_sol is improved in this cycle, set to true
        improved = False

        for i in range(nEmployed):
            (current_E, cost, noimp) = E[i]
            new_E = generate_neighboring_solution(G, current_E)
            if new_E is None:
                new_E = random_solution(G)
                E[i] = (new_E, calculate_cost(new_E), 0)
            else:
                new_cost = calculate_cost(new_E)
                if new_cost < cost:
                    E[i] = (new_E, new_cost, 0)
                else:
                    noimp += 1
                    if noimp >= noimp_limit:
                        new_E = random_solution(G)
                        E[i] = (new_E, calculate_cost(new_E), 0)
                    else:
                        E[i] = (current_E, cost, noimp)

        S = []
        for i in range(nOnlooker):
            p = select_and_return_index(E)
            (current_E, cost, noimp) = E[i]
            new_S = generate_neighboring_solution(current_E)
            new_cost = float('inf') if new_S is None else calculate_cost(new_S)
            if new_cost < best_cost:
                best_sol = new_S
                best_cost = new_cost
            S.append((p, new_S, new_cost))

        for (p, s, cost) in S:
            if cost < E[p][1]:
                E[p] = (s, cost, 0)
                improved = True

        if not improved:
            global_noimp += 1

    return best_sol, best_cost

# Random instance
def random_solution(G):
    # Solution picking mode
    use_square = random.random() < p_sq
    def calculate_prob(w):
        if use_square:
            return 1 / w ** 2
        else:
            return 1 / w

    current_v = choice(G.nodes())
    vertices = set([current_v])
    edges = dict([(e, calculate_prob(e[2]['weight'])) for e in G.edges(current_v)])
    weight_sum = sum(edges.values())
    # Start running the algorithm
    for i in range(G.number_of_nodes() - 1):
        e = np.random.choice(np.array([edges.keys()]), p=np.array([edges.values()]) / weight_sum)
        current_v = e[0] if e[1] in vertices else e[1]
        vertices.add(current_v)
        for (u, v, w) in G.edges(current_v):
            neighbor = u if v == current_v else v
            if neighbor not in vertices:
                weight_sum += w['weight']
                edges[(u, v, w)] = calculate_prob(w['weight'])
    # Create graph
    res = nx.Graph()
    res.add_nodes_from(vertices)
    res.add_edges_from(edges.keys())
    return res

# Calculate the cost of T.
def calculate_cost(T):
    visited = dict([(v, False) for v in T.nodes()])
    res = 0

# See the comment in the paper
def generate_neighboring_solution(G, E, E_list):
    res = E.copy()
    no_sol = True
    for i in range(t_k):
        # Remove random edge
        e = random.choice(res.edges())
        res.remove_edge(e[0], e[1])
        # Detected partition
        part = set(nx.dfs_tree(res, source=e[0]).nodes())
        # Random other solution
        F = random.choice(E_list)[0]
        # Check all edges across the cut in F
        cut = []
        for e_F in F.edges():
            if ((e_F[0] in part) ^ (e_F[1] in part)) and e_F != e:
                cut.append(e_F)
        if cut is None:
            # No solution
            res.add_edge(e)
        else:
            no_sol = False
            cost = []
            for e_c in cut:
                res.add_edge(e_c)
                cost.append(calculate_cost(e_c))
                res.remove_edge(e_c[0], e_c[1])
            e_sol = cut[np.argmin(np.array(cost))]
            res.add_edge(e_sol)
            # Got a solution, return
            break
    if no_sol:
        res = None
    return res

# See the comment in the paper
def select_and_return_index(G, E_list):
    a_ind = random.randrange(len(E_list))
    b_ind = random.randrange(len(E_list))
    choose_better = random.random() < p_better
    if choose_better:
        return a_ind if E_list[a_ind][1] < E_list[b_ind][1] else b_ind
    else:
        return b_ind if E_list[a_ind][1] < E_list[b_ind][1] else a_ind

# Local search. See the paper for details. See the main code for input and output
# format.
def local_search(G, T, cost):
    pass

# search for leaf edges that can reduce cost if removed, Similar to local search.
def leaf_search(G, T):
    pass

# main
if  __name__ == "__main__":
    # Parse the input file into a graph
    G = None # TODO: Replace this
    # RUn ABC
    tree, cost = ABC(G)
    # Local search
    tree, cost = local_search(G, tree, cost)
    # Leaves removal
    tree, cost = leaf_search(G, tree, cost)
    # Print G into the output
