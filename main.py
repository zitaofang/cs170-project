import networkx as nx
import numpy as np
import random
import re
from argparse import ArgumentParser
import os

# Arguments
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
    best_sol, best_cost, _ = E[np.argmax(np.array([Ei[1] for Ei in E]))]
    global_noimp = 0
    counter = 0
    while global_noimp < 5 * G.number_of_nodes():
        # If the best_sol is improved in this cycle, set to true
        improved = False

        for i in range(nEmployed):
            (current_E, cost, noimp) = E[i]
            new_E = generate_neighboring_solution(G, current_E, E)
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
            p = select_and_return_index(G, E)
            (current_E, cost, noimp) = E[p]
            new_S = generate_neighboring_solution(G, current_E, E)
            new_cost = float('inf') if new_S is None else calculate_cost(new_S)
            if new_cost < best_cost:
                best_sol = new_S
                best_cost = new_cost
                improved = True
            S.append((p, new_S, new_cost))

        for (p, s, cost) in S:
            if cost < E[p][1]:
                E[p] = (s, cost, 0)

        if not improved:
            global_noimp += 1
        counter += 1

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

    vertices = set()
    edges = list()
    candidate_edges = dict()
    current_v = random.choice(list(G.nodes()))
    current_e = None
    # Start running the algorithm
    vertices.add(current_v)
    for i in range(G.number_of_nodes() - 1):
        # Add new candidate edges to the set for next random selection
        for (u, v, w) in G.edges(current_v, data=True):
            if v not in vertices:
                candidate_edges[(u, v)] = ((u, v, w), calculate_prob(w['weight']))
            else:
                # Remove back edges from candidate list
                assert (v, u) in candidate_edges
                candidate_edges.pop((v, u))
        # Select a random edge to add to the tree in the next cycle (may be null at the end)
        candidate_list, p = zip(*candidate_edges.values())
        keys = np.empty(len(candidate_list), dtype=object)
        keys[:] = candidate_list
        p = np.array(p)
        current_e = np.random.choice(keys, p=p / p.sum())
        # Update the current vertex
        current_v = current_e[0] if current_e[1] in vertices else current_e[1]
        # Add the connecting vertex to the tree
        vertices.add(current_v)
        # Add the current edges to the list
        edges.append(current_e)
    # Create graph
    res = nx.Graph()
    res.add_nodes_from(vertices)
    res.add_edges_from(edges)
    return res

# Calculate the cost of T.
def calculate_cost(T):
    assert nx.is_tree(T), "Graph is not a Tree"
    cost_matrix = np.array([d for sublist in dict(nx.all_pairs_shortest_path_length(T)).values() for d in sublist.values()])
    return cost_matrix.sum() / cost_matrix.shape[0]

# See the comment in the paper
def generate_neighboring_solution(G, E, E_list):
    res = E.copy()
    no_sol = True
    for i in range(t_k):
        # Remove random edge
        e = random.choice(list(res.edges(data=True)))
        res.remove_edge(e[0], e[1])
        # Detected partition
        part = set(nx.dfs_tree(res, source=e[0]).nodes())
        # Random other solution
        F = random.choice(E_list)[0]
        # Check all edges across the cut in F
        cut = [e_F for e_F in F.edges(data=True) if ((e_F[0] in part) ^ (e_F[1] in part)) and e_F != e]
        if len(cut) == 0:
            # No solution
            res.add_edges_from([e])
        else:
            no_sol = False
            cost = []
            for e_c in cut:
                res.add_edges_from([e_c])
                cost.append(calculate_cost(res))
                res.remove_edge(e_c[0], e_c[1])
            e_sol = cut[np.argmin(np.array(cost))]
            res.add_edges_from([e_sol])
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
'''
    Note:
    1. We don't need to keep the original T; if it's optimal, then it won't change
    after this algorithm. You also don't need to write a branch for optimal cases
    (like len(cut) == 0), though it's not the case in generate_neighboring_solution()
    where the edge in E that I pulled out is not necessarily in the other solution
    F. I need to keep the original e in that case.
    2. Try to maintain a consistent naming convention with other code and the
    networkX library.
    3. add_edges_from() is a better alternative than add_edge() since it accepts
    the edges format used by edges(). I remove all the weight variables to keep
    the code conscise.
'''
def local_search(G, T, T_cost):
    tree_edges = list(T.edges(data=True))
    all_edges = list(G.edges(data=True))

    for u, v, w in tree_edges:
        # For every edge, remove the edge and add the lightest edges across the
        # resulting cut.
        best_edge = (u, v, w)
        T.remove_edge(u, v)
        part = set(nx.dfs_tree(T, source=u).nodes())

        # Find all edges in cut
        # Use XOR to ensure that only one endpoint is in 'part'
        cut = [(a, b, w) for (a, b, w) in all_edges if (a in part) ^ (b in part)]

        # Look for the minimum cost edges
        for x, y, wc in cut:
            T.add_edges_from([(x, y, wc)])
            cost = calculate_cost(T)
            if cost < T_cost:
                T_cost = cost
                best_edge = (x, y, wc)
            T.remove_edge(x, y)
        T.add_edges_from([best_edge])

    assert nx.is_tree(T) and nx.is_connected(T)
    return T, T_cost
# search for leaf edges that can reduce cost if removed, Similar to local search.
# def leaf_search(G, T, T_cost):

def valid_tree_solution(G, T):
    '''
    Check if all vertices not in T are
    neighbors to at least one vertex in T.

    Returns a boolean.
    '''
    verticesG = set(G.nodes)
    verticesT = list(T.nodes)
    tempSet = set()

    for vertex in verticesT:
        neighbors = list(G.neighbors(vertex))

        for neighbor in neighbors:
            tempSet.add(neighbor)

    if tempSet == verticesG:
        return True

    return False

def read_input_file(path, max_size=None):
    """
    Parses and validates an input file
    :param path: str, a path
    :return: networkx Graph is the input is well formed, AssertionError thrown otherwise
    """
    with open(path, "r") as fo:
        n = fo.readline().strip()
        assert n.isdigit()
        n = int(n)

        lines = fo.read().splitlines()
        fo.close()

        # validate lines
        for line in lines:
            tokens = line.split(" ")

            assert len(tokens) == 3
            assert tokens[0].isdigit() and int(tokens[0]) < n
            assert tokens[1].isdigit() and int(tokens[1]) < n
            assert bool(re.match(r"(^\d+\.\d{1,3}$|^\d+$)", tokens[2]))
            assert 0 < float(tokens[2]) < 100

        G = nx.parse_edgelist(lines, nodetype=int, data=(("weight", float),))
        G.add_nodes_from(range(n))

        assert nx.is_connected(G)

        if max_size is not None:
            assert len(G) <= max_size

        return G

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg

def write_output_file(T, path):
    with open(path, "w") as fo:
        fo.write(" ".join(map(str, T.nodes)) + "\n")
        lines = nx.generate_edgelist(T, data=False)
        fo.writelines("\n".join(lines))
        fo.close()
# main
if  __name__ == "__main__":
    # Parse the input file into a graph
    parser = ArgumentParser(description="Graph Solver")
    parser.add_argument("-i", dest="filename", required=True, help="input file with graph", metavar="FILE", type=lambda x: is_valid_file(parser, x))
    args = parser.parse_args()

    G = read_input_file(args.filename)
    # RUn ABC
    tree, cost = abc(G)
    # Local search
    tree, cost = local_search(G, tree, cost)
    # Leaves removal
    # tree, cost = leaf_search(G, tree, cost)
    # Verify graph
    print(valid_tree_solution(G, tree))
    # Print G into the output
    write_output_file(tree, args.filename.replace(".in", ".out"))
