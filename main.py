import networkx as nx
import numpy as np
import random
import re
from argparse import ArgumentParser
from collections import deque
import math
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
    T = nx.relabel.convert_node_labels_to_integers(T)
    n_total = T.number_of_nodes()
    subtree_node_list = np.ones(n_total)
    sum = 0
    # Nodes to be explored
    node_queue = deque([x for x in T.nodes if T.degree(x) == 1])

    # Calculate sum of all pairwise distance
    while node_queue:
        current_node = node_queue.pop()
        n_subtree_node = subtree_node_list[current_node]
        current_edges = list(T.edges(current_node, data=True))
        # If there is no more edge, we have finished the graph, return
        if not current_edges:
            break
        # Extract weight and update cost
        _, v, weight = current_edges[0]
        weight = weight['weight']
        sum += weight * (n_total - n_subtree_node) * n_subtree_node
        # Update tree
        T.remove_node(current_node)
        subtree_node_list[v] += subtree_node_list[current_node]
        # Put v to the vertices to explore if it becomes a leaf
        if T.degree(v) == 1:
            node_queue.append(v)
    # Divide it by the number of pairs and retrun
    return sum / (n_total * (n_total - 1) / 2)

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
    # Introduce randomness to avoid being stuck at local optimal
    random.shuffle(tree_edges)

    global_noimp = False
    while not global_noimp:
        global_noimp = True
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
                    global_noimp = False
                T.remove_edge(x, y)
            T.add_edges_from([best_edge])

    return T, T_cost

# search for leaf edges that can reduce cost if removed, Similar to local search.
def leaf_search(G, T, T_cost):
    leaf_nodes = [x for x in T.nodes if T.degree(x) == 1]
    # Introduce randomness to avoid being stuck at local optimal
    random.shuffle(leaf_nodes)
    leaf_nodes = deque(leaf_nodes)
    tree_nodes = set(T.nodes)
    # second queue of leaf in case of new improvement after scanning for the
    # first time; this would only include nodes excluded for no improvment
    leaf_nodes_second = deque()
    noimp_nodes = 0

    # If the current queue's elements are all from the previous loop, no improvement,
    # Return
    while noimp_nodes != len(leaf_nodes):
        noimp_nodes = 0
        while leaf_nodes:
            leaf_node = leaf_nodes.pop()
            u, v, w = list(T.edges(leaf_node, data=True))[0]
            T.remove_node(leaf_node)
            tree_nodes.remove(leaf_node)
            neighbors = list(G.neighbors(leaf_node))

            # Check if removing this leaf will make any vertex no longer adjacent to
            # the tree
            vertex_disconnected = False
            for neighbor in neighbors:
                if neighbor in tree_nodes:
                    continue
                adjacent_vertices = list(G.neighbors(neighbor))
                vertex_disconnected = True
                for adjacent_vertex in adjacent_vertices:
                    if adjacent_vertex in tree_nodes:
                        vertex_disconnected = False
                        break
                if vertex_disconnected:
                    break
            if vertex_disconnected:
                T.add_node(leaf_node)
                tree_nodes.add(leaf_node)
                T.add_edges_from([(u, v, w)])
                continue

            # Check cost
            new_cost = calculate_cost(T)
            if new_cost > T_cost:
                T.add_node(leaf_node)
                tree_nodes.add(leaf_node)
                T.add_edges_from([(u, v, w)])
                # Add the node back to the second queue and
                leaf_nodes_second.append(leaf_node)
                noimp_nodes += 1
            else:
                T_cost = new_cost
                if T.degree(v) == 1:
                    leaf_nodes.append(v)

        # Exchange the two queue
        leaf_nodes, leaf_nodes_second = leaf_nodes_second, leaf_nodes

    return T, T_cost

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
    # Run ABC
    tree, cost = abc(G)
    print("ABC done")
    assert valid_tree_solution(G, tree)
    # Local search
    tree, cost = local_search(G, tree, cost)
    print("Local search done")
    assert valid_tree_solution(G, tree)
    # Leaves removal
    tree, cost = leaf_search(G, tree, cost)
    print("Leaf search done")
    assert valid_tree_solution(G, tree)
    # Test cost
    reference_cost = average_shortest_path_length(tree, weight='weight')
    print(reference_cost)
    print(cost)
    assert reference_cost == cost
    # Print G into the output
    write_output_file(tree, args.filename.replace(".in", ".out"))
