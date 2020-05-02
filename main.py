import networkx as nx
import numpy as np
import random
import re
from argparse import ArgumentParser
from collections import deque
import math
import os
import threading
import queue
import datetime

# Arguments
nEmployed = 50
nOnlooker = 150
noimp_limit_coeff = 5
global_noimp_limit_coeff = 0.25
p_sq = 0.25
p_better = 0.95
t_k = 5

# Multithread queue
file_queue = queue.Queue()
# Log file
log_file = None
log_lock = threading.Lock()

# Solution tree class for fast computation
class Solution:
    def __init__(self, n, vertices, edges):
        self.n = n
        self.m = len(edges)
        self.next_i = self.m
        self.nodes = vertices
        self.neighbors = [[] for _ in range(n)]
        self.edges = dict()
        self.edges_reverse = dict()
        for i, (u, v, w) in enumerate(edges):
            assert (u, v) not in self.edges_reverse and (v, u) not in self.edges_reverse
            self.neighbors[u].append((v, w['weight']))
            self.neighbors[v].append((u, w['weight']))
            self.edges[i] = (u, v, w['weight'])
            self.edges_reverse[u, v] = i
        self.pop_u = None
        self.pop_v = None
        self.disabled_edge = None
        self.disabled_node = False

    def copy(self):
        res = Solution(self.n, [], [])
        res.m = self.m
        res.next_i = self.next_i
        res.nodes = self.nodes.copy()
        res.neighbors = [l.copy() for l in self.neighbors]
        res.edges = self.edges.copy()
        res.edges_reverse = self.edges_reverse.copy()
        return res

    def degree_list(self):
        res = [len(l) for l in self.neighbors]
        if self.disabled_edge is not None:
            res[self.disabled_edge[0]] -= 1
            res[self.disabled_edge[1]] -= 1
        return res

    def connected_component(self, v):
        # Since this is only called on one of the endpoints of disabled_edge...
        visited = np.zeros(self.n, dtype=bool)
        res = set()
        def explore(u):
            visited[u] = True
            res.add(u)
            for next, _ in self.neighbors[u]:
                if not visited[next] and not (u == self.disabled_edge[0] and next == self.disabled_edge[1]):
                    explore(next)
        for u, _ in self.neighbors[v]:
            if not (v == self.disabled_edge[0] and u == self.disabled_edge[1]):
                explore(u)
        return res

    def number_of_nodes(self):
        res = len(self.nodes)
        if self.disabled_node:
            res -= 1
        return res

    def disable_node(self):
        assert not self.disabled_node
        self.disabled_node = True

    def reenable_node(self):
        assert self.disabled_node
        self.disabled_node = False

    def remove_node(self, v):
        assert not self.neighbors[v]
        if self.disabled_node:
            self.reenable_node()
        self.nodes.remove(v)

    def disable_edge(self, u, v):
        self.disabled_edge = (u, v)

    def disabled(self, u, v):
        return ((u, v) == self.disabled_edge or (v, u) == self.disabled_edge)

    def reenable_edge(self):
        self.disabled_edge = None

    def remove_edge(self, u, v):
        if (u, v) == self.disabled_edge:
            self.disabled_edge = None
        self.m -= 1
        if (v, u) in self.edges_reverse:
            v, u = u, v
        i = self.edges_reverse[u, v]
        del self.edges_reverse[u, v]
        del self.edges[i]
        self.neighbors[u] = [a for a in self.neighbors[u] if a[0] != v]
        self.neighbors[v] = [a for a in self.neighbors[v] if a[0] != u]

    def add_edge(self, e):
        u, v, w = e
        assert (u, v) not in self.edges_reverse and (v, u) not in self.edges_reverse
        i = self.next_i
        self.next_i += 1
        self.edges_reverse[u, v] = i
        self.edges[i] = (u, v, w)
        self.neighbors[u].append((v, w))
        self.neighbors[v].append((u, w))
        # Save for pop_edge
        self.pop_u = u
        self.pop_v = v
        self.m += 1

    # Remove the last edge inserted - quick version
    def pop_edge(self):
        self.m -= 1
        i = self.edges_reverse[self.pop_u, self.pop_v]
        del self.edges_reverse[self.pop_u, self.pop_v]
        del self.edges[i]
        self.neighbors[self.pop_u].pop()
        self.neighbors[self.pop_v].pop()
        self.pop_u = None
        self.pop_v = None

    def to_nx(self):
        res = nx.Graph()
        res.add_nodes_from(self.nodes)
        res.add_edges_from([(u, v, { 'weight' : w }) for (u, v, w) in self.edges.values()])
        return res

# ABC algorithm core
def abc(G):
    # Setting
    n = G.number_of_nodes()
    noimp_limit = noimp_limit_coeff * n

    E = [random_solution(G) for i in range(nEmployed)]
    best_sol, best_cost, _ = E[np.argmax(np.array([Ei[1] for Ei in E]))]
    global_noimp = 0

    while global_noimp < 25:
        # If the best_sol is improved in this cycle, set to true
        improved = False

        for i in range(nEmployed):
            (current_E, cost, noimp) = E[i]
            new_E, new_cost = generate_neighboring_solution(G, current_E, cost, E)
            if new_E is None:
                E[i] = random_solution(G)
            else:
                if new_cost < cost:
                    E[i] = (new_E, new_cost, 0)
                else:
                    noimp += 1
                    if noimp >= noimp_limit:
                        E[i] = random_solution(G)
                    else:
                        E[i] = (current_E, cost, noimp)

        S = []
        for i in range(nOnlooker):
            p = select_and_return_index(G, E)
            (current_E, cost, noimp) = E[p]
            new_S, new_cost = generate_neighboring_solution(G, current_E, cost, E)
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
        else:
            global_noimp = 0

    return best_sol, best_cost

# Random instance
def random_solution(G):
    n = G.number_of_nodes()
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
    res = Solution(n, vertices, edges)
    # Calculate its cost and return in the required format (a tuple)
    return (res, calculate_cost(res, n), 0) # 0 is cycles without improvement

# Calculate the cost of T.
def calculate_cost(T, n_G):
    n = T.number_of_nodes()
    subtree_node_list = np.ones(n_G)
    sum = 0
    # Nodes to be explored
    visited = np.zeros(n_G, dtype=bool)
    degrees = np.array(T.degree_list())
    node_queue = deque([x for x in T.nodes if degrees[x] == 1])

    # Calculate sum of all pairwise distance
    while node_queue:
        current_node = node_queue.pop()
        n_subtree_node = subtree_node_list[current_node]
        current_edges = [e for e in T.neighbors[current_node] if not visited[e[0]] and not T.disabled(current_node, e[0])]
        # If there is no more edge, we have finished the graph, return
        if not current_edges:
            break
        # Extract weight and update cost
        v, weight = current_edges[0]
        sum += weight * (n - n_subtree_node) * n_subtree_node
        # Update tree
        visited[current_node] = True
        degrees[v] -= 1
        subtree_node_list[v] += subtree_node_list[current_node]
        # Put v to the vertices to explore if it becomes a leaf
        if degrees[v] == 1:
            node_queue.append(v)
    # Divide it by the number of pairs and retrun
    return sum / (n * (n - 1) / 2)

# See the comment in the paper
def generate_neighboring_solution(G, E, E_cost, E_list):
    n = G.number_of_nodes()
    res = E.copy()
    res_cost = E_cost
    no_sol = True
    for i in range(t_k):
        # Remove random edge
        e = random.choice(list(res.edges.values()))
        res.disable_edge(e[0], e[1])
        # Detected partition
        part = res.connected_component(e[0])
        # Random other solution
        F = random.choice(E_list)[0]
        # Check all edges across the cut in F
        cut = [e_F for e_F in F.edges.values() if ((e_F[0] in part) ^ (e_F[1] in part)) and e_F != e]
        if len(cut) == 0:
            # No solution
            res.reenable_edge()
        else:
            no_sol = False
            # Commit change
            res.remove_edge(e[0], e[1])
            cost = []
            for e_c in cut:
                res.add_edge(e_c)
                cost.append(calculate_cost(res, n))
                res.pop_edge()
            cut_ind = np.argmin(np.array(cost))
            e_sol = cut[cut_ind]
            res.add_edge(e_sol)
            res_cost = cost[cut_ind]
            # Got a solution, return
            break
    if no_sol:
        res, res_cost = None, float('inf')
    return res, res_cost

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
    n = G.number_of_nodes()
    tree_edges = list(T.edges.values())
    all_edges = [(u, v, w['weight']) for (u, v, w) in G.edges(data=True)]
    # Introduce randomness to avoid being stuck at local optimal
    random.shuffle(tree_edges)

    global_noimp = False
    while not global_noimp:
        global_noimp = True
        for u, v, w in tree_edges:
            # For every edge, remove the edge and add the lightest edges across the
            # resulting cut.
            best_edge = (u, v, w)
            # Call disable first to make connected_component() works
            T.disable_edge(u, v)
            part = T.connected_component(u)
            T.remove_edge(u, v)

            # Find all edges in cut
            # Use XOR to ensure that only one endpoint is in 'part'
            cut = [(a, b, wc) for (a, b, wc) in all_edges if (a in part) ^ (b in part)]

            # Look for the minimum cost edges
            for x, y, wc in cut:
                T.add_edge((x, y, wc))
                cost = calculate_cost(T, n)
                if cost < T_cost:
                    T_cost = cost
                    best_edge = (x, y, wc)
                    global_noimp = False
                T.pop_edge()
            T.add_edge(best_edge)

    return T, T_cost

# search for leaf edges that can reduce cost if removed, Similar to local search.
def leaf_search(G, T, T_cost):
    n = G.number_of_nodes()
    degrees = np.array(T.degree_list())
    leaf_nodes = [x for x in T.nodes if degrees[x] == 1]
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
            v, w = list(T.neighbors[leaf_node])[0]
            T.disable_edge(leaf_node, v)
            T.disable_node()
            # T.remove_node(leaf_node)
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
                # T.add_node(leaf_node)
                T.reenable_node()
                tree_nodes.add(leaf_node)
                T.reenable_edge()
                continue

            # Check cost
            new_cost = calculate_cost(T, n)
            if new_cost > T_cost:
                # Failed, revert
                # T.add_node(leaf_node)
                T.reenable_node()
                tree_nodes.add(leaf_node)
                T.reenable_edge()
                # Add the node back to the second queue for next cycle
                leaf_nodes_second.append(leaf_node)
                noimp_nodes += 1
            else:
                # Success, commit change
                T.remove_edge(leaf_node, v)
                T.remove_node(leaf_node)
                T_cost = new_cost
                if len(T.neighbors[v]) == 1:
                    leaf_nodes.append(v)

        # Exchange the two queue
        leaf_nodes, leaf_nodes_second = leaf_nodes_second, leaf_nodes

    return T, T_cost

def remove_self_loops(G):
    for (u, v) in list(G.edges):
        if u == v:
            G.remove_edge(u, v)
    return G

def valid_tree_solution(G, T):
    '''
    Check if all vertices not in T are
    neighbors to at least one vertex in T.

    Returns a boolean.
    '''
    if not nx.is_tree(T.to_nx()) or T.disabled_edge or T.disabled_node:
        return False
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

def is_valid_path(string):
    if os.path.isdir(string):
        return string
    else:
        parser.error("The file %s does not exist!" % string)

def write_output_file(T, path):
    with open(path, "w") as fo:
        fo.write(" ".join(map(str, T.nodes)) + "\n")
        lines = nx.generate_edgelist(T, data=False)
        fo.writelines("\n".join(lines))

def run_on_all_files(num):
    while True:
        # Get a file from the list
        try:
            item = file_queue.get(block=False)
        except queue.Empty:
            print('Thread ' + str(num) + ' exits!')
            return
        # Fault protection: continue to the next file on exception and dump error
        # to a file
        print('Thread ' + str(num) + ' now at ' + item + '!')
        try:
            G = read_input_file(item)
            G = remove_self_loops(G)
            min_tree, min_cost = None, float("inf")
            for i in range(5):
                # Run ABC
                tree, cost = abc(G)
                assert valid_tree_solution(G, tree)
                # Local search
                tree, cost = local_search(G, tree, cost)
                assert valid_tree_solution(G, tree)
                # Leaves removal
                tree, cost = leaf_search(G, tree, cost)
                assert valid_tree_solution(G, tree)
                # Test cost
                if cost < min_cost:
                    min_tree = tree
                    min_cost = cost
                print('Thread ' + str(num) + ' finished cycle ' + str(i) + '!')
            # Print G into the output
            print("Minimum cost: " + str(cost))
            min_tree = min_tree.to_nx()
            write_output_file(min_tree, item.replace(".in", ".out"))
        except Exception as e:
            # print debug message
            log_lock.acquire()
            log_file.write(str(e) + '\n')
            log_lock.release()
            print('Thread ' + str(num) + ' triggered an exception at ' + item + '!')
        file_queue.task_done()

# main
if  __name__ == "__main__":
    # Parse the input file into a graph
    parser = ArgumentParser(description="Graph Solver")
    parser.add_argument("-p", dest="path", required=True, help="input folder with graphs", type=lambda x: is_valid_path(x))
    args = parser.parse_args()
    # Print timestamp
    start_time = datetime.datetime.now()
    # Open log file
    log_file = open('main.log', "w")
    # Put all *.in file into the queue
    regex = re.compile('.*\.in')
    for file in os.listdir(args.path):
        if regex.match(file):
            # If *.out already exists, skip
            if not os.path.isfile(os.path.join(args.path, file).replace(".in", ".out")):
                file_queue.put(os.path.join(args.path, file))
    # Run threads
    for i in range(8):
        threading.Thread(target=run_on_all_files, args=(i,), daemon=True).start()
    file_queue.join()
    # Success: print current time
    log_file.close()
    print("all files processed")
    end_time = datetime.datetime.now()
    print("Start at: " + str(start_time))
    print("End at: " + str(end_time))
    print("Total time: " + str(end_time - start_time))
