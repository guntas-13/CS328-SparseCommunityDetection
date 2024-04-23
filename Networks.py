import networkx as nx
from pyvis.network import Network
import community
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from latex import latexify
latexify(columns = 2)



def loadEUCommunity(path):
    community = defaultdict(list)
    with open(path) as f:
         for line in f:
            node, comm_id = map(int, line.strip().split())
            community[comm_id].append(node)
    return community

def loadEUGraph(path):
    G = nx.Graph()
    with open(path) as f:
        for line in f:
            u, v = map(int, line.strip().split())
            if (v != u):
                G.add_edge(u, v)
            else:
                G.add_node(u)
    return G

def loadGraph(path):
    G = nx.Graph()
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            u, v = map(int, line.strip().split("\t"))
            G.add_edge(u, v)
    return G

def loadCommunity(path, k = None):
    community = {}
    node = set()
    with open(path) as f:
        for id, nodes in enumerate(f):
            if (k is not None) and (id >= k):
                break
            community[id] = list(map(int, nodes.strip().split("\t"))) 
            node.update(set(community[id]))
    return community, node

def inducedSubgraph(G, nodes):
    H = G.subgraph(nodes)
    return H

def plotRandomCommunity(G, community, title = None):
    id = np.random.randint(len(community))
    H = G.subgraph(community[id])
    plt.figure(figsize = (12, 8))
    nx.draw(H, with_labels = False, node_size = 50, node_color = "darkblue", edge_color = "black", pos = nx.spring_layout(H, scale = 4))
    plt.title(f"Community {id + 1} {title}")
    plt.show()


# COMMUNITY STRUCTURES
def get_community_dict(communities):
    community_dict = {}
    for i, community in enumerate(communities):
        for node in community:
            community_dict[node] = i
    return community_dict


def get_communities(community_dict):
    communities = defaultdict(list)
    for node, comm_id in community_dict.items():
        communities[comm_id].append(node)
    return list(communities.values())


# EVALUATION METRICS
def edge_betweenness_sparsification(G, k):
    edge_betweenness = nx.edge_betweenness_centrality(G)
    edges_to_remove = sorted(edge_betweenness, key = edge_betweenness.get)[:int((1 - k) * G.number_of_edges())]
    H = G.copy()
    for edge in edges_to_remove:
        H.remove_edge(*edge)
    return H

def edge_random_sparsification(G, k):
    edges = list(G.edges())
    np.random.shuffle(edges)
    H = G.copy()
    for i in range(int((1 - k) * G.number_of_edges())):
        H.remove_edge(*edges[i])
    return H

def edge_jaccard_sparsification(G, k):
    edge_jaccard = nx.jaccard_coefficient(G)
    edges = list(edge_jaccard)
    edges = sorted(edges, key = lambda x: x[2], reverse = True)
    G_sparse = set()
    H = nx.Graph()
    for edge in edges[:int(k * G.number_of_edges())]:
        G_sparse.add(edge[:2])
    
    for node in G.nodes():
        H.add_node(node)
    
    for edge in G_sparse:
        H.add_edge(edge[0], edge[1])
    return H

def edge_L_Spar_sparsification(G, r):
    G_sparse = set()
    H = nx.Graph()
    for node in G.nodes():
        edges = list(G.edges(node))
        edge_sim = nx.jaccard_coefficient(G, edges)
        edges = sorted(edge_sim, key = lambda x: x[2], reverse = True)
        for edge in edges[:int(len(edges)**r)]:
            G_sparse.add(edge[:2])
    for node in G.nodes():
        H.add_node(node)
    for edge in G_sparse:
        H.add_edge(edge[0], edge[1])
    return H

def run_louvain(G):
    communities = community.community_louvain.best_partition(G)
    return communities

def run_lpa(G):
    communities = nx.community.label_propagation.label_propagation_communities(G)
    return communities

def metrics(ground_truth, predicted):
    ari = adjusted_rand_score(list(ground_truth.values()), list(predicted.values()))
    nmi = normalized_mutual_info_score(list(ground_truth.values()), list(predicted.values()))
    return ari, nmi

def modularity(G, communities):
    m = nx.community.modularity(G, communities)
    return m



# MAJOR PLOT FUNCTIONS
def plot_metrics_sparse(G, ground_truth, sparseFunctions, k_values, AlgoFunction, flag, networkName = None, AlgoName = None):
    
    """_summary_

    Args:
        G (nx.Graph): Original Graph
        ground_truth (dictionary): dictionary mapping nodes to communities
        sparseFunctions (list): list of tuples containing the name and the function to generate the sparse graph
        k_values (list): list of values for the percentage of edges to retain
        AlgoFunction (function): Community Detection Algorithm
        flag (int): 0 if the output of the algorithm is a dictionary, 1 if the output is a list of communities
        networkName (str, optional): Name of Network to go in the Title of the Plots. Defaults to None.
        AlgoName (str, optional): Algorithm Name to go in the Title of the Plots. Defaults to None.

    Returns:
        list: List of Sparse Graphs (nx.Graph)
    """
    
    ari_values = [[0] * len(k_values) for _ in sparseFunctions]
    modularity_values = [[0] * len(k_values) for _ in sparseFunctions]
    nmi_values = [[0] * len(k_values) for _ in sparseFunctions]
    names = [name for name, _ in sparseFunctions]
    SparseGraphs = [[0] * len(k_values) for _ in sparseFunctions]
    
    for idx, (_, function) in enumerate(sparseFunctions):
        for i, k in enumerate(k_values):
            H = function(G, k)
            predicted = AlgoFunction(H)
            if (flag == 1):
                predicted = get_community_dict(predicted)
            ari, nmi = metrics(ground_truth, predicted)
            modularity_values[idx][i] = modularity(H, get_communities(predicted))
            ari_values[idx][i] = ari
            nmi_values[idx][i] = nmi
            SparseGraphs[idx][i] = H
    
    plt.figure(figsize = (12, 8))
    plt.xticks(range(len(k_values)), [f"{100 * k}%" for k in k_values])
    for idx in range(len(sparseFunctions)):
        plt.plot(ari_values[idx], label = names[idx], marker = "o")
        for i, txt in enumerate(ari_values[idx]):
            plt.annotate(f"{txt:.4f}", (i, ari_values[idx][i]))      
    plt.xlabel("Percentage Retention of Edges")
    plt.ylabel("ARI")
    plt.title(f"ARI for {AlgoName} vs k for {networkName}")
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(figsize = (12, 8))
    plt.xticks(range(len(k_values)), [f"{100 * k}%" for k in k_values])
    for idx in range(len(sparseFunctions)):
        plt.plot(nmi_values[idx], label = names[idx], marker = "o")
        for i, txt in enumerate(nmi_values[idx]):
            plt.annotate(f"{txt:.4f}", (i, nmi_values[idx][i]))      
    plt.xlabel("Percentage Retention of Edges")
    plt.ylabel("NMI")
    plt.title(f"NMI for {AlgoName} vs k for {networkName}")
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(figsize = (12, 8))
    plt.xticks(range(len(k_values)), [f"{100 * k}%" for k in k_values])
    for idx in range(len(sparseFunctions)): 
        plt.plot(modularity_values[idx], label = names[idx], marker = "o")
        for i, txt in enumerate(modularity_values[idx]):
            plt.annotate(f"{txt:.4f}", (i, modularity_values[idx][i]))

    plt.axhline(y = modularity(G, get_communities(ground_truth)), color = "black", linestyle = "--", label = "Original Graph")
    plt.xlabel("Percentage Retention of Edges")
    plt.ylabel("Modularity")
    plt.title(f"Modularity for {AlgoName} vs k for {networkName}")
    plt.legend()
    plt.grid()
    plt.show()
    
    return SparseGraphs


print("Function Description:\n1. plotRandomCommunity(G, community, title = None)\n2. get_community_dict(communities)\n3. get_communities(community_dict)\n4. run_louvain(G)\n5. metrics(ground_truth, predicted)\n6. plot_metrics_sparse(G, ground_truth, sparseFunctions, k_values, AlgoFunction, flag, networkName = None, AlgoName = None)\n\nSampling Methods:\n1. edge_betweenness_sparsification(G, k)\n2. edge_random_sparsification(G, k)\n3. edge_jaccard_sparsification(G, k)\n4. edge_L_Spar_sparsification(G, r)\n\n")


# DBLP GRAPH

G_DBLP = loadGraph("./Networks/DBLP/com-dblp.ungraph.txt")
communitiesDBLP, nodes = loadCommunity("./Networks/DBLP/com-dblp.top5000.cmty.txt", k = 150)
G_ind_DBLP = inducedSubgraph(G_DBLP, nodes)

print("\n=============================================================================")
print("ORIGINAL GRAPH: G_DBLP, INDUCED SUBGRAPH: G_ind_DBLP, COMMUNITIES: communitiesDBLP")
print("Number of nodes: ", G_DBLP.number_of_nodes())
print("Number of edges: ", G_DBLP.number_of_edges())
print("Number of communities: ", len(communitiesDBLP))
print("Number of nodes in induced subgraph: ", G_ind_DBLP.number_of_nodes())
print("Number of edges in induced subgraph: ", G_ind_DBLP.number_of_edges())



# AMAZON GRAPH

G_Amz = loadGraph("./Networks/AmazonCoPurchase/com-amazon.ungraph.txt")
communitiesAmazon, nodes = loadCommunity("./Networks/AmazonCoPurchase/com-amazon.top5000.cmty.txt", k = 300)
G_ind_Amz = inducedSubgraph(G_Amz, nodes)

print("\n=============================================================================")
print("ORIGINAL GRAPH: G_Amz, INDUCED SUBGRAPH: G_ind_Amz, COMMUNITIES: communitiesAmazon")
print("Number of nodes: ", G_Amz.number_of_nodes())
print("Number of edges: ", G_Amz.number_of_edges())
print("Number of communities: ", len(communitiesAmazon))
print("Number of nodes in induced subgraph: ", G_ind_Amz.number_of_nodes())
print("Number of edges in induced subgraph: ", G_ind_Amz.number_of_edges())



# YOUTUBE GRAPH

G_YT = loadGraph("./Networks/YouTube/com-youtube.ungraph.txt")
communitiesYT, nodes = loadCommunity("./Networks/YouTube/com-youtube.top5000.cmty.txt", k = 100)
G_ind_YT = inducedSubgraph(G_YT, nodes)

print("\n=============================================================================")
print("ORIGINAL GRAPH: G_YT, INDUCED SUBGRAPH: G_ind_YT, COMMUNITIES: communitiesYT")
print("Number of nodes: ", G_YT.number_of_nodes())
print("Number of edges: ", G_YT.number_of_edges())
print("Number of communities: ", len(communitiesYT))
print("Number of nodes in induced subgraph: ", G_ind_YT.number_of_nodes())
print("Number of edges in induced subgraph: ", G_ind_YT.number_of_edges())



# EU EMAIL GRAPH

G_eu = loadEUGraph("./Networks/EmailEUCore/email-Eu-core.txt")
comm_eu = loadEUCommunity("./Networks/EmailEUCore/email-Eu-core-department-labels.txt")

print("\n=============================================================================")
print("ORIGINAL GRAPH: G_eu, COMMUNITIES: comm_eu")
print("Number of nodes: ", G_eu.number_of_nodes())
print("Number of edges: ", G_eu.number_of_edges())
print("Number of communities: ", len(comm_eu))



# FACEBOOK GRAPH

G_FB = loadEUGraph("./Networks/FacebookCircles/facebook_combined.txt")

print("\n=============================================================================")
print("ORIGINAL GRAPH: G_FB")
print("Number of nodes: ", G_FB.number_of_nodes())
print("Number of edges: ", G_FB.number_of_edges())