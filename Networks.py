import networkx as nx
from pyvis.network import Network
import community
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from cdlib import algorithms
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

def clustering_coeffs_edge_sampling(graph, sampling_ratio):
    H = nx.Graph()
    edge_cc_prod = []
    edges = []
    clustering_coeffs = nx.clustering(graph)
    for edge in graph.edges():
        edges.append(edge)
        edge_cc_prod.append(clustering_coeffs[edge[0]]*clustering_coeffs[edge[1]])
    indices = np.argsort(edge_cc_prod)[::-1]
    sampled_edges = np.array(edges)[indices][: int(graph.number_of_edges()*sampling_ratio)]
    H.add_edges_from(sampled_edges)
    H.add_nodes_from(graph.nodes)
    return H

def metropolis_hastings_algorithm(graph:nx.Graph, sampling_ratio):
    seed_node = np.random.choice(np.array(graph.nodes()))
    current_node = seed_node
    num_edges = 0
    patience = 0
    alpha = 0.3

    H = nx.Graph()
    while num_edges < int(graph.number_of_edges()*sampling_ratio):
        if patience == 10:
            current_node = np.random.choice(np.array(graph.nodes()))
            patience = 0
            alpha = alpha**1.5
        neighbors = graph.neighbors(current_node)
        new_node = np.random.choice(list(neighbors))
        if not H.has_edge(current_node, new_node):
            r = np.random.uniform(alpha, 1)
            if r < (graph.degree(current_node)/graph.degree(new_node)):
                H.add_edges_from([(current_node, new_node)])
                current_node = new_node
                num_edges += 1
            else:
                current_node = current_node
        else:
            current_node = current_node
            patience += 1
    
    H.add_nodes_from(graph.nodes)
    return H

def effective_resistance_sampling_2(graph:nx.Graph, sampling_ratio):
    list_sampled_edges = []
    l = 0
    for component in list(nx.connected_components(graph)):
        subgraph = nx.subgraph(graph, component)
        edges = np.array(subgraph.edges())
        resistances = []
        for i in range(edges.shape[0]):
            resistances.append(nx.resistance_distance(subgraph, edges[i,0], edges[i,1]))
        sorted_edges = edges[np.argsort(resistances)[::-1]]
        list_sampled_edges.append(sorted_edges[:int(subgraph.number_of_edges()*sampling_ratio)])
        l += int(subgraph.number_of_edges()*sampling_ratio)
    sampled_edges = np.zeros(shape=(l, 2))
    p = 0
    for arr in list_sampled_edges:
        sampled_edges[p: p+len(arr), :] = arr
        p += len(arr)
    H = nx.Graph()
    H.add_edges_from(sampled_edges)
    H.add_nodes_from(graph.nodes)
    return H

def run_louvain(G):
    communities = community.community_louvain.best_partition(G)
    return communities

def run_lpa(G):
    communities = nx.community.label_propagation.label_propagation_communities(G)
    return communities

def run_walktrap(G):
    return algorithms.walktrap(G)

def run_infomap(G):
    return algorithms.infomap(G)

def metrics(ground_truth, predicted):
    ari = adjusted_rand_score(list(ground_truth.values()), list(predicted.values()))
    nmi = normalized_mutual_info_score(list(ground_truth.values()), list(predicted.values()))
    return ari, nmi

def modularity(G, communities):
    m = nx.community.modularity(G, communities)
    return m

def clustering_coefficient(G):
    return nx.average_clustering(G)



# MAJOR PLOT FUNCTIONS
def plot_metrics_sparse(G, ground_truth, sparseFunctions, k_values, AlgoFunction, flag, networkName = None, AlgoName = None):
    
    """_summary_

    Args:
        G (nx.Graph): Original Graph
        ground_truth (dictionary): dictionary mapping nodes to communities
        sparseFunctions (list): list of tuples containing the name and the function to generate the sparse graph
        k_values (list): list of values for the percentage of edges to retain
        AlgoFunction (function): Community Detection Algorithm
        flag (int): 0 if the output of the algorithm is a dictionary, 1 if the output is a list of communities, 2 if the output is a community object
        networkName (str, optional): Name of Network to go in the Title of the Plots. Defaults to None.
        AlgoName (str, optional): Algorithm Name to go in the Title of the Plots. Defaults to None.

    Returns:
        list: List of Sparse Graphs (nx.Graph)
    """
    
    ari_values = [[0] * len(k_values) for _ in sparseFunctions]
    modularity_values = [[0] * len(k_values) for _ in sparseFunctions]
    nmi_values = [[0] * len(k_values) for _ in sparseFunctions]
    clust_coeff_values = [[0] * len(k_values) for _ in sparseFunctions]
    names = [name for name, _ in sparseFunctions]
    SparseGraphs = [[0] * len(k_values) for _ in sparseFunctions]
    
    for idx, (_, function) in enumerate(sparseFunctions):
        for i, k in enumerate(k_values):
            H = function(G, k)
            predicted = AlgoFunction(H)
            if (flag == 1):
                predicted = get_community_dict(predicted)
            elif (flag == 2):
                predicted = get_community_dict(predicted.communities)
            ari, nmi = metrics(ground_truth, predicted)
            modularity_values[idx][i] = modularity(H, get_communities(predicted))
            clust_coeff_values[idx][i] = clustering_coefficient(H)
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
    plt.savefig(f"{AlgoName}_{networkName}_ARI.png")
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
    plt.savefig(f"{AlgoName}_{networkName}_NMI.png")
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
    plt.savefig(f"{AlgoName}_{networkName}_Modularity.png")
    plt.show()
    
    plt.figure(figsize = (12, 8))
    plt.xticks(range(len(k_values)), [f"{100 * k}%" for k in k_values])
    for idx in range(len(sparseFunctions)): 
        plt.plot(clust_coeff_values[idx], label = names[idx], marker = "o")
        for i, txt in enumerate(clust_coeff_values[idx]):
            plt.annotate(f"{txt:.4f}", (i, clust_coeff_values[idx][i]))

    plt.axhline(y = clustering_coefficient(G), color = "black", linestyle = "--", label = "Original Graph")
    plt.xlabel("Percentage Retention of Edges")
    plt.ylabel("Clustering Coefficient")
    plt.title(f"Clustering Coefficient for {AlgoName} vs k for {networkName}")
    plt.legend()
    plt.grid()
    plt.savefig(f"{AlgoName}_{networkName}_Clustering_Coefficients.png")
    plt.show()
    
    return SparseGraphs, ari_values, nmi_values, modularity_values, clust_coeff_values

def createDataFrames(ari_values, nmi_values, modularity_values, clust_coeff_values, k_values, sparseFunctionNames, algorithm, dataset):
    ari_df = pd.DataFrame(ari_values, columns = k_values, index = sparseFunctionNames)
    nmi_df = pd.DataFrame(nmi_values, columns = k_values, index = sparseFunctionNames)
    modularity_df = pd.DataFrame(modularity_values, columns = k_values, index = sparseFunctionNames)
    clust_coeff_df = pd.DataFrame(clust_coeff_values, columns = k_values, index = sparseFunctionNames)
    
    ari_df.to_csv(f"{algorithm}_{dataset}_ARI.csv")
    nmi_df.to_csv(f"{algorithm}_{dataset}_NMI.csv")
    modularity_df.to_csv(f"{algorithm}_{dataset}_Modularity.csv")
    clust_coeff_df.to_csv(f"{algorithm}_{dataset}_Clustering_Coefficients.csv")
    print("DataFrames Created Successfully")

print("Function Description:\n1. plotRandomCommunity(G, community, title = None)\n2. get_community_dict(communities)\n3. get_communities(community_dict)\n4. metrics(ground_truth, predicted)\n5. plot_metrics_sparse(G, ground_truth, sparseFunctions, k_values, AlgoFunction, flag, networkName = None, AlgoName = None)\n6. createDataFrames(ari_values, nmi_values, modularity_values, clust_coeff_values, k_values, sparseFunctionNames, algorithm, dataset)\n\nSampling Methods:\n1. edge_betweenness_sparsification(G, k)\n2. edge_random_sparsification(G, k)\n3. edge_jaccard_sparsification(G, k)\n4. edge_L_Spar_sparsification(G, r)\n5. clustering_coeffs_edge_sampling(G, k)\n6. metropolis_hastings_algorithm(G, r)\n7. effective_resistance_sampling_2(G, r)\n\nCommunity Detection Algorithms:\n1. run_louvain(G)\n2. run_lpa(G)\n3. run_walktrap(G)\n4. run_infomap(G)\n\n")


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