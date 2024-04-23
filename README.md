# SparseCommunityDetection

$$ \text{Modularity} $$

$$ Q(\mathcal{C}) = \frac{1}{2m} \sum*{C \in \mathcal{C}} \sum*{u \in C, v \in C} \left( A\_{u, v} - \frac{d_u d_v}{2m} \right) $$ <br>

$$ \text{Betweenness Centrality} $$

$$ \text{Betweenness}(v) = \sum\_{s \neq v,t \neq v} \frac{|\sigma\*{s, t}(v)|}{|\sigma\_{s, t}|}$$ <br>

$$ \text{Jaccard Similarity} $$

$$ \texttt{An edge (i, j) is likely to lie within a cluster if} $$
$$ \text{the vertices i and j have adjancency lists with high overlap} $$

$$ \text{J}(i, j) = \frac{|\text{Adj}(i) \cap \text{Adj}(j)|}{|\text{Adj}(i) \cup \text{Adj}(j)|} $$

![](https://github.com/guntas-13/CS328-SparseCommunityDetection/blob/main/Media/KarateGraph.png)
![](https://github.com/guntas-13/CS328-SparseCommunityDetection/blob/main/Media/Karate.gif)

## RESULTS (As of Now)

![](https://github.com/guntas-13/CS328-SparseCommunityDetection/blob/main/Media/DBLP_ARI.png)
![](https://github.com/guntas-13/CS328-SparseCommunityDetection/blob/main/Media/DBLP_NMI.png)
![](https://github.com/guntas-13/CS328-SparseCommunityDetection/blob/main/Media/DBLP_Mod.png)

## God Funtion

```python
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

```

## Some Additional Sources:

[Zachary's Karate Club Dataset Analysis](https://www.youtube.com/watch?v=uE2U4QHYmNE) <br>
[Modularity - Justin Ruth](https://www.youtube.com/watch?v=lRX5CvK3JpY) <br>
[Community Detection - IITM NPTEL](https://www.youtube.com/watch?v=Jck7WTLQxM8) <br>
[Although not needed - Introduction to GNN](https://distill.pub/2021/gnn-intro/)<br>

## Handy Libraries

[Networkx](https://networkx.org/documentation/stable/reference/index.html)<br>
[igraph](https://python.igraph.org/en/stable/analysis.html#clustering)<br>
[Liitle Ball of Fur](https://little-ball-of-fur.readthedocs.io/en/latest/index.html)
