# SparseCommunityDetection

## $$ \text{Modularity} $$

### $$ Q(\mathcal{C}) = \frac{1}{2m} \sum*{C \in \mathcal{C}} \sum*{u \in C, v \in C} \left( A\_{u, v} - \frac{d_u d_v}{2m} \right) $$

## $$ \text{Betweenness Centrality} $$

### $$ \text{Betweenness}(v) = \sum*{s \neq v,t \neq v} \frac{|\sigma*{s, t}(v)|}{|\sigma\_{s, t}|}$$

## $$ \text{Jaccard Similarity} $$

$$ \texttt{An edge (i, j) is likely to lie within a cluster if} $$
$$ \text{the vertices i and j have adjancency lists with high overlap} $$

### $$ \text{J}(i, j) = \frac{|\text{Adj}(i) \cap \text{Adj}(j)|}{|\text{Adj}(i) \cup \text{Adj}(j)|} $$

![](https://github.com/guntas-13/CS328-SparseCommunityDetection/blob/main/Media/KarateGraph.png)
![](https://github.com/guntas-13/CS328-SparseCommunityDetection/blob/main/Media/Karate.gif)

## Some Additional Sources:

[Zachary's Karate Club Dataset Analysis](https://www.youtube.com/watch?v=uE2U4QHYmNE) <br>
[Modularity - Justin Ruth](https://www.youtube.com/watch?v=lRX5CvK3JpY) <br>
[Community Detection - IITM NPTEL](https://www.youtube.com/watch?v=Jck7WTLQxM8) <br>
[Although not needed - Introduction to GNN](https://distill.pub/2021/gnn-intro/)<br>

## Handy Libraries

[Networkx](https://networkx.org/documentation/stable/reference/index.html)<br>
[igraph](https://python.igraph.org/en/stable/analysis.html#clustering)<br>
[Liitle Ball of Fur](https://little-ball-of-fur.readthedocs.io/en/latest/index.html)
