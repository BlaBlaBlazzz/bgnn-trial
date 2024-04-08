# import dgl
# import networkx as nx

# def networkx_to_torch(networkx_graph):
#     import dgl
#     # graph = dgl.DGLGraph()
#     graph = dgl.from_networkx(networkx_graph)
#     print(graph)
#     graph = dgl.remove_self_loop(graph)
#     graph = dgl.add_self_loop(graph)
    
#     return graph


# networkx_graph = nx.read_graphml('datasets/optdigit/graph.graphml')
# networkx_graph = nx.relabel_nodes(networkx_graph, {str(i): i for i in range(len(networkx_graph))})

# for node in networkx_graph.nodes():
#     print(type(node))

# graph = dgl.from_networkx(networkx_graph)

a = [[1]*3]