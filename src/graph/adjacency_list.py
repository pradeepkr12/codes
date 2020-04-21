# adjacency list

# implementation 1
# dictionary approach

class Graph:
    def __init__(self, graph_type = 'directed'):
        self.graph_type = graph_type
        self.graph = {}

    def add_edge_(self, src, dest, cost):
        if self.graph.get(src) is None:
            node_ = {}
            node_[dest] = cost
        else:
            node_ = self.graph[src]
            if node_.get(dest) is None:
                node_[dest] = cost
            else:
                raise ValueError

        self.graph[src] = node_

    def add_edge(self, src, dest, cost = 0):
        self.add_edge_(src, dest, cost)
        # for undirected graph
        if self.graph_type == 'undirected':
            self.add_edge_(dest, src, cost)

    def print_graph_(self, edges, vertex):
        for v_ in edges.keys():
            print(vertex, '->', v_, ' = ', edges[v_])

    def print_graph(self):
        for vertex in self.graph.keys():
            self.print_graph_(self.graph[vertex], vertex)
        print()

def run():
    graph_type = 'directed'
    graph = Graph(graph_type)
    graph.add_edge(0, 1)
    graph.add_edge(0, 4)
    graph.add_edge(1, 2)
    graph.add_edge(1, 3)
    graph.add_edge(1, 4)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)

    graph.print_graph()