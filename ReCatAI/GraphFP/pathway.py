import networkx as nx
from bitarray import bitarray
import hashlib

class PathwayFingerprint:
    def __init__(self, minPath=1, maxPath=7, fpSize=2048, nBitsPerHash=2, useBondOrder=True):
        self.minPath = minPath
        self.maxPath = maxPath
        self.fpSize = fpSize
        self.nBitsPerHash = nBitsPerHash
        self.useBondOrder = useBondOrder

    def generate_fingerprint(self, graph):
        """
        Generates a fingerprint for a NetworkX graph.

        Parameters:
        graph (networkx.Graph): The graph representing the molecule.

        Returns:
        bitarray: The resulting fingerprint as a bit vector.
        """
        fingerprint = bitarray(self.fpSize)
        fingerprint.setall(0)

        # Iterate over all nodes to generate paths
        for node in graph.nodes:
            for length in range(self.minPath, self.maxPath + 1):
                paths = nx.single_source_shortest_path_length(graph, node, cutoff=length)
                for target, path_len in paths.items():
                    if path_len == length:
                        actual_path = nx.shortest_path(graph, node, target)
                        path_hash = self._complex_path_hash(graph, actual_path)
                        for _ in range(self.nBitsPerHash):
                            fingerprint[path_hash % self.fpSize] = 1
                            path_hash = (path_hash + 1) % self.fpSize

        return fingerprint

    def _complex_path_hash(self, graph, path):
        """
        Generates a complex hash for a given path in the graph.

        Parameters:
        graph (networkx.Graph): The graph.
        path (list): The path as a list of node indices.

        Returns:
        int: The hash value.
        """
        path_str = ''
        for i in range(len(path)):
            node = path[i]
            node_attr = graph.nodes[node].get('element', 'C')  # Defaulting to 'C'
            path_str += node_attr

            if i < len(path) - 1:
                next_node = path[i + 1]
                if self.useBondOrder:
                    bond_order = graph.edges[node, next_node].get('order', 1)  # Default to single bond
                    path_str += str(bond_order)

        hash_object = hashlib.sha256(path_str.encode())
        hash_hex = hash_object.hexdigest()
        return int(hash_hex, 16)


if __name__ == "__main__":
    # Example usage
    # Create a simple graph with attributes
    G = nx.Graph()
    G.add_node(0, attribute='C')
    G.add_node(1, attribute='O')
    G.add_node(2, attribute='N')
    G.add_edge(0, 1)
    G.add_edge(1, 2)

    pathway_fp_generator = PathwayFingerprint()
    fingerprint = pathway_fp_generator.generate_fingerprint(G)

    print(fingerprint)
    print(len(fingerprint))