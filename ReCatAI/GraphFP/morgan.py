import networkx as nx
from collections import defaultdict
from bitarray import bitarray
import hashlib

class NetworkXMorganFingerprint:
    def __init__(self, radius=2, nBits=2048, invariants=None, fromAtoms=None, useChirality=False, useBondTypes=False, useFeatures=False, bitInfo=None):
        self.radius = radius
        self.nBits = nBits
        self.invariants = invariants
        self.fromAtoms = fromAtoms
        self.useChirality = useChirality
        self.useBondTypes = useBondTypes
        self.useFeatures = useFeatures
        self.bitInfo = bitInfo if bitInfo is not None else {}

    def generate_fingerprint(self, graph):
        """
        Generates a Morgan-like fingerprint for a NetworkX graph.

        Parameters:
        graph (networkx.Graph): The graph representing the molecule.

        Returns:
        bitarray: The resulting fingerprint as a bit vector.
        """
        fingerprint = bitarray(self.nBits)
        fingerprint.setall(0)

        # Initialize atom invariants
        if self.invariants:
            if len(self.invariants) != graph.number_of_nodes():
                raise ValueError("Length of invariants list must match the number of nodes in the graph.")
            node_invariants = {node: [self.invariants[node]] for node in graph.nodes}
        else:
            # Use a simple invariant (degree) if no custom invariants are provided
            node_invariants = {node: [graph.degree(node)] for node in graph.nodes}

        # Define starting atoms for fingerprint generation
        if self.fromAtoms:
            starting_nodes = self.fromAtoms
        else:
            starting_nodes = graph.nodes

        # Iteratively update the invariants
        for _ in range(self.radius):
            new_invariants = defaultdict(list)
            for node in graph.nodes:
                neighborhood_invariant = [tuple(node_invariants[neighbor]) for neighbor in graph[node]]
                neighborhood_invariant.sort()  # Ensure determinism
                new_invariants[node] = node_invariants[node] + [hash(tuple(neighborhood_invariant))]
            node_invariants = new_invariants

        # Generate the fingerprint
        for node in starting_nodes:
            if node not in graph:
                continue
            invariant = node_invariants[node]
            hash_val = self._hash_invariant(invariant)
            for _ in range(2):  # Example: set 2 bits per hash
                bit_position = hash_val % self.nBits
                fingerprint[bit_position] = 1
                self.bitInfo[bit_position] = self.bitInfo.get(bit_position, []) + [node]
                hash_val = (hash_val + 1) % self.nBits

        return fingerprint

    def _hash_invariant(self, invariant):
        """
        Hashes the invariant using SHA-256 and returns an integer hash value.

        Parameters:
        invariant (list): The invariant to be hashed.

        Returns:
        int: The hash value.
        """
        hash_object = hashlib.sha256(str(invariant).encode())
        hash_hex = hash_object.hexdigest()
        return int(hash_hex, 16)

if __name__ == "__main__":
    # Example usage
    G = nx.Graph()
    G.add_node(0, attribute='C')
    G.add_node(1, attribute='O')
    G.add_node(2, attribute='N')
    G.add_edge(0, 1)
    G.add_edge(1, 2)

    fingerprint_generator = NetworkXMorganFingerprint(radius=2, nBits=2048)
    fingerprint = fingerprint_generator.generate_fingerprint(G)

    print(fingerprint)
    print(len(fingerprint))
