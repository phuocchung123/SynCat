import networkx as nx
import numpy as np
import struct
from hashlib import sha1
from typing import List, Set

class MHFPEncoder:
    """A class for encoding molecular graphs as MHFP fingerprints."""

    prime = (1 << 61) - 1
    max_hash = (1 << 32) - 1

    def __init__(self, n_permutations: int = 2048, seed: int = 42):
        """
        Initialize the MHFPEncoder instance.

        Parameters:
        n_permutations (int): The size of the binary vector and the number of permutations used for hashing.
        seed (int): The seed value for numpy's random number generator.
        """
        self.n_permutations = n_permutations
        self.seed = seed

        np.random.seed(seed)
        self.permutations_a = np.random.randint(1, self.max_hash, size=n_permutations, dtype=np.uint64)
        self.permutations_b = np.random.randint(0, self.max_hash, size=n_permutations, dtype=np.uint64)

    def shingling_from_graph(self, graph: nx.Graph, radius: int = 3, min_radius: int = 1) -> List[bytes]:
        """
        Generate shingles from a NetworkX graph.

        Parameters:
        graph (nx.Graph): The NetworkX graph.
        radius (int): The maximum radius to consider for subgraphs.
        min_radius (int): The minimum radius to consider for subgraphs.

        Returns:
        List[bytes]: A list of encoded shingles.
        """
        shingling: Set[str] = set()

        for node in graph.nodes:
            for r in range(min_radius, radius + 1):
                subgraph = nx.ego_graph(graph, node, radius=r)
                shingle = ''.join(sorted(str(subgraph.nodes[n]['attribute']) for n in subgraph.nodes))
                shingling.add(shingle)

        return [s.encode('utf-8') for s in shingling]

    def from_molecular_shingling(self, tokens: List[bytes]) -> np.ndarray:
        """
        Generate hash values from a list of shingles.

        Parameters:
        tokens (List[bytes]): A list of encoded shingles.

        Returns:
        np.ndarray: An array containing the hash values.
        """
        hash_values = np.full(self.n_permutations, self.max_hash, dtype=np.uint32)
        for token in tokens:
            token_hash = struct.unpack("<I", sha1(token).digest()[:4])[0]
            combined_hashes = (self.permutations_a * token_hash + self.permutations_b) % self.prime
            hash_values = np.minimum(hash_values, combined_hashes % self.max_hash)
        return hash_values.astype(np.uint32)

    def encode_graph(self, graph: nx.Graph, radius: int = 3, min_radius: int = 1) -> np.ndarray:
        """
        Encode a NetworkX graph to a MHFP fingerprint.

        Parameters:
        graph (nx.Graph): The NetworkX graph to encode.
        radius (int): The maximum radius for shingling.
        min_radius (int): The minimum radius for shingling.

        Returns:
        np.ndarray: The resulting MHFP fingerprint.
        """
        shingling = self.shingling_from_graph(graph, radius=radius, min_radius=min_radius)
        hash_values = self.from_molecular_shingling(shingling)
        return self.fold(hash_values)

    def fold(self, hash_values: np.ndarray) -> np.ndarray:
        """
        Folds the hash values to a binary vector of a given length.

        Parameters:
        hash_values (np.ndarray): An array containing the hash values.

        Returns:
        np.ndarray: The folded fingerprint.
        """
        folded = np.zeros(self.n_permutations, dtype=np.uint8)
        for value in hash_values.flatten():  # Ensure hash_values is a 1-D array
            index = int(value) % self.n_permutations
            folded[index] = 1
        return folded

# Example usage
if __name__ == "__main__":
    # Create a simple graph with attributes
    G = nx.Graph()
    G.add_node(0, attribute='C')
    G.add_node(1, attribute='O')
    G.add_node(2, attribute='N')
    G.add_edge(0, 1)
    G.add_edge(1, 2)

    # Initialize encoder
    encoder = MHFPEncoder(n_permutations=1024)  

    # Generate fingerprint
    fingerprint = encoder.encode_graph(G)
    print("MHFP Fingerprint:", fingerprint)
    print("Length of Fingerprint:", len(fingerprint))
