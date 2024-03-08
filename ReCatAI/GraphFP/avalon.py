from collections import defaultdict
import numpy as np
import networkx as nx

class AvalonFP:
    """
    This class generates Avalon-style fingerprints for molecules represented as graphs.
    It captures information about atom types, bonds, and aromatic rings.
    """

    def __init__(self, fingerprint_size=1024):
        """
        Initializes the fingerprint generator with a desired fingerprint size.

        Args:
          fingerprint_size (int, optional): The size of the fingerprint bit vector. Defaults to 1024.
        """
        self.fingerprint_size = fingerprint_size
        self.aromatic_atom_types = {"C", "N", "O"}  # Extend this set for more aromatic atom types

    def generate_fingerprint(self, graph):
        """
        Generates a fingerprint for a given graph representation of a molecule.

        Args:
          graph (nx.Graph): The graph representation of the molecule.

        Returns:
          np.array: An array of integers representing the fingerprint bit vector.
        """
        fingerprint = np.zeros(self.fingerprint_size, dtype=int)

        # Pre-compute aromatic ring information
        #aromatic_rings = self.find_aromatic_rings(graph)

        # Iterate through nodes in the graph
        for node in graph.nodes():
            atom_type = graph.nodes[node]["attribute"]
            self.set_atom_type_bits(fingerprint, atom_type, node)

            # Loop through connected neighbors
            for neighbor in graph.neighbors(node):
                bond_type = "single"  # Assuming all bonds are single for now
                neighbor_type = graph.nodes[neighbor]["attribute"]
                self.set_bond_bits(fingerprint, atom_type, bond_type, neighbor_type, node, neighbor)

        return fingerprint

    def set_atom_type_bits(self, fingerprint, atom_type, node_index):
        """
        Sets bits in the fingerprint corresponding to the provided atom type.

        This method is a simplistic version. Consider using a hashing function for a more distributed representation.
        """
        base_index = hash(atom_type) % self.fingerprint_size
        fingerprint[base_index] = 1

    def set_bond_bits(self, fingerprint, atom1_type, bond_type, atom2_type, atom1_index, atom2_index):
        """
        Sets bits in the fingerprint based on the bond type and neighboring atom.

        Accounts for aromaticity if applicable. This is a simplistic implementation that should be expanded.
        """
        sorted_atoms = sorted([atom1_type, atom2_type])
        bond_repr = f"{sorted_atoms[0]}-{bond_type}-{sorted_atoms[1]}"
        base_index = hash(bond_repr) % self.fingerprint_size
        fingerprint[base_index] = 1

        if bond_type == "aromatic":
            fingerprint[(base_index + 1) % self.fingerprint_size] = 1

if __name__ == "__main__":
    # Example usage
    G = nx.Graph()
    G.add_node(0, attribute='C')
    G.add_node(1, attribute='O')
    G.add_node(2, attribute='N')
    G.add_edge(0, 1)
    G.add_edge(1, 2)

    avalon_fp_generator = AvalonFP(2048)
    fingerprint = avalon_fp_generator.generate_fingerprint(G)

    print(fingerprint)
    print(len(fingerprint))
