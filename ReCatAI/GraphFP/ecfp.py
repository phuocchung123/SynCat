import networkx as nx
from collections import defaultdict, namedtuple


class XCFPFingerprinter:
    """
    Class for generating Extended Connectivity Fingerprints (ECFPs) from graphs.
    Allows customization of the hashing algorithm via `hashf` attribute.

    Attributes
    ----------
    hashf : function
        Function used for hashing (default is `hash`).
    identifier : namedtuple
        Namedtuple with fields `value` and `edges`, holding the hash value and
        corresponding edges.
    radius : int
        Number of iterations to perform (radius, not diameter).
    invariant : callable
        Function to calculate node invariants, returning a hashable object.
    bond_order : callable
        Function to determine bond order between two nodes, returning a hashable object.

    Parameters
    ----------
    radius : int
        Iteration count, where `radius=3` corresponds to ECFP6.
    invariant : callable, optional
        Function for node invariants, defaults to `default_invariant`.
    bond_order : callable, optional
        Function for bond order, defaults to `default_bond_order`.
    """
    hashf = hash
    identifier = namedtuple('Identifier', ['value', 'edges'])

    def __init__(self, radius, invariant=None, bond_order=None):
        self.invariant = invariant if invariant else self.default_invariant
        self.bond_order = bond_order if bond_order else self.default_bond_order
        self.radius = radius

    def fingerprint(self, graph):
        """
        Computes the graph's fingerprint.

        Parameters
        ----------
        graph : nx.Graph
            Graph to compute the fingerprint for.

        Returns
        -------
        dict
            Fingerprint where keys are feature hashes and values are their counts.
        """
        self._initialize(graph)
        for iternum in range(self.radius):
            self._update(graph, iternum)
            self._store()
        return dict(self._fingerprint)

    def _initialize(self, graph):
        """
        Initializes fingerprint computation.
        """
        self._fingerprint = defaultdict(int)
        self._known_features = {}
        self._per_node = defaultdict(list)
        self._identifiers = {}
        self._covered_edges = defaultdict(set)

        for node_key in graph:
            hash_val = self.hashf(self.invariant(graph, node_key))
            self._identifiers[node_key] = self.identifier(hash_val, set())
        self._store()

    def _update(self, graph, iternum):
        """
        Performs an iteration of the fingerprint update.
        """
        new_identifiers = {}
        new_covered_edges = {}

        for node_key in graph:
            ids = [iternum, self._identifiers[node_key].value]
            edges = self._update_node(graph, node_key, ids)
            new_identifiers[node_key] = self.identifier(self.hashf(tuple(ids)), edges)
            new_covered_edges[node_key] = edges

        self._identifiers = new_identifiers
        self._covered_edges = new_covered_edges

    def _update_node(self, graph, node_key, ids):
        """
        Updates identifiers and edges for a single node.
        """
        order = self.sort_neighbours(graph, node_key)
        edges = self._covered_edges[node_key].copy()

        for neighbour in order:
            ids.extend([self.bond_order(graph, node_key, neighbour), self._identifiers[neighbour].value])
            edges.add(frozenset((node_key, neighbour)))
            edges.update(self._covered_edges[neighbour])

        return edges

    def _store(self):
        """
        Stores the identifiers from the current round in the fingerprint.
        """
        for node_key, identifier in sorted(self._identifiers.items(), key=lambda i: i[1].value):
            feature = frozenset(identifier.edges)
            if not feature or feature not in self._known_features:
                self._fingerprint[identifier.value] += 1
                self._known_features[feature] = identifier.value
            self._per_node[node_key].append(self._known_features.get(feature, identifier.value))

    def sort_neighbours(self, graph, root_node):
        """
        Sorts a node's neighbours for the xCFP algorithm.

        Returns
        -------
        list
            Sorted neighbours.
        """
        return sorted(graph[root_node], key=lambda node_key: (self.bond_order(graph, root_node, node_key), self._identifiers[node_key].value))

    @staticmethod
    def default_invariant(graph, node_key):
        """
        Default method to calculate a node's invariant.
        """
        cycles = nx.cycle_basis(graph)
        return (len(graph[node_key]), graph.nodes[node_key]['element'], graph.nodes[node_key].get('charge', 0), int(any(node_key in cycle for cycle in cycles)))

    @staticmethod
    def default_bond_order(graph, node_key1, node_key2):
        """
        Default method to determine bond order.
        """
        order = graph.edges[node_key1, node_key2].get('order', 1)
        return 4 if order == 1.5 else order


# Demonstration of usage:
if __name__ == '__main__':
    mol = nx.Graph()
    mol.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (3, 5)])
    for k in mol:
        mol.nodes[k]['element'] = 'C'
        mol.nodes[k]['charge'] = 0

    mol.nodes[5]['element'] = 'O'
    mol.nodes[4]['element'] = 'N'
    ecfp = XCFPFingerprinter(1)  # Change the radius as needed
    fingerprint = ecfp.fingerprint(mol)
    print(fingerprint)
    print(ecfp._per_node)
