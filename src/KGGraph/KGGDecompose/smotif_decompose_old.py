from rdkit import Chem
from typing import List, Set, Tuple
from collections import defaultdict


class SMotifDecomposition:
    @staticmethod
    def generate_mark_pattern(mol: Chem.Mol) -> Set[int]:
        """
        Generate marks for atoms that are part of identified functional groups in a molecule.

        Args:
        mol (Chem.Mol): RDKit molecule object to analyze.

        Returns:
        Set[int]: A set of atom indices that are part of identified functional groups.
        """
        PATTERNS = {
            "HETEROATOM": "[!#6]",  # Matches atoms that are not carbon (heteroatoms).
            "DOUBLE_TRIPLE_BOND": "*=,#*",  # Matches double or triple bonds.
            "ACETAL": "[CX4]([O,N,S])[O,N,S]",  # Matches acetal functional groups.
            "PROPYL": "[C;X4]([#6,#9,#17,#35,#53])([#6,#9,#17,#35,#53])([#6,#7,#8,#16,#9,#17,#35,#53])",
        }
        PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in PATTERNS.items()}

        marks = set()

        for name, patt in PATTERNS.items():
            for sub in mol.GetSubstructMatches(patt):
                if name not in ["PROPYL", "DOUBLE_TRIPLE_BOND"]:
                    marks.update(sub)
                elif name == "DOUBLE_TRIPLE_BOND":
                    bond = mol.GetBondBetweenAtoms(sub[0], sub[1])
                    if not bond.IsInRing():
                        marks.update(sub)
                else:
                    atom = mol.GetAtomWithIdx(sub[1])
                    if atom.IsInRing():
                        break
                    else:
                        marks.add(sub[1])
                        break

        return marks

    @staticmethod
    def merge_rings(mol: Chem.Mol) -> List[Set[int]]:
        """
        Merges rings in a molecule that share more than one atom, identifying fused ring systems.

        Args:
        mol (Chem.Mol): RDKit molecule object to analyze.

        Returns:
        List[Set[int]]: A list of sets, each representing a unique or fused ring system.
        """
        rings = [set(x) for x in Chem.GetSymmSSSR(mol)]  # get simple rings
        flag = True  # flag == False: no rings can be merged
        while flag:
            flag = False
            for i in range(len(rings)):
                if len(rings[i]) == 0:
                    continue
                for j in range(i + 1, len(rings)):
                    shared_atoms = rings[i] & rings[j]
                    if len(shared_atoms) > 1:
                        rings[i].update(rings[j])
                        rings[j] = set()
                        flag = True
        return [list(r) for r in rings if len(r) > 0]

    @staticmethod
    def find_carbonyl(mol: Chem.Mol) -> Tuple[List[List[int]], List[Tuple[int]]]:
        """
        Identify carbonyl groups in the molecule and merge adjacent or overlapping CO groups.

        Args:
        mol (Chem.Mol): RDKit molecule object.

        Returns:
        Tuple[List[List[int]], List[Tuple[int]]]:
        List of merged CO groups and list of merged COO groups.
        """
        CO = list(
            mol.GetSubstructMatches(
                Chem.MolFromSmarts("[C;D3](=O)([#0,#6,#7,#8,#17,#35,#53])")
            )
        )
        COO = []

        for idx, sub1 in enumerate(CO):
            for sub2 in CO[idx + 1 :]:
                if sub1[0] == sub2[0]:
                    merge = sub1 + tuple(set(sub2).difference(set(sub1)))
                    CO.insert(idx, merge)
                    CO.remove(sub1)
                    CO.remove(sub2)

        for idx, sub1 in enumerate(CO):
            for sub2 in CO[idx + 1 :]:
                (a1, a2) = (sub1[0], sub2[0])
                bond = mol.GetBondBetweenAtoms(a1, a2)
                if bond is not None:
                    COO.append(tuple(set(sub1 + sub2)))
                    if sub1 in CO:
                        CO.remove(sub1)
                    if sub2 in CO:
                        CO.remove(sub2)

        return CO, COO

    @staticmethod
    def fix_carbonyl_and_cluster_atoms(
        mol: Chem.Mol,
        CO: List[List[int]],
        COO: List[Tuple[int]],
        rings: List[Set[int]],
        marks: Set[int],
    ) -> List[List[int]]:
        """
        Merge carbonyl groups and cluster atoms based on their connectivity.

        Args:
        mol (Chem.Mol): RDKit molecule object.
        CO (List[List[int]]): List of merged CO groups.
        COO (List[Tuple[int]]): List of merged COO groups.
        rings (List[Set[int]]): List of ring atoms.
        marks (Set[int]): Set of marks indicating atoms that have not been clustered yet.

        Returns:
        List[List[int]]: A list of clusters of atoms.
        """
        pre_cluster = []

        for value in CO:
            if mol.GetAtomWithIdx(value[0]).IsInRing():
                if not mol.GetAtomWithIdx(value[1]).IsInRing():
                    for i in rings:
                        if value[0] in i:
                            i.append(value[1])
                for idx in value:
                    if idx in marks:
                        marks.remove(idx)
                continue

            atom_carbonyl = set((value[0], value[1]))
            for i in list(atom_carbonyl):
                if i in marks:
                    marks.remove(i)

            check = set(value) - atom_carbonyl

            for k in check:
                atom = mol.GetAtomWithIdx(k)
                if atom.GetSymbol() == "C":
                    if k in marks:
                        if not atom.IsInRing():
                            nei = [c.GetIdx() for c in atom.GetNeighbors()]
                            pre_cluster.append(list(set(nei + list(value))))
                            marks.remove(k)
                        else:
                            marks.remove(k)
                    else:
                        if list(value) not in pre_cluster:
                            pre_cluster.append(list(value))
                else:
                    if list(value) not in pre_cluster:
                        pre_cluster.append(list(value))
                    if k in marks:
                        marks.remove(k)

        cluster = []
        for i in range(len(pre_cluster)):
            is_subset = False
            for j in range(len(pre_cluster)):
                if i != j and set(pre_cluster[i]).issubset(set(pre_cluster[j])):
                    is_subset = True
                    break
            if not is_subset:
                cluster.append(pre_cluster[i])

        for coo in COO:
            cluster.append(list(coo))
        cluster.extend(rings)

        return cluster, marks

    @staticmethod
    def cluster_atoms_and_identify_functional_groups(
        mol: Chem.Mol, marks: Set[int], cluster: List[Set[int]]
    ) -> List[Set[int]]:
        """
        Cluster atoms and identify functional groups.

        Args:
        mol (Chem.Mol): RDKit molecule object.
        marks (Set[int]): Set of marks indicating atoms that have not been clustered yet.
        cluster (List[Set[int]]): List of clusters of atoms.

        Returns:
        List[Set[int]]: A list of identified functional groups.
        """
        atom2fg = [[] for _ in range(mol.GetNumAtoms())]
        fgs = []

        for atom in marks:
            fgs.append({atom})
            atom2fg[atom] = [len(fgs) - 1]

        for bond in mol.GetBonds():
            if bond.IsInRing():
                continue
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

            found_in_cluster = False
            for c in cluster:
                if (a1 in c) and (a2 in c):
                    found_in_cluster = True
                    break
            if found_in_cluster:
                continue

            if a1 in marks and a2 in marks:
                assert a1 != a2
                assert len(atom2fg[a1]) == 1 and len(atom2fg[a2]) == 1
                fgs[atom2fg[a1][0]].update(fgs[atom2fg[a2][0]])
                fgs[atom2fg[a2][0]] = set()
                atom2fg[a2] = atom2fg[a1]
            elif a1 in marks:
                assert len(atom2fg[a1]) == 1
                fgs[atom2fg[a1][0]].add(a2)
                atom2fg[a2].extend(atom2fg[a1])
            elif a2 in marks:
                assert len(atom2fg[a2]) == 1
                fgs[atom2fg[a2][0]].add(a1)
                atom2fg[a1].extend(atom2fg[a2])
            else:
                fgs.append({a1, a2})
                atom2fg[a1].append(len(fgs) - 1)
                atom2fg[a2].append(len(fgs) - 1)

        return [fg for fg in fgs if fg]

    @staticmethod
    def finalize_function_groups(
        fgs: List[Set[int]], cluster: List[Set[int]], mol: Chem.Mol
    ) -> List[str]:
        """
        Finalize the identified functional groups and convert them to SMILES representations.

        Args:
        fgs (List[Set[int]]): List of identified functional groups.
        cluster (List[Set[int]]): List of atom clusters.
        mol (Chem.Mol): RDKit molecule object.

        Returns:
        List[str]: List of SMILES representations of the final functional groups.
        """
        tmp = [
            list(fg)
            for fg in fgs
            if fg and (len(fg) > 1 or not mol.GetAtomWithIdx(list(fg)[0]).IsInRing())
        ]
        fgs = tmp
        fgs.extend(cluster)

        for i, fg in enumerate(fgs):
            for fg2 in fgs[i + 1 :]:
                inter = set(fg) & set(fg2)
                if len(inter) > 1:
                    fgs[i].extend(list(set(fg2).difference(set(fg))))
                    fgs.remove(fg2)

        if 0 not in fgs[0]:
            for i, cls in enumerate(fgs):
                if 0 in cls:
                    fgs = [fgs[i]] + fgs[:i] + fgs[i + 1 :]
                    break

        atom_cls = [[] for _ in range(mol.GetNumAtoms())]
        for i, fg in enumerate(fgs):
            for atom in fg:
                atom_cls[atom].append(i)

        edges = defaultdict(int)
        for atom, nei_cls in enumerate(atom_cls):
            for i, c1 in enumerate(nei_cls):
                for c2 in nei_cls[i + 1 :]:
                    inter = set(fgs[c1]) & set(fgs[c2])
                    edges[(c1, c2)] = len(inter)
        edges_list = [k for k, v in edges.items()]

        return list(fgs), edges_list

    @staticmethod
    def defragment(mol: Chem.Mol) -> List[str]:
        """
        Perform defragmentation of a molecule into functional groups.

        Args:
        mol (Chem.Mol): RDKit molecule object.

        Returns:
        List[str]: List of SMILES strings representing the identified functional groups.
        """
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1:
            return [[0]], []

        marks = SMotifDecomposition.generate_mark_pattern(mol)
        rings = SMotifDecomposition.merge_rings(mol)

        CO, COO = SMotifDecomposition.find_carbonyl(mol)
        for val in COO:
            marks.difference_update(val)

        cluster, marks = SMotifDecomposition.fix_carbonyl_and_cluster_atoms(
            mol, CO, COO, rings, marks
        )
        fgs = SMotifDecomposition.cluster_atoms_and_identify_functional_groups(
            mol, marks, cluster
        )

        return SMotifDecomposition.finalize_function_groups(fgs, cluster, mol)
