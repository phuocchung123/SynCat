"""
Structural validation of reaction SMILES strings.

The four-level hierarchical model groups the supplied molecules by reaction
side, so it can only be trained or evaluated on reaction strings whose structure
is unambiguous: exactly one ">>", a non-empty reactant and product side, and no
empty dot-separated component. Legacy architectures accept whatever the parser
happens to do, so this validation is opt-in and its default preserves that
behaviour.

This module deliberately has no RDKit import so that the string-level checks can
be exercised even when RDKit's native libraries cannot be loaded.
"""

from typing import List, Optional


SEPARATOR = ">>"


def validate_reaction_structure(rsmi: str, index: Optional[int] = None) -> None:
    """
    Check the structure of one reaction SMILES string.

    Parameters
    ----------
    rsmi : str
        The reaction SMILES string, as stored after the "~" -> "-" replacement.
    index : int, optional
        Sample index used in the error message when the string comes from a
        list (default is None, in which case no index is reported).

    Raises
    ------
    ValueError
        If the string does not contain exactly one ">>", if either side is
        empty, or if a side contains an empty dot-separated component.
    """
    label = "reaction" if index is None else "reaction %d" % index
    n_separators = rsmi.count(SEPARATOR)
    if n_separators != 1:
        raise ValueError(
            "%s %r must contain exactly one %r, found %d"
            % (label, rsmi, SEPARATOR, n_separators)
        )
    reactants_smi, products_smi = rsmi.split(SEPARATOR)
    for side_name, side in (("reactant", reactants_smi), ("product", products_smi)):
        if side == "":
            raise ValueError(
                "%s %r has an empty %s side" % (label, rsmi, side_name)
            )
        if any(component == "" for component in side.split(".")):
            raise ValueError(
                "%s %r has an empty %s component"
                % (label, rsmi, side_name)
            )


def validate_reaction_strings(rsmi_list: List[str]) -> None:
    """
    Check the structure of every reaction SMILES of a dataset.

    The strings are validated in order and the first failure is raised, so an
    invalid record is reported instead of being silently dropped.

    Parameters
    ----------
    rsmi_list : list of str
        Reaction SMILES strings.

    Raises
    ------
    ValueError
        Describing the first malformed reaction.
    """
    for index, rsmi in enumerate(rsmi_list):
        validate_reaction_structure(str(rsmi), index)
