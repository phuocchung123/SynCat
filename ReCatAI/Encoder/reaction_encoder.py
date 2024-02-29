from joblib import Parallel, delayed
import numpy as np
from rdkit.DataStructs import cDataStructs
from ReCatAI.Encoder.smiles_featurizer import SmilesFeaturizer
class ReactionEncoder:
    def __init__(self, reaction_col: str = 'smiles', symbols: str = '>>', fingerprint_type: str = 'maccs', abs: bool = True, 
                 n_jobs: int = -1, verbose: int = 1):
        """
        Initialize the ReactionEncoder class with default parameters for reaction fingerprint generation.

        Parameters:
        - reaction_col (str): Column name in the dictionary where the reaction SMILES is stored.
        - symbols (str): The symbol used to separate reactants and products in the reaction SMILES string.
        - fingerprint_type (str): The type of fingerprint to generate (e.g., 'maccs', 'ecfp').
        - abs (bool): Whether to take the absolute value of the reaction fingerprint difference.
        - n_jobs (int): The number of parallel jobs to run. -1 means using all processors.
        """
        self.reaction_col = reaction_col
        self.symbols = symbols
        self.fingerprint_type = fingerprint_type
        self.abs = abs
        self.n_jobs = n_jobs
        self.verbose=verbose

    @staticmethod
    def convert_arr2vec(arr):
        """
        Converts a numpy array to a RDKit ExplicitBitVect.

        Parameters:
        - arr (numpy.ndarray): The input array.

        Returns:
        - RDKit ExplicitBitVect: The converted bit vector.
        """
        arr_tostring = "".join(arr.astype(str))
        EBitVect2 = cDataStructs.CreateFromBitString(arr_tostring)
        return EBitVect2

    @staticmethod
    def reaction_fps(reaction_dict, reaction_col, symbols, fingerprint_type, abs, **kwargs):
        """
        Generates a reaction fingerprint for a single reaction dictionary as a static method.

        Parameters:
        - reaction_dict (dict): A dictionary containing reaction information.
        - reaction_col (str): Column name in the dictionary where the reaction SMILES is stored.
        - symbols (str): The symbol used to separate reactants and products in the reaction SMILES string.
        - fingerprint_type (str): The type of fingerprint to generate (e.g., 'maccs', 'ecfp').
        - abs (bool): Whether to take the absolute value of the reaction fingerprint difference.

        Returns:
        - dict: The input dictionary updated with the reaction fingerprint ('rfp').
        """
        react, prod = reaction_dict[reaction_col].split(symbols)
        react_fps = None
        for s in react.split('.'):
            if react_fps is None:
                react_fps = SmilesFeaturizer.featurize_smiles(s, fingerprint_type, **kwargs)
            else:
                react_fps += SmilesFeaturizer.featurize_smiles(s, fingerprint_type, **kwargs)

        prod_fps = None
        for s in prod.split('.'):
            if prod_fps is None:
                prod_fps = SmilesFeaturizer.featurize_smiles(s, fingerprint_type, **kwargs)
            else:
                prod_fps += SmilesFeaturizer.featurize_smiles(s, fingerprint_type, **kwargs)
        
        # Placeholder for actual fingerprint operation
        reaction_smiles = np.subtract(prod_fps, react_fps)  
        if abs:
            reaction_smiles = np.abs(reaction_smiles)
        
        reaction_smiles_obj = ReactionEncoder.convert_arr2vec(reaction_smiles)
        reaction_dict['rfp'] = reaction_smiles_obj
        return reaction_dict

    def process_reaction_fps(self, reaction_dicts, **kwargs):
        """
        Processes a list of reaction dictionaries to generate reaction fingerprints in parallel.

        Parameters:
        - reaction_dicts (List[dict]): List of dictionaries containing reaction information.

        Returns:
        - List[dict]: The list of dictionaries, each updated with the reaction fingerprint ('rfp').
        """
        processed_reactions = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(ReactionEncoder.reaction_fps)(
                d, 
                self.reaction_col, 
                self.symbols, 
                self.fingerprint_type, 
                self.abs,
                **kwargs
            ) for d in reaction_dicts
        )
        return processed_reactions