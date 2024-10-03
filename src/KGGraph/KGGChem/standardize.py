from rdkit.Chem.MolStandardize import rdMolStandardize
from joblib import Parallel, delayed
from tqdm import tqdm


class SmileStandardizer:

    @staticmethod
    def standardize(input_file, output_file, n_jobs=8):
        # Load SMILES from file
        with open(input_file) as f:
            smile_list = [line.strip().split()[0] for line in f]

        # Use tqdm outside the parallel processing for clearer progress update
        processed_smile_list = tqdm(smile_list, desc="Standardizing SMILES")

        # Execute standardization in parallel with error handling
        standsmi_list = Parallel(n_jobs=n_jobs)(
            delayed(SmileStandardizer.safe_standardize_smiles)(smile)
            for smile in processed_smile_list
        )

        # Write the results to file using efficient list handling
        with open(output_file, "w") as f:
            f.writelines(s + "\n" for s in standsmi_list if s)

        assert len(smile_list) == len(
            standsmi_list
        ), "Mismatch in number of input and output SMILES"

    @staticmethod
    def safe_standardize_smiles(smiles):
        """Safely standardize SMILES with error handling."""
        try:
            return rdMolStandardize.StandardizeSmiles(smiles)
        except Exception as e:
            print(f"Error processing {smiles}: {e}")
            return None
