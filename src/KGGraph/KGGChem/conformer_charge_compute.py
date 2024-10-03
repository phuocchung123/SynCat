import os
import shutil
import subprocess
from pandas import DataFrame
from rdkit import Chem
from rdkit.Chem import AllChem


class ConformerChargeGeneration:
    """
    A class for optimizing molecules using xTB and converting SMILES strings to 3D structure files.

    Attributes:
        input_dir (str): Directory to store input files for optimization.
        output_dir (str): Directory where optimized molecule files will be saved.
    """

    def __init__(self, input_dir: str, output_dir: str, charge_dir: str) -> None:
        """
        Initializes the ConformerGeneration class with directories for input and output.

        Parameters:
            input_dir (str): Directory to store input files for optimization.
            output_dir (str): Directory where optimized molecule files will be saved.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.charge_dir = charge_dir
        if not os.path.isdir(self.input_dir):
            os.mkdir(self.input_dir)
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        if not os.path.isdir(self.charge_dir):
            os.mkdir(charge_dir)

    @staticmethod
    def smiles2file_df(
        data: DataFrame,
        smiles_col: str,
        ID_col: str,
        lig_dir: str,
        output_format: str = "pdb",
    ) -> None:
        """
        Converts SMILES strings from a DataFrame to 3D structure files in PDB or SDF format.

        Args:
            data (pandas.DataFrame): DataFrame containing SMILES strings and IDs.
            smiles_col (str): Name of the column containing the SMILES strings.
            ID_col (str): Name of the column containing the molecule IDs.
            lig_dir (str): Directory to save the generated 3D structure files.
            output_format (str): The output file format ('pdb' or 'sdf'). Defaults to 'pdb'.
        """
        if output_format not in ["pdb", "sdf"]:
            raise ValueError("output_format must be 'pdb' or 'sdf'")

        for num in range(data.shape[0]):
            id = data.loc[num, ID_col]
            mol = Chem.MolFromSmiles(data.loc[num, smiles_col])
            if mol is not None:
                mol_h = Chem.AddHs(mol)
                params = AllChem.ETKDGv3()
                params.randomSeed = 42
                embed_success = AllChem.EmbedMolecule(mol_h, params)
                if embed_success != -1:
                    AllChem.MMFFOptimizeMolecule(mol_h, maxIters=10000)
                    file_path = os.path.join(lig_dir, f"{id}.{output_format}")
                    try:
                        if output_format == "pdb":
                            Chem.MolToPDBFile(mol_h, file_path)
                        else:  # output_format == "sdf"
                            Chem.MolToMolFile(mol_h, file_path)
                    except Exception as e:
                        print(f"Error writing {id} to {output_format.upper()}: {e}")
            else:
                print(f"Invalid SMILES for ID {id}")

    @staticmethod
    def smiles2file(
        smiles: list or str, lig_dir: str, ID: list = None, output_format: str = "pdb"
    ) -> None:
        """
        Convert SMILES to file in the specified format and save to the provided ligand directory.

        Args:
            smiles (list or str): A list of SMILES strings or a single SMILES string.
            lig_dir (str): The directory to save the output files.
            ID (list, optional): A list of IDs for the molecules. If None, default IDs will be assigned.
            output_format (str, optional): The format for the output files, either "pdb" or "sdf". Defaults to "pdb".

        Returns:
            None
        """
        if output_format not in ["pdb", "sdf"]:
            raise ValueError("output_format must be 'pdb' or 'sdf'")

        if isinstance(smiles, str):
            smiles = [smiles]

        if ID is None:
            ID = list(range(len(smiles)))
        else:
            if len(ID) != len(smiles):
                raise ValueError("ID and smiles must be the same length")

        for id, s in zip(ID, smiles):
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                mol_h = Chem.AddHs(mol)
                params = AllChem.ETKDGv3()
                params.randomSeed = 42
                embed_success = AllChem.EmbedMolecule(mol_h, params)
                if embed_success != -1:
                    AllChem.MMFFOptimizeMolecule(mol_h, maxIters=10000)
                    file_path = os.path.join(lig_dir, f"{id}.{output_format}")
                    try:
                        if output_format == "pdb":
                            Chem.MolToPDBFile(mol_h, file_path)
                        else:  # output_format == "sdf"
                            Chem.MolToMolFile(mol_h, file_path)
                    except Exception as e:
                        print(f"Error writing {id} to {output_format.upper()}: {e}")
            else:
                print(f"Invalid SMILES for ID {id}")

    @staticmethod
    def geometry_charge_compute(
        input_file: str,
        file_name: str,
        level: str,
        output_dir: str,
        charge_dir: str,
        num_cpus: int = 1,
    ) -> None:
        """
        Optimizes the geometry of a molecule using xtb.

        Args:
            input_file (str): Path to the input file containing the molecule to be optimized.
            file_name (str): Name of the output file for the optimized molecule.
            level (str): Optimization level to be used with xtb.
            output_dir (str): Directory where the optimized molecule file will be saved.
            num_cpus (int): Number of CPUs to use for the optimization. Defaults to 1.
        """

        geometric_command = (
            f"xtb {input_file} --opt {level} --chrg 0 --parallel {num_cpus}"
        )
        geometric_process = subprocess.Popen(
            geometric_command, shell=True, stdout=subprocess.PIPE
        )
        geometric_output, geometric_err = geometric_process.communicate()

        if geometric_process.returncode != 0:
            print(f"Error in xtb: {geometric_err}")
            return None

        # Rename and move the optimized geometric file to the specified directory
        if os.path.exists("xtbopt.pdb"):
            os.rename("xtbopt.pdb", file_name)
        if os.path.isfile(f"{output_dir}/{file_name}"):
            os.remove(f"{output_dir}/{file_name}")
        shutil.move(file_name, output_dir)

        # Rename and move the partial charge file to the specified directory
        if os.path.exists("charges"):
            name_without_extension, extension = os.path.splitext(file_name)
            charge_rename = f"{name_without_extension}.txt"
            os.rename("charges", charge_rename)
        if os.path.isfile(f"{charge_dir}/{charge_rename}"):
            os.remove(f"{charge_dir}/{charge_rename}")
        shutil.move(charge_rename, charge_dir)

        # Remove any temporary files generated by xtb
        temp_files = [
            "wbo",
            "xtbrestart",
            "hessian",
            "xtbtopo.mol",
            "xtbopt.log",
            ".xtboptok",
            "g98.out",
            "vibspectrum",
            "xtbopt.log",
        ]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def run_geometry_charge_df(
        self,
        data: DataFrame,
        smiles_col: str,
        ID_col: str,
        xtb: bool = True,
        level: str = "normal",
        num_cpus: int = 1,
        format: str = "pdb",
    ) -> None:
        """
        Converts SMILES to 3D structures and optimizes their geometry.

        This method first converts SMILES strings from a DataFrame to 3D structure files and
        then optimizes the geometry of each structure using xtb.

        Args:
            data (pandas.DataFrame): DataFrame containing SMILES strings and IDs.
            smiles_col (str): Name of the column containing the SMILES strings.
            ID_col (str): Name of the column containing the molecule IDs.
            xtb (bool): Whether to use xtb for geometry optimization. Defaults to True.
            level (str): Optimization level to be used with xtb.
            num_cpus (int): Number of CPUs to use for the optimization. Defaults to 1.
            format (str): The format of the output files ('pdb' or 'sdf'). Defaults to 'pdb'.
        """
        self.smiles2file_df(
            data, smiles_col, ID_col, self.input_dir, output_format=format
        )
        if xtb:
            # Iterate over all files in the specified directory
            for file_name in os.listdir(self.input_dir):
                if file_name.endswith(
                    "." + format
                ):  # Ensures we are only processing specified format files
                    input_file = os.path.join(self.input_dir, file_name)
                    self.geometry_charge_compute(
                        input_file,
                        file_name,
                        level,
                        self.output_dir,
                        self.charge_dir,
                        num_cpus=num_cpus,
                    )
                else:
                    print(f"Skipping non-{format} file: {file_name}")

    def run_geometry_charge(
        self,
        smiles: list or str,
        ID: list = None,
        xtb: bool = True,
        level: str = "normal",
        num_cpus: int = 1,
        format: str = "pdb",
    ) -> None:
        """
        Runs geometry optimization for the given SMILES strings using xTB or a specified level of theory.

        Args:
            smiles (list or str): The SMILES string or list of SMILES strings to be optimized.
            ID (list, optional): List of identifiers for the input SMILES strings. Defaults to None.
            xtb (bool, optional): Flag to indicate whether to use xTB for optimization. Defaults to True.
            level (str, optional): The level of theory for the geometry optimization. Defaults to 'normal'.
            num_cpus (int, optional): The number of CPUs to use for the optimization. Defaults to 1.
            format (str, optional): The output format for the optimized structures. Defaults to "pdb".

        Returns:
            None
        """
        self.smiles2file(
            smiles=smiles, ID=ID, lig_dir=self.input_dir, output_format=format
        )
        if xtb:
            # Iterate over all files in the specified directory
            for file_name in os.listdir(self.input_dir):
                if file_name.endswith(
                    "." + format
                ):  # Ensures we are only processing specified format files
                    input_file = os.path.join(self.input_dir, file_name)
                    self.geometry_charge_compute(
                        input_file,
                        file_name,
                        level,
                        self.output_dir,
                        self.charge_dir,
                        num_cpus=num_cpus,
                    )
                else:
                    print(f"Skipping non-{format} file: {file_name}")


if __name__ == "__main__":
    import time

    test_data = {
        "Sorafenib": "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1",
        "Pazopanib": "Cc1ccc(Nc2nccc(N(C)c3ccc4c(C)n(C)nc4c3)n2)cc1S(N)(=O)=O",
        "Sunitinib": "CCN(CC)CCNC(=O)c1c(C)[nH]c(C=C2C(=O)Nc3ccc(F)cc32)c1C",
        "Cabozantinib": "COc1cc2nccc(Oc3ccc(NC(=O)C4(C(=O)Nc5ccc(F)cc5)CC4)cc3)c2cc1OC",
        "Axitinib": "CNC(=O)c1ccccc1Sc1ccc2c(C=Cc3ccccn3)n[nH]c2c1",
        "Lenvatinib": "COc1cc2nccc(Oc3ccc(NC(=O)NC4CC4)c(Cl)c3)c2cc1C(N)=O",
        "Regorafenib": "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)c(F)c2)ccn1",
        "Vandetanib": "COc1cc2c(Nc3ccc(Br)cc3F)ncnc2cc1OCC1CCN(C)CC1",
        "Tivozanib": "COc1cc2nccc(Oc3ccc(NC(=O)Nc4cc(C)on4)c(Cl)c3)c2cc1OC",
    }
    input_dir = "./data/random_conform"
    output_dir = "./data/optim_conform"
    charge_dir = "./data/partial_charge"
    start_time = time.time()
    geo_optim = ConformerChargeGeneration(input_dir, output_dir, charge_dir)
    geo_optim.run_geometry_charge(
        smiles=test_data.values(),
        ID=test_data.keys(),
        level="normal",
        num_cpus=-1,
        format="pdb",
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for run_geometry_charge: {elapsed_time} seconds")
