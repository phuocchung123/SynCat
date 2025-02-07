import pandas as pd
from rdkit import Chem

def count_atom(data_path,rxn_col):
    df=pd.read_csv(data_path)
    rxn=df[rxn_col].values
    dct={}
    for i in rxn:
        rct_and_pdt= i.split('>>')
        compounds=[]
        for j in rct_and_pdt:
            compound=j.split('.')
            compounds.extend(compound)
        for m in compounds:
            mol= Chem.MolFromSmiles(m)
            for atom in mol.GetAtoms():
                num_atom=atom.GetAtomicNum()
                if num_atom not in dct.keys():
                    dct[num_atom]=0
                else:
                    dct[num_atom]+=1
    return dct
            
            
        