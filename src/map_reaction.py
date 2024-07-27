import pandas as pd
import numpy as np
from rxnmapper import RXNMapper
from rdkit import Chem
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def map_reaction(args):
    mapper = RXNMapper()
    data = pd.read_csv(args.Data_folder+args.data_csv,index_col=0)
    print(data.shape)
    rsmi=data[args.reaction_column].values
    for i in tqdm(rsmi):
        try:
            mapped_rsmi=mapper.get_attention_guided_atom_maps([i])[0]['mapped_rxn']
            precusor, product = i.split('>>')
            precusor1, product1=mapped_rsmi.split('>>')

            #Choose mapped precusor
            ele_react=precusor.split('.')
            ele_react1=precusor1.split('.')
            precusor_main=[j for j in ele_react if j not in ele_react1]
            precusor_str='.'.join(precusor_main)
            reagent_1=[j for j in ele_react if j in ele_react1]

            #Choose mapped product
            ele_product=product.split('.')
            ele_product1=product1.split('.')
            product_main=[j for j in ele_product if j not in ele_product1]
            product_str='.'.join(product_main)
            reagent_2=[j for j in ele_product if j in ele_product1]

            reagent=reagent_1 + reagent_2
            reagent='.'.join(reagent)

            new_react=precusor_str+'>>'+product_str

        except:
            print('error')
            new_react=np.nan
            reagent=np.nan
        data.loc[data[args.reaction_column]==i,args.mapped_reaction_column]=new_react
        data.loc[data[args.reaction_column]==i,args.reagent_column]=reagent
    print(data.shape)
    data=data.dropna(subset=[args.mapped_reaction_column,args.reagent_column])
    print(data.shape)
    data.to_csv(args.Data_folder+args.mapped_data_csv)





    