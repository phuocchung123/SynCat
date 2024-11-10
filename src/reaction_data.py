import numpy as np
from rdkit import Chem
from preprocess_utils import add_mol, add_dummy, dict_list_to_numpy
from tqdm import tqdm


def mol_dict():
    return {
        "n_node": [],
        "n_edge": [],
        "dummy":[],
        "node_attr": [],
        "edge_attr": [],
        "src": [],
        "dst": [],
    }


def get_graph_data(
    args,
    rsmi_list,
    y_list,
    filename,
    rmol_max_cnt,
    pmol_max_cnt,
    reagent=None,
    rgmol_max_cnt=None,
):
    rmol_max_cnt = rmol_max_cnt
    pmol_max_cnt = pmol_max_cnt

    rmol_dict = [mol_dict() for _ in range(rmol_max_cnt)]
    pmol_dict = [mol_dict() for _ in range(pmol_max_cnt)]

    reaction_dict = {"y": [], "rsmi": []}

    print("--- generating graph data for %s" % filename)
    if args.reagent_option:
        rgmol_max_cnt = rgmol_max_cnt
        rgmol_dict = [mol_dict() for _ in range(rgmol_max_cnt)]
        print(
            "--- n_reactions: %d, reactant_max_cnt: %d, product_max_cnt: %d, rgmol_max_cnt: %d"
            % (len(rsmi_list), rmol_max_cnt, pmol_max_cnt, rgmol_max_cnt)
        )
    else:
        print(
            "--- n_reactions: %d, reactant_max_cnt: %d, product_max_cnt: %d"
            % (len(rsmi_list), rmol_max_cnt, pmol_max_cnt)
        )

    for i in tqdm(range(len(rsmi_list))):
        rsmi = rsmi_list[i].replace("~", "-")
        y = y_list[i]

        [reactants_smi, products_smi] = rsmi.split(">>")

        # processing reactants
        reactants_smi_list = reactants_smi.split(".")
        for _ in range(rmol_max_cnt - len(reactants_smi_list)):
            reactants_smi_list.append("")
        for j, smi in enumerate(reactants_smi_list):
            if smi == "":
                rmol_dict[j] = add_dummy(rmol_dict[j])
                rmol_dict[j]["dummy"].append(False)

            else:
                rmol_dict[j]['dummy'].append(True)
                rmol = Chem.MolFromSmiles(smi)
                rs = Chem.FindPotentialStereo(rmol)
                for element in rs:
                    if (
                        str(element.type) == "Atom_Tetrahedral"
                        and str(element.specified) == "Specified"
                    ):
                        rmol.GetAtomWithIdx(element.centeredOn).SetProp(
                            "Chirality", str(element.descriptor)
                        )
                    elif (
                        str(element.type) == "Bond_Double"
                        and str(element.specified) == "Specified"
                    ):
                        rmol.GetBondWithIdx(element.centeredOn).SetProp(
                            "Stereochemistry", str(element.descriptor)
                        )

                rmol = Chem.RemoveHs(rmol)
                rmol_dict[j] = add_mol(rmol_dict[j], rmol)

        # processing products
        products_smi_list = products_smi.split(".")
        for _ in range(pmol_max_cnt - len(products_smi_list)):
            products_smi_list.append("")
        for j, smi in enumerate(products_smi_list):
            if smi == "":
                pmol_dict[j] = add_dummy(pmol_dict[j])
                pmol_dict[j]['dummy'].append(False)
            else:
                pmol_dict[j]['dummy'].append(True)
                pmol = Chem.MolFromSmiles(smi)
                ps = Chem.FindPotentialStereo(pmol)
                for element in ps:
                    if (
                        str(element.type) == "Atom_Tetrahedral"
                        and str(element.specified) == "Specified"
                    ):
                        pmol.GetAtomWithIdx(element.centeredOn).SetProp(
                            "Chirality", str(element.descriptor)
                        )
                    elif (
                        str(element.type) == "Bond_Double"
                        and str(element.specified) == "Specified"
                    ):
                        pmol.GetBondWithIdx(element.centeredOn).SetProp(
                            "Stereochemistry", str(element.descriptor)
                        )

                pmol = Chem.RemoveHs(pmol)
                pmol_dict[j] = add_mol(pmol_dict[j], pmol)

        if args.reagent_option:
            # processing reagents
            reagents_smi = reagent[i]
            reagents_smi_list = reagents_smi.split(".")
            for _ in range(rgmol_max_cnt - len(reagents_smi_list)):
                reagents_smi_list.append("")
            for j, smi in enumerate(reagents_smi_list):
                if smi == "":
                    rgmol_dict[j] = add_dummy(rgmol_dict[j])
                else:
                    rgmol = Chem.MolFromSmiles(smi)
                    rgs = Chem.FindPotentialStereo(rgmol)
                    for element in rgs:
                        if (
                            str(element.type) == "Atom_Tetrahedral"
                            and str(element.specified) == "Specified"
                        ):
                            rgmol.GetAtomWithIdx(element.centeredOn).SetProp(
                                "Chirality", str(element.descriptor)
                            )
                        elif (
                            str(element.type) == "Bond_Double"
                            and str(element.specified) == "Specified"
                        ):
                            rgmol.GetBondWithIdx(element.centeredOn).SetProp(
                                "Stereochemistry", str(element.descriptor)
                            )

                    rgmol = Chem.RemoveHs(rgmol)
                    rgmol_dict[j] = add_mol(rgmol_dict[j], rgmol)

        # yield and reaction SMILES
        reaction_dict["y"].append(y)
        reaction_dict["rsmi"].append(rsmi)

        # monitoring
        if (i + 1) % 10000 == 0:
            print("--- %d/%d processed" % (i + 1, len(rsmi_list)))

    # datatype to numpy
    for j in range(rmol_max_cnt):
        rmol_dict[j] = dict_list_to_numpy(rmol_dict[j])
    for j in range(pmol_max_cnt):
        pmol_dict[j] = dict_list_to_numpy(pmol_dict[j])
    reaction_dict["y"] = np.array(reaction_dict["y"])
    if args.reagent_option:
        for j in range(rgmol_max_cnt):
            rgmol_dict[j] = dict_list_to_numpy(rgmol_dict[j])
        # save file
        np.savez_compressed(
            filename,
            rmol=rmol_dict,
            pmol=pmol_dict,
            rgmol=rgmol_dict,
            reaction=reaction_dict,
        )
    else:
        # save file
        np.savez_compressed(
            filename, rmol=rmol_dict, pmol=pmol_dict, reaction=reaction_dict
        )
