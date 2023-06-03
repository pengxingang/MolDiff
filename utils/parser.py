import os
import numpy as np
from rdkit import Chem
# from rdkit.Chem.rdchem import BondType
# from rdkit.Chem import ChemicalFeatures
# from rdkit import RDConfig

# from utils.mol2frag import mol2frag

# ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable', 'ZnBinder']
# ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
# BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}
# BOND_NAMES = {i: t for i, t in enumerate(BondType.names.keys())}


def parse_conf_list(conf_list, smiles=None):
    # data_list = [parse_drug3d_mol(conf) for conf in conf_list]
    
    element = []
    bond_index = []
    bond_type = []
    pos_all_confs = []
    i_conf_list = []
    num_atoms = 0
    num_bonds = 0  # NOTE: the num of bonds is not symtric
    for i_conf,  conf in enumerate(conf_list):
        data = parse_drug3d_mol(conf, smiles=smiles)
        if data is None:
            continue
        # check element
        if len(element) == 0:
            element = data['element']
            num_atoms = data['num_atoms']
        else:
            if data['num_atoms'] != num_atoms:
                print('Skipping conformer with different number of atoms')
                continue
            if not np.all(element == data['element']):
                print('Skipping conformer with different element order')
                continue
        # check bond
        if len(bond_index) == 0:
            bond_index = data['bond_index']
            bond_type = data['bond_type']
            num_bonds = data['num_bonds']
        else:
            if data['num_bonds'] != num_bonds:
                print('Skipping conformer with different number of bonds')
                continue
            if not np.all(bond_index == data['bond_index']):
                print('Skipping conformer with different bond index')
                continue
            if not np.all(bond_type == data['bond_type']):
                print('Skipping conformer with different bond type')
                continue
        pos_all_confs.append(data['pos'])
        i_conf_list.append(i_conf)

    return {
        'element': np.array(element),
        'bond_index': np.array(bond_index),
        'bond_type': np.array(bond_type),
        'pos_all_confs': np.array(pos_all_confs, dtype=np.float32),
        'num_atoms': num_atoms,
        'num_bonds': num_bonds,
        'i_conf_list': i_conf_list,
        'num_confs': len(i_conf_list),
    }


def parse_drug3d_mol(mol, smiles=None):
    if smiles is not None: # check smiles
        if smiles != Chem.MolToSmiles(mol):
            return None
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    conf = mol.GetConformer()
    ele_list = []
    pos_list = []
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        ele = atom.GetAtomicNum()
        pos_list.append(list(pos))
        ele_list.append(ele)
    
    row, col = [], []
    bond_type = []
    for bond in mol.GetBonds():
        b_type = int(bond.GetBondType())
        assert b_type in [1, 2, 3, 12], 'Bond can only be 1,2,3,12 bond'
        b_type = b_type if b_type != 12 else 4
        b_index = [
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx()
        ]
        bond_type += 2*[b_type]
        row += [b_index[0], b_index[1]]
        col += [b_index[1], b_index[0]]
    
    bond_type = np.array(bond_type, dtype=np.long)
    bond_index = np.array([row, col],dtype=np.long)

    perm = (bond_index[0] * num_atoms + bond_index[1]).argsort()
    bond_index = bond_index[:, perm]
    bond_type = bond_type[perm]

    data = {
        'element': np.array(ele_list, dtype=np.int64),
        'pos': np.array(pos_list, dtype=np.float32),
        'bond_index': np.array(bond_index, dtype=np.int64),
        'bond_type': np.array(bond_type, dtype=np.int64),
        'num_atoms': num_atoms,
        'num_bonds': num_bonds,
    }
    return data