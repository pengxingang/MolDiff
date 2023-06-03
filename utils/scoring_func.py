import os

import numpy as np
import itertools
import torch
import pickle
from tqdm import tqdm
from copy import deepcopy
from multiprocessing import Pool
from rdkit import Chem, DataStructs
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
from rdkit.Chem.QED import qed
from utils.sascorer import compute_sa_score
from utils.dataset import get_dataset
from rdkit.Chem.FilterCatalog import *
# from utils.visualize import show_mols, show
# from utils.reconstruct import reconstruct_from_generated_with_edges
params_pain = FilterCatalogParams()
params_pain.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
catalog_pain = FilterCatalog(params_pain)


# class ScoringTool:
#     def __init__(self) -> None:
#         pass

def is_pains(mol):
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    entry = catalog_pain.GetFirstMatch(mol)
    if entry is None:
        return False
    else:
        return True


def obey_lipinski(mol):
    # mol = deepcopy(mol)
    # Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    rule_4 = (logp:=Crippen.MolLogP(mol)>=-2) & (logp<=5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])
    

def get_basic(mol):
    n_atoms = len(mol.GetAtoms())
    n_bonds = len(mol.GetBonds())
    n_rings = len(Chem.GetSymmSSSR(mol))
    weight = Descriptors.ExactMolWt(mol)
    return n_atoms, n_bonds, n_rings, weight

def get_rdkit_rmsd(mol, n_conf=100, random_seed=42):
    """
    Calculate the alignment of generated mol and rdkit predicted mol
    Return the rmsd (max, min, median) of the `n_conf` rdkit conformers
    """
    # mol = deepcopy(mol)
    # Chem.SanitizeMol(mol)
    mol3d = Chem.AddHs(deepcopy(mol))
    rmsd_list = []
    # predict 3d
    confIds = AllChem.EmbedMultipleConfs(mol3d, n_conf, randomSeed=random_seed)
    for confId in confIds:
        AllChem.UFFOptimizeMolecule(mol3d, confId=confId)
        # AllChem.MMFFOptimizeMolecule(mol3d, confId=confId)
        rmsd = Chem.rdMolAlign.GetBestRMS(mol, mol3d, refId=confId)
        rmsd_list.append(rmsd)
    # mol3d = Chem.RemoveHs(mol3d)
    rmsd_list = np.array(rmsd_list)
    return [np.max(rmsd_list), np.min(rmsd_list), np.median(rmsd_list)]


def get_logp(mol):
    return Crippen.MolLogP(mol)


def get_chem(mol):
    qed_score = qed(mol)
    sa_score = compute_sa_score(mol)
    logp_score = Crippen.MolLogP(mol)
    hacc_score = Lipinski.NumHAcceptors(mol)
    hdon_score = Lipinski.NumHDonors(mol)
    return qed_score, sa_score, logp_score, hacc_score, hdon_score


class SimilarityWithMe:
    def __init__(self, mol) -> None:
        self.mol = deepcopy(mol)
        self.mol = Chem.MolFromSmiles(Chem.MolToSmiles(self.mol))
        self.fp= Chem.RDKFingerprint(self.mol)
    
    def get_sim(self, mol):
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))  # automately sanitize 
        fg_query = Chem.RDKFingerprint(mol)
        sims = DataStructs.TanimotoSimilarity(self.fp, fg_query)
        return sims

class SimilarityAnalysis:
    def __init__(self, cfg_dataset) -> None:
        self.cfg_dataset = cfg_dataset
        self.smiles_path = os.path.join(self.cfg_dataset.root, self.cfg_dataset.train_smiles)
        self.finger_path = os.path.join(self.cfg_dataset.root, self.cfg_dataset.train_finger)
        self.smiles_path_val = os.path.join(self.cfg_dataset.root, self.cfg_dataset.val_smiles)
        self.finger_path_val = os.path.join(self.cfg_dataset.root, self.cfg_dataset.val_finger)
        # self.train_smiles = None
        # self.train_fingers = None
        self._get_train_mols()
        self._get_val_mols()
        
        
    def _get_train_mols(self):
        file_not_exists = ((not os.path.exists(self.smiles_path))  or
                            (not os.path.exists(self.finger_path)))
        if file_not_exists:
            _, subsets = get_dataset(config = self.cfg_dataset)
            train_set = subsets['train']
            self.train_smiles = []
            self.train_finger = []
            for data in tqdm(train_set, desc='Prepare train set fingerprint'):  # calculate fingerprint and smiles of train data
                smiles = data.smiles
                mol = Chem.MolFromSmiles(smiles)
                fg = Chem.RDKFingerprint(mol)
                self.train_finger.append(fg)
                self.train_smiles.append(smiles)
            self.train_smiles = np.array(self.train_smiles)
            # self.train_finger = np.array(self.train_finger)
            torch.save(self.train_smiles, self.smiles_path)
            with open(self.finger_path, 'wb') as f:
                pickle.dump(self.train_finger, f)
        else:
            self.train_smiles = torch.load(self.smiles_path)
            self.train_smiles = np.array(self.train_smiles)
            with open(self.finger_path, 'rb') as f:
                self.train_finger = pickle.load(f)

    def _get_val_mols(self):
        file_not_exists = ((not os.path.exists(self.smiles_path_val))  or
                            (not os.path.exists(self.finger_path_val)))
        if file_not_exists:
            _, subsets = get_dataset(config = self.cfg_dataset)
            val_set = subsets['val']
            self.val_smiles = []
            self.val_finger = []
            for data in tqdm(val_set, desc='Prepare val set fingerprint'):  # calculate fingerprint and smiles of val data
                smiles = data.smiles
                mol = Chem.MolFromSmiles(smiles)
                fg = Chem.RDKFingerprint(mol)
                self.val_finger.append(fg)
                self.val_smiles.append(smiles)
            self.val_smiles = np.array(self.val_smiles)
            # self.val_finger = np.array(self.val_finger)
            torch.save(self.val_smiles, self.smiles_path_val)
            with open(self.finger_path_val, 'wb') as f:
                pickle.dump(self.val_finger, f)
        else:
            self.val_smiles = torch.load(self.smiles_path_val)
            self.val_smiles = np.array(self.val_smiles)
            with open(self.finger_path_val, 'rb') as f:
                self.val_finger = pickle.load(f)


    def get_novelty_and_uniqueness(self, mols):
        n_in_train = 0
        smiles_list = []
        for mol in tqdm(mols, desc='Calculate novelty and uniqueness'):
            smiles = Chem.MolToSmiles(mol)
            smiles_list.append(smiles)
            if smiles in self.train_smiles:
                n_in_train += 1
        novelty = 1 - n_in_train / len(mols)
        unique = len(np.unique(smiles_list)) / len(mols)
        return {'novelty': novelty, 'uniqueness': unique}
    
    def get_sim_with_train(self, mols, parallel=False):
        mol_finger = [Chem.RDKFingerprint(mol) for mol in mols]
        finger_pair = list(itertools.product(mol_finger, self.train_finger))
        if not parallel:
            similarity_list = []
            for fg1, fg2 in tqdm(finger_pair, desc='Calculate similarity with train'):
                similarity_list.append(get_similarity((fg1, fg2)))
        else:
            with Pool(102) as pool:
                similarity_list = list(tqdm(pool.imap(get_similarity, finger_pair), 
                                            total=len(mol_finger)*len(self.train_finger)))
                
        # calculate the max similarity of each mol with train data
        similarity_max = np.reshape(similarity_list, (len(mols), -1)).max(axis=1)
        return np.mean(similarity_max)
    
    def get_sim_with_val(self, mols, parallel=False):
        mol_finger = [Chem.RDKFingerprint(mol) for mol in mols]
        finger_pair = list(itertools.product(mol_finger, self.val_finger))
        if not parallel:
            similarity_list = []
            for fg1, fg2 in tqdm(finger_pair, desc='Calculate similarity with val'):
                similarity_list.append(get_similarity((fg1, fg2)))
        else:
            with Pool(102) as pool:
                similarity_list = list(tqdm(pool.imap(get_similarity, finger_pair), 
                                            total=len(mol_finger)*len(self.val_finger)))
                
        # calculate the max similarity of each mol with train data
        similarity_max = np.reshape(similarity_list, (len(mols), -1)).max(axis=1)
        return np.mean(similarity_max)
    
    def get_diversity(self, mols, parallel=False):
        fgs = [Chem.RDKFingerprint(mol) for mol in mols]
        all_fg_pairs = list(itertools.combinations(fgs, 2))
        if not parallel:
            similarity_list = []
            for fg1, fg2 in tqdm(all_fg_pairs, desc='Calculate diversity'):
                similarity_list.append(TanimotoSimilarity(fg1, fg2))
        else:
            with Pool(102) as pool:
                similarity_list = pool.imap_unordered(TanimotoSimilarity, all_fg_pairs)
        return 1 - np.mean(similarity_list)

def get_similarity(fg_pair):
    return TanimotoSimilarity(fg_pair[0], fg_pair[1])