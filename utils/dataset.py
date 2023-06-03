import pickle
import os

import torch
import numpy as np
import pandas as pd
import lmdb
from rdkit import Chem
from tqdm import tqdm

from torch.utils.data import Subset, Dataset
from .parser import parse_conf_list
from .data import Drug3DData, torchify_dict


def get_dataset(config, *args, **kwargs):
    name = config.name
    root = config.root
    if name == 'drug3d':
        dataset = Drug3DDataset(root, config.path_dict, *args, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)
    
    if 'split' in config:
        split_by_molid = torch.load(os.path.join(root, config.split))
        split = {
            k: [dataset.molid2idx[mol_id] for mol_id in mol_id_list if mol_id in dataset.molid2idx]
            for k, mol_id_list in split_by_molid.items()
        }
        subsets = {k:Subset(dataset, indices=v) for k, v in split.items()}
        print('Num of samples:', *{(k, len(v)) for k,v in split.items()})
        return dataset, subsets
    else:
        return dataset


class Drug3DDataset(Dataset):

    def __init__(self, root, path_dict, transform=None):
        super().__init__()
        self.root = root
        self.sdf_path = os.path.join(root, path_dict['sdf'])
        self.summary_path = os.path.join(root, path_dict['summary'])
        
        self.processed_path = os.path.join(root, path_dict['processed'])
        self.molid2idx_path = self.processed_path[:self.processed_path.find('.lmdb')]+'_molid2idx.pt'
        # self.filter = filter

        self.transform = transform
        self.db = None
        self.keys = None

        if (not os.path.exists(self.processed_path)) or (not os.path.exists(self.molid2idx_path)):
            self._process()
            self._precompute_molid2idx()
        self.molid2idx = torch.load(self.molid2idx_path)

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
        
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False, # Writable
        )
        
        # read summary
        df_summary = pd.read_csv(self.summary_path, index_col=0)
        
        # filter 
        df_use = df_summary[df_summary['pass_size'] & df_summary['pass_element'] &
                            (~df_summary['broken']) & (~df_summary['error_mol'])]
        
        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for _, line in tqdm(df_use.iterrows(), total=len(df_use), desc='Preprocessing data'):
                # mol info
                mol_id = line['mol_id']
                smiles = line['smiles']
                
                try:
                    # load all confs of the mol
                    suppl = Chem.SDMolSupplier(os.path.join(self.sdf_path, 'mol_%d.sdf' % mol_id))
                    confs_list = []
                    for i_conf in range(len(suppl)):
                        mol = Chem.MolFromMolBlock(suppl.GetItemText(i_conf).replace(
                            "RDKit          3D", "RDKit          2D"
                        ))  # removeHs=True is default
                        mol = Chem.RemoveAllHs(mol)
                        confs_list.append(mol)
                    
                    # build data
                    ligand_dict = parse_conf_list(confs_list, smiles=smiles)
                    if ligand_dict['num_confs'] == 0:
                        raise ValueError('No conformers found')
                    ligand_dict = torchify_dict(ligand_dict)
                    data = Drug3DData.from_drug3d_dicts(ligand_dict)

                    data.smiles = smiles
                    data.mol_id = mol_id
                    
                    txn.put(
                        key = str(mol_id).encode(),
                        value = pickle.dumps(data)
                    )
                except:
                    num_skipped += 1
                    print('Skipping (%d) Num: %s, %s' % (num_skipped, mol_id, smiles))
                    continue
        db.close()
        print('Processed %d molecules' % (len(df_use) - num_skipped), 'Skipped %d molecules' % num_skipped)


    def _precompute_molid2idx(self):
        molid2idx = {}
        for i in tqdm(range(self.__len__()), 'Indexing'):
            try:
                data = self.__getitem__(i)
            except AssertionError as e:
                print(i, e)
                continue
            mol_id = data.mol_id
            molid2idx[mol_id] = i
        torch.save(molid2idx, self.molid2idx_path)

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        # data.id = idx
        if self.transform is not None:
            data = self.transform(data)
        return data

