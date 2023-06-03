import copy
import torch
import numpy as np
from torch_geometric.data import Data, Batch
# from torch_geometric.loader import DataLoader


class Drug3DData(Data):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_drug3d_dicts(ligand_dict=None, **kwargs):
        instance = Drug3DData(**kwargs)

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance[key] = item
            instance['orig_keys'] = list(ligand_dict.keys())

        # instance['nbh_list'] = {i.item():[j.item() for k, j in enumerate(instance.ligand_bond_index[1]) if instance.ligand_bond_index[0, k].item() == i] for i in instance.ligand_bond_index[0]}
        return instance

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'bond_index':
            return len(self['node_type'])
        elif key == 'edge_index':
            return len(self['node_type'])
        elif key == 'halfedge_index':
            return len(self['node_type'])
        else:
            return super().__inc__(key, value, *args, **kwargs)


def torchify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output

    