import sys
import os
import argparse
import pandas as pd
import pickle
from tqdm.auto import tqdm
sys.path.append('.')

from utils.reconstruct import *
from utils.misc import *
from utils.scoring_func import *
from utils.evaluation import *
from utils.dataset import get_dataset
from easydict import EasyDict
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 


cfg_dataset = EasyDict({
        'name': 'drug3d',
        'root': './data/geom_drug',
        'path_dict':{
            'sdf': 'sdf',
            'summary': 'mol_summary.csv',
            'processed': 'processed.lmdb',},
        'split': 'split_by_molid.pt',
        'train_smiles': 'train_smiles.pt',
        'train_finger': 'train_finger.pkl',
        'val_smiles': 'val_smiles.pt',
        'val_finger': 'val_finger.pkl',
    })
def load_mols_from_dataset(dataset_type):
    dataset, subsets = get_dataset(cfg_dataset)
    
    # load sdf
    subset = subsets[dataset_type]
    mol_list = []
    n_all = 0
    n_failed = 0
    failed_list = []
    for idx_data, data in tqdm(enumerate(subset), total=len(subset)):
        data.atom_pos = data.pos_all_confs[0]
        try:
            n_all += 1
            mol = reconstruct_from_generated_with_edges(data)
        except MolReconsError:
            failed_list.append(idx_data)
            n_failed += 1
            continue

        mol_list.append(mol)
    print('Load dataset', dataset_type, 'all:', n_all, 'failed:', n_failed)
    mol_dict = {idx: mol for idx, mol in enumerate(mol_list)}
    
    metrics_dir = os.path.join(cfg_dataset.root, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    df_path = os.path.join(metrics_dir, dataset_type+'.csv')
    if os.path.exists(df_path):
        df = pd.read_csv(df_path, index_col=0)
    else:
        df = pd.DataFrame(index=mol_dict.keys())
        
    return mol_dict, df, metrics_dir, df_path


def load_mols_from_generated(exp_name, result_root):
    # prepare data path
    all_exp_paths = os.listdir(result_root)
    sdf_dir = [path for path in all_exp_paths
                      if (path.startswith(exp_name) and path.endswith('_SDF'))]
    assert len(sdf_dir) == 1, f'Found more than one or none sdf directory of sampling with prefix `{exp_name}` and suffix `_SDF` in {result_root}: {sdf_dir}'
    sdf_dir = sdf_dir[0]
    
    sdf_dir = os.path.join(args.result_root, sdf_dir)
    metrics_dir = sdf_dir.replace('_SDF', '')
    df_path = os.path.join(metrics_dir, 'mols.csv')
    mol_names = [mol_name for mol_name in os.listdir(sdf_dir) if (mol_name[-4:] == '.sdf') and ('traj' not in mol_name) ]
    mol_ids = np.sort([int(mol_name[:-4]) for mol_name in mol_names])
        
    # load sdfs
    mol_dict_raw = {mol_id:Chem.MolFromMolFile(os.path.join(sdf_dir, '%d.sdf' % mol_id))
                for mol_id in mol_ids}
    mol_dict = {mol_id:mol for mol_id, mol in mol_dict_raw.items() if mol is not None}
    print('Load success:', len(mol_dict), 'failed:', len(mol_dict_raw)-len(mol_dict))

    # load df
    if os.path.exists(df_path):
        df = pd.read_csv(df_path, index_col=0)
    else:
        df = pd.DataFrame(index=list(mol_dict.keys()))
        df.index.name = 'mol_id'
    return mol_dict, df, metrics_dir, df_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # generated
    parser.add_argument('--from_where', type=str, default='generated',
                        help='be `generated` or `dataset`')
    parser.add_argument('--exp_name', type=str, default='sample_MolDiff_20230602',
                        help='For `generated`, it is the name of the config file of the sampling experiment (e.g., sample_MolDiff)'
                        'For `dataset`, it is one of train/val/test')
    parser.add_argument('--result_root', type=str, default='./outputs',
                        help='The root directory of the generated data and sdf files.')
    args = parser.parse_args()

    from_where = args.from_where
    metrics_list = [
        'drug_chem',  # qed, sa, logp, lipinski
        'count_prop',  # n_atoms, n_bonds, n_rings, n_rotatable, weight, n_hacc, n_hdon
        'global_3d',  # rmsd_max, rmsd_min, rmsd_median
        'frags_counts',  # cnt_eleX, cnt_bondX, cnt_ringX(size)
        
        'local_3d',  # bond length, bond angle, dihedral angle
        
        'validity',  # validity, connectivity
        'similarity', # sim_with_train, uniqueness, diversity

        'ring_type', # cnt_ring_type_{x}, top_n_freq_ring_type
    ]

    if from_where == 'dataset':
        dataset_type = args.exp_name
        mol_dict, df, metrics_dir, df_path = load_mols_from_dataset(dataset_type)
        logger = get_logger('eval_'+dataset_type, metrics_dir)
        metrics_list = ['count_prop', 'frags_counts', 'local_3d', 'ring_type']
    elif from_where == 'generated':
        exp_name = args.exp_name
        mol_dict, df, metrics_dir, df_path = load_mols_from_generated(exp_name, args.result_root)
        logger = get_logger('eval_'+exp_name, metrics_dir)

    for metric_name in metrics_list:
        logger.info(f'Computing {metric_name} metrics...')
        if metric_name in ['drug_chem', 'count_prop', 'global_3d', 'frags_counts', 'groups_counts']:
            parallel =True
            results_list = get_metric(mol_dict.values(), metric_name, parallel=parallel)
            if list(results_list[0].keys())[0] not in df.columns:
                df = pd.concat([df, pd.DataFrame(results_list, index=mol_dict.keys())], axis=1)
            else:
                df.loc[mol_dict.keys(), results_list[0].keys()] = pd.DataFrame(
                    results_list, index=mol_dict.keys())
            df.to_csv(df_path)
        elif metric_name == 'local_3d':
            local3d = Local3D()
            local3d.get_predefined()
            logger.info(f'Computing local 3d - bond lengths metric...')
            lengths = local3d.calc_frequent(mol_dict.values(), type_='length', parallel=False)
            logger.info(f'Computing local 3d - bond angles metric...')
            angles = local3d.calc_frequent(mol_dict.values(), type_='angle', parallel=False)
            logger.info(f'Computing local 3d - dihedral angles metric...')
            dihedral = local3d.calc_frequent(mol_dict.values(), type_='dihedral', parallel=False)
            save_path = df_path.replace('.csv', '_local3d.pkl')
            local3d = {'lengths': lengths, 'angles': angles, 'dihedral': dihedral}
            with open(save_path, 'wb') as f:
                f.write(pickle.dumps(local3d))
        elif metric_name == 'validity':
            validity = calculate_validity(
                output_dir=os.path.dirname(df_path),
                is_edm=('e3_diffusion_for_molecules' in args.result_root),
            )
            with open(df_path.replace('.csv', '_validity.pkl'), 'wb') as f:
                f.write(pickle.dumps(validity))
            logger.info(f'Validity : {validity}')
        elif metric_name == 'similarity':
            sim = SimilarityAnalysis(cfg_dataset)
            uniqueness = sim.get_novelty_and_uniqueness(mol_dict.values())
            diversity = sim.get_diversity(mol_dict.values())
            uniqueness['diversity'] = diversity
            sim_with_val = sim.get_sim_with_val(mol_dict.values())
            uniqueness['sim_with_val'] = sim_with_val
            save_path = df_path.replace('.csv', '_similarity.pkl')
            with open(save_path, 'wb') as f:
                f.write(pickle.dumps(uniqueness))
            logger.info(f'Similarity : {uniqueness}')
        elif metric_name == 'ring_type':
            ring_analyzer = RingAnalyzer()
            # cnt of ring type (common in val set)
            # cnt_ring_type = ring_analyzer.get_count_ring(mol_dict.values())
            # if list(cnt_ring_type.keys())[0] not in df.columns:
            #     df = pd.concat([df, pd.DataFrame(cnt_ring_type, index=mol_dict.keys())], axis=1)
            # else:
            #     df.loc[mol_dict.keys(), cnt_ring_type.keys()] = pd.DataFrame(
            #         cnt_ring_type, index=mol_dict.keys())
            # df.to_csv(df_path)
            # top n freq ring type
            freq_dict = ring_analyzer.get_freq_rings(mol_dict.values())
            with open(df_path.replace('.csv', '_freq_ring_type.pkl'), 'wb') as f:
                f.write(pickle.dumps(freq_dict))
            
            
    logger.info(f'Saving metrics to {df_path}')
    logger.info(f'Done.')
