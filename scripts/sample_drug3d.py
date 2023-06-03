import os
import sys
import shutil
import argparse
sys.path.append('.')

import torch
import numpy as np
import torch.utils.tensorboard
from easydict import EasyDict
from rdkit import Chem

from models.model import MolDiff
from models.bond_predictor import BondPredictor
from utils.sample import seperate_outputs
from utils.transforms import *
from utils.misc import *
from utils.reconstruct import *

def print_pool_status(pool, logger):
    logger.info('[Pool] Finished %d | Failed %d' % (
        len(pool.finished), len(pool.failed)
    ))


def data_exists(data, prevs):
    for other in prevs:
        if len(data.logp_history) == len(other.logp_history):
            if (data.ligand_context_element == other.ligand_context_element).all().item() and \
                (data.ligand_context_feature_full == other.ligand_context_feature_full).all().item() and \
                torch.allclose(data.ligand_context_pos, other.ligand_context_pos):
                return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/sample/sample_MolDiff.yml')
    parser.add_argument('--outdir', type=str, default='./outputs')
    parser.add_argument('--device', type=str, default='cuda:7')
    parser.add_argument('--batch_size', type=int, default=0)
    args = parser.parse_args()

    # # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.sample.seed + np.sum([ord(s) for s in args.outdir]))
    # load ckpt and train config
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    train_config = ckpt['config']

    # # Logging
    log_root = args.outdir.replace('outputs', 'outputs_vscode') if sys.argv[0].startswith('/data') else args.outdir
    log_dir = get_new_log_dir(log_root, prefix=config_name)
    logger = get_logger('sample', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

    # # Transform
    logger.info('Loading data placeholder...')
    featurizer = FeaturizeMol(train_config.chem.atomic_numbers, train_config.chem.mol_bond_types,
                            use_mask_node=train_config.transform.use_mask_node,
                            use_mask_edge=train_config.transform.use_mask_edge,)
    max_size = None
    add_edge = getattr(config.sample, 'add_edge', None)
    
    # # Model
    logger.info('Loading diffusion model...')
    if train_config.model.name == 'diffusion':
        model = MolDiff(
                    config=train_config.model,
                    num_node_types=featurizer.num_node_types,
                    num_edge_types=featurizer.num_edge_types
                ).to(args.device)
    else:
        raise NotImplementedError
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    # # Bond predictor adn guidance
    if 'bond_predictor' in config:
        logger.info('Building bond predictor...')
        ckpt_bond = torch.load(config.bond_predictor, map_location=args.device)
        bond_predictor = BondPredictor(ckpt_bond['config']['model'],
                featurizer.num_node_types,
                featurizer.num_edge_types-1 # note: bond_predictor not use edge mask
        ).to(args.device)
        bond_predictor.load_state_dict(ckpt_bond['model'])
        bond_predictor.eval()
    else:
        bond_predictor = None
    if 'guidance' in config.sample:
        guidance = config.sample.guidance  # tuple: (guidance_type[entropy/uncertainty], guidance_scale)
    else:
        guidance = None


    pool = EasyDict({
        'failed': [],
        'finished': [],
    })
    # # generating molecules
    while len(pool.finished) < config.sample.num_mols:
        if len(pool.failed) > 3 * (config.sample.num_mols):
            logger.info('Too many failed molecules. Stop sampling.')
            break
        
        # prepare batch
        batch_size = args.batch_size if args.batch_size > 0 else config.sample.batch_size
        n_graphs = min(batch_size, (config.sample.num_mols - len(pool.finished))*2)
        batch_holder = make_data_placeholder(n_graphs=n_graphs, device=args.device, max_size=max_size)
        batch_node, halfedge_index, batch_halfedge = batch_holder['batch_node'], batch_holder['halfedge_index'], batch_holder['batch_halfedge']
        
        # inference
        outputs = model.sample(
            n_graphs=n_graphs,
            batch_node=batch_node,
            halfedge_index=halfedge_index,
            batch_halfedge=batch_halfedge,
            bond_predictor=bond_predictor,
            guidance=guidance,
        )
        outputs = {key:[v.cpu().numpy() for v in value] for key, value in outputs.items()}
        
        # decode outputs to molecules
        batch_node, halfedge_index, batch_halfedge = batch_node.cpu().numpy(), halfedge_index.cpu().numpy(), batch_halfedge.cpu().numpy()
        try:
            output_list = seperate_outputs(outputs, n_graphs, batch_node, halfedge_index, batch_halfedge)
        except:
            continue
        gen_list = []
        for i_mol, output_mol in enumerate(output_list):
            mol_info = featurizer.decode_output(
                pred_node=output_mol['pred'][0],
                pred_pos=output_mol['pred'][1],
                pred_halfedge=output_mol['pred'][2],
                halfedge_index=output_mol['halfedge_index'],
            )  # note: traj is not used
            try:
                rdmol = reconstruct_from_generated_with_edges(mol_info, add_edge=add_edge)
            except MolReconsError:
                pool.failed.append(mol_info)
                logger.warning('Reconstruction error encountered.')
                continue
            mol_info['rdmol'] = rdmol
            smiles = Chem.MolToSmiles(rdmol)
            mol_info['smiles'] = smiles
            if '.' in smiles:
                logger.warning('Incomplete molecule: %s' % smiles)
                pool.failed.append(mol_info)
            else:   # Pass checks!
                logger.info('Success: %s' % smiles)
                p_save_traj = np.random.rand()  # save traj
                if p_save_traj <  config.sample.save_traj_prob:
                    traj_info = [featurizer.decode_output(
                        pred_node=output_mol['traj'][0][t],
                        pred_pos=output_mol['traj'][1][t],
                        pred_halfedge=output_mol['traj'][2][t],
                        halfedge_index=output_mol['halfedge_index'],
                    ) for t in range(len(output_mol['traj'][0]))]
                    mol_traj = []
                    for t in range(len(traj_info)):
                        try:
                            mol_traj.append(reconstruct_from_generated_with_edges(traj_info[t], False, add_edge=add_edge))
                        except MolReconsError:
                            mol_traj.append(Chem.MolFromSmiles('O'))
                    mol_info['traj'] = mol_traj
                gen_list.append(mol_info)
                # pool.finished.append(mol_info)

        # # Save sdf mols
        sdf_dir = log_dir + '_SDF'
        os.makedirs(sdf_dir, exist_ok=True)
        with open(os.path.join(log_dir, 'SMILES.txt'), 'a') as smiles_f:
            for i, data_finished in enumerate(gen_list):
                smiles_f.write(data_finished['smiles'] + '\n')
                rdmol = data_finished['rdmol']
                Chem.MolToMolFile(rdmol, os.path.join(sdf_dir, '%d.sdf' % (i+len(pool.finished))))

                if 'traj' in data_finished:
                    with Chem.SDWriter(os.path.join(sdf_dir, 'traj_%d.sdf' % (i+len(pool.finished)))) as w:
                        for m in data_finished['traj']:
                            try:
                                w.write(m)
                            except:
                                w.write(Chem.MolFromSmiles('O'))
        pool.finished.extend(gen_list)
        print_pool_status(pool, logger)

    torch.save(pool, os.path.join(log_dir, 'samples_all.pt'))
    