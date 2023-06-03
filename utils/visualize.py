import os
import py3Dmol
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole


def show(x):
    """
    Show one molecule with 3D in notebook
    """
    print(Chem.MolToSmiles(x))
    IPythonConsole.drawMol3D(x)
    return x

def show_mols(mols, molsPerRow=8, subImgSize=(250,200)):
    """
    Show many molecules with 2D in mesh
    """
    mols2d = [Chem.MolFromSmiles(Chem.MolToSmiles(x)) for x in mols]
    return Chem.Draw.MolsToGridImage(mols2d, molsPerRow=molsPerRow, subImgSize=subImgSize)