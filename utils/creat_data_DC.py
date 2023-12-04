from rdkit import Chem
import numpy as np
import torch
from rdkit.Chem import AllChem



atom_type_max = 100
atom_f_dim = 133
atom_features_define = {
    'atom_symbol': list(range(atom_type_max)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'charity_type': [0, 1, 2, 3],
    'hydrogen': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ], }

smile_changed = {}


def get_atom_features_dim():
    return atom_f_dim


def onek_encoding_unk(key, length):
    encoding = [0] * (len(length) + 1)
    index = length.index(key) if key in length else -1
    encoding[index] = 1

    return encoding


def get_atom_feature(atom):
    feature = onek_encoding_unk(atom.GetAtomicNum() - 1, atom_features_define['atom_symbol']) + \
              onek_encoding_unk(atom.GetTotalDegree(), atom_features_define['degree']) + \
              onek_encoding_unk(atom.GetFormalCharge(), atom_features_define['formal_charge']) + \
              onek_encoding_unk(int(atom.GetChiralTag()), atom_features_define['charity_type']) + \
              onek_encoding_unk(int(atom.GetTotalNumHs()), atom_features_define['hydrogen']) + \
              onek_encoding_unk(int(atom.GetHybridization()), atom_features_define['hybridization']) + \
              [1 if atom.GetIsAromatic() else 0] + \
              [atom.GetMass() * 0.01]
    return feature


class GraphOne:
    def __init__(self, mol):
        self.mol = mol
        self.atom_feature = []


        self.atom_num = mol.GetNumAtoms()

        for i, atom in enumerate(mol.GetAtoms()):
            self.atom_feature.append(get_atom_feature(atom))
        self.atom_feature = [self.atom_feature[i] for i in range(self.atom_num)]

def create_graph(smile):
    if smile in smile_changed:
        graph = smile_changed[smile]
    else:
        graph = GraphOne(smile)
    return graph

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])
    #atom.GetDegree()
    #atom.GetTotalNumHs()
    #atom.GetImplicitValence()
    #atom.GetIsAromatic()

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_edge_feature(mol):
    edges = []
    for i in range(mol.GetNumAtoms()):
        edge_x = []
        for j in range(mol.GetNumAtoms()):
            bond_ij = mol.GetBondBetweenAtoms(i, j)
            if bond_ij is None:
                edge_x.append([i, j, 4])
                continue
            if bond_ij.GetBondType() == Chem.rdchem.BondType.SINGLE:
                edge_x.append([i, j, 1])
            elif bond_ij.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                edge_x.append([i, j, 2])
            elif bond_ij.GetBondType() == Chem.rdchem.BondType.TRIPLE:
                edge_x.append([i, j, 3])
            else:
                edge_x.append([i, j, 5])
        edges.append(edge_x)
    return torch.tensor(edges)

def get_3d(mol3d):
    # use EmbedMolecule function
    N = mol3d.GetNumAtoms()
    AllChem.EmbedMolecule(mol3d, randomSeed=1)
    statis = Chem.MolToMolBlock(mol3d)

    statis = statis.split('\n', 4 + N)[4:4 + N]

    coors = []
    for s in statis:
        data = s.lstrip().split(maxsplit=3)[:3]
        data = list(map(float, data))
        coors.append(data)
    return torch.tensor(coors)

def get_adj(mol):
    adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
    return torch.tensor(adj)


def smile_to_graph(smiles):
    atom_no=0
    atoms_index=[]
    feats_batch=torch.tensor([])
    edges_batch=torch.tensor([])
    coors_batch=torch.tensor([])
    adj_batch = torch.tensor([])
    mask_batch=torch.tensor([])
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        atom_num = mol.GetNumAtoms()
        mol=Chem.AddHs(mol)

        atoms_index.append([atom_no,atom_no+atom_num])
        atom_no=atom_no+atom_num

        #atom represantation
        feats=torch.tensor(create_graph(mol).atom_feature)
        feats_batch=torch.cat((feats_batch,feats[:atom_num]),0)

        #edge representation
        edges=get_edge_feature(mol)
        edges_batch=torch.cat((edges_batch,edges[:atom_num,:atom_num].reshape(-1,3)),dim=0)

        #3D coordinate representation
        coors=get_3d(mol)
        coors_batch=torch.cat((coors_batch,coors.reshape(-1,3)[:atom_num]),dim=0)

        #adj
        adj=get_adj(mol)
        adj_batch=torch.cat((adj_batch,adj[:atom_num,:atom_num].reshape(-1)),0)
        #mask
        mask = torch.ones(1, feats.shape[0]).bool()
        mask_batch=torch.cat((mask_batch,mask.reshape(-1)[:atom_num]),dim=0)
    print()
    return torch.tensor(atoms_index).cuda(),feats_batch.cuda(),edges_batch.cuda(),coors_batch.cuda(),adj_batch.cuda(),mask_batch.cuda()


if __name__ == '__main__':
    smile = 'C[S](=O)(=O)c1ccc(cc1)[C@@H](O)[C@@H](CO)NC(=O)C(Cl)Cl'
    c_size, features, edge_index, atoms = smile_to_graph(smile)
    print(atoms)