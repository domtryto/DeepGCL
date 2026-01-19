from rdkit import Chem
import networkx as nx
import numpy as np
from Bio.PDB import PDBParser

def compute_distance(atom1, atom2):
    return np.linalg.norm(np.array(atom1) - np.array(atom2))
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),# 原子的化学元素
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +#原子的度 该原子直接连接其它原子的数量
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + # 获取原子上连接的氢原子的数量 包括显式和隐士的氢原子
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + #原子的隐式化合价
                    one_of_k_encoding_unk(str(atom.GetChiralTag()), #原子的手性标签
                                          ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER',
                                           'misc']) +
                    one_of_k_encoding_unk(atom.GetFormalCharge(), [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc']) + #形式电荷
                    one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1, 2, 3, 4, 'misc']) + #自由基电子数
                    one_of_k_encoding_unk(str(atom.GetHybridization()), ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc']) + #原子的杂化状态
                    [atom.IsInRing()] + #判断某个原子是否属于一个环结构
                    [atom.GetIsAromatic()] #判断原子是否在芳香环中
                    )
def mol2_to_graph(smile):
    # if_h = None
    mol = Chem.MolFromMol2File(smile)
    c_size = mol.GetNumAtoms()
    features = []
    adj_matrix = []
    for atom in mol.GetAtoms():
        # if atom.GetSymbol() == 'H' :
        #     if_h = True
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []

    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index
def pdb_to_graph(smile):
    # if_h = None
    mol = Chem.MolFromPDBFile(smile)
    if mol is None:
        mol = Chem.MolFromPDBFile(smile,sanitize=False) # 跳过净化处理
    c_size = mol.GetNumAtoms() # 返回分子对象mol中的原子数量
    features = []
    for atom in mol.GetAtoms():
        # if atom.GetSymbol() == 'H' :
        #     if_h = True
        feature = atom_features(atom)
        features.append(feature / sum(feature)) # 将原子特征进行归一化处理
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]) # 将化学分子中每个键的起始原子和结束原子的索引作为一个边（edge）
    g = nx.Graph(edges).to_directed() # 将无向图转换成有向图
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index

# 定义函数来计算两个原子之间的距离
def calculate_distance(atom1, atom2):
    x1, y1, z1 = atom1
    x2, y2, z2 = atom2
    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
    return distance


# 从PDB文件中读取原子坐标
def read_pdb_file(pdb_filename):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_filename)
    atom_coordinates = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_coordinates.append(atom.get_coord())
    return atom_coordinates
def read_pdb_coordinates(pdb_file):
    atom_coordinates = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                if line[66:78].strip() == 'H':
                    continue
                # 提取坐标信息
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                atom_coordinates.append([x, y, z])
    return atom_coordinates
# 从MOL2文件中读取原子坐标
def read_mol2_file(mol2_filename):
    atom_coordinates = []
    read_coordinates = False

    with open(mol2_filename, 'r') as mol2_file:
        start_reading = False
        for line in mol2_file:
            if line.strip() == '@<TRIPOS>ATOM':
                start_reading = True
                continue
            if line.strip() == '@<TRIPOS>BOND':
                break
            if start_reading:
                if line.strip().split()[5]=='H':
                    continue
                # print(line.split())
                x = float(line.strip().split()[2])
                y = float(line.strip().split()[3])
                z = float(line.strip().split()[4])
                atom_coordinates.append((x, y, z))

            # if read_coordinates:
            #     parts = line.split()
            #     print(parts)
            #     if len(parts) >= 6:
            #         try:
            #             x = float(parts[2])
            #             y = float(parts[3])
            #             z = float(parts[4])
            #             atom_coordinates.append((x, y, z))
            #         except ValueError:
            #             continue
            #
            # elif line.strip() == '@<TRIPOS>ATOM':
            #     read_coordinates = True
    return atom_coordinates
# 计算两个分子之间的原子距离小于10Å的原子对数
def find_close_atoms(atom_coordinates1, atom_coordinates2, threshold):
    close_atom_pairs = []
    distances = []
    for i, atom1 in enumerate(atom_coordinates1):
        for j, atom2 in enumerate(atom_coordinates2):
            distance = calculate_distance(atom1, atom2)
            # print(distance)
            if distance < threshold:
                close_atom_pairs.append([i, j])
                distances.append(distance)
    return close_atom_pairs,distances



def read_atom_parts(pdb_filename,mol2_filename,threshold):
    # 读取原子坐标
    pdb_atom_coordinates = read_pdb_coordinates(pdb_filename)
    mol2_atom_coordinates = read_mol2_file(mol2_filename)
    # print(mol2_atom_coordinates)
    # 计算原子距离小于10Å的原子对数
    count_pairs,distances = find_close_atoms(pdb_atom_coordinates, mol2_atom_coordinates,threshold)
    return count_pairs,distances

def read_complex_graph(pdb,mol2,threshold=None):
    pdb_c_index, pdb_features, pdb_edge_index = pdb_to_graph(pdb)
    mol2_c_index, mol2_features, mol2_edge_index = mol2_to_graph(mol2)
    # if_h = None
    # if pdb_if_h == True or mol2_if_h == True:
    #     if_h = True
    #读取权重
    pdb_corrdinates = read_pdb_coordinates(pdb)  #原子坐标
    mol2_corrdinates = read_mol2_file(mol2)
    mol2_distances = []
    pdb_distances = []
    for part in mol2_edge_index:
        distance = compute_distance(mol2_corrdinates[part[0]],mol2_corrdinates[part[1]])
        mol2_distances.append(distance)
    for part in pdb_edge_index:
        distance = compute_distance(pdb_corrdinates[part[0]],pdb_corrdinates[part[1]])
        pdb_distances.append(distance)

    mol2_edge_index = [[x + pdb_c_index for x in sublist] for sublist in mol2_edge_index]
    c_size = mol2_c_index + pdb_c_index

    # mol2_edge_index = [x + [val] for x, val in zip(mol2_edge_index, mol2_distances)]
    # pdb_edge_index = [x + [val] for x, val in zip(pdb_edge_index, pdb_distances)]
    edge_index = pdb_edge_index + mol2_edge_index
    features = pdb_features + mol2_features
    parts,parts_distances = read_atom_parts(pdb, mol2,threshold)
    parts = [[x[0], x[1] + pdb_c_index] for x in parts]
    # parts = [x + [val] for x, val in zip(parts, parts_distances)]

    edge_index = edge_index + parts
    # complex_edge_index = [x + [val] for x, val in zip(edge_index, edge_weight)]
    complex_edge_index = sorted(edge_index, key=lambda x: x[0])     #暂时没有用上

    # edge_index = [sublist[:2] for sublist in complex_edge_index]
    # edge_weight = [sublist[2] for sublist in complex_edge_index]
    # edge_weights = [ np.exp(-sublist / 5.0)for sublist in edge_weight]
    return c_size , features , edge_index
# a,b,c,d = read_complex_graph('1a30_pocket.pdb','3wc5_ligand.mol2',threshold=6)
