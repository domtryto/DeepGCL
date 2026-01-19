import pandas as pd
from get_complex_graph import read_complex_graph,mol2_to_graph
from dataset import *

def create_dataset(data_select):
    data_select = data_select
    '''
    mol2_path = '/home/zhc/pycharm/pdbbind_2016_2013/pdbbind2020/'+data_select+'/mol2/'
    pocket_path = '/home/zhc/pycharm/pdbbind_2016_2013/pdbbind2020/'+data_select+'/pocket/'
    protein_path = '/home/zhc/pycharm/pdbbind_2016_2013/pdbbind2020/'+data_select+'/protein/'
    protein_df = pd.read_csv('/home/zhc/pycharm/pdbbind_2016_2013/sequence/'+data_select+'_protein.csv')
    proteins = {i["id"]: i["seq"] for _, i in protein_df.iterrows()}
    ligands_df = pd.read_csv('/home/zhc/pycharm/pdbbind_2016_2013/smiles/'+ data_select + '_smi.csv')
    ligands = {i["pdbid"]: i["smiles"] for _, i in ligands_df.iterrows()}
    '''

    path = 'Dataset/PDBBind/'
    sequence = 'sequence/'
    smiles = 'smiles/'
    mol2_path = path+data_select+'/mol2/'
    pocket_path = path+data_select+'/pocket/'
    protein_path = path+data_select+'/protein/'
    protein_df = pd.read_csv(sequence+data_select+'_protein.csv')
    proteins = {i["id"]: i["seq"] for _, i in protein_df.iterrows()}
    ligands_df = pd.read_csv(smiles+ data_select + '_smi.csv')
    ligands = {i["pdbid"]: i["smiles"] for _, i in ligands_df.iterrows()}

    def pro_cat(prot):
        x = np.zeros(1000)
        for i, ch in enumerate(prot[:1000]):
            x[i] = seq_dict[ch]
        return x

    seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
    seq_dict_len = len(seq_dict)

    mol2_file_list = os.listdir(mol2_path)
    complex_graph = {}
    protein_dic = {}
    error = []
    pocket_dic = {}
    smiles_dic = {}
    ligand_graph = {}
    for smile in tqdm(mol2_file_list, desc="Processing"):
        try:
            name = smile.split('_')[0]
            # print(name)
            g = read_complex_graph(pocket_path + name +'_pocket.pdb',mol2_path+smile,threshold=6.0)
            complex_graph[name] = g
            protein_dic[name] = pro_cat(proteins[name])

            ligand_g = mol2_to_graph(mol2_path+smile)
            ligand_graph[name] = ligand_g
        except:
            error.append(smile.split('_')[0])
            continue


    print("error===",error)
    print(len(error))
    affinity = {}
    affinity_path = 'affinity_data.csv'
    # affinity_df = pd.read_csv('/home/zhc/pycharm/pdbbind_2016_2013/data/affinity_data.csv')
    affinity_df = pd.read_csv(affinity_path)
    for _, row in affinity_df.iterrows():
        affinity[row[0]] = row[1]

    test_data = TestbedDataset(root='data', dataset=data_select,pro = protein_dic,y=affinity,complex_graph=complex_graph,ligand_graph=ligand_graph)
    print(test_data)



    # train = error=== ['3w8o', '2z97', '6rrm', '6frj', '3wc5', '6h3q']