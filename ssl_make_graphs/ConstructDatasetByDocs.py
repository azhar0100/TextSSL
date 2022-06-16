from typing import Dict, Sequence, Tuple
import pandas as pd
import networkx as nx
import numpy as np
from scipy import sparse
import torch, sys
Your_path = '/code/'
sys.path.append(Your_path+'ssl_make_graphs')
sys.path.append(Your_path+'ssl_graphmodels')
from PairData import PairData
pd.set_option('display.max_columns', None)
import os.path as osp, os
from tqdm import tqdm


def combine_same_word_pair(df, col_name):
    dfs = []
    for w1, w1_df in df.groupby(by='word1'):
        for w2, w2_df in w1_df.groupby(by='word2'):
            freq_sum = w2_df['freq'].sum() / len(w2_df)
            dfs.append([w2_df['word1'].values[0], w2_df['word2'].values[0], freq_sum])
    dfs = pd.DataFrame(dfs, columns=['word1', 'word2', col_name])
    return dfs


def graph_to_torch_sparse_tensor(G_true:nx.Graph, edge_attr:str=None, node_attr:Sequence[str]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Converts the networx graph into a torch sparse tensor

    Args:
        G_true (nx.Graph): The original networkx graph.
        edge_attr (str, optional): _description_. If given, edge_attrs will be a torch Tensor containing N x edge_number elements). Defaults to None.
        node_attr (Sequence[str], optional: If given, it should be a list containing one of 'paragraph_id' or 'batch_id'. Nothing else. Repeated values would lead to bugs. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: edge_index, edge_attrs, x, batch_n, pos_n
                                    edge_index: PyG style edge index. Shape is 2 x edge_number. Every tuple of nodes is connected.
                                    edge_attrs: torch Tensor, of shape N x (dimension of given edge attr). Only if edge_attr is not None
                                    x: torch.Tensor. Of shape N x (node embedding dimension)
                                    batch_n: A one dimensional torch tensor, returned if 'paragraph_id' is in node_attr, and contains the same value
                                    pos_n: A one dimensional torch tensor, returned if 'node_pos' is in node_attr, and contains the same_value
    """    

    G = nx.convert_node_labels_to_integers(G_true, label_attribute='word_name')
    # Gives every node in G an integer label.
    # The old name is stored in word_name.
    # G_true contains nodes via word_name

    A_G = np.array(nx.adjacency_matrix(G).todense())


    sparse_mx = sparse.csr_matrix(A_G).tocoo()
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    # sparse_mx contains a row array, and a col array and a data array
    # such that in the adjacency matrix A_G[row[i]][col[i]] = data[i]
    # Every other element is defined to be zero, unless explicitly set
    # to something else other than zero.
    

    edge_attrs = []
    if edge_attr != None:
        for i in range(sparse_mx.row.shape[0]):
            edge_attrs.append(G.edges[sparse_mx.row[i], sparse_mx.col[i]][edge_attr])
            # networx allows graph edges to be accessed as G.edges[u,v], where u and v are node indices

    edge_index = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.long))
    # Most likely a PyG matrix, containing edges in the form 2 x N_e

    edge_attrs = torch.from_numpy(np.array(edge_attrs)).to(torch.float32)
    x = []
    batch_n = []
    pos_n = []
    for node in range(len(G)):
        # word_name = G.nodes[node]['word_name']
        x.append(G.nodes[node]['node_emb'])
        if node_attr != None:
            for attr in node_attr:
                if attr == 'paragraph_id':
                    batch_n.append(G.nodes[node][attr])
                elif attr == 'node_pos':
                    pos_n.append(G.nodes[node][attr])
                else:
                    print('sth wrong with node attribute')
    x = np.array(x)
    if len(x.shape) != 2:
        print(x.shape)
    x = torch.from_numpy(np.array(x)).to(torch.float32)
    batch_n = torch.from_numpy(np.array(batch_n)).to(torch.long)
    pos_n = torch.from_numpy(np.array(pos_n)).to(torch.long)

    return edge_index, edge_attrs, x, batch_n, pos_n



def set_word_id_to_node(G:nx.Graph, dictionary:Sequence[str], node_emb:str, word_embeddings:Dict[str,Sequence[float]]) -> nx.Graph:
    """Turns a graph of words into a graph of word embeddings.

    Args:
        G (nx.Graph): Original graph of words.
        dictionary (Sequence[str]): A list of allowed words to be used. Any word in the graph outside this sequence would cause an AssertionError
        node_emb (str): The key against which the word embedding will be stored inside the networkx graph G.
        word_embeddings (Dict[str,Sequence[float]]): A dictionary from word to corresponding word embeddings. Random embedding will be assigned to each word
                                                    without an embedding.

    Returns:
        nx.Graph: The original graph, which would now contain word embeddings for each word in the node_attr, node_emb
    """    
    for node in G:
        if node in dictionary:
            ind = np.array([dictionary.index(node)])
            emb = np.array(word_embeddings[node]) if node in word_embeddings else np.random.uniform(-0.01, 0.01, 300)
            emb = np.concatenate([ind, emb]).reshape(301)
            G.nodes[node][node_emb] = emb
        else:
            print('no!!')
            print(f"Node is {node}")
            assert node in dictionary
    return G


class ConstructDatasetByDocs():
    def __init__(self, pre_path, split, dictionary, pt):
        self.pre_path = pre_path
        self.split = split
        self.dictionary = dictionary
        self.pt = pt
        super(ConstructDatasetByDocs).__init__()
        self.all_cats = []
        self.word_embeddings = {}
        if pt == "":
            print('importing glove.6B.300d pretrained word representation...')
            with open(Your_path+'ssl_graphmodels/config/glove.6B.300d.txt', 'r') as f:
                for line in f.readlines():
                    data = line.split()
                    self.word_embeddings[str(data[0])] = list(map(float, data[1:]))
        else:
            print('pre trained is not available!')

    def generate_doc_graph(self, df):
        # print('\nraw---df: ', len(df))
        result_df = combine_same_word_pair(df, col_name='global_freq')
        result_df['edge_attr'] = 1
        result_graph = nx.from_pandas_edgelist(result_df, 'word1', 'word2', 'edge_attr')
        return result_graph

    def construct_datalist(self):
        Data_list = []
        cooc_path = osp.join(self.pre_path, self.split+'_cooc')
        for y_id, y in enumerate(os.listdir(cooc_path)):
            patients = os.listdir(os.path.join(cooc_path, y))
            for patient in tqdm(patients, desc='Iterating over patients in {}_{}_cooc'.format(y, self.split)):
                p_df = pd.read_csv(osp.join(cooc_path, y, patient), sep='\t', header=0)
                G_p = self.generate_doc_graph(p_df)
                G_p = set_word_id_to_node(G_p, self.dictionary, node_emb='node_emb', word_embeddings=self.word_embeddings)
                edge_index_p, edge_attrs_p, x_p, _, _ = graph_to_torch_sparse_tensor(G_p, edge_attr='edge_attr')
                y_p = torch.from_numpy(np.array([y_id])).to(torch.long)

                G_n_list = []
                y_n_list = []
                for n_id, n_df in p_df.groupby(by='paragraph_id'):
                    n_df = n_df.dropna(axis=0)
                    G_n = nx.from_pandas_edgelist(n_df, 'word1', 'word2', ['freq'])
                    attrs = {}
                    for node in G_n:
                        attrs[node] = {'node_pos': list(G_p.nodes).index(node), 'paragraph_id': n_id}
                    nx.set_node_attributes(G_n, attrs)
                    G_n = set_word_id_to_node(G_n, self.dictionary, 'node_emb', self.word_embeddings)
                    G_n_list.append(G_n)
                    y_n_list.append([n_id])

                G_n = nx.disjoint_union_all(G_n_list)
                edge_index_n, _, x_n, batch_n, pos_n = graph_to_torch_sparse_tensor(G_n, node_attr=['paragraph_id', 'node_pos'])
                y_n = torch.from_numpy(np.array(y_n_list)).to(torch.long)

                # print(x_n, edge_index_n, y_n, batch_n)
                # print(x_p, edge_index_p, y_p, edge_attrs_p)
                data = PairData(x_n, edge_index_n, y_n, batch_n, pos_n, x_p, edge_index_p, y_p, edge_attrs_p)
                assert data.x_p.squeeze().max().item() < len(self.dictionary)
                Data_list.append(data)
        return Data_list

if __name__ == '__main__':
    pass
