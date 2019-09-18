import h5py
import numpy as np

from scipy import ndimage
import networkx as nx
import os
from pandas.core.frame import DataFrame


def ReadH5(filename, datasetname='main'):
    fid=h5py.File(filename,'r')
    if isinstance(datasetname, (list,)):
        out = [None] *len(datasetname)
        for i,dd in enumerate(datasetname):
            out[i] = np.array(fid[dd])
    else:
        sz = len(fid[datasetname].shape)
        out = np.array(fid[datasetname])
    return out

def WriteH5(filename, dtarray, datasetname='main'):
    fid=h5py.File(filename,'w')
    if isinstance(datasetname, (list,)):
        for i,dd in enumerate(datasetname):
            ds = fid.create_dataset(dd, dtarray[i].shape, compression="gzip", dtype=dtarray[i].dtype)
            ds[:] = dtarray[i]
    else:
        ds = fid.create_dataset(datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype)
        ds[:] = dtarray
    fid.close()

def in_range(xyz, range_a, range_b):
    a = np.array(xyz)-np.array(range_a)
    b = np.array(range_b)-np.array(xyz)
    if (a<0).sum()==0 and (b<0).sum()==0:
        return True
    else:
        return False

def process(SKEL_FOLDER, all_nodes,Graph,root_xyz, id, CSV_PATH):
    new_G = Graph

    range_a = np.array(root_xyz) - 100
    range_b = np.array(root_xyz) + 100

    del_node = []
    adj_node = []
    red_nodes = []

    # del_nodes: nodes to be deleted
    # adj_nodes: nodes connected to del_nodes, need to connect to root directly
    # red_nodes: all the reduced_nodes
    for p, t, w in new_G.edges(data=True):
        p_xyz = [all_nodes[p][2], all_nodes[p][1], all_nodes[p][0]]
        t_xyz = [all_nodes[t][2], all_nodes[t][1], all_nodes[t][0]]
        red_nodes.append(p)
        red_nodes.append(t)
        if in_range(p_xyz, range_a, range_b) == True:
            del_node.append(p)
            if in_range(t_xyz, range_a, range_b) == True:
                del_node.append(t)
            else:
                adj_node.append(t)
        else:
            if in_range(t_xyz, range_a, range_b) == True:
                del_node.append(t)
                adj_node.append(p)
            else:
                continue

    #print(np.unique(np.array(red_nodes)))
    # delete the nodes in the range
    a = np.unique(np.array(del_node))

    for i in range(len(a)):
        new_G.remove_node(a[i])

    if 0 in red_nodes:
        root_idx = 1
        assert (1 not in red_nodes)
    else:
        root_idx = 0

    # relabel all_nodes
    all_nodes[root_idx] = [root_xyz[2],root_xyz[1],root_xyz[0]]
    WriteH5(os.path.join(SKEL_FOLDER, 'new_node_pos.h5'), all_nodes)

    new_G.add_node(root_idx)

    # connect adj_nodes to root
    for i in range(len(adj_node)):
        i_xyz = [all_nodes[adj_node[i]][2], all_nodes[adj_node[i]][1], all_nodes[adj_node[i]][0]]
        distance = np.sqrt(np.sum(np.square(np.asarray(i_xyz) - np.asarray(root_xyz))))
        new_G.add_edge(root_idx, adj_node[i], weight=distance)

    nx.write_gpickle(new_G, os.path.join(SKEL_FOLDER, 'new_graph.obj'))

    for p, t, w in new_G.edges(data=True):
        p_xyz = [all_nodes[p][2], all_nodes[p][1], all_nodes[p][0]]
        t_xyz = [all_nodes[t][2], all_nodes[t][1], all_nodes[t][0]]
        w['weight'] = np.sqrt(np.sum(np.square(np.asarray(p_xyz) - np.asarray(t_xyz))))
        w['pos_a'] = p_xyz
        w['pos_b'] = t_xyz

    north = np.array([0, 0, 1])
    for p, t, w in new_G.edges(data=True):
        p_xyz = [all_nodes[p][2], all_nodes[p][1], all_nodes[p][0]]
        t_xyz = [all_nodes[t][2], all_nodes[t][1], all_nodes[t][0]]
        shortest_len_p = nx.shortest_path_length(new_G, source=root_idx, target=p)
        shortest_len_t = nx.shortest_path_length(new_G, source=root_idx, target=t)
        if shortest_len_p > shortest_len_t:
            w['rank'] = shortest_len_t
            w['distance'] = nx.dijkstra_path_length(new_G, root_idx, t, 'weight')
            vector = np.asarray(p_xyz) - np.asarray(t_xyz)
            w['orientation'] = np.arccos(vector.dot(north) / w['weight']) * 360 / 2 / np.pi
        else:
            w['rank'] = shortest_len_p
            w['distance'] = nx.dijkstra_path_length(new_G, root_idx, p, 'weight')
            vector = np.asarray(p_xyz) - np.asarray(t_xyz)
            w['orientation'] = vector.dot(north) / w['weight']

    rank_list = []
    distance_list = []
    orientation_list = []
    pos_a_list = []
    pos_b_list = []

    for p, t, w in new_G.edges(data=True):
        rank_list.append(w['rank'])
        distance_list.append(w['distance'])
        orientation_list.append(w['orientation'])
        pos_a_list.append(w['pos_a'])
        pos_b_list.append(w['pos_b'])
    # %%
    new_edge_list = list(new_G.edges())
    # %%
    id_list =[id]*len(rank_list)
    c = {"cell_id":id_list, "edge": new_edge_list, "rank": rank_list,
         "distance": distance_list, "orientation": orientation_list,
         "pos_a":pos_a_list,"pos_b":pos_b_list}

    edge_dict = DataFrame(c)

    edge_dict.to_csv(path_or_buf=CSV_PATH,index=False)



if __name__ == "__main__":
    bfs = 'bfs'
    edgTh = [40, 1]  # threshold
    outside_ids = [9, 10, 12, 13, 16, 17,
                   19, 20, 21, 22, 23, 24, 37,
                   39, 42, 46, 71, 72,
                   82, 85, 96, 98, 101, 103, 111,
                   113, 114, 121,
                   122, 126, 127, 128, 130, 131,
                   132, 134]

    seg_dir = '/home/xingyu/PycharmProjects/skeleton/neuron/'
    for i in range(len(outside_ids)):
        # get seg
        seg_fn = seg_dir + 'cell' + str(outside_ids[i]) + '_d.h5'
        seg = ReadH5(seg_fn, 'main')
        seg = seg.astype(int)
        assert (np.max(seg)==1)

        # upsample to identify the root point
        upsampled_seg = seg[::10, ::10, ::10]
        k = np.ones([5, 5, 5])
        new = ndimage.convolve(upsampled_seg, k, mode='constant', cval=0.0)

        z,x,y = np.where(new==np.max(new))
        if len(z)!=1:
            z = np.mean(z)
            x = np.mean(x)
            y = np.mean(y)

        root_xyz = [int(y)*10,int(x)*10,int(z)*10]
        print('root',root_xyz)
        SKEL_FOLDER = '/home/xingyu/PycharmProjects/skeleton/outputs/' + str(outside_ids[i])
        CSV_PATH = SKEL_FOLDER + '/analysis.csv'

        # read the reduced nodes graph
        Graph = nx.read_gpickle(os.path.join(SKEL_FOLDER, 'graph-%s-40-10.obj' % (bfs)))

        # read all_nodes volume
        all_nodes = ReadH5(os.path.join(SKEL_FOLDER, 'node_pos.h5'))

        # process to get statistical analysis
        process(SKEL_FOLDER,all_nodes, Graph, root_xyz, outside_ids[i], CSV_PATH)

        print('finshed:', i, '/', len(outside_ids))



