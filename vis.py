import os,sys
import h5py
import numpy as np
import networkx as nx
import neuroglancer

from T_util import readh5,ngLayer,relabel

Do='file:///var/www/html/dataset/jwr15_df/'


ip='localhost'
pp=10000+int(np.random.random()*10000)


neuroglancer.set_server_bind_address(bind_address=ip,bind_port=pp)
viewer=neuroglancer.Viewer()






res = [128,256,120]
bfs= 'bfs'
edgTh = [40,1] # threshold

# modify these two for the skeleton you want to visualize
seg = readh5('/home/xingyu/PycharmProjects/skeleton/neuron/cell9_d.h5').astype(np.uint16)
SKEL_FOLDER = '/home/xingyu/PycharmProjects/skeleton/outputs/9'



all_nodes = readh5(os.path.join(SKEL_FOLDER,'new_node_pos.h5'))

reduced_nodes = readh5(os.path.join(SKEL_FOLDER,'graph-bfs-%d-10.h5'%(edgTh[0])))


Graph = nx.read_gpickle(os.path.join(SKEL_FOLDER,'graph-%s.obj'%(bfs)))

new_G = nx.read_gpickle(os.path.join(SKEL_FOLDER,'new_graph.obj'))
edge_list = new_G.edges()



with viewer.txn() as s:
    s.layers.append(name="den",layer=ngLayer(seg,res))
    s.layers['den'].segments.update(range(1,30))
    s.layers['den'].visible = True

    s.layers.append(name="nodes",layer=ngLayer(reduced_nodes,res))
    s.layers['nodes'].segments.update(range(1,3))
    s.layers['nodes'].visible = True

    s.layers.append(name='edges',layer=neuroglancer.AnnotationLayer(voxelSize=res))
    annotations = s.layers[-1].annotations
    line_id = 1
    for edge in edge_list:
        annotations.append(
            neuroglancer.LineAnnotation(
                id=line_id,
                point_a=[all_nodes[edge[0]][2],all_nodes[edge[0]][1],all_nodes[edge[0]][0]],
                point_b=[all_nodes[edge[1]][2],all_nodes[edge[1]][1],all_nodes[edge[1]][0]]
                )
            )


        line_id += 1



print(viewer)
