import torch
import dgl
import numpy as np
import random

def build_graph(vocabulary, examples, hyperedge_dropout, device, args):
    selected = int((1 - hyperedge_dropout) * len(examples))
    #random.shuffle(examples)
    examples = examples[:selected]
    triple_train = []
    for example in examples:
        triple_train.append([vocabulary.convert_tokens_to_ids([example.head])[0]-2-args.num_relations, 
                             vocabulary.convert_tokens_to_ids([example.relation])[0]-2, 
                             vocabulary.convert_tokens_to_ids([example.tail])[0]-2-args.num_relations])
    s_gat = []
    t_gat = []
    s = []
    t = []
    r = []
    is_atomic = []
    print(f'num_entities:{args.num_entities}')
    for hyperedge, example in enumerate(examples):
        s_gat.append(hyperedge+args.num_entities+args.num_relations) 
        t_gat.append(vocabulary.convert_tokens_to_ids([example.head])[0]-2-args.num_relations)

        s_gat.append(hyperedge+args.num_entities+args.num_relations)
        t_gat.append(vocabulary.convert_tokens_to_ids([example.relation])[0]-2+args.num_entities)

        s_gat.append(hyperedge+args.num_entities+args.num_relations)
        t_gat.append(vocabulary.convert_tokens_to_ids([example.tail])[0]-2-args.num_relations)

        s.append(hyperedge+args.num_entities+args.num_relations)
        t.append(vocabulary.convert_tokens_to_ids([example.head])[0]-2-args.num_relations)
        r.append(args.num_relations*2)
        is_atomic.append(0)

        s.append(hyperedge+args.num_entities+args.num_relations)
        t.append(vocabulary.convert_tokens_to_ids([example.relation])[0]-2+args.num_entities)
        r.append(args.num_relations*2+1)
        is_atomic.append(0)

        s.append(hyperedge+args.num_entities+args.num_relations)
        t.append(vocabulary.convert_tokens_to_ids([example.tail])[0]-2-args.num_relations)
        r.append(args.num_relations*2+2)
        is_atomic.append(0)

        s.append(vocabulary.convert_tokens_to_ids([example.head])[0]-2-args.num_relations)
        t.append(vocabulary.convert_tokens_to_ids([example.tail])[0]-2-args.num_relations)
        r.append(vocabulary.convert_tokens_to_ids([example.relation])[0]-2)
        is_atomic.append(1)

        if example.auxiliary_info:
            for k,v in example.auxiliary_info.items():
                k_id = vocabulary.convert_tokens_to_ids([k])[0]-2
                v_id = vocabulary.convert_tokens_to_ids(v)
                for i in range(len(v)):    
                    s_gat.append(hyperedge+args.num_entities+args.num_relations)
                    t_gat.append(v_id[i]-2-args.num_relations)
                for i in range(len(v)): 
                    s.append(hyperedge+args.num_entities+args.num_relations)
                    t.append(v_id[i]-2-args.num_relations)
                    r.append(k_id)
                    is_atomic.append(0)

    #逆关系
    for hyperedge, example in enumerate(examples):
        # 逆关系
        t_gat.append(hyperedge+args.num_entities+args.num_relations)
        s_gat.append(vocabulary.convert_tokens_to_ids([example.head])[0]-2-args.num_relations)

        t_gat.append(hyperedge+args.num_entities+args.num_relations)
        s_gat.append(vocabulary.convert_tokens_to_ids([example.relation])[0]-2+args.num_entities)

        t_gat.append(hyperedge+args.num_entities+args.num_relations)
        s_gat.append(vocabulary.convert_tokens_to_ids([example.tail])[0]-2-args.num_relations)

        t.append(hyperedge+args.num_entities+args.num_relations)
        s.append(vocabulary.convert_tokens_to_ids([example.head])[0]-2-args.num_relations)
        r.append(args.num_relations*2+3)
        is_atomic.append(0)

        t.append(hyperedge+args.num_entities+args.num_relations)
        s.append(vocabulary.convert_tokens_to_ids([example.relation])[0]-2+args.num_entities)
        r.append(args.num_relations*2+4)
        is_atomic.append(0)

        t.append(hyperedge+args.num_entities+args.num_relations)
        s.append(vocabulary.convert_tokens_to_ids([example.tail])[0]-2-args.num_relations)
        r.append(args.num_relations*2+5)
        is_atomic.append(0)

        s.append(vocabulary.convert_tokens_to_ids([example.tail])[0]-2-args.num_relations)
        t.append(vocabulary.convert_tokens_to_ids([example.head])[0]-2-args.num_relations)
        r.append(vocabulary.convert_tokens_to_ids([example.relation])[0]-2+args.num_relations)
        is_atomic.append(1)
        if example.auxiliary_info:
            for k,v in example.auxiliary_info.items():
                k_id = vocabulary.convert_tokens_to_ids([k])[0]-2
                v_id = vocabulary.convert_tokens_to_ids(v)
                # 逆关系
                for i in range(len(v)): 
                    t_gat.append(hyperedge+args.num_entities+args.num_relations)
                    s_gat.append(v_id[i]-2-args.num_relations)
                for i in range(len(v)): 
                    t.append(hyperedge+args.num_entities+args.num_relations)
                    s.append(v_id[i]-2-args.num_relations)
                    r.append(k_id+args.num_relations)
                    is_atomic.append(0)

    graph_gat = dgl.graph((s_gat, t_gat), num_nodes=selected+args.num_entities+args.num_relations)
    graph = dgl.graph((s, t), num_nodes=selected+args.num_entities+args.num_relations)
    node_norm = comp_deg_norm(graph, -1)
    edge_norm = node_norm_to_edge_norm(graph,node_norm)
    return torch.tensor(triple_train), graph_gat.to(device), graph.to(device), torch.tensor(r), torch.tensor(edge_norm), selected, torch.tensor(is_atomic)

def comp_deg_norm(graph, power=-1):
    graph = graph.local_var()
    print(f'graph_num_nodes:{graph.number_of_nodes()}')
    print(f'graph_num_edges:{graph.number_of_edges()}')

    in_deg = graph.in_degrees(range(graph.number_of_nodes())).float().numpy()
    norm = in_deg.__pow__(power)
    norm[np.isinf(norm)] = 0
    return torch.from_numpy(norm)

def node_norm_to_edge_norm(graph, node_norm):
    graph.ndata['norm'] = node_norm
    graph.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    norm = graph.edata.pop('norm').squeeze()
    return norm
