import pickle, random
import torch

# padd for the same length
from torch.nn.utils.rnn import pad_sequence

def data(file_name, use_ratio):
    '''
        return: num_features -> int, feature_matrix -> tensor[float], edge_list -> tensor[long], traces_x -> tensor[float], traces_y -> tensor[long]
    '''

    with open(f'./data/{file_name}_journeys.pkl', 'rb') as f:
        journeys = pickle.load(f)
        journeys = random.sample(journeys, int(len(journeys) * use_ratio))

    with open(f'./data/{file_name}_combined_graph.pkl', 'rb') as f:
        graph = pickle.load(f)

    # check the node types
    types = set()
    for i in range(len(journeys[0])):
        types.add(graph.nodes[journeys[0][i]]['node_type'])

    # create type and label -> id mapping
    type2id = {}
    for t in types:
        type2id[t] = len(type2id)

    label2id = {}
    for i, node in enumerate(graph):
        if graph.nodes[node]['label'] not in label2id:
            label2id[graph.nodes[node]['label']] = len(label2id)

    # create node idx to id mapping
    node2id = {}
    for i, node in enumerate(graph):
        node2id[node] = i + 1

    # create edge list
    edge_list_dep = []
    edge_list_rev = []

    for node in graph.nodes:
        for neighbor in graph.neighbors(node):
            edge_list_dep.append(node2id[node] - 1)
            edge_list_rev.append(node2id[neighbor] - 1)
            
    # create node feature list
    featureMatrix = []

    for node in graph.nodes:
        featureMatrix.append([type2id[graph.nodes[node]['node_type']], label2id[graph.nodes[node]['label']]])

    num_features = len(featureMatrix[0])

    # jounrey to node id
    journeys_id = []

    for journey in journeys:
        journey_id = []
        for node in journey:
            journey_id.append(node2id[node])
        journeys_id.append(torch.tensor(journey_id, dtype=torch.long))
    
    # trace in journey to fm
    journeys_fm = []

    for journey in journeys_id:
        journey_fm = []
        for node in journey:
            journey_fm.append(featureMatrix[node])
        journeys_fm.append(torch.tensor(journey_fm, dtype=torch.float))

    # Pad the sequences so they are the same length
    traces_x = pad_sequence(journeys_fm, batch_first=True)
    traces_y = pad_sequence(journeys_id, batch_first=True)

    return num_features, torch.tensor(featureMatrix, dtype=torch.float), torch.tensor([edge_list_dep, edge_list_rev], dtype=torch.long), traces_x, traces_y






    
    