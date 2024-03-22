import sys, json, pickle
import networkx as nx
import argparse
import matplotlib.pyplot as plt
import random

graph = None
journeys_RW = []
max_length = 0
end_id = None

def main(args):
    global graph
    global journeys_RW
    global max_length
    global endId
    global num_journeys

    # get arguments
    graph_file = args.graph_file
    system_file = args.system_file
    output_name = args.output_name    
    max_length = args.max_length
    num_journeys = args.num_journeys
    allow_cycle = True

    # read the graph with pickle
    with open(f"{graph_file}", "rb") as f:
        graph = pickle.load(f)

    # get the id of start and the end node
    with open(f"{system_file}", "r") as f:
        funcs = json.load(f)
        for func in funcs:
            if func["function name"] == "main":
                for node in func["nodes"]:
                    if node["type"] == "ePoint":
                        start_id = node["id"]
                    if node["type"] == "retCall":
                        end_id = node["id"]    

    count = 0
    while len(journeys_RW) < num_journeys:
        count += 1
        if count % 5000 == 0:
            print(len(journeys_RW) / num_journeys * 100, "% done")
        randomWalk(graph, start_id, end_id, max_length, allow_cycle)

    # save the journeys
    with open(f"{output_name}.pkl", "wb") as f:
        pickle.dump(journeys_RW, f)

    # plot the graph
    if args.plot:
        # check the distribution of the length of the journies
        lens = []
        for journey in journeys_RW:
            lens.append(len(journey))
        
        plt.hist(lens, bins=100)
        plt.savefig(f"{output_name}.png")

        print("min: ", min(lens))
        print("max: ", max(lens))
        print("avg: ", sum(lens) / len(lens))


def randomWalk(graph, start_id, end_id, max_length, allow_cycle=True):
    global journeysRW
    
    path = [start_id]
    current_node = start_id
    while current_node != end_id and len(path) < max_length:
        neighbors = list(graph.neighbors(current_node))
        if not neighbors:
            break
        next_node = random.choice(neighbors)
        if not allow_cycle and next_node in path:
            break
        path.append(next_node)
        current_node = next_node

    if current_node == end_id:
        journeys_RW.append(path)

if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_file", type=str)
    parser.add_argument("--system_file", type=str)
    parser.add_argument("--output_name", type=str, default="output")
    parser.add_argument("--max_length", type=int, default=1000)
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--num_journeys", type=int, default=10e2)

    args = parser.parse_args()
    main(args) 