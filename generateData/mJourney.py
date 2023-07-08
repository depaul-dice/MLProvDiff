import sys, json, pickle
import networkx as nx
import argparse
import matplotlib.pyplot as plt

graph = None
journeys_dfs = []
maxLen = 0
endId = None

def main(args):
    global graph
    global journeys_dfs
    global max_length
    global endId
    global num_journeys

    # get arguments
    graph_file = args.graph_file
    system_file = args.system_file
    output_name = args.output_name    
    max_length = args.max_length
    num_journeys = args.num_journeys

    sys.setrecursionlimit(1000000)

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
                        startId = node["id"]
                    if node["type"] == "retCall":
                        endId = node["id"]    

    stack = [startId]
    dfs(stack, [])

    # save the journeys
    with open(f"{output_name}.pkl", "wb") as f:
        pickle.dump(journeys_dfs, f)

    # plot the graph
    if args.plot:
        # check the distribution of the length of the journies
        lens = []
        for journey in journeys_dfs:
            lens.append(len(journey))
        
        plt.hist(lens, bins=100)
        plt.savefig(f"{output_name}.png")

        print("min: ", min(lens))
        print("max: ", max(lens))
        print("avg: ", sum(lens) / len(lens))


def dfs(stack, journey):
    global graph
    global journeys_dfs
    global max_length
    global endId
    global num_journeys

    node = stack.pop()

    if len(journey) > max_length or len(journeys_dfs) >= num_journeys:
        return
    elif node == endId:
        journeys_dfs.append(journey.copy())
        return
    
    for neighbor in graph.neighbors(node):
        stack.append(neighbor)
        journey.append(neighbor)
        dfs(stack, journey)
        journey.pop()


if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_file", type=str)
    parser.add_argument("--system_file", type=str)
    parser.add_argument("--output_name", type=str, default="output")
    parser.add_argument("--max_length", type=int, default=80)
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--num_journeys", type=int, default=10e3)

    args = parser.parse_args()
    main(args)