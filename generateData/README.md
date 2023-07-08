# How to use ```simple.ipynb```
- Use ```simple.ipynb``` to produce traces from simple graph without loops. 
- Change the paramters ```fan out``` and ```number of samples required``` in the first block.
- The code automatically calculates the required depth to produce the ```required number of samples``` with the given ```fan out```.
- Executing the file will produce outputs in ```.pkl```.

# How to use ```mJourney.py```
- Use ```mJourney.py``` to produce traces of nodes from a combined graph
- Requirements:
```
matplotlib==3.7.1
networkx==3.1
```
- Command example:
```
python mJourney.py --graph_file ../../Dominators/graphs/cat_combined_graph.pkl --system_file ../../Traces/progSpec/cat.json --plot True --num_journeys 10000
```
- Parameters: ```--max_length``` decides the maximum length of the trace, ```--num_journeys``` decides the number of traces to be produced, ```--plot``` decides whether to plot the graph or not, ```--graph_file``` is the path to the combined graph, ```--system_file``` is the path to the system file.


