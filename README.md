# Progress

# Update Log
### 11th, Apr 2023
- Meeting
  1. More data production required: 10 -> 10000 entries
  2. Need for process tracking: use github readme
  3. Need for segmented intergrity check
- Meeting Follow Up
  1. Produced more Data. Sent it via email, made it reproducible, wrote how to use on subfolder readme.

### 12th, Apr 2023
- Data Request
  1. Dhruv's Request for small sized data
  2. Full automization
- Integrity check

### 25th, Apr 2023
- Graphsage Intergrity Check Pass
  1. Test data with 100 traces and 200 nodes
  2. 100% Accuracy
  3. To replicate, 
    - Run ```generateData/simple.ipynb``` to generate data
    - Run ```integrityCheck/graphSage.ipynb``` to run train and test
- BiLSTM + Graphsage Integrity Check Pass
  1. Test data with 100 traces and 200 nodes
  2. 100% Accuracy
  3. To replicate, 
    - Run ```generateData/simple.ipynb``` to generate data
    - Run ```integrityCheck/graphSage+BiLSTM.ipynb``` to run train and test
- Findings
  1. MLP can be redundant. We can just use graphsage embedding to get feature vectors corresponding to each node
  2. BiLSTM is necessary for encapsulating the sequence information of the traces

### 26th, Apr 2023
- Stage 1 Integrity Check Pass
  1. The model can accept varying trace length
  2. 100% accuracy for 50+ traces
- Stage 3 Integrity Check Start
  1. Need to look up masking. Training or inferencing?
  - Usually masking is done during the inference. However, we have a different problem here, where the choices (for which will be the next node) are very limited. When we do a conventional masking, the inference of the labels of the later nodes does not take advantage of the narrow choices because the dot product is done altogether by a matrix to matrix multiplication.
  - We might benefit if the inference of the label for each node is sequential, in order to build up on the prior knowledge. 
  2. I will proceed the experiement without masking, but this is something we should consider.


### 2th, May 2023
- Stage 3 Integrity Check Finish
  1. To replicate
    - Run ```generateData/stage3.ipynb``` to generate data
    - Run ```integrityCheck/stage3.ipynb``` to run train and test
  2. To see the results of the experiment, see ```midterm PT.pdf```

### 10th, May 2023
- Goals
  1. Check with complex data
  2. Check with different structure
  3. Apply masking
- Progress
  1. Simpler and faster structure
  2. Can try masking, but has two choices: 
    - Normal masking at test stage.
    - Sequential masking using masked information
    - Should be tested on real data.
