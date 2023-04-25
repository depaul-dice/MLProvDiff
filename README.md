# Progress

# HY Worklog
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
### 26th, Apr 2023
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