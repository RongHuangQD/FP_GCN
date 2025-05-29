## Table of Contents
- [Directory Structure](#directory Structure)
- [Data preprocessing](#data preprocessing)
- [Code](#code)
- [Reproducing Research Results](#reproducing-research-results)

## Directory Structure
- `code/`: This directory contains the code used by the project.
- `data/`: This directory contains the data used for the project and information about the higher-order petal adjacency matrices which are subsequently generated.
- `exp/`: All generated results, including classification accuracy, confusion matrix, and true and predicted category labels for functional zones, will be saved in this directory after running the code.

## Data preprocessing
1. Download the POI data: The POI data of Chaoyang District is obtained through the API provided by Gaode Map (https://www.amap.com/) in 2020, totaling 177,165 entries. The format is as follows:

| FID | type                                   | longitude           | latitude            |
|-----|----------------------------------------|---------------------|---------------------|
| 0   | supermarket                            | 116.468352765000010 | 39.808660980799999  |
| 1   | building materials and hardware market | 116.468136942000000 | 39.809233351300001  |
| 2   | comprehensive automotive maintenance   | 116.467987226999990 | 39.810792828200000  |
| 3   | motorcycle service-related             | 116.468247112000000 | 39.810950521700001  |
- Change the downloaded data from table format to text format, i.e. data/poi.data/allPOIs_2411.txt.
- File content: Including the number, name, x-coordinate, y-coordinate, and functional zone number of 177,165 POIs in Chaoyang District.

2. Executing `data/generate_features.py` to generate a file data/poi.data/place2vec.txt for POI type feature vectors.
- File content: Includes name and 200-dimensional feature vectors of 592 POI categories in Chaoyang District.

3. Executing `data/form_content.py` to generate a file data/poi.data/poi.data.graphcontent.txt for all POI feature vectors.
- File content: Includes the number, 200-dimensional feature vectors, name and functional zone number of 177,165 POIs in Chaoyang District.

4. Loading the manually labeled urban functional zone category labels file data/poi.data/poi.data.ZoneTypes.txt.
- File content: Includes the number and category labels of the urban functional zones in the Chaoyang District.


## Code
### Environment Requirements
- Python version: Python 3.11.3
- Required libraries: pytorch  pytorch-geometric  networkx  numpy

### Code List
- `data_loading.py`: Loading the graph dataset.
- `form.graph.py`: Creating an adjacency list of graph based on POIs within each functional zone and generating edge-indexed list file for each functional zone.
- `HiGCN.py`: A graph convolutional neural network model that classifies graphs based on features learned from the graph structures.
- `HL_utils.py`: Constructing higher-order petal adjacency matrices.
- `parser.py`: Parameter settings. 
- `run.py`: The main program file that loads the data, trains the model, evaluates performance, and saves the results.
- `train_utils.py`: Training and evaluating the model.
- `tu.py`: Loading the dataset and splitting the dataset into training and testing sets.
- `tu_utils.py`: Loading, preprocessing, and converting a dataset of graphs into a format suitable for graph neural networks.

## Reproducing Research Results
1. Data Preparation: Load the data.
2. Run the Code: Firstly, Executing `code/form_graph.py` to construct a POI graph and generate an edge list file for each functional zone. Then, Executing `code/run.py` to generate the results.
3. View Results: Obtaining classification accuracy, confusion matrix containing precision (that is the heatmap shown in Figure 6 of the paper) and predicted category labels for functional zones (in the exp/.../sorted_true_pred.txt file).
4. Visualization: The predicted category labels can be imported into the attribute table of Chaoyang District's functional zones in ArcMap, which will generate the results shown in Figure 5 of the paper.