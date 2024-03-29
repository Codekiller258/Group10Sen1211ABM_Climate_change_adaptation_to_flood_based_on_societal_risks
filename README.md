## Climate Adaptation to flood based on societal risks

### Introduction
To see the conceptual flowchart of the model, see `other` file.

This directory contains an agent-based model (ABM) implemented in Python, focused on simulating household adaptation to flood events in a social network context, based on the societal risks. It uses the Mesa framework for ABM and incorporates geographical data processing for flood depth and damage calculations.

The RBB we developed are also used in the model, mainly in the government agent part. 

### Installation
To set up the project environment, follow these steps:
1. Make sure you have installed a recent Python version, like 3.11 or 3.12.
2. Install the latest Mesa version (2.1.5 or above) with `pip install -U mesa`
2. Clone the repository to your local machine.
3. Install required dependencies:
   ```bash
   pip install -U geopandas shapely rasterio networkx
   ```

### File descriptions
The `analysis` directory contains some of the analysis
The `output` directory contains experimental output
The `other` directory contains flowcharts
The `input_data` directory contains necessary input data for the model
The `model` directory contains the actual Python code for the minimal model. It has the following files:
- `agents.py`: Defines the `Households` agent class, each representing a household in the model. These agents have attributes related to flood depth and damage, and their behavior is influenced by these factors. It also contains the `Government` agent, which have the social-related data. Their responsibility is to make collective adaptation strategies and policies, the government's behavior is mainly based on the FN theory. These two agents have interactions.
- `functions.py`: Contains utility functions for the model, including setting initial values, calculating flood damage, and processing geographical data. These functions are essential for data handling and mathematical calculations within the model.
- `model.py`: The central script that sets up and runs the simulation. It integrates the agents, geographical data, and network structures to simulate the complex interactions and adaptations of households to flooding scenarios. Some of the important data will also be collected here.
- `demo.ipynb`: A Jupyter notebook titled "Flood Adaptation based on societal risks". It demonstrates running a model and analyzing and plotting some results.
There is also a directory `input_data` that contains the geographical data used in the model. You don't have to touch it, but it's used in the code and there if you want to take a look.

### Usage
This model is used to simulate the households' adapatation to flood based on the societal risk and FN theory.
