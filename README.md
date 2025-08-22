# \textit{TalkToAgent}: A Human-centric Explanation of Reinforcement Learning Agents with Large Language Models
Welcome to the TalkToAgent page. scChat is a pioneering AI assistant designed to enhance single-cell RNA sequencing (scRNA-seq) analysis by incorporating research context into the workflow. Powered by a large language model (LLM), scChat goes beyond standard tasks like cell annotation by offering advanced capabilities such as research context-based experimental analysis, hypothesis validation, and suggestions for future experiments.

## Table of Contents
- [Overview](#overview)
- [Tutorial](#tutorial)
- [Chat Example](#chat-example)
- [Datasets](#datasets)
- [Citation](#citation)

# Overview
<a name="overview"></a>
## Motivation


## Scope


# Tutorial 

To set up the project environment and run the server, follow these steps:

1. Install the required dependencies:
   ```bash
   pip3 install -r requirements.txt

Follow these steps to utilize the application effectively:
### Step 1: Set the OPENAI Key Environment Variable 
- Type and enter export OPENAI_API_KEY='your_openai_api_key' in your terminal

  
### Step 2: Download Neo4j Desktop 2
- Download Neo4j Desktop 2 (https://neo4j.com/download/)
- Download required dump files (https://drive.google.com/drive/folders/17UCKv95G3tFqeyce1oQAo3ss2vS7uZQE)
- Create a new instance on Neo4j (this step asks you set the password)
- Import the dump files as new databases in the created instance.
- Start the database

### Step 3: Upload and update files
- Upload scRNA-seq adata file (.h5ad)
- Upload the pathway vector-based model (.pkl and .faiss), which can be found in this link: https://drive.google.com/drive/u/4/folders/1OklM2u5T5FsjiUvvYRYyWxrssQIb84Ky
- Update specification_graph.json with your Neo4j username, password, system and organ relevant to the database you are using with specific format
- Update sample_mapping.json with adata file corresponding "Sample name", which can be found in adata.obs, and write descriptions for each condition.

  
### Step 4: Initialize the Application
- Run python3 manage.py runserver

### Step 5: Access the Application
- Open your web browser and navigate to:
  `http://127.0.0.1:8000/schatbot`
  
- **Recommended:** Use Google Chrome for the best experience.


## Chat Example
<a name="chat-example"></a>
<p align="center">
<!-- <img src="images/Chatbot_eg_highPPI.png" alt="drawing" width="700"/> -->
</p>

# Datasets
The datasets used for testing 


### Available Explainable Reinforcement Learning Methods


## Citation




