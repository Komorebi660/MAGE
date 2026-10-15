#!/bin/bash

# Install Environment Dependencies via `conda`
conda install -c conda-forge openjdk=11 python=3.8.13 -y
conda install -c conda-forge mkl intel-openmp -y
conda install -c conda-forge faiss-cpu=1.7.4 -y

# Install Python Dependencies
pip install -r requirements.txt

# Download spaCy large NLP model
python -m spacy download en_core_web_lg

# Download dataset into `data` folder
mkdir -p data
cd data
wget https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/main/items_shuffle.json
wget https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/main/items_ins_v2.json
wget https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/main/items_human_ins.json
wget https://huggingface.co/datasets/ZexueHe/memoryarena/resolve/main/bundled_shopping/data.jsonl
cd ..

# Build search engine index
cd search_engine
mkdir -p resources resources_100 resources_1k resources_100k
python convert_product_file_format.py # convert items.json => required doc format
mkdir -p indexes
./run_indexing.sh
cd ..
