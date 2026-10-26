# MAGE: Memory as Agent-Guided Exploration

This repository is the official implementation for our paper ["Beyond Semantic Organization: Memory as Execution State Management for Long-Horizon Agents"](https://arxiv.org/pdf/2606.06090). MAGE treats memory as active execution-state management rather than a passive archive of retrievable facts, enabling long-horizon agents to maintain a compact, coherent representation of their execution state. Its key components are:
- **Hierarchical Execution State Tree.** A two-layer tree connects raw action–observation traces with subgoal summaries. The agent constructs its context from the active root-to-current path, preserving decision dependencies rather than assembling disconnected memory entries.
- **Boundary-Aware Compression and Validation.** `Grow` records new interactions, `Compress` summarizes completed subgoals, and `Maintain` validates summaries against their underlying traces before they inform subsequent decisions.
- **Selective Revision and Error Isolation.** `Revise` restores a relevant decision boundary and resumes execution on a new branch after detecting an error. Failed branches remain available as exploration hints, excluding from being part of the current execution state.


<div align="center">
  <img src="asserts/method.png" width="700"/>
  <p><em>MAGE Overview.</em></p>
</div>


## 🛠️ Installation

Install the `agentcore` package via conda:

```bash
conda create -n mage python=3.12
conda activate mage
pip install -e .
```

### vLLM Model Serving

The underlying LLM (e.g., Qwen3.6-27B) for MAGE can be served locally via vLLM. Recommand to use CUDA 13.0, install the vLLM package:

```bash
pip install vllm==0.20.0
```

You can download the model by [download_models.sh](agentcore/llm/download_models.sh) and serve it by [serve_llm.sh](agentcore/llm/serve_llm.sh).


## 🚀 Evaluation

We evaluate MAGE mainly on [MemoryArena](https://arxiv.org/pdf/2602.16313). The shared entry points are [evaluate.py](benchmark/evaluate.py). You can use [run.sh](benchmark/run.sh) for evaluations, and [parse_results.py](benchmark/parse_results.py) for parsing evaluation results.

```bash
cd benchmark
bash run.sh webshop
bash run.sh websearch
bash run.sh math
bash run.sh phys
bash run.sh travelplan

# you can set base_url/llm_name if you run on different model/URL.
bash run.sh math http://localhost:8004/v1 Qwen/Qwen3.6-27B
```

The results will be generated in `logs/<dataset>/<timestamp>/<model>_mage/` by default, then you can use the following command to parse the results:

```bash
python parse_results.py <log_dir>
# e.g., python parse_results.py logs/webshop/20260101-000000/Qwen3.6-27B_mage
```

> [!IMPORTANT]
> Before running the evaluations, make sure to set up the environment and download the necessary datasets as described below.

### Webshop

You need create another conda environment to run the WebShop server.

```bash
cd benchmark/webshop/WebShopEnv
conda create -n webshop python=3.8.13
conda activate webshop
bash ./setup.sh  # this will download the dataset and set up the WebShop server

# recommended to run in tmux: `tmux new -s webshop_server` in WebShopEnv/ folder and in `webshop` environment:
bash ./run_dev.sh
```

### WebSearch

BrowseComp-Plus require `JDK 21+` for running the Java-based components.

```bash
conda install -c conda-forge "openjdk=21"
```

Hybrid/dense search uses `Qwen/Qwen3-Embedding-8B` model, you can download and serve it using scripts provided in the [agentcore/llm](agentcore/llm) directory. The embedding model is served by default at `http://localhost:8001/v1/embeddings`. Set `VLLM_EMBEDDING_URL` in [run.sh](benchmark/run.sh) to change the embedding model's endpoint.

Download the dataset and build index:

```bash
cd benchmark/websearch/WebSearchEnv
bash download_data_and_index.sh

# Optional: You can get golden doc IDs of each MemoryArena query after running this script.
python -u find_search.py --build-map
```

### TravelPlan

Download the dataset:

```bash
cd benchmark/travelplan
mkdir -p data && cd data
wget https://huggingface.co/datasets/ZexueHe/memoryarena/resolve/main/group_travel_planner/data.jsonl -O multitraveler.jsonl
wget https://huggingface.co/datasets/osunlp/TravelPlanner/resolve/main/train.csv -O train.csv
```

### Formal Reasoning

Download the datasets:

```bash
cd benchmark/math
mkdir -p data && cd data && wget https://huggingface.co/datasets/ZexueHe/memoryarena/resolve/main/formal_reasoning_math/data.jsonl && cd ../..

cd benchmark/phys
mkdir -p data && cd data && wget https://huggingface.co/datasets/ZexueHe/memoryarena/resolve/main/formal_reasoning_phys/data.jsonl && cd ../..
```

## 📘 Acknowledgements

If you find this work useful, please consider citing our paper:
```bibtex
@inproceedings{chen-etal-2026-beyond-semantic,
    title = "Beyond Semantic Organization: Memory as Execution State Management for Long-Horizon Agents",
    author = "Chen, Yaoqi  and
      Lai, Haibin  and
      Feng, Yuru  and
      Han, Chuyu  and
      Zhang, Qianxi  and
      Lu, Baotong  and
      Li, Menghao  and
      Wang, Xinjiang  and
      Jin, Zewen  and
      Wu, Hao  and
      Li, Cheng  and
      Chen, Qi",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    year = "2026",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.828/",
    pages = "19255--19272",
    ISBN = "979-8-89176-459-0"
}
```

The `agentcore` package is modified from [AgentLite](https://github.com/SalesforceAIResearch/AgentLite). The dataset we use to evaluate MAGE is [MemoryArena](https://huggingface.co/datasets/ZexueHe/memoryarena). The environments we used to test MAGE are based on [WebShop](https://github.com/princeton-nlp/webshop), [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) and [TravelPlanner](https://github.com/OSU-NLP-Group/TravelPlanner).
