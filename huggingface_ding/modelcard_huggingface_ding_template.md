---
{{ card_data }}
---

# Play **{{ task_name | default("[More Information Needed]", true)}}** with **{{ algo_name | default("[More Information Needed]", true)}}** Policy

## Model Description
<!-- Provide a longer summary of what this model is. -->
This is a simple **{{ algo_name | default("[More Information Needed]", true)}}** implementation to {{ benchmark_name | default("[More Information Needed]", true)}} **{{ task_name | default("[More Information Needed]", true)}}** using the [DI-engine library](https://github.com/opendilab/di-engine) and the [DI-zoo](https://github.com/opendilab/DI-engine/tree/main/dizoo).

**DI-engine** is a python library for solving general decision intelligence problems, which is based on implementations of reinforcement learning framework using PyTorch or JAX. This library aims to standardize the reinforcement learning framework across different algorithms, benchmarks, environments, and to support both academic researches and prototype applications. Besides, self-customized training pipelines and applications are supported  by reusing different abstraction levels of DI-engine reinforcement learning framework.

{{ model_description | default("", false)}}

## Model Usage

<details open>
<summary>(Click to Collapse)</summary>

```shell
# Install huggingface_ding
git clone https://github.com/opendilab/huggingface_ding.git
pip3 install -e ./huggingface_ding/
# Install Dependencies If Needed
pip3 install -r requirements.txt
{{ installation_guide | default("", false)}}
# Running with Trained Model
python3 run.py
```
</details>

**run.py**
<details close>
<summary>(Click for Details)</summary>

```python
{{ python_code_for_usage | default("[# More Information Needed]", true)}}
```
</details>

## Model Training

<details close>
<summary>(Click to Details)</summary>

```shell
# Install huggingface_ding
git clone https://github.com/opendilab/huggingface_ding.git
pip3 install -e ./huggingface_ding/
# Install Dependencies If Needed
pip3 install -r requirements.txt
#Training Your Own Agent
python3 training.py
```
</details>

**Configuration**
<details close>
<summary>(Click for Details)</summary>


```python
{{ python_config | default("[More Information Needed]", true)}}
```
</details>

**Training Procedure** 
<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->
- **Weights & Biases (wandb):** [monitor link]({{ wandb_url | default("[More Information Needed]", true)}})

## Model Information
<!-- Provide the basic links for the model. -->
- **Github Repository:** [repo link]({{ github_repo_url | default("https://github.com/opendilab/DI-engine", true)}})
- **Doc**: [DI-engine-docs Algorithm link]({{ github_doc_model_url | default("https://di-engine-docs.readthedocs.io/", true)}})
- **Configuration:** [config link]({{ config_file_url | default("[More Information Needed]", true)}})
- **Demo:** [video]({{ video_demo_url | default("[More Information Needed]", true)}})
<!-- Provide the size information for the model. -->
- **Parameters total size:** {{ parameters_total_size | default("[More Information Needed]", true)}}
- **Last Update Date:** {{ date | default("[More Information Needed]", true)}}

## Environments
<!-- Address questions around what environment the model is intended to be trained and deployed at, including the necessary information needed to be provided for future users. -->
- **Benchmark:** {{ benchmark_name | default("[More Information Needed]", true)}}
- **Task:** {{ task_name | default("[More Information Needed]", true)}}
- **Gym version:** {{ gym_version | default("[More Information Needed]", true)}}
- **DI-engine version:** {{ di_engine_version | default("[More Information Needed]", true)}}
- **PyTorch version:** {{ pytorch_version | default("[More Information Needed]", true)}}
- **Doc**: [DI-engine-docs Environments link]({{ github_doc_env_url | default("https://di-engine-docs.readthedocs.io/", true)}})

