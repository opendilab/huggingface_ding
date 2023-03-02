---
{{ card_data }}
---

<!-- Provide a quick summary of what the model is/does. -->


# Details

## Model Description
<!-- Provide a longer summary of what this model is. -->
This is a simple PPO implementation to OpenAI/Gym/Box2d LunarLander-v2.

- **Developed by:** {{ developers | default("[More Information Needed]", true)}}
- **License:** {{ license | default("[More Information Needed]", true)}}
- **DI-engine version:** {{ di_engine_version | default("[More Information Needed]", true)}}
- **Gym version:** {{ gym_version | default("[More Information Needed]", true)}}
- **PyTorch version:** {{ pytorch_version | default("[More Information Needed]", true)}}
- **Last Update Date:** {{ date | default("[More Information Needed]", true)}}


## Model Sources
<!-- Provide the basic links for the model. -->
- **Github Repository:** [repo link]({{ github_repo_url | default("[More Information Needed]", true)}})
- **Configuration:** [config link]({{ configuration_path | default("[More Information Needed]", true)}})
- **Demo:** [video]({{ demo | default("[More Information Needed]", true)}})
<!-- Provide the size information for the model. -->
- **Parameters total size:** {{ parameters_total_size | default("[More Information Needed]", true)}}

# Environments

<!-- Address questions around what environment the model is intended to be trained and deployed at, including the necessary information needed to be provided for future users. -->


- **Benchmark:** {{ benchmark_name | default("[More Information Needed]", true)}}
- **Task:** {{ task_name | default("[More Information Needed]", true)}}
- **Doc**: [doc link](https://di-engine-docs.readthedocs.io/en/latest/index.html)

# Training Details

## Configuration
<details close>
<summary>(Click for Details)</summary>


```python
{{ python_config | default("[More Information Needed]", true)}}
```
</details>

## Training Procedure 

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

- **Weights & Biases (wandb):** [monitor link]({{ wandb_url | default("[More Information Needed]", true)}})