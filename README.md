# Hugging Face 🤗 x OpenDILab/DI-engine

A library to push and pull models from the Huggingface Hub.

## Installation
### With pip
```
pip install -e .
```

## Examples
### Case 1: I want to download a model from the Hub
```python
from ding.bonus import TD3OffPolicyAgent
from huggingface_ding import pull_model_from_hub
# Pull model from Hugggingface hub
policy_state_dict, cfg=pull_model_from_hub(repo_id="OpenDILabCommunity/Hopper-v3-TD3")
# Create the environment
env_id = "hopper"
# Instantiate the agent
agent = TD3OffPolicyAgent(env=env_id,exp_name="hopper-td3-from-huggingface", cfg=cfg.exp_config, policy_state_dict=policy_state_dict)
# Continue training
agent.train(step=500000)
# Render the new agent performance
agent.deploy(enable_save_replay=True)
```

### Case 2: I trained an agent and want to upload it to the Hub
```python
from ding.bonus import TD3OffPolicyAgent
from huggingface_ding import push_model_to_hub

# Create the environment
env_id = "hopper"
exp_name = "hopper-td3"
# Instantiate the agent
agent = TD3OffPolicyAgent(env_id, exp_name=exp_name)
# Train the agent
return_ = agent.train(step=int(200000), collector_env_num=4, evaluator_env_num=4)
# Train the agent
push_model_to_hub(
    agent=agent,
    env_name="OpenAI/Gym/MuJoCo",
    task_name="Hopper-v3",
    algo_name="TD3",
    wandb_url=return_.wandb_url,
    github_repo_url="https://github.com/opendilab/DI-engine",
    model_description="This is a simple TD3 implementation to OpenAI/Gym/MuJoCo Hopper-v3.",
    usage_file_path="./benchmark/TD3Agent_Download.py",
    repo_id="OpenDILabCommunity/Hopper-v3-TD3"
)
```
