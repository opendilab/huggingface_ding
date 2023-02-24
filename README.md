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
from ding.bonus import PPOF
from huggingface_ding import push_model_from_hub

policy_state_dict, cfg = push_model_from_hub(repo_id="OpenDILabCommunity/LunarLander-v2-ppo")
# Create the environment
env_id = "lunarlander_discrete"
# Instantiate the agent
agent = PPOF(env=env_id, cfg=cfg.exp_config)
agent.load_policy(policy_state_dict=policy_state_dict, config=cfg.exp_config)
agent.train(step=5000)

```

### Case 2: I trained an agent and want to upload it to the Hub
```python
from ding.bonus import PPOF
from huggingface_ding import push_model_to_hub

# Create the environment
env_id = "lunarlander_discrete"
# Instantiate the agent
agent = PPOF(env_id, exp_name=env_id)
# Train the agent
return_of_trainning = agent.train(step=int(20000))
push_model_to_hub(
    agent=agent,
    env_name="OpenAI/Gym/Box2d",
    task_name="LunarLander-v2",
    algo_name="PPO",
    wandb_url=return_of_trainning["wandb_url"],
    github_repo_url="https://github.com/opendilab/DI-engine",
    model_description="This is a simple PPO implementation to OpenAI/Gym/Box2d LunarLander-v2.",
    repo_id="OpenDILabCommunity/LunarLander-v2-ppo"
)

```
