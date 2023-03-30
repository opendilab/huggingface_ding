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
from ding.bonus import TD3Agent
from huggingface_ding import pull_model_from_hub

# Pull model from Hugggingface hub
policy_state_dict, cfg=pull_model_from_hub(repo_id="OpenDILabCommunity/LunarLander-v2-TD3")
# Instantiate the agent
agent = TD3Agent(env="lunarlander_continuous",exp_name="LunarLander-v2-TD3", cfg=cfg.exp_config, policy_state_dict=policy_state_dict)
# Continue training
agent.train(step=5000)
# Render the new agent performance
agent.deploy(enable_save_replay=True)

```

### Case 2: I trained an agent and want to upload it to the Hub
```python
from ding.bonus import TD3Agent
from huggingface_ding import push_model_to_hub

# Instantiate the agent
agent = TD3Agent("lunarlander_continuous", exp_name="LunarLander-v2-TD3")
# Train the agent
return_ = agent.train(step=int(4000000), collector_env_num=4, evaluator_env_num=4)
# Push model to huggingface hub
push_model_to_hub(
    agent=agent,
    env_name="OpenAI/Gym/Box2d",
    task_name="LunarLander-v2",
    algo_name="TD3",
    wandb_url=return_.wandb_url,
    github_repo_url="https://github.com/opendilab/DI-engine",
    github_doc_model_url="https://di-engine-docs.readthedocs.io/en/latest/12_policies/td3.html",
    github_doc_env_url="https://di-engine-docs.readthedocs.io/en/latest/13_envs/lunarlander.html",
    installation_guide="pip3 install DI-engine[common_env,fast]",
    usage_file_by_git_clone="./dizoo/common/td3/lunarlander_td3_deploy.py",
    usage_file_by_huggingface_ding="./dizoo/common/td3/lunarlander_td3_download.py",
    train_file="./dizoo/common/td3/lunarlander_td3.py",
    repo_id="OpenDILabCommunity/LunarLander-v2-TD3"
)
```
