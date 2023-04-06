# Hugging Face 🤗 x OpenDILab/DI-engine

English | [简体中文](./README.cn.md)

> Huggingface_ding is a library for OpenDILab/DI-engine user to push and pull models from the Huggingface Hub.

<!-- toc -->

- [Installation](#Installation)
- [Examples](#examples)
  - [Download Model](#download-model)
  - [Upload Model](#upload-model)
- [API](#api)

# Installation
## With pip
```
pip install -e .
```

# Examples
## Download Model

I want to download a model from the Hub
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

## Upload Model

I trained an agent and want to upload it to the Hub
```python
from ding.bonus import TD3Agent
from huggingface_ding import push_model_to_hub

# Instantiate the agent
agent = TD3Agent("lunarlander_continuous", exp_name="LunarLander-v2-TD3")
# Train the agent
return_ = agent.train(step=int(4000000), collector_env_num=4, evaluator_env_num=4)
# Push model to huggingface hub
push_model_to_hub(
    agent=agent.best,
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

# API

**pull_model_from_hub**

Arguments:

- agent (:obj:`object`): the agent instance to be uploaded.
- env_name (:obj:`str`): the name of environment in which the task is contained. 
- task_name (:obj:`str`): the name of task for which the agent is designed. 
- algo_name (:obj:`str`): the policy name of the agent.
- wandb_url (:obj:`str`): the wandb url of the trainning process.
- repo_id (:obj:`str`): the repository id of Huggingface Hub where the model is stored.
- usage_file_by_git_clone (:obj:`str`): the path of a python file which describes ways to use the OpenDILab/DI-engine model that git cloned from huggingface hub.
- usage_file_by_huggingface_ding (:obj:`str`): the path of a python file which describes ways to use the OpenDILab/DI-engine model that downloaded by huggingface ding.
- train_file (:obj:`str`): the path of a python file which describes how this model is trained.
- github_repo_url (:obj:`str`): the github url of the DI-engine repository which the model is used.
- github_doc_model_url (:obj:`str`): the github or document url of the model used.
- github_doc_env_url (:obj:`str`): the github or document url of the environment.
- model_description (:obj:`str`): a paragraph of description to the model.
- installation_guide (:obj:`str`): the guide for installation.
- create_repo (:obj:`bool`): whether to create a new repository in huggingface hub.

**push_model_to_hub**

Arguments:

- repo_id (:obj:`str`): the repository id of Huggingface Hub where the model is stored.
