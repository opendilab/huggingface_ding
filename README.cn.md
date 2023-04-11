# Hugging Face 🤗 x OpenDILab/DI-engine

[English](./README.md) | 简体中文

> Huggingface_ding 代码库用于提供 Huggingface Hub 的 API 封装，可以用于快速拉取 OpenDILab/DI-engine 的公开模型，或是可以将使用 OpenDILab/DI-engine 训练的模型推送至 Huggingface Hub.

<!-- toc -->

- [安装方法](#安装方法)
- [案例](#案例)
  - [下载模型](#下载模型)
  - [上传模型](#上传模型)
- [API说明](#api说明)

# 安装方法
## 使用 pip 安装
```
pip install -e .
```

# 案例
## 下载模型

从 Huggingface Hub 下载一个模型，部署评估，并渲染生成对应的视频回放：
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

## 上传模型

使用 DI-engine 训练了一个模型，并将其推送至 Huggingface Hub，自动制作生成 Modelcard
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
    usage_file_by_git_clone="./td3/lunarlander_td3_deploy.py",
    usage_file_by_huggingface_ding="./td3/lunarlander_td3_download.py",
    train_file="./td3/lunarlander_td3.py",
    repo_id="OpenDILabCommunity/LunarLander-v2-TD3"
)
```

# API说明

**pull_model_from_hub**

参数列表:

- agent (:obj:`object`): 需要上传的智能体
- env_name (:obj:`str`): 任务所属的环境名
- task_name (:obj:`str`): 智能体决策的任务名
- algo_name (:obj:`str`): 算法名
- wandb_url (:obj:`str`): wandb 网页地址，用于记录训练过程
- repo_id (:obj:`str`): Huggingface Hub 仓库 ID
- usage_file_by_git_clone (:obj:`str`): 描述从 Huggingface Hub 使用 git clone 方法下载至本地的模型应该如何使用的文件的路径
- usage_file_by_huggingface_ding (:obj:`str`): 描述使用 huggingface ding 下载的模型应该如何使用的文件的路径
- train_file (:obj:`str`): 描述该模型是如何被训练的文件的路径
- github_repo_url (:obj:`str`): github 网页地址，用于指示该模型所依赖的 DI-engine 仓库
- github_doc_model_url (:obj:`str`): github 或相关文档的网页地址，用于描述模型
- github_doc_env_url (:obj:`str`): github 或相关文档的网页地址，用于描述任务环境
- model_description (:obj:`str`): 一段描述该模型的简单介绍
- installation_guide (:obj:`str`): 描述任务环境的安装方法
- create_repo (:obj:`bool`): 是否需要创建 Huggingface Hub 仓库

**push_model_to_hub**

参数列表:

- repo_id (:obj:`str`): Huggingface Hub 仓库 ID