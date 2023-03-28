from ding.bonus import TD3OffPolicyAgent
from huggingface_ding import push_model_to_hub

# Instantiate the agent
agent = TD3OffPolicyAgent("lunarlander_continuous", exp_name="LunarLander-v2-TD3")
# Train the agent
return_ = agent.train(step=int(200000), collector_env_num=4, evaluator_env_num=4)
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
    usage_file_path="./dizoo/common/td3/lunarlander_td3_download.py",
    train_file_path="./dizoo/common/td3/lunarlander_td3.py",
    repo_id="OpenDILabCommunity/LunarLander-v2-TD3"
)
