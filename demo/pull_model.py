from ding.bonus import PPOF
from huggingface_ding import pull_model_from_hub

policy_state_dict, cfg = pull_model_from_hub(repo_id="OpenDILabCommunity/LunarLander-v2-ppo")
# Create the environment
env_id = "lunarlander_discrete"
# Instantiate the agent
agent = PPOF(env=env_id, cfg=cfg.exp_config)
agent.load_policy(policy_state_dict=policy_state_dict, config=cfg.exp_config)
return_of_trainning = agent.train(step=5000)
print(return_of_trainning["wandb_url"])
