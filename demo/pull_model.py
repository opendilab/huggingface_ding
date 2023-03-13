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
