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
