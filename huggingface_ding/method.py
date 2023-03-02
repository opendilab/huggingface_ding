import os
import tempfile
import numpy as np
import torch
from pathlib import Path
from huggingface_hub import ModelCard, ModelCardData
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download, snapshot_download

from datetime import date
import gym
import ding
from ding.config import Config, save_config_py
from easydict import EasyDict


def _find_video_file_path(record_path):
    file_list = []
    for p in os.listdir(record_path):
        if os.path.splitext(p)[-1] == ".mp4":
            file_list.append(p)
    file_list.sort(key=lambda fn: os.path.getmtime(os.path.join(record_path, fn)))
    video_path = os.path.join(record_path, file_list[-2])
    return video_path


def _calculate_model_params(model):
    Total_params = 0
    for param_tensor in model:
        mulValue = np.prod(model[param_tensor].size())
        Total_params += mulValue
    return Total_params


# This method save, evaluate, generate a model card and record a replay video of your agent before pushing the repo to the hub
def push_model_to_hub(
    agent, env_name, task_name, algo_name, wandb_url, github_repo_url, repo_id, model_description, create_repo=True
):

    with tempfile.TemporaryDirectory() as workfolder:
        huggingface_api = HfApi()

        if create_repo:
            huggingface_api.create_repo(
                repo_id=repo_id,
                private=True,
            )

        torch.save(agent.policy.state_dict(), os.path.join(Path(workfolder), "model.pth"))
        model_file_url = huggingface_api.upload_file(
            path_or_fileobj=os.path.join(Path(workfolder), "model.pth"),
            path_in_repo="model.pth",
            repo_id=repo_id,
        )

        agent.batch_evaluate(1, render=True, replay_video_path=os.path.join(Path(workfolder), 'video'))
        demo_file_url = huggingface_api.upload_file(
            path_or_fileobj=_find_video_file_path(os.path.join(Path(workfolder), 'video')),
            path_in_repo="demo_video.mp4",
            repo_id=repo_id,
        )

        save_config_py(agent.cfg, os.path.join(Path(workfolder), 'policy_config.py'))
        config_file_url = huggingface_api.upload_file(
            path_or_fileobj=os.path.join(Path(workfolder), 'policy_config.py'),
            path_in_repo='policy_config.py',
            repo_id=repo_id,
        )

        with open(os.path.join(Path(workfolder), 'policy_config.py'), 'r') as file:
            python_config = file.read()

        card_data = ModelCardData(
            language='en',
            license='apache-2.0',
            library_name='pytorch',
            benchmark_name=env_name,
            task_name=task_name,
            tags=["deep-reinforcement-learning", "reinforcement-learning", "DI-engine", task_name],
            pipeline_tag="reinforcement-learning",
        )

        card = ModelCard.from_template(
            card_data,
            model_id='{}-{}-{}'.format(env_name, task_name, algo_name),
            model_description=model_description,
            developers="OpenDILab",
            configuration_path=config_file_url,
            di_engine_version=ding.__version__,
            gym_version=gym.__version__,
            pytorch_version=torch.__version__,
            date=date.today(),
            demo=demo_file_url,
            parameters_total_size=str(_calculate_model_params(agent.policy.state_dict()["model"])),
            wandb_url=wandb_url,
            github_repo_url=github_repo_url,
            python_config=python_config,
            template_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "modelcard_huggingface_ding_template.md"
            ),
        )

        try:
            card.validate()
            card.save('README.md')
            card.push_to_hub(repo_id=repo_id)
        except:
            raise ValueError("model card info is invalid. please check.")


def pull_model_from_hub(repo_id):

    with tempfile.TemporaryDirectory() as workfolder:

        model_file = hf_hub_download(repo_id=repo_id, filename="model.pth", cache_dir=Path(workfolder))
        policy_state_dict = torch.load(model_file)

        config_file = hf_hub_download(repo_id=repo_id, filename="policy_config.py", cache_dir=Path(workfolder))
        config = Config.file_to_dict(config_file)

    return policy_state_dict, EasyDict(config.cfg_dict)
