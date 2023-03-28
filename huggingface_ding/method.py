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
    assert len(file_list)>=1, "No replay rendered."
    if len(file_list)==1:
        video_path = os.path.join(record_path, file_list[0])
    else: 
        # sometimes the late media file is a one-frame video generated after first to reset and then close the environment.
        video_path = os.path.join(record_path, file_list[-2])
    return video_path


def _calculate_model_params(model):
    Total_params = 0
    for param_tensor in model:
        mulValue = np.prod(model[param_tensor].size())
        Total_params += mulValue
    return Total_params

def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _get_agent_policy_state_dict(agent):
    if hasattr(agent.policy,"state_dict"):
        return agent.policy.state_dict()
    elif hasattr(agent.policy,"_state_dict_learn"):
        return agent.policy._state_dict_learn()
    else:
        raise ValueError("No state_dict method available for this Policy.")
    
# This method save, evaluate, generate a model card and record a replay video of your agent before pushing the repo to the hub
def push_model_to_hub(
    agent,
    env_name,
    task_name,
    algo_name,
    wandb_url,
    repo_id,
    usage_file_path=None,
    train_file_path=None,
    github_repo_url=None,
    github_doc_model_url=None,
    github_doc_env_url=None,
    model_description=None,
    installation_guide=None,
    create_repo=True
):

    with tempfile.TemporaryDirectory() as workfolder:
        huggingface_api = HfApi()

        torch.save(_get_agent_policy_state_dict(agent), os.path.join(Path(workfolder), "model.pth"))
        agent.deploy(enable_save_replay=True, replay_save_path=os.path.join(Path(workfolder), 'videos'))
        save_config_py(agent.cfg, os.path.join(Path(workfolder), 'policy_config.py'))
        eval_return=agent.batch_evaluate()
        with open(os.path.join(Path(workfolder), 'policy_config.py'), 'r') as file:
            python_config = file.read()
        if usage_file_path is not None and os.path.exists(usage_file_path):
            with open(usage_file_path, 'r') as file:
                python_code_for_usage = file.read()
        else:
            python_code_for_usage = ""

        if train_file_path is not None and os.path.exists(train_file_path):
            with open(train_file_path, 'r') as file:
                python_code_for_train = file.read()
        else:
            python_code_for_train = ""

        model_size = str(round(_calculate_model_params(_get_agent_policy_state_dict(agent)["model"]) / 256.0, 2)) + " KB"

        if model_description is None:
            model_description=""

        if installation_guide is None:
            installation_guide=""

        if github_repo_url is None:
            github_repo_url="https://github.com/opendilab/DI-engine"

        if github_doc_model_url is None:
            github_repo_url="https://di-engine-docs.readthedocs.io"

        if github_doc_env_url is None:
            github_repo_url="https://di-engine-docs.readthedocs.io"

        if create_repo:
            huggingface_api.create_repo(
                repo_id=repo_id,
                private=True,
            )

        model_file_url = huggingface_api.upload_file(
            path_or_fileobj=os.path.join(Path(workfolder), "model.pth"),
            path_in_repo="model.pth",
            repo_id=repo_id,
        )

        demo_file_url = huggingface_api.upload_file(
            path_or_fileobj=_find_video_file_path(os.path.join(Path(workfolder), 'videos')),
            path_in_repo="replay.mp4",
            repo_id=repo_id,
        )

        
        config_file_url = huggingface_api.upload_file(
            path_or_fileobj=os.path.join(Path(workfolder), 'policy_config.py'),
            path_in_repo='policy_config.py',
            repo_id=repo_id,
        )

        metric=[
            {
                "name":"mean_reward",
                "value":str(round(eval_return.eval_value,2))+" +/- "+str(round(eval_return.eval_value_std,2)),
                "type":"mean_reward",
            }
        ]

        card_data = ModelCardData(
            language='en',
            license='apache-2.0',
            library_name='pytorch',
            benchmark_name=env_name,
            task_name=task_name,
            tags=["deep-reinforcement-learning", "reinforcement-learning", "DI-engine", task_name],
            **{
                "pipeline_tag":"reinforcement-learning",
                "model-index":[{
                    "name":algo_name,
                    "results":[{
                        "task":{
                            "name":"reinforcement-learning",
                            "type":"reinforcement-learning",
                        },
                        "dataset":{
                            "name":'{}-{}'.format(env_name, task_name),
                            "type":'{}-{}'.format(env_name, task_name),
                        },
                        "metrics":metric
                    },]
                },]
            }
        )

        card = ModelCard.from_template(
            card_data,
            model_id='{}-{}-{}'.format(env_name, task_name, algo_name),
            algo_name=algo_name,
            model_description=model_description,
            installation_guide=installation_guide,
            developers="OpenDILab",
            config_file_url=config_file_url,
            di_engine_version=ding.__version__,
            gym_version=gym.__version__,
            pytorch_version=torch.__version__,
            date=date.today(),
            video_demo_url=demo_file_url,
            parameters_total_size=model_size,
            wandb_url=wandb_url,
            github_repo_url=github_repo_url,
            github_doc_model_url=github_doc_model_url,
            github_doc_env_url=github_doc_env_url,
            python_config=python_config,
            python_code_for_usage=python_code_for_usage,
            python_code_for_train=python_code_for_train,
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


def pull_model_from_hub(repo_id:str):

    with tempfile.TemporaryDirectory() as workfolder:

        model_file = hf_hub_download(repo_id=repo_id, filename="model.pth", cache_dir=Path(workfolder))
        policy_state_dict = torch.load(model_file, map_location=torch.device("cpu"))

        config_file = hf_hub_download(repo_id=repo_id, filename="policy_config.py", cache_dir=Path(workfolder))
        config = Config.file_to_dict(config_file)

    return policy_state_dict, EasyDict(config.cfg_dict)
