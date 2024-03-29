import os
import tempfile
import numpy as np
import torch
import logging
from pathlib import Path
from huggingface_hub import ModelCard, ModelCardData
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download, snapshot_download

from datetime import date
import gym
import ding
from ding.config import Config, save_config_py
from easydict import EasyDict


def _find_video_file_path(record_path, file_name=None):
    if file_name is not None:
        video_path = os.path.join(record_path, file_name)
        assert os.path.exists(video_path), "No replay rendered."
        return video_path
    else:
        file_list = []
        for p in os.listdir(record_path):
            if os.path.splitext(p)[-1] == ".mp4":
                file_list.append(p)
        file_list.sort(key=lambda fn: os.path.getmtime(os.path.join(record_path, fn)))
        assert len(file_list) >= 1, "No replay rendered."
        if len(file_list) == 1:
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
    if hasattr(agent.policy, "state_dict"):
        return agent.policy.state_dict()
    elif hasattr(agent.policy, "_state_dict_learn"):
        return agent.policy._state_dict_learn()
    else:
        raise ValueError("No state_dict method available for this Policy.")


def _huggingface_api_upload_file(huggingface_api, path_or_fileobj, path_in_repo, repo_id, retry=5):
    while retry > 0:
        try:
            file_url=huggingface_api.upload_file(
                path_or_fileobj=path_or_fileobj,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
            )
            return(file_url)
        except:
            retry -= 1
            assert retry > 0, "Huggingface Hub upload retry exceeded limit."


# This method save, evaluate, generate a model card and record a replay video of your agent before pushing the repo to the hub
def push_model_to_hub(
    agent,
    env_name,
    task_name,
    algo_name,
    repo_id,
    wandb_url=None,
    usage_file_by_git_clone=None,
    usage_file_by_huggingface_ding=None,
    train_file=None,
    github_repo_url=None,
    github_doc_model_url=None,
    github_doc_env_url=None,
    model_description=None,
    installation_guide=None,
    platform_info=None,
    create_repo=True
):
    """
    Overview:
        Push OpenDILab/DI-engine models into Huggingface Hub
    Arguments:
        - agent (:obj:`object`): the agent instance to be uploaded.
        - env_name (:obj:`str`): the name of environment in which the task is contained. 
        - task_name (:obj:`str`): the name of task for which the agent is designed. 
        - algo_name (:obj:`str`): the policy name of the agent.
        - repo_id (:obj:`str`): the repository id of Huggingface Hub where the model is stored.
        - wandb_url (:obj:`str`): the wandb url of the trainning process.
        - usage_file_by_git_clone (:obj:`str`): the path of a python file which describes ways to use the \
            OpenDILab/DI-engine model that git cloned from huggingface hub.
        - usage_file_by_huggingface_ding (:obj:`str`): the path of a python file which describes ways to use \
            the OpenDILab/DI-engine model that downloaded by huggingface ding.
        - train_file (:obj:`str`): the path of a python file which describes how this model is trained.
        - github_repo_url (:obj:`str`): the github url of the DI-engine repository which the model is used.
        - github_doc_model_url (:obj:`str`): the github or document url of the model used.
        - github_doc_env_url (:obj:`str`): the github or document url of the environment.
        - model_description (:obj:`str`): a paragraph of description to the model.
        - installation_guide (:obj:`str`): the guide for installation.
        - platform_info (:obj:`str`): the platform information.
        - create_repo (:obj:`bool`): whether to create a new repository in huggingface hub.
    """
    with tempfile.TemporaryDirectory() as workfolder:
        huggingface_api = HfApi()

        torch.save(_get_agent_policy_state_dict(agent), os.path.join(Path(workfolder), "pytorch_model.bin"))
        deploy_return_ = agent.deploy(
            enable_save_replay=True, 
            concatenate_all_replay=True, 
            replay_save_path=os.path.join(Path(workfolder), f'videos'),
            seed=[0,1,2,3,4,5,6,7,8,9]
        )
        best_video_path = _find_video_file_path(os.path.join(Path(workfolder), f'videos'),file_name='deploy.mp4')
        if hasattr(agent, "origin_cfg"):
            save_config_py(agent.origin_cfg, os.path.join(Path(workfolder), 'policy_config.py'))
        elif hasattr(agent, "cfg"):
            save_config_py(agent.cfg, os.path.join(Path(workfolder), 'policy_config.py'))
        else:
            raise ValueError("No config available for this Agent.")
        with open(os.path.join(Path(workfolder), 'policy_config.py'), 'r') as file:
            python_config = file.read()
        if usage_file_by_git_clone is not None and os.path.exists(usage_file_by_git_clone):
            with open(usage_file_by_git_clone, 'r') as file:
                usage_by_git_clone = file.read()
        else:
            usage_by_git_clone = ""

        if usage_file_by_huggingface_ding is not None and os.path.exists(usage_file_by_huggingface_ding):
            with open(usage_file_by_huggingface_ding, 'r') as file:
                usage_by_huggingface_ding = file.read()
        else:
            usage_by_huggingface_ding = ""

        if train_file is not None and os.path.exists(train_file):
            with open(train_file, 'r') as file:
                python_code_for_train = file.read()
        else:
            python_code_for_train = ""

        model_size = str(
            round(_calculate_model_params(_get_agent_policy_state_dict(agent)["model"]) / 256.0, 2)
        ) + " KB"

        if model_description is None:
            model_description = ""

        if installation_guide is None:
            installation_guide = "<TODO>"

        if wandb_url is None:
            wandb_url = "<TODO>"

        if github_repo_url is None:
            github_repo_url = "<TODO>"

        if github_doc_model_url is None:
            github_doc_model_url = "<TODO>"

        if github_doc_env_url is None:
            github_doc_env_url = "<TODO>"

        if create_repo:
            huggingface_api.create_repo(
                repo_id=repo_id,
                private=True,
            )

        model_file_url = _huggingface_api_upload_file(
            huggingface_api=huggingface_api,
            path_or_fileobj=os.path.join(Path(workfolder), "pytorch_model.bin"),
            path_in_repo="pytorch_model.bin",
            repo_id=repo_id,
        )

        demo_file_url = _huggingface_api_upload_file(
            huggingface_api=huggingface_api,
            path_or_fileobj=best_video_path,
            path_in_repo="replay.mp4",
            repo_id=repo_id,
        )

        config_file_url = _huggingface_api_upload_file(
            huggingface_api=huggingface_api,
            path_or_fileobj=os.path.join(Path(workfolder), 'policy_config.py'),
            path_in_repo='policy_config.py',
            repo_id=repo_id,
        )

        metric = [
            {
                "name": "mean_reward",
                "value": str(round(deploy_return_.eval_value, 2)) + " +/- " + str(round(deploy_return_.eval_value_std, 2)),
                "type": "mean_reward",
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
                "pipeline_tag": "reinforcement-learning",
                "model-index": [
                    {
                        "name": algo_name,
                        "results": [
                            {
                                "task": {
                                    "name": "reinforcement-learning",
                                    "type": "reinforcement-learning",
                                },
                                "dataset": {
                                    "name": task_name,
                                    "type": task_name,
                                },
                                "metrics": metric
                            },
                        ]
                    },
                ]
            }
        )

        card = ModelCard.from_template(
            card_data,
            model_id='{}-{}-{}'.format(env_name, task_name, algo_name),
            algo_name=algo_name,
            platform_info=platform_info,
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
            usage_by_git_clone=usage_by_git_clone,
            usage_by_huggingface_ding=usage_by_huggingface_ding,
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


def pull_model_from_hub(repo_id: str):
    """
    Overview:
        Pull public available models from Huggingface Hub
    Arguments:
        - repo_id (:obj:`str`): the repository id of Huggingface Hub where the model is stored.
    """
    with tempfile.TemporaryDirectory() as workfolder:

        model_file = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin", cache_dir=Path(workfolder))
        policy_state_dict = torch.load(model_file, map_location=torch.device("cpu"))

        config_file = hf_hub_download(repo_id=repo_id, filename="policy_config.py", cache_dir=Path(workfolder))
        config = Config.file_to_dict(config_file)

    return policy_state_dict, EasyDict(config.cfg_dict)
