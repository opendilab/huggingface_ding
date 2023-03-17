from setuptools import setup

install_requires = [
    "huggingface_hub",
    "DI-engine",
    "easydict",
    "torch",
]

setup(
    name='huggingface_ding',
    version='0.0.1',
    packages=['huggingface_ding'],
    url='https://github.com/opendilab/huggingface_ding',
    license='Apache',
    author='OpenDILab Team',
    author_email='opendilab@pjlab.org.cn',
    description='API for DI-engine to push and pull models from the Huggingface Hub.',
    install_requires=install_requires,
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="huggingface reinforcement learning deep reinforcement learning RL",
)
