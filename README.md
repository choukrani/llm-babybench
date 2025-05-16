# LLM-BabyBench

## Create and Activate Virtual Environment

To begin, create a Python virtual environment and activate it.

1. **Create the virtual environment:**

```
python -m venv llm-babybench
```

2. **Activate the environment:**

```
source llm-babybench/bin/activate
```

3. **Install the requirements:**

```
pip install -r requirement
```

## Install Minigrid

Next, clone the Minigrid repository and install the necessary dependencies.

1. **Clone the Minigrid repository:**

```
git clone https://github.com/Farama-Foundation/Minigrid.git
```

1. **Clone the Minigrid repository:**

```
git clone https://github.com/Farama-Foundation/Minigrid.git
```

2. **Checkout the specific commit:**

After cloning the repository, navigate into the project directory and checkout the commit hash:

```
cd Minigrid
git checkout 6e713afef8d23d5280ebf28fb3fcf635d40d6a7f
```
This ensures that you are using the exact version of the repository.

3. **Navigate into the Minigrid directory:**

```
cd Minigrid
```

4. **Install Minigrid in editable mode:**

```
python3 -m pip install -e .
```

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{svrbench2025,
  author = {Choukrani, Omar and Malek, Idriss and Orel, Daniil and Xie, Zhuohan and Iklassov, Zangir and Takáč, Martin and Lahlou, Salem},
  title = {SVRPBench: A Benchmark for Stochastic Vehicle Routing Problems},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/choukrani/llm-babybench}}
}
``` 
