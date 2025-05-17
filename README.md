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
pip install -r requirements.txt
```

## Install Minigrid

Next, clone the Minigrid repository and install the necessary dependencies.

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

This ensures that you are using the exact version of this repository.

3. **Install Minigrid in editable mode:**

In the same directory, run:

```
python3 -m pip install -e .
```
