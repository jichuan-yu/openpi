## Installation

**Install uv:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Create conda env:**

```bash
conda create -y -n pi python=3.11
conda activate pi
```

**Env setup using uv:**

```
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```



## Training

**Specify your own DataConfig and TrainConfig in `training/config.py`**











