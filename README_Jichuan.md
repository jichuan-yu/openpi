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

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```



## Training

**Specify your train config in `training/config.py`**

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_kinova

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_kinova --exp-name=20260326_0010 --overwrite
```









