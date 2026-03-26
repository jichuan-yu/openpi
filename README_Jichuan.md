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

**[Optional] Install lerobot v3:**

```bash
cd ..
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout eff8a6fd12715edcb0a0facc10f17d53a258fa2c
pip install -e ".[all]" --use-deprecated=legacy-resolver
```

**Install lerobot dataset converter:**

```bash
cd ..
git clone https://github.com/Tavish9/any4lerobot.git
```


**Install ffmpeg:**
```bash
conda install ffmpeg=6.0 -c conda-forge
```

## Training

**国内有时需要搭梯子：**

```bash
export https_proxy=http://127.0.0.1:7890
```

**1. Specify your train config in `training/config.py`**

**2. Convert dataset version:**

```bash
cd /home/jichuan/projects/any4lerobot/ds_version_convert/v30_to_v21

uv run convert_dataset_v30_to_v21.py \
    --repo-id=dummy \
    --root=/home/jichuan/projects/openpi/dataset/20260221_T00-00-01-00_merge_last_frame
```

**3. Train**

```bash
cd /home/jichuan/projects/openpi

export HF_LEROBOT_HOME=/home/jichuan/projects/openpi/dataset
uv run scripts/compute_norm_stats.py --config-name pi0_kinova

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_kinova --exp-name=20260326_0010 --overwrite
```



## Sync Data

From my laptop to 801:
```bash
rsync -avz --progress ./dataset/20260221_T00-00-01-00_merge_last_frame jichuan@183.173.89.223:/home/jichuan/projects/openpi/dataset
```





