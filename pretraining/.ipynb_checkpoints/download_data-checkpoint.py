from huggingface_hub import snapshot_download
import os
import argparse
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument("--access-token", type=str)
args = parser.parse_args()

access_token = args.access_token

local_dir = os.getcwd()

snapshot_download(
    repo_id="ECoG-fm-team/ECoG-fm-sandbox",
    repo_type="dataset",
    revision="main",
    cache_dir="./cache",
    local_dir="./",
    local_dir_use_symlinks=False,
    resume_download=False,
    token=access_token,
)

with zipfile.ZipFile(f"{local_dir}/dataset.zip", "r") as zip_ref:
    zip_ref.extractall(local_dir)

with zipfile.ZipFile(f"{local_dir}/signal.zip", "r") as zip_ref:
    zip_ref.extractall(local_dir)

with zipfile.ZipFile(f"{local_dir}/word-embeddings.zip", "r") as zip_ref:
    zip_ref.extractall(local_dir)
