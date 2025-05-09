# Run all commands in one shell
.ONESHELL:

USR := $(shell whoami | head -c 2)
DT := $(shell date +"%Y%m%d-%H:%M:%S")

ACCESS_TOKEN = "your hf access token"

download-data:
	$ python ECoG_MAE/download_data.py \
		--access-token $(ACCESS_TOKEN)

PREFIX = "video-mae-train"
JOB_NAME = "$(PREFIX)-$(USR)-$(DT)"

CONFIG_FILE = "configs/video_mae_train.yml"

CMD = sbatch --job-name=$(JOB_NAME) submit.sh
# to debug, request interactive gpu node via salloc and select this option:
# CMD = python

# for commands debug, use-contrastive-loss, use-cls-token: add to arguments = True, leave out = False
# --debug -> just enables verbose print out for debugging
# --env -> compute power envelope
# --dataset-path="dataset" -> Sets where training looks for the dataset. Should be a relative path.
# --train-data-proportion=0.8 -> Sets proportion of data assigned to train split. All remaining data is assigned to test.
# --use-cls-token (not implemented yet!)
# --use-contrastive-loss (not implemented yet!)
# --running-cell-masking -> specific type of decoder masking (not properly tested yet!)

model-train:
	mkdir -p logs
	$(CMD) ECoG_MAE/main.py \
		--config-file $(CONFIG_FILE);
