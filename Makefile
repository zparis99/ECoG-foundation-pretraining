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

MODEL_NAMES := patch_dim_1_small patch_dim_1_medium patch_dim_1_large patch_dim_2_small patch_dim_2_medium patch_dim_2_large

foundation-model-list:
	@for item in $(MODEL_NAMES); do \
		$(MAKE) model-train PREFIX=$$item MODEL_NAME=$$item; \
	done

model-train:
	mkdir -p logs
	$(CMD) pretraining/main.py \
		--config=configs/video_mae_train.yml \
		--video_mae_task_config.model_name=$(MODEL_NAME);

model-analysis:
	mkdir -p logs
	sbatch --job-name=$(JOB_NAME) high_mem_submit.sh pretraining/analyze_file_outputs.py \
		--checkpoint_dir=checkpoints/$(TRIAL_NAME)/best_checkpoint \
		--samples_dir=results/samples --out=$(TRIAL_NAME).pth
