# bash experiments/cifar100-s10.sh
# 10 tasks
# experiment settings
DATASET=mnist
N_CLASS=10

# save directory
OUTDIR=outputs/${DATASET}/5-task-new

# hard coded inputs
GPUID='0 1 2 3'
CONFIG=configs/MNIST10_prompt-s5.yaml
REPEAT=1
OVERWRITE=1

###############################################################

# process inputs
mkdir -p $OUTDIR

python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name MSPrompt \
    --prompt_param 1 8 0 0 10 0 4 \
    --clip_type ViT-B/32 \
    --memory 200 \
    --log_dir ${OUTDIR}/msp \
    --use_clip_encoder