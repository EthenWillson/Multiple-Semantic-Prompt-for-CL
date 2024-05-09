# bash experiments/cifar100-s10.sh
# 10 tasks
# experiment settings
DATASET=cifar100
N_CLASS=100

# save directory
OUTDIR=outputs/${DATASET}/10-task-new

# hard coded inputs
GPUID='0 1 2 3'
CONFIG=configs/cifar-100_prompt.yaml
REPEAT=1
OVERWRITE=1

###############################################################

# process inputs
mkdir -p $OUTDIR

python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name MSPrompt \
    --prompt_param 5 8 0 0 20 0 4 \
    --log_dir ${OUTDIR}/msp \
    --use_clip_encoder