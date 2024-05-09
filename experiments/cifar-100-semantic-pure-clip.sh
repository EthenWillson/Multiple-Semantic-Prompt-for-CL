# bash experiments/cifar100-s1.sh
# 1 task
# experiment settings
DATASET=cifar100
N_CLASS=100

# save directory
OUTDIR=outputs/${DATASET}/1-task

# hard coded inputs
GPUID='0 1 2 3'
CONFIG=configs/cifar-100_prompt_s1.yaml
REPEAT=3
OVERWRITE=1

###############################################################

# process inputs
mkdir -p $OUTDIR

python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name MSPrompt \
    --prompt_param 2 8 0 0 20 0 4 \
    --log_dir ${OUTDIR}/msp \
    --use_clip_encoder \
    --semantic_exp_pure_clip \
    --use_clip_encoder \
    --use_label_encoder