# bash experiments/imagenet-r-s1.sh
# experiment settings
DATASET=ImageNet_R
N_CLASS=200

# save directory
OUTDIR=outputs/${DATASET}/1-task-new

# hard coded inputs
GPUID='0 1 2 3'
CONFIG=configs/imnet-r_prompt-s1.yaml
REPEAT=1
OVERWRITE=1

###############################################################

# process inputs
mkdir -p $OUTDIR

# MSP
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name MSPrompt \
    --prompt_param 5 8 0 0 20 400 4 \
    --log_dir ${OUTDIR}/msp \
    --use_clip_encoder