GPUID=$1
OUTDIR=outputs/split_IMAGENET_incre_class
NUM_TRAIN_SAMS=1000
REPEAT=10
mkdir -p $OUTDIR


python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --num_train_sams $NUM_TRAIN_SAMS   --force_out_dim 100 --no_class_remap --first_split_size 10 --other_split_size 10 --schedule 20 --batch_size 128 --model_name LeNetC --agent_type customization  --agent_name EWC_mnist        --lr 0.001 --reg_coef 600      | tee ${OUTDIR}/EWC.log
python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --num_train_sams $NUM_TRAIN_SAMS   --force_out_dim 100 --no_class_remap --first_split_size 10 --other_split_size 10 --schedule 20 --batch_size 128 --model_name LeNetC --agent_type customization  --agent_name EWC_online_mnist --lr 0.001 --reg_coef 100      | tee ${OUTDIR}/EWC_online.log
python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --num_train_sams $NUM_TRAIN_SAMS    --force_out_dim 100 --no_class_remap --first_split_size 10 --other_split_size 10 --schedule 20 --batch_size 128 --model_name LeNetC --agent_type regularization --agent_name SI  --lr 0.001 --reg_coef 600      | tee ${OUTDIR}/SI.log
python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam     --num_train_sams $NUM_TRAIN_SAMS  --force_out_dim 100 --no_class_remap --first_split_size 10 --other_split_size 10 --schedule 20 --batch_size 128 --model_name LeNetC --agent_type regularization --agent_name L2  --lr 0.001 --reg_coef 100      | tee ${OUTDIR}/L2.log
python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --num_train_sams $NUM_TRAIN_SAMS   --force_out_dim 100 --no_class_remap --first_split_size 10 --other_split_size 10 --schedule 20 --batch_size 128 --model_name LeNetC --agent_type regularization --agent_name MAS --lr 0.001 --reg_coef 1        |tee  ${OUTDIR}/MAS.log
python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer SGD    --num_train_sams $NUM_TRAIN_SAMS    --force_out_dim 100 --no_class_remap --first_split_size 10 --other_split_size 10 --schedule 20 --batch_size 128 --model_name LeNetC                                              --lr 0.001                      | tee ${OUTDIR}/SGD.log
