GPUID=$1
MODELNAME=ICARL-KLD
LR=8e-04

MODELMODE=3
ORIPROTOSIZE=450
ORIPROTOEACHSIZE=450
DATASETNEXTEPSIODES=450
EPOCHSPERTASK=2

CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 1 --model-name $MODELNAME  --model-mode $MODELMODE  --oriproto-size $ORIPROTOSIZE --oriproto-eachsize $ORIPROTOEACHSIZE --dataset-nextepisodes $DATASETNEXTEPSIODES --epochs-per-task $EPOCHSPERTASK  --lr $LR
CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 2 --model-name $MODELNAME  --model-mode $MODELMODE  --oriproto-size $ORIPROTOSIZE --oriproto-eachsize $ORIPROTOEACHSIZE --dataset-nextepisodes $DATASETNEXTEPSIODES --epochs-per-task $EPOCHSPERTASK --lr $LR
CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 3 --model-name $MODELNAME  --model-mode $MODELMODE  --oriproto-size $ORIPROTOSIZE --oriproto-eachsize $ORIPROTOEACHSIZE --dataset-nextepisodes $DATASETNEXTEPSIODES --epochs-per-task $EPOCHSPERTASK --lr $LR
CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 4 --model-name $MODELNAME  --model-mode $MODELMODE  --oriproto-size $ORIPROTOSIZE --oriproto-eachsize $ORIPROTOEACHSIZE --dataset-nextepisodes $DATASETNEXTEPSIODES --epochs-per-task $EPOCHSPERTASK --lr $LR
CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 5 --model-name $MODELNAME  --model-mode $MODELMODE  --oriproto-size $ORIPROTOSIZE --oriproto-eachsize $ORIPROTOEACHSIZE --dataset-nextepisodes $DATASETNEXTEPSIODES --epochs-per-task $EPOCHSPERTASK --lr $LR
#CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 6 --model-name $MODELNAME  --model-mode $MODELMODE  --oriproto-size $ORIPROTOSIZE --oriproto-eachsize $ORIPROTOEACHSIZE --dataset-nextepisodes $DATASETNEXTEPSIODES --epochs-per-task $EPOCHSPERTASK --lr $LR
#CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 7 --model-name $MODELNAME  --model-mode $MODELMODE  --oriproto-size $ORIPROTOSIZE --oriproto-eachsize $ORIPROTOEACHSIZE --dataset-nextepisodes $DATASETNEXTEPSIODES --epochs-per-task $EPOCHSPERTASK --lr $LR
#CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 8 --model-name $MODELNAME  --model-mode $MODELMODE  --oriproto-size $ORIPROTOSIZE --oriproto-eachsize $ORIPROTOEACHSIZE --dataset-nextepisodes $DATASETNEXTEPSIODES --epochs-per-task $EPOCHSPERTASK --lr $LR
#CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 9 --model-name $MODELNAME  --model-mode $MODELMODE  --oriproto-size $ORIPROTOSIZE --oriproto-eachsize $ORIPROTOEACHSIZE --dataset-nextepisodes $DATASETNEXTEPSIODES --epochs-per-task $EPOCHSPERTASK --lr $LR
#CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 10 --model-name $MODELNAME  --model-mode $MODELMODE  --oriproto-size $ORIPROTOSIZE --oriproto-eachsize $ORIPROTOEACHSIZE --dataset-nextepisodes $DATASETNEXTEPSIODES --epochs-per-task $EPOCHSPERTASK --lr $LR
