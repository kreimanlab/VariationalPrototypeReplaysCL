GPUID=$1
MODELNAME=ICARL-MSE
MODELMODE=1
LR=5e-03

CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 1 --model-name $MODELNAME  --model-mode $MODELMODE --lr $LR
CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 2 --model-name $MODELNAME  --model-mode $MODELMODE --lr $LR
CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 3 --model-name $MODELNAME  --model-mode $MODELMODE --lr $LR
CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 4 --model-name $MODELNAME  --model-mode $MODELMODE --lr $LR
CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 5 --model-name $MODELNAME  --model-mode $MODELMODE --lr $LR
#CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 6 --model-name $MODELNAME  --model-mode $MODELMODE --lr $LR
#CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 7 --model-name $MODELNAME  --model-mode $MODELMODE --lr $LR
#CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 8 --model-name $MODELNAME  --model-mode $MODELMODE --lr $LR
#CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 9 --model-name $MODELNAME  --model-mode $MODELMODE --lr $LR
#CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 10 --model-name $MODELNAME  --model-mode $MODELMODE --lr $LR
