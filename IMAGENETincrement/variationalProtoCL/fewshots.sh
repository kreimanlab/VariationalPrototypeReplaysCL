GPUID=$1
MODELNAME=VP

CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 1
CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 2
CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 3
CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 4
CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 5
CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 6
CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 7
CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 8
CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 9
CUDA_VISIBLE_DEVICES=$GPUID python main.py --n-repeats 10
