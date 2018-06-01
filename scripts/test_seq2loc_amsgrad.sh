cd ..
python train_seq2loc.py --GPU_ids $1 --lr 1E-4 --seq_dropout 0 --loss L1Loss --amsgrad True
