cd ..
python train_seq2loc.py --GPU_ids $1 \
			--lr 1E-5 \
			--seq_dropout 0 \
			--loss L1Loss \
			--model tiramisu \
			--model_seq transformer \
			--batch_size 16 \
			--seq_nout 1024 \
			--patch_size 64
