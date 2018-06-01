cd ..
python train_seq2loc.py --GPU_ids $1 \
			--lr 1E-5 \
			--seq_dropout 0 \
			--loss L1Loss \
			--model tiramisu \
			--batch_size 64 \
			--seq_dropout 0.2 \
			--seq_ch_intermed 16 \
			--seq_nout 2048 \
			--seq_resid cat \
			--seq_pooling avg \
			--patch_size 64
