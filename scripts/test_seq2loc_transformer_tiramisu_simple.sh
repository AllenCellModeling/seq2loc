cd ..
python train_seq2loc.py --GPU_ids $1 \
			--lr 1E-3 \
			--seq_dropout 0 \
			--seq_layers_deep 5 \
			--loss L1Loss \
			--model tiramisu_simple \
			--model_seq transformer \
			--batch_size 16 \
			--seq_nout 1024 \
			--patch_size 64
