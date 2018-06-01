cd ..
python train_classifier.py --GPU_ids $1 --ch_intermed 16 --layers_deep 100 --downsize_on_nth 5 --resid_type cat --pooling_type max --kernel_size 3 --lr 1E-3 --dropout 0
