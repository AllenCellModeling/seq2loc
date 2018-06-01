cd ..
python train_classifier.py --GPU_ids $1 --ch_intermed 512 --layers_deep 15 --downsize_on_nth 2 --resid_type sum --kernel_size 5 --lr 1E-4 --sequence_type newsgroups
