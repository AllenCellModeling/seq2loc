cd ..
python train_classifier.py --GPU_ids $1 --trim_to_firstlast True --ch_intermed 1024 --layers_deep 30 --downsize_on_nth 3 --resid_type sum --kernel_size 5 --lr 1E-4
