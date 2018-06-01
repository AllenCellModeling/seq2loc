cd ..
python train_classifier.py --GPU_ids $1 --ch_intermed 256 --layers_deep 30 --downsize_on_nth 3 --resid_type sum --pooling_type max --kernel_size 3 --lr 1E-4 --dropout 0.5 --column_name "Uniprot GO id nuc or cyto"
