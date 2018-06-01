cd ..
python train_classifier.py --GPU_ids $1 --model_type densenet --lr 1E-3 --column_name "Uniprot GO id nuc or cyto" --dropout 0.5
