nvidia-docker run \
	-it \
	-v /allen:/root/allen \
	-v /allen/aics/modeling/gregj/projects:/root/projects \
	-v /allen/aics/modeling/gregj/data:/root/data \
	-v /allen/aics/modeling/gregj/results:/root/results  \
	gregj/pytorch_extras:jupyter_dgx bash 
