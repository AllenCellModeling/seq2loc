nvidia-docker run -d \
        -v /allen:/root/allen \
        -v /allen/aics/modeling/gregj/projects:/root/projects \
        -v /allen/aics/modeling/gregj/data:/root/data \
        -v /allen/aics/modeling/gregj/results:/root/results  \
        gregj/pytorch_extras:jupyter_dgx bash -c "pip install joblib; cd ~/projects/attention-is-all-you-need-pytorch; pip install -e ./; cd ~/projects/seq2loc/scripts;  bash $1 $2"
