export PYTHONBUFFERED=1
export PATH=$PATH
export LD_LIBRARY_PATH=/home/mkocabas/raisim_grasp/build/lib
export LD_LIBRARY_PATH=/home/mkocabas/raisim_grasp/raisim_build/lib
export PYTHONPATH=/home/mkocabas/raisim_grasp/build/lib
export PYTHON="/home/mkocabas/miniconda3/envs/raisim/bin/python"
export DATAPATH="raisimGymTorch/exp_rebuttal"

echo $LD_LIBRARY_PATH
echo $PYTHONPATH

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/mkocabas/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/mkocabas/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/mkocabas/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/mkocabas/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate raisim
echo "Activated raisim conda env"
pwd

$PYTHON setup.py develop
conda install -c anaconda dropbox