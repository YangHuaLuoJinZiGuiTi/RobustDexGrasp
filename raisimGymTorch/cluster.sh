export PYTHONBUFFERED=1
export PATH=$PATH
export LD_LIBRARY_PATH=/home/mkocabas/raisim_grasp/build/lib
export LD_LIBRARY_PATH=/home/mkocabas/raisim_grasp/raisim_build/lib
export PYTHONPATH=/home/mkocabas/raisim_grasp/build/lib
export PYTHON="/home/mkocabas/miniconda3/envs/raisim/bin/python"
export DATAPATH="raisimGymTorch/cvpr_rebuttal_contactopt"

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

case $1 in
	20)
		echo "Case 20"
		$PYTHON raisimGymTorch/env/envs/mano_pred/runner.py -o 2 -e '003_crackerbox_contactopt' -c 'cfg_reg.yaml' -d $DATAPATH -nr 10
		;;
	21)
		echo "Case 21"
		$PYTHON raisimGymTorch/env/envs/mano_pred/runner.py -o 3 -e '004_sugarbox_contactopt' -c 'cfg_reg.yaml' -d $DATAPATH -nr 10
		;;
	22)
		echo "Case 22"
		$PYTHON raisimGymTorch/env/envs/mano_pred/runner.py -o 5 -e '006_mustard_contactopt' -c 'cfg_reg.yaml' -d $DATAPATH  -nr 10
		;;
	23)
		echo "Case 23"
		$PYTHON raisimGymTorch/env/envs/mano_pred/runner.py -o 9 -e '010_pottedcan_contactopt' -c 'cfg_reg.yaml' -d $DATAPATH  -nr 10
		;;
	24)
		echo "Case 24"
		$PYTHON raisimGymTorch/env/envs/mano_pred/runner.py -o 10 -e '011_banana_contactopt' -c 'cfg_reg.yaml' -d $DATAPATH  -nr 10
		;;
	25)
		echo "Case 25"
		$PYTHON raisimGymTorch/env/envs/mano_pred/runner.py -o 12 -e '021_bleach_contactopt' -c 'cfg_reg.yaml' -d $DATAPATH   -nr 10
		;;
	26)
		echo "Case 26"
		$PYTHON raisimGymTorch/env/envs/mano_pred/runner.py -o 14 -e '025_mug_contactopt' -c 'cfg_reg.yaml' -d $DATAPATH   -nr 10
		;;
	27)
		echo "Case 27"
		$PYTHON raisimGymTorch/env/envs/mano_pred/runner.py -o 15 -e '035_powerdrill_contactopt' -c 'cfg_reg.yaml' -d $DATAPATH    -nr 10
		;;
	28)
		echo "Case 28"
		$PYTHON raisimGymTorch/env/envs/mano_pred/runner.py -o 17 -e '037_scissors_contactopt' -c 'cfg_reg.yaml' -d $DATAPATH    -nr 10
		;;

esac

exit 1