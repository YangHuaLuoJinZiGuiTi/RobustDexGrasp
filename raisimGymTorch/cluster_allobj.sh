export PYTHONBUFFERED=1
export PATH=$PATH
export LD_LIBRARY_PATH=/home/mkocabas/raisim_grasp/build/lib
export LD_LIBRARY_PATH=/home/mkocabas/raisim_grasp/raisim_build/lib
export PYTHONPATH=/home/mkocabas/raisim_grasp/build/lib
export PYTHON="/home/mkocabas/miniconda3/envs/raisim/bin/python"
export DATAPATH="raisimGymTorch/cvpr_post_rebuttal"

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
		$PYTHON raisimGymTorch/env/envs/mano_pred/runner.py  -e 'alllabelsx1_contactopt_mi' -c 'cfg_allobj.yaml' -ao -nr 1 -df 1  -d $DATAPATH -itr 10001 -mi -to 3
		;;
  21)
    echo "Case 21"
		$PYTHON raisimGymTorch/env/envs/mano_pred/runner.py  -e 'alllabelsx1_contactopt_mi' -c 'cfg_allobj.yaml' -ao -nr 1  -df 1  -d $DATAPATH -itr 10001 --seed 2 -mi -to 3
		;;
#	12)
#		echo "Case 12"
#		$PYTHON raisimGymTorch/env/envs/mano_gen/runner.py  -e 'alllabelsx2_corrected' -c 'cfg_allobj.yaml' -ao -nr 2   -d $DATAPATH -itr 10001 -to 10
#		;;
#  13)
#    echo "Case 13"
#		$PYTHON raisimGymTorch/env/envs/mano_gen/runner.py -e 'alllabelsx2_corrected' -c 'cfg_allobj.yaml' -ao  -nr 2 -d $DATAPATH -itr 10001 --seed 2 -to 10
#		;;
#  14)
#    echo "Case 14"
#		$PYTHON raisimGymTorch/env/envs/mano_gen/runner.py -e 'alllabelsx2_corrected' -c 'cfg_allobj.yaml' -ao  -nr 2 -d $DATAPATH -itr 10001 --seed 3 -to 10
#		;;
#  15)
#    echo "Case 15"
#		$PYTHON raisimGymTorch/env/envs/mano_gen/runner.py -e 'alllabelsx2_corrected' -c 'cfg_allobj.yaml' -ao  -nr 2 -d $DATAPATH -itr 10001 --seed 4 -to 10
#		;;
#	16)
#		echo "Case 16"
#		$PYTHON raisimGymTorch/env/envs/mano_pred/runner.py  -e 'alllobjx2_gtta' -c 'cfg_allobj.yaml' -ao -nr 2 -df 1  -d $DATAPATH -itr 10001
#		;;
#  17)
#    echo "Case 17"
#		$PYTHON raisimGymTorch/env/envs/mano_pred/runner.py  -e 'alllobjx2_gtta' -c 'cfg_allobj.yaml' -ao -nr 2 -df 1  -d $DATAPATH -itr 10001 --seed 2
#		;;
#  18)
#    echo "Case 18"
#		$PYTHON raisimGymTorch/env/envs/mano_pred/runner.py  -e 'alllobjx2_gtta_mi' -c 'cfg_allobj.yaml' -ao -nr 2 -df 1  -d $DATAPATH -itr 10001 -mi
#		;;
#  19)
#    echo "Case 19"
#		$PYTHON raisimGymTorch/env/envs/mano_pred/runner.py  -e 'alllobjx2_gtta_mi' -c 'cfg_allobj.yaml' -ao -nr 2 -df 1  -d $DATAPATH -itr 10001 -mi  --seed 2
#		;;
#	1)
#		echo "Case 84"
#		$PYTHON raisimGymTorch/env/envs/mano_gen/runner.py -e 'allobjx2_corrected_t8' -c 'cfg_allobj.yaml' -ao -to 8 -nr 2 -d $DATAPATH -itr 10001
#		;;
#  85)
#    echo "Case 85"
#		$PYTHON raisimGymTorch/env/envs/mano_gen/runner.py -e 'allobjx2_corrected_t9' -c 'cfg_allobj.yaml' -ao -to 9 -nr 2 -d $DATAPATH -itr 10001
#		;;
#	16)
#		echo "Case 16"
#		$PYTHON raisimGymTorch/env/envs/mano_pred/runner.py  -e 'allobjx2_contactopt_t1' -c 'cfg_allobj.yaml' -ao -nr 2 -to 1 -df 1  -d $DATAPATH -itr 10001
#		;;
#  17)
#    echo "Case 17"
#		$PYTHON raisimGymTorch/env/envs/mano_pred/runner.py  -e 'allobjx2_contactopt_t0' -c 'cfg_allobj.yaml' -ao -nr 2 -to 0 -df 1  -d $DATAPATH -itr 10001
#		;;
#  18)
#    echo "Case 18"
#		$PYTHON raisimGymTorch/env/envs/mano_pred/runner.py  -e 'allobjx2_contactopt_t2' -c 'cfg_allobj.yaml' -ao -nr 2 -to 2 -df 1  -d $DATAPATH -itr 10001
#		;;
#	19)
#		echo "Case 18"
#		$PYTHON raisimGymTorch/env/envs/mano_gen/runner.py -e 'allobjx2_corrected_t7' -c 'cfg_allobj.yaml' -ao -to 7 -nr 2 -d $DATAPATH -itr 10001
#		;;
#	80)
#		echo "Case 19"
#		$PYTHON raisimGymTorch/env/envs/mano_gen/runner.py -e 'allobjx2_corrected_t8' -c 'cfg_allobj.yaml' -ao -to 8 -nr 2 -d $DATAPATH -itr 10001
#		;;
#	20)
#		echo "Case 20"
#		$PYTHON raisimGymTorch/env/envs/mano_gen/runner.py -e 'allobjx2_corrected_t9' -c 'cfg_allobj.yaml' -ao -to 9 -nr 2 -d $DATAPATH -itr 10001
#		;;

esac

exit 1