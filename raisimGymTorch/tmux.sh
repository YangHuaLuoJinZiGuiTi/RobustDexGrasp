for i in $(seq 20 21); do
    echo $i;
    TMUX=''
    tmux \
        new -s "dgrasp_exp_$i" \; \
        send-keys "sh int_sess.sh" Enter \; \
        send-keys "sh cluster_allobj.sh $i 2>&1 | tee -a log_$i" Enter \; \
        detach-client \; pipe-pane "cat > /home/mkocabas/raisim_grasp_ssh/raisimGymTorch/logs/tmux_log_$i.txt"
done

exit 1