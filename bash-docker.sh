docker run -it \
  --device /dev/dri --device /dev/kfd \
  --device=/dev/infiniband --network host --ipc host \
  --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --privileged \
  -v $HOME:$HOME -v $(pwd):$(pwd) -w $(pwd) --shm-size 64G --name primus_hybrid_26.5 \
  unifiedtrainingdockers.azurecr.io/utd/ci:primus_the_rock_ci_ff632df_20260627
 
