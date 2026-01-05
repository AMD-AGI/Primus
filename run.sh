


./runner/primus-cli slurm srun \
       -N "${NNODES:-2}" \
    --exclusive \
    --export ALL \
    --nodelist="node[05,06,08]" \
    --ntasks-per-node=1 \
    --cpus-per-task="${CPUS_PER_TASK:-128}" \
    -- container \
        --clean \
        --image "docker.io/tasimage/primus:pr-464-v25.09-ainic" \
    -- \
        --env "NCCL_SOCKET_IFNAME=enp193s0f1np1" \
        --env "GLOO_SOCKET_IFNAME=enp193s0f1np1" \
        --env "runner/helpers/envs/enable_ainic.sh" \
    preflight 

