#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -x

BASE_IMAGE=${BASE_IMAGE:-docker.io/rocm/primus:v25.10}
ROCSHMEM_COMMIT=${ROCSHMEM_COMMIT:-release/rocm-rel-7.2}

build_rocshmem() {
	# build rocSHMEM
	apt update && apt install rdma-core libibverbs-dev ibverbs-utils -y
	cd /tmp && git clone https://github.com/ROCm/rocSHMEM.git
	cd rocSHMEM
	git checkout "${ROCSHMEM_COMMIT}"

	mkdir build && cd build

	MPI_ROOT=/opt/ompi UCX_ROOT=/opt/ucx INSTALL_PREFIX=/io/rocshmem ../scripts/build_configs/gda \
		-DGDA_IONIC=ON \
		-DGDA_MLX5=ON \
		-DGDA_BNXT=ON \
		-DUSE_IPC=ON

}


docker run --rm \
	--ipc=host \
	--network=host \
	--device=/dev/kfd \
	--device=/dev/dri \
	--device=/dev/infiniband \
	--cap-add=SYS_PTRACE \
	--cap-add=CAP_SYS_ADMIN \
	--security-opt seccomp=unconfined \
	--group-add video \
	--privileged \
	-v "$PWD":/io \
	-e ROCSHMEM_COMMIT="${ROCSHMEM_COMMIT}" \
	-e FUNCTION_DEF="$(declare -f build_rocshmem)" \
	"$BASE_IMAGE" /bin/bash -c '
    set -euo pipefail
    eval "$FUNCTION_DEF"
    build_rocshmem
    echo "build rocshmem finish!"
    '
