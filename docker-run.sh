#!/bin/bash
set -eEuo pipefail

docker build -t quality docker 1>&2
mkdir -p docker-home

args=(
	docker run
	-i
	--rm # !
	--user=1000 # change this if you also changed it in docker/Dockerfile
	--network=host
	--device=/dev/kfd
	--device=/dev/dri
	--ipc=host
	--shm-size 16G
	--group-add video
	--cap-add=SYS_PTRACE
	--security-opt seccomp=unconfined
	-v "$PWD":/dockerx
	--workdir /dockerx

	# Allow caching stuff
	--env HOME=/home/user
	-v "$PWD"/docker-home:/home/user

	# These allow the D programs to ask to run the code
	# on the CPU (optionally single-threaded)
	--env CUDA_VISIBLE_DEVICES
	--env OMP_NUM_THREADS
	--env TF_NUM_INTRAOP_THREADS
	--env TF_NUM_INTEROP_THREADS
)

if [[ -h tests ]]
then
	args+=(
		# Allow reading JSON files
		# Assume "tests" is a symlink to your bulk storage device
		# (i.e. your HDD and not your SSD)
		-v "$(readlink tests)":"$(readlink tests)"
	)
fi

if [ -t 0 ]
then
	args+=(-t)
fi

args+=(
	quality

	"$@"
)

exec "${args[@]}"
