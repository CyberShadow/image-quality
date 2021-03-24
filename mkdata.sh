#!/bin/bash
set -eu

rdmd -O -inline -release --build-only mkdata

ulimit -n 32768

function mkdata_fmt() {
	./mkdata --block-width 8 --block-height 8 --image-samples $((128*4)) --sample-order=entropy --filter-samples=false --label-format=u8 "$@"
}

mkdata_fmt --pixel-format=u8
mkdata_fmt --pre dct --pixel-format f32
