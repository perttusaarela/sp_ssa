#!/bin/sh

mkdir -p data
mkdir -p data/rank
mkdir -p data/rank/results
mkdir -p data/subspace
mkdir -p data/subspace/setting1
mkdir -p data/subspace/setting2
mkdir -p data/subspace/setting3
mkdir -p data/subspace/setting4
mkdir -p data/subspace/results
mkdir -p data/kola

Rscript get_kola_data.R

cd c_src
make
