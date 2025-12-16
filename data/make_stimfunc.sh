#!/bin/zsh

# zsh make_stimfunc.sh

source "$FREESURFER_HOME/SetUpFreeSurfer.sh"

optseq2 --ntp 1000 --tr 2 --tnullmin 4 --tnullmax 8 --psdwin 0 12 2 \
--ev Q-sound_M-pic 2 50 --ev Q-sound_M-word 2 50 --ev Q-1000yrs_M-pic 2 50 \
--ev Q-1000yrs_M-word 2 50 --ev Q-comp_M-pic 2 50 --ev Q-comp_M-word 2 50 \
--evc  1 -1  1 -1  1 -1 \
--evc  1  1 -1 -1  0  0 \
--evc  1  1  0  0 -1 -1 \
--evc  1 -1 -1  1  0  0 \
--nkeep 10 --o stimfunc --nsearch 10000

if [ ! -d "optseq2" ]; then
  mkdir optseq2
fi

mv stimfunc-*.* optseq2
mv stimfunc.* optseq2