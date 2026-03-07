#!/bin/bash

set -x

echo "args: $@"

# set the dataset dir via `DATADIR_JetClass`
DATADIR=${DATADIR_JetClass}
[[ -z $DATADIR ]] && DATADIR='./datasets/JetClass'

weaver --predict \
  --data-config data/JetClass/JetClass_full.yaml \
  --network-config networks/example_ParticleTransformer_ptq_w8a8.py \
  --model-prefix models/ParT_full_w8a8_ptq.pt \
  --data-test "HToBB:${DATADIR}/Pythia/test_20M/HToBB_*.root" \
  --data-test "HToCC:${DATADIR}/Pythia/test_20M/HToCC_*.root" \
  --data-test "HToGG:${DATADIR}/Pythia/test_20M/HToGG_*.root" \
  --data-test "HToWW2Q1L:${DATADIR}/Pythia/test_20M/HToWW2Q1L_*.root" \
  --data-test "HToWW4Q:${DATADIR}/Pythia/test_20M/HToWW4Q_*.root" \
  --data-test "TTBar:${DATADIR}/Pythia/test_20M/TTBar_*.root" \
  --data-test "TTBarLep:${DATADIR}/Pythia/test_20M/TTBarLep_*.root" \
  --data-test "WToQQ:${DATADIR}/Pythia/test_20M/WToQQ_*.root" \
  --data-test "ZToQQ:${DATADIR}/Pythia/test_20M/ZToQQ_*.root" \
  --data-test "ZJetsToNuNu:${DATADIR}/Pythia/test_20M/ZJetsToNuNu_*.root" \
  --batch-size 512 --num-workers 2 --gpus 0 \
  --predict-output pred_ptq.root

# weaver --predict \
#   --data-config data/JetClass/JetClass_full.yaml \
#   --network-config networks/example_ParticleTransformer.py \
#   --model-prefix ParT/particle_transformer/models/ParT_full.pt \
#   --data-test "HToBB:${DATADIR}/Pythia/test_20M/HToBB_*.root" \
#   --data-test "HToCC:${DATADIR}/Pythia/test_20M/HToCC_*.root" \
#   --data-test "HToGG:${DATADIR}/Pythia/test_20M/HToGG_*.root" \
#   --data-test "HToWW2Q1L:${DATADIR}/Pythia/test_20M/HToWW2Q1L_*.root" \
#   --data-test "HToWW4Q:${DATADIR}/Pythia/test_20M/HToWW4Q_*.root" \
#   --data-test "TTBar:${DATADIR}/Pythia/test_20M/TTBar_*.root" \
#   --data-test "TTBarLep:${DATADIR}/Pythia/test_20M/TTBarLep_*.root" \
#   --data-test "WToQQ:${DATADIR}/Pythia/test_20M/WToQQ_*.root" \
#   --data-test "ZToQQ:${DATADIR}/Pythia/test_20M/ZToQQ_*.root" \
#   --data-test "ZJetsToNuNu:${DATADIR}/Pythia/test_20M/ZJetsToNuNu_*.root" \
#   --batch-size 512 --num-workers 2 --gpus 0 \
#   --predict-output pred_fp.root