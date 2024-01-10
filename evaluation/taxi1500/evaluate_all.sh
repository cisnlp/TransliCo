#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#MODEL=${1:-cis-lmu/glot500-base}
MODEL="xlm-roberta-base"
GPU=${2:-3}

export CUDA_VISIBLE_DEVICES=$GPU
MODEL_TYPE="xlmr"

NUM_PRIMITIVE=768
max_checkpoint_num=20000

OUTPUT_DIR="/mounts/data/proj/ayyoobbig/transliteration_modeling/evaluation/taxi1500/results/"
init_checkpoint="/mounts/data/proj/ayyoobbig/transliteration_modeling/trained_models/model_five_percent_TCM_with_latn_pool_weight_0.5_no_lm"

python -u evaluate_all.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL \
    --output_dir $OUTPUT_DIR \
    --epochs 40 \
    --max_checkpoint_num $max_checkpoint_num \
    --init_checkpoint $init_checkpoint
