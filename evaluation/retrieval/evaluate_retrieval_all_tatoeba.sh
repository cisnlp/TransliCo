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

# MODEL=${1:-cis-lmu/glot500-base}
MODEL="xlm-roberta-base"
GPU=${2:-2}

export CUDA_VISIBLE_DEVICES=$GPU
MODEL_TYPE="xlmr"

MAX_LENGTH=512
LC=""
BATCH_SIZE=128
DIM=768
NLAYER=12
LAYER=7

DATA_DIR="/mounts/data/proj/linpq/datasets/retrieval_tatoeba/"
OUTPUT_DIR="/mounts/data/proj/ayyoobbig/transliteration_modeling/evaluation/retrieval/tatoeba/"
tokenized_dir="/mounts/data/proj/ayyoobbig/transliteration_modeling/evaluation/retrieval/tatoeba_tokenized"
init_checkpoint="/mounts/data/proj/ayyoobbig/transliteration_modeling/trained_models/model_five_percent_TCM_with_latn_pool_weight_0.5_no_lm"

python -u evaluate_retrieval_all_tatoeba.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --embed_size $DIM \
    --batch_size $BATCH_SIZE \
    --max_seq_len $MAX_LENGTH \
    --num_layers $NLAYER \
    --dist cosine $LC \
    --specific_layer $LAYER \
    --tokenized_dir $tokenized_dir \
    --init_checkpoint $init_checkpoint
