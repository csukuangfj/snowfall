#!/usr/bin/env bash

set -xe

for epoch in $(seq 9 9); do
  # python3 ./mmi_bigram_embeddings_decode.py --epoch ${epoch} --enable_second_pass_decoding 0
  python3 ./mmi_bigram_embeddings_decode.py --epoch ${epoch} --enable_second_pass_decoding 1

  # gdb -ex r --args python3 ./mmi_bigram_embeddings_decode.py --epoch ${epoch} --enable_second_pass_decoding 1
done
