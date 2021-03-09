#!/usr/bin/env bash

for epoch in $(seq 0 9); do
  echo "python3 ./mmi_bigram_embeddings_decode.py --epoch ${epoch} --enable_second_pass_decoding 0"
  python3 ./mmi_bigram_embeddings_decode.py --epoch ${epoch} --enable_second_pass_decoding 0

  echo "python3 ./mmi_bigram_embeddings_decode.py --epoch ${epoch} --enable_second_pass_decoding 1"
  python3 ./mmi_bigram_embeddings_decode.py --epoch ${epoch} --enable_second_pass_decoding 1
done
