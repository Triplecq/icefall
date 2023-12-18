#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

stage=-1
stop_stage=100
perturb_speed=true


# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/aishell4
#      You can find four directories:train_S, train_M, train_L and test.
#      You can download it from https://openslr.org/111/
#
#  - $dl_dir/musan
#      This directory contains the following directories downloaded from
#       http://www.openslr.org/17/
#
#     - music
#     - noise
#     - speech

dl_dir=$PWD/download

. shared/parse_options.sh || exit 1

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Download data"

  # If you have pre-downloaded it to /path/to/aishell4,
  # you can create a symlink
  #
  #   ln -sfv /path/to/aishell4 $dl_dir/aishell4
  #
  if [ ! -f $dl_dir/aishell4/train_L ]; then
    lhotse download aishell4  $dl_dir/aishell4
  fi

  # If you have pre-downloaded it to /path/to/musan,
  # you can create a symlink
  #
  #   ln -sfv /path/to/musan $dl_dir/musan
  #
  if [ ! -d $dl_dir/musan ]; then
    lhotse download musan $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare aishell4 manifest"
  # We assume that you have downloaded the aishell4 corpus
  # to $dl_dir/aishell4
  if [ ! -f data/manifests/aishell4/.manifests.done ]; then
    mkdir -p data/manifests/aishell4
    lhotse prepare aishell4 $dl_dir/aishell4 data/manifests/aishell4
    touch data/manifests/aishell4/.manifests.done
  fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Process aishell4"
  if [ ! -f data/fbank/aishell4/.fbank.done ]; then
    mkdir -p data/fbank/aishell4
    lhotse prepare aishell4 $dl_dir/aishell4 data/manifests/aishell4
    touch data/fbank/aishell4/.fbank.done
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Prepare musan manifest"
  # We assume that you have downloaded the musan corpus
  # to data/musan
  if [ ! -f data/manifests/.musan_manifests.done ]; then
    log "It may take 6 minutes"
    mkdir -p data/manifests
    lhotse prepare musan $dl_dir/musan data/manifests
    touch data/manifests/.musan_manifests.done
  fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Compute fbank for musan"
  if [ ! -f data/fbank/.msuan.done ]; then
    mkdir -p data/fbank
    ./local/compute_fbank_musan.py
    touch data/fbank/.msuan.done
  fi
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Compute fbank for aishell4"
  if [ ! -f data/fbank/.aishell4.done ]; then
    mkdir -p data/fbank
    ./local/compute_fbank_aishell4.py --perturb-speed ${perturb_speed}
    touch data/fbank/.aishell4.done
  fi
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Prepare char based lang"
  lang_char_dir=data/lang_char
  mkdir -p $lang_char_dir

  # Prepare text.
  # Note: in Linux, you can install jq with the  following command:
  # wget -O jq https://github.com/stedolan/jq/releases/download/jq-1.6/jq-linux64
  gunzip -c data/manifests/aishell4/aishell4_supervisions_train_S.jsonl.gz \
    | jq ".text" | sed 's/"//g' \
    | ./local/text2token.py -t "char" > $lang_char_dir/text_S

  gunzip -c data/manifests/aishell4/aishell4_supervisions_train_M.jsonl.gz \
    | jq ".text" | sed 's/"//g' \
    | ./local/text2token.py -t "char" > $lang_char_dir/text_M

  gunzip -c data/manifests/aishell4/aishell4_supervisions_train_L.jsonl.gz \
    | jq ".text" | sed 's/"//g' \
    | ./local/text2token.py -t "char" > $lang_char_dir/text_L

  for r in text_S text_M text_L ; do
    cat $lang_char_dir/$r >> $lang_char_dir/text_full
  done

  # Prepare text normalize
  python ./local/text_normalize.py \
    --input $lang_char_dir/text_full \
    --output $lang_char_dir/text

  # Prepare words segments
  python ./local/text2segments.py \
    --input $lang_char_dir/text \
    --output $lang_char_dir/text_words_segmentation

  cat $lang_char_dir/text_words_segmentation | sed "s/ /\n/g" \
    | sort -u | sed "/^$/d" \
    | uniq > $lang_char_dir/words_no_ids.txt

  # Prepare words.txt
  if [ ! -f $lang_char_dir/words.txt ]; then
    ./local/prepare_words.py \
      --input-file $lang_char_dir/words_no_ids.txt \
      --output-file $lang_char_dir/words.txt
  fi

  if [ ! -f $lang_char_dir/L_disambig.pt ]; then
    ./local/prepare_char.py
  fi
fi
