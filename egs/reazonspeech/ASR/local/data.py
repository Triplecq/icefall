import getopt
import glob
import os
import random
import shlex
import sys

from datasets import load_dataset


def get_symbols():
    with open(os.path.join(os.path.dirname(__file__), "symbol.txt")) as fp:
        return {ord(c.rstrip("\n")): "" for c in fp}


SYMBOLS = get_symbols()
HANKAKU = "".join(chr(x) for x in range(0x30, 0x7B))
ZENKAKU = "".join(chr(x) for x in range(0xFF10, 0xFF5B))
HAN2ZEN = str.maketrans(HANKAKU, ZENKAKU)


def normalize(text):
    return text.translate(HAN2ZEN).translate(SYMBOLS)


def get_duration(items):
    ret = 0
    for item in items:
        ret += len(item["audio"]["array"]) / item["audio"]["sampling_rate"]
    return ret


def partition(ds):
    items = []
    for idx, item in enumerate(ds):
        if idx % 2 == 0:
            yield [item]
        else:
            items.append(item)
            if get_duration(items) > 20:
                yield items
                items = []
    if items:
        yield items


def get_command(musan_dir, audio_filepaths):
    arg0 = os.path.join(os.path.dirname(__file__), "augment.py")
    rand = random.random()
    if rand < 0.375:
        command = [arg0, "-m", musan_dir, *audio_filepaths]
    elif rand < 0.75:
        command = [arg0, "-c", *audio_filepaths]
    else:
        command = [arg0, *audio_filepaths]

    return "%s |" % shlex.join(command)


def save_kaldi_format(outdir, musan_dir, ds):
    os.makedirs(outdir, exist_ok=True)
    fp_text = open(os.path.join(outdir, "text"), "w")
    fp_wav = open(os.path.join(outdir, "wav.scp"), "w")
    fp_utt2spk = open(os.path.join(outdir, "utt2spk"), "w")
    fp_spk2utt = open(os.path.join(outdir, "spk2utt"), "w")
    with fp_text, fp_wav, fp_utt2spk, fp_spk2utt:
        for item in ds.sort("name"):
            path = item["audio"]["path"]

            # '11時のニュースです。' -> '１１時
            text = normalize(item["transcription"])

            # '000/e7fb3323c280c.flac' -> '000e7fb3323c280c'
            name = os.path.splitext(item["name"].replace("/", ""))[0]
            uttid = "uttid%s" % name
            spkid = "spkid%s" % name
            print(uttid, text, file=fp_text)
            print(uttid, path, file=fp_wav)
            print(uttid, spkid, file=fp_utt2spk)
            print(spkid, uttid, file=fp_spk2utt)


def main():
    if len(sys.argv) != 3:
        print("Usage: %s <musan_dir> <download_dir>" % sys.argv[0], file=sys.stderr)
        return 1

    musan_dir = sys.argv[1]
    download_dir = sys.argv[2]

    ds = load_dataset("reazon-research/reazonspeech", "small", cache_dir=download_dir)[
        "train"
    ]
    save_kaldi_format("data/dev", musan_dir, ds.select(range(1000)))
    save_kaldi_format("data/test", musan_dir, ds.select(range(1000, 1100)))
    save_kaldi_format("data/train", musan_dir, ds.select(range(1100, ds.num_rows)))


if __name__ == "__main__":
    main()
