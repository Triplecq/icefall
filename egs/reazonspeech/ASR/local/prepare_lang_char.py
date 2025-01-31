#!/usr/bin/env python3
# Copyright    2022  The University of Electro-Communications  (Author: Teo Wen Shen)  # noqa
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import logging
from pathlib import Path

from lhotse import CutSet

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "train_cut", metavar="train-cut", type=Path, help="Path to the train cut"
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default=Path("data/lang_char"),
        help=(
            "Name of lang dir. "
            "If not set, this will default to lang_char_{trans-mode}"
        ),
    )

    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(
        format=("%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"),
        level=logging.INFO,
    )

    sysdef_string = set(["<blk>", "<unk>", "<sos/eos>", " "])

    token_set = set()
    logging.info(f"Creating vocabulary from {args.train_cut}.")
    train_cut: CutSet = CutSet.from_file(args.train_cut)
    for cut in train_cut:
        for sup in cut.supervisions:
            token_set.update(sup.text)

    token_set = ["<blk>"] + sorted(token_set - sysdef_string) + ["<unk>", "<sos/eos>"]
    args.lang_dir.mkdir(parents=True, exist_ok=True)
    (args.lang_dir / "tokens.txt").write_text(
        "\n".join(f"{t}\t{i}" for i, t in enumerate(token_set))
    )

    (args.lang_dir / "lang_type").write_text("char")
    logging.info("Done.")


if __name__ == "__main__":
    main()
