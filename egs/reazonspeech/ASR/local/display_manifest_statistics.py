#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#              2023  Reazon Human Interaction Lab (authors: Qi CHEN)
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

"""
This file displays duration statistics of utterances in a manifest.
You can use the displayed value to choose minimum/maximum duration
to remove short and long utterances during the training.
See the function `remove_short_and_long_utt()`
in ../../../librispeech/ASR/transducer/train.py
for usage.
"""


from lhotse import load_manifest_lazy


def main():
    paths = [
        "./data/fbank/reazonspeech_cuts_train.jsonl.gz",
        "./data/fbank/reazonspeech_cuts_dev.jsonl.gz",
        "./data/fbank/reazonspeech_cuts_test.jsonl.gz",
    ]

    for path in paths:
        print(f"Starting display the statistics for {path}")
        cuts = load_manifest_lazy(path)
        cuts.describe()


if __name__ == "__main__":
    main()

"""
Starting display the statistics for ./data/fbank/reazonspeech_cuts_train.jsonl.gz                                                                                            
Cut statistics:                                                                                                                                                              
________________________________________                                                                                                                                     
_ Cuts count:               _ 962      _                                                                                                                                     
________________________________________                                                                                                                                     
_ Total duration (hh:mm:ss) _ 01:50:49 _                                                                                                                                     
________________________________________                                                                                                                                     
_ mean                      _ 6.9      _                                                                                                                                     
________________________________________                                                                                                                                     
_ std                       _ 4.9      _                                                                                                                                     
________________________________________                                                                                                                                     
_ min                       _ 0.6      _                                                                                                                                     
________________________________________                                                                                                                                     
_ 25%                       _ 3.4      _                                                                                                                                     
________________________________________                                                                                                                                     
_ 50%                       _ 5.7      _
________________________________________
_ 75%                       _ 9.2      _
________________________________________
_ 99%                       _ 20.9     _
________________________________________
_ 99.5%                     _ 23.4     _
________________________________________
_ 99.9%                     _ 33.0     _
________________________________________
_ max                       _ 61.2     _
________________________________________
_ Recordings available:     _ 962      _
________________________________________
_ Features available:       _ 962      _
________________________________________
_ Supervisions available:   _ 962      _
________________________________________
Speech duration statistics:
__________________________________________________________________
_ Total speech duration        _ 01:50:49 _ 100.00% of recording _
__________________________________________________________________
_ Total speaking time duration _ 01:50:49 _ 100.00% of recording _
__________________________________________________________________
_ Total silence duration       _ 00:00:00 _ 0.00% of recording   _
__________________________________________________________________

Starting display the statistics for ./data/fbank/reazonspeech_cuts_dev.jsonl.gz
Cut statistics:
________________________________________
_ Cuts count:               _ 133      _
________________________________________
_ Total duration (hh:mm:ss) _ 00:13:05 _
________________________________________
_ mean                      _ 5.9      _
________________________________________
_ std                       _ 4.5      _
________________________________________
_ min                       _ 1.0      _
________________________________________
_ 25%                       _ 3.3      _
________________________________________
_ 50%                       _ 4.8      _
________________________________________
_ 75%                       _ 7.5      _
________________________________________
_ 99%                       _ 26.1     _
________________________________________
_ 99.5%                     _ 29.1     _
________________________________________
_ 99.9%                     _ 30.4     _
________________________________________
_ max                       _ 30.8     _
________________________________________
_ Recordings available:     _ 133      _
________________________________________
_ Features available:       _ 133      _
________________________________________
_ Supervisions available:   _ 133      _
________________________________________
Speech duration statistics:
__________________________________________________________________
_ Total speech duration        _ 00:13:05 _ 100.00% of recording _
__________________________________________________________________
_ Total speaking time duration _ 00:13:05 _ 100.00% of recording _
__________________________________________________________________
_ Total silence duration       _ 00:00:00 _ 0.00% of recording   _
__________________________________________________________________

Starting display the statistics for ./data/fbank/reazonspeech_cuts_test.jsonl.gz
Cut statistics:
________________________________________
_ Cuts count:               _ 64       _
________________________________________
_ Total duration (hh:mm:ss) _ 00:06:54 _
________________________________________
_ mean                      _ 6.5      _
________________________________________
_ std                       _ 4.8      _
________________________________________
_ min                       _ 1.0      _
________________________________________
_ 25%                       _ 3.0      _
________________________________________
_ 50%                       _ 5.6      _
________________________________________
_ 75%                       _ 8.3      _
________________________________________
_ 99%                       _ 20.8     _
________________________________________
_ 99.5%                     _ 25.4     _
________________________________________
_ 99.9%                     _ 29.1     _
________________________________________
_ max                       _ 30.0     _
________________________________________
_ Recordings available:     _ 64       _
________________________________________
_ Features available:       _ 64       _
________________________________________
_ Supervisions available:   _ 64       _
________________________________________
Speech duration statistics:
_ Total speech duration        _ 00:06:54 _ 100.00% of recording _
__________________________________________________________________
_ Total speaking time duration _ 00:06:54 _ 100.00% of recording _
__________________________________________________________________
_ Total silence duration       _ 00:00:00 _ 0.00% of recording   _
__________________________________________________________________
"""
