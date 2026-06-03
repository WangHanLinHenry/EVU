# Copyright 2024 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Create a simple multi-turn dataset for testing
"""

import argparse
import os

import pandas as pd

import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/code/EUV/other_env/baselines/vagen")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    # json_path = "/code/EUV/other_env/sciworld_sft_filtered.json"
    json_path  = "/code/EUV/other_env/baselines/vagen/sciworld_vagen_data.json"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conversations = []

    for j in range(len(data)):
        each_piece = []
        for i in range(len(data[j]['conversations'])):
            if i%2 == 0:
                each_piece.append({"role":"user","content":data[j]['conversations'][i]['value']})
            else:
                each_piece.append({"role":"assistant","content":data[j]['conversations'][i]['value']})
        conversations.append({"messages":each_piece})

    train_data = conversations
    test_data = conversations[int(len(conversations)*0.95):]

    # Create output directory
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Save to parquet files
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_df.to_parquet(os.path.join(local_dir, "test.parquet"))

    # Handle HDFS if specified
    if args.hdfs_dir is not None:
        try:
            from verl.utils.hdfs_io import copy, makedirs

            makedirs(args.hdfs_dir)
            copy(src=local_dir, dst=args.hdfs_dir)
        except ImportError:
            print("Warning: HDFS support not available. Skipping HDFS copy.")

    # Print statistics
    print(f"Train dataset size: {len(train_df)}")
    print(f"Test dataset size: {len(test_df)}")
    print(f"Data saved to {local_dir}")


if __name__ == "__main__":
    main()
