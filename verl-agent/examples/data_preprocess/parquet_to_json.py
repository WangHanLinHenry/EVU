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
将parquet文件反向转换为原始JSON格式
"""

import argparse
import os
import pandas as pd
import json


def main():
    parser = argparse.ArgumentParser(description="将parquet文件转换为原始JSON格式")
    parser.add_argument("--parquet_file", default="/code/EUV/verl-agent/trial_data/sciworld_our_sft/train.parquet", help="输入的parquet文件路径")
    parser.add_argument("--output_json", default="/code/EUV/verl-agent/trial_data/sciworld_our_sft/transverse_train.json", help="输出的JSON文件路径")
    args = parser.parse_args()

    # 读取parquet文件
    print(f"正在读取parquet文件: {args.parquet_file}")
    df = pd.read_parquet(args.parquet_file)
    
    # 转换数据格式
    original_data = []
    
    for idx, row in df.iterrows():
        messages = row['messages']
        conversations = []
        
        # 将messages格式转换回conversations格式
        for msg in messages:
            # 原始格式中，conversations的每个元素都有'from'和'value'字段
            # role: "user" -> from: "human"
            # role: "assistant" -> from: "gpt"
            if msg['role'] == 'user':
                from_role = 'human'
            elif msg['role'] == 'assistant':
                from_role = 'gpt'
            else:
                # 如果遇到其他role，保持原样或使用默认值
                from_role = msg['role']
            
            conversations.append({
                "from": from_role,
                "value": msg['content']
            })
        
        # 构建原始JSON格式
        # 检查是否有id和index字段，如果没有则使用默认值
        item = {"conversations": conversations}
        
        # 如果parquet中有id字段，保留它；否则使用索引作为id
        if 'id' in df.columns and pd.notna(row['id']):
            item['id'] = str(row['id'])
        else:
            item['id'] = str(idx)
        
        # 如果parquet中有index字段，保留它；否则使用行索引
        if 'index' in df.columns and pd.notna(row['index']):
            item['index'] = int(row['index'])
        else:
            item['index'] = int(idx)
        
        original_data.append(item)
    
    # 保存为JSON文件
    print(f"正在保存JSON文件: {args.output_json}")
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(original_data, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print(f"转换完成！")
    print(f"共转换 {len(original_data)} 条数据")
    print(f"JSON文件已保存到: {args.output_json}")


if __name__ == "__main__":
    main()

