# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

from typing import List
import re

def scienceworld_projection(actions: List[str]):
    """
    A function to process the actions for ScienceWorld environment.
    actions: the list of actions to be processed, it is a list of strings.
    Expected format:
        Action: <action_text>
    """
    
    valids = [0] * len(actions)
    
    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        try:
            llm_output = actions[i].strip()
            # Extract action from "Action: <action_text>" format
            pattern = re.compile(r"Action:\s*(.*)", re.DOTALL)
            matches = re.findall(pattern, llm_output)
            if matches:
                action = matches[0].strip()
                actions[i] = action
                valids[i] = 1
            else:
                # If no "Action:" prefix found, try to use the whole string
                actions[i] = llm_output
                valids[i] = 0
        except Exception as e:
            # If parsing fails, use the last part of the string
            temp_value = len(actions[i].strip()) // 3
            actions[i] = actions[i].strip()[-temp_value:] if temp_value > 0 else actions[i].strip()
            valids[i] = 0
        
        # Check if contains any Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0
    
    return actions, valids

