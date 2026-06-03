from typing import List
import re

def virtualhome_projection(actions: List[str], action_pools: List[List[str]]):
    """
    An function to process the actions
    actions: the list of actions to be processeed, it is a list of strings.
    action_pools: the list of action pools, each pool is a list of strings.
    """

    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        try:
            
            llm_output = actions[i].strip()
            pattern = re.compile(r"Action:\s?(.*)", re.DOTALL)
            action = re.findall(pattern, llm_output)[0]
            actions[i] = action.lower()
            valids[i] = 1

        except:
            temp_value = len(actions[i].strip()) // 3
            actions[i] = actions[i].strip()[-temp_value:]

        # check if contains any Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

    return actions, valids