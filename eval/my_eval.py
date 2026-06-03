# 导入相关的库
import os
import json
import logging
import pathlib
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from colorama import Fore

import sys
sys.path.append("/code/EUV/other_env")
import eval_agent.tasks as tasks
import eval_agent.agents as agents
import eval_agent.envs as envs
from eval_agent.utils.datatypes import State

from fastchat.model.model_adapter import get_conversation_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict, Any, Mapping

class LMAgent:
    """Base class for an agent."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        # The agent should not generate observations or expert feedback
        self.stop_words = ["\nObservation:", "\nTask:", "\n---"]

    def __call__(self) -> str:
        pass

    def add_system_message(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        # Prepend the prompt with the system message
        first_msg = messages[0]
        assert first_msg["role"] == "user"
        system, examples, task = first_msg["content"].split("\n---\n")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": examples + "\n---\n" + task},
        ] + messages[1:]
        return messages

def _add_to_set(s, new_stop):
    if not s:
        return
    if isinstance(s, str):
        new_stop.add(s)
    else:
        new_stop.update(s)


class LocalAgent(LMAgent):
    """An agent that loads and runs model locally while following the same conversation format"""
    
    def __init__(self, config) -> None:
        super().__init__(config)
        self.model_name = config["model_name"]
        self.temperature = config.get("temperature", 0.8)
        self.max_new_tokens = config.get("max_new_tokens", 512) 
        self.top_p = config.get("top_p", 1)
        self.do_sample = config.get("do_sample", False)
        
        # Load model and tokenizer locally
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

    def __call__(self, messages: List[dict]) -> str:
        # Use same conversation template as FastChat
        conv = get_conversation_template(self.model_name)
        
        # Format conversation history
        for history_item in messages:
            role = history_item["role"]
            content = history_item["content"]
            if role == "user":
                conv.append_message(conv.roles[0], content)
            elif role == "assistant":
                conv.append_message(conv.roles[1], content)
            else:
                raise ValueError(f"Unknown role: {role}")
                
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Handle stop words
        new_stop = set()
        _add_to_set(self.stop_words, new_stop)
        _add_to_set(conv.stop_str, new_stop)
        
        # print('input prompt:', repr(prompt))
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.do_sample,
            # pad_token_id=self.tokenizer.eos_token_id,
            # stopping_criteria=new_stop if new_stop else None,
            # stop_token_ids=conv.stop_token_ids if hasattr(conv, 'stop_token_ids') else None
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant's response
        response = response.split(conv.roles[1])[-1].strip()
        
        # Apply stop words
        for stop in new_stop:
            if stop in response:
                response = response[:response.index(stop)]
                
        return response.strip()

# Initialize agent
agent_config = {
    "model_name": "/code/EUV/checkpoints/global_step_33",  # Change to your model
    "temperature": 0.8,
    "max_new_tokens": 512,
    "top_p": 1,
    "do_sample": False  # Explicitly set do_sample=False for greedy decoding
}

agent = LocalAgent(agent_config)

def interactive_loop(
    task,
    agent,
    env_config):
    
    env = getattr(envs, env_config["env_class"])(task, **env_config)
    # reset the environment and set the prompt
    observation, state = env.reset()

    init_msg = observation

    cur_step = 1
    while not state.finished:
        cur_step += 1
        # agent act
        try:
            llm_output: str = agent(state.history)

            number_n = llm_output.count('\n')
            if number_n>=2:
                first_n = llm_output.find('\n')
                second_n = llm_output.find('\n', first_n+1)
                llm_output = llm_output[:second_n]

        except Exception as e:
            state.success = False
            state.finished = True
            state.terminate_reason = "exceeding maximum input length"
            break
        # environment step
        observation, state = env.step(llm_output)
        # color the state in blue
        # if not state.finished:
        #     # color the observation in blue
        #     print(
        #         f"\n{observation}\n"
        #     )

        # if state.finished:
        #     break

    if state.reward is not None:
        print(
            f"Task finished in {state.steps} steps. Success: {state.success}. Reward: {state.reward}"
        )
    else:
        print(
            f"Task finished in {state.steps} steps. Success: {state.success}"
        )

    return state

with open("/code/EUV/other_env/eval_agent/configs/task/alfworld.json") as f:
    exp_config = json.load(f)
    
with open("/code/EUV/other_env/eval_agent/configs/model/fastchat.json") as f:
    agent_config = json.load(f)
    
env_config = exp_config["env_config"]

task_config = exp_config["task"]
task_class = getattr(tasks, task_config["task_class"])

output_path = '/code/EUV/evaluation/verl_alfworld_unseen_llama_3b_sft_another_inference'
split = 'test'
part_num = 1 
part_idx = 0
all_tasks, n_tasks = task_class.load_tasks(split, part_num, part_idx)

# initialize the agent
# agent = getattr(agents, agent_config["agent_class"])(
#     agent_config["config"]
# )

state_list = []
done_task_id = []
if os.path.exists(output_path):
    for file in os.listdir(output_path):
        if not file.endswith('json'):
            continue
        done_task_id.append(file.split('.')[0])

pbar = tqdm(total=n_tasks)
for i, task in enumerate(all_tasks):

    # skip done tasks
    if task.task_id in done_task_id or str(task.task_id) in done_task_id:
        continue

    state = interactive_loop(
        task, agent, env_config
    )

    state_list.append(state)
    json.dump(state.to_dict(), open(os.path.join(output_path, f"{task.task_id}.json"), 'w'), indent=4)

    pbar.update(1)
pbar.close()

# calculate metrics
reward_list = []
success_list = []
for state in state_list:
    if state.reward is not None:
        reward_list.append(state.reward)
    success_list.append(state.success)