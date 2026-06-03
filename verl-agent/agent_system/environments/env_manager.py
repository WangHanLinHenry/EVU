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

from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory, SearchMemory
from omegaconf import OmegaConf

import json
import re

from torch._inductor.utils import pass_execution_and_save

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        self.memory.reset(batch_size=len(obs))

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy()
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            if init or self.config.env.history_length <= 0:
                obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i]
                )
            else:
                obs_i = SEARCH_TEMPLATE.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask
            

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        # 读取instruction
        with open('/code/EUV/other_env/eval_agent/prompt/instructions/alfworld_inst.txt') as f:
            self.alfworld_instruction = f.read()
        # 读取icl examples
        with open('/code/EUV/other_env/eval_agent/prompt/icl_examples/alfworld_icl.json') as f:
            self.alfworld_icl = json.load(f)[0]
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        # print(text_actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': text_actions})
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        # 调试：打印动作映射信息
        import logging
        logging.debug(f"[动作映射] 原始动作: {text_actions}")
        logging.debug(f"[动作映射] 映射后动作: {actions}")
        logging.debug(f"[动作映射] 动作有效性: {valids}")
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        # self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        # self.memory.store({'text_obs': self.pre_text_obs, 'action': text_actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        # print("full_text_obs: ", full_text_obs)
        # print("actions: ", actions)

        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        # if not init and self.config.env.history_length > 0:
        #     memory_contexts, valid_lens = self.memory.fetch(
        #             self.config.env.history_length,
        #             obs_key="text_obs",
        #             action_key="action")
        memory_contexts = []
        valid_lens = []
        for i in range(len(self.memory._data)):
            each_env_actions = []
            each_env_observations = []
            for j in range(len(self.memory._data[i])):
                each_env_actions.append(self.memory._data[i][j]['action'])
                each_env_observations.append("Observation: " + self.memory._data[i][j]['text_obs'])
            memory_contexts.append([each_env_actions, each_env_observations])
            
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            # if init or self.config.env.history_length <= 0:
            #     obs = ALFWORLD_TEMPLATE_NO_HIS.format(
            #         current_observation=text_obs[i],
            #         admissible_actions=reformatted_admissible_actions
            #     )
            # else:
            #     obs = ALFWORLD_TEMPLATE.format(
            #         task_description=self.tasks[i],
            #         step_count=len(self.memory[i]),
            #         history_length=valid_lens[i],
            #         action_history=memory_contexts[i],
            #         current_step=len(self.memory[i]) + 1,
            #         current_observation=text_obs[i],
            #         admissible_actions=reformatted_admissible_actions
            #     )

            new_observation = text_obs[i]
            each_observation_memory = memory_contexts[i][1]
            each_action_memory = memory_contexts[i][0]

            prompt_method = self.config.env.alfworld.baseline  # 默认为 'baseline'
            model_name = self.config.env.model_name
            # baseline的方法
            if prompt_method == 'baseline':
                obs = self.get_fastchat_prompt(each_observation_memory, each_action_memory, new_observation, model_name)
            # 我的方法
            elif prompt_method == 'our_method':
                obs = self.get_prompt_our_method(self.tasks[i], each_action_memory, new_observation, model_name)
            # elif prompt_method == 'our_method_w_two_steps':
            #     obs = self.get_prompt_our_method_w_two_steps(self.tasks[i], each_action_memory, each_observation_memory, new_observation, 'qwen')
            # elif prompt_method == 'baseline_again':
            #     obs = self.get_prompt_baseline(self.tasks[i], each_action_memory, each_observation_memory, new_observation, 'qwen')
            elif prompt_method == 'prompting_baseline':
                obs = self.prompting_method_get_fastchat_prompt(each_observation_memory, each_action_memory, new_observation, model_name)
            elif prompt_method == 'prompting_our_method':
                obs = self.prompting_method_get_prompt_our_method(self.tasks[i], each_action_memory, new_observation, model_name)
            else:
                raise ValueError(f"Unknown prompt method: {prompt_method}")

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def get_prompt_our_method(self, task, actions, new_observation, model_name):
        
        pure_actions, _ = self.projection_f(actions[:-1], self.envs.get_admissible_commands)
        each_action_history = ", ".join(pure_actions)
        # each_action_history = "\n"
        # for idx, action in enumerate(pure_actions, 1):
        #     each_action_history += f"Step {idx}: {action}\n"

        last_turn_information = ''
        if len(actions) > 0:
            last_turn_information += '\n' + actions[-1] + '\nObservation: ' + new_observation
        else:
            last_turn_information += '\n' + new_observation[:new_observation.find('Your task is to: ')]
        
        prompt = ALFWORLD_TEMPLATE_OUR_METHOD.format(
            task_goal=task,
            action_history=each_action_history,
            # step_num=len(pure_actions) + 1,
            last_turn_information=last_turn_information
        )

        if model_name == 'llama':
            ret = "<|begin_of_text|>"
            ret += "<|start_header_id|>user<|end_header_id|>\n\n" + prompt + "<|eot_id|>"
            ret += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        elif model_name == 'qwen':
            ret = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            ret += '<|im_start|>user' + '\n' + prompt + '<|im_end|>' + '\n'
            ret += '<|im_start|>assistant' + '\n'
        elif model_name == 'gemma':
            ret = ""
            ret += '<start_of_turn>user' + '\n' + prompt + '<end_of_turn>\n'
            ret += '<start_of_turn>model' + '\n'
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return ret

    def prompting_method_get_prompt_our_method(self, task, actions, new_observation, model_name):
        
        pure_actions, _ = self.projection_f(actions[:-1], self.envs.get_admissible_commands)
        each_action_history = ", ".join(pure_actions)
        # each_action_history = "\n"
        # for idx, action in enumerate(pure_actions, 1):
        #     each_action_history += f"Step {idx}: {action}\n"

        last_turn_information = ''
        if len(actions) > 0:
            last_turn_information += '\n' + actions[-1] + '\nObservation: ' + new_observation
        else:
            last_turn_information += '\n' + new_observation[:new_observation.find('Your task is to: ')]
        
        ret = prompting_method_ALFWORLD_TEMPLATE_OUR_METHOD.format(
            task_goal=task,
            action_history=each_action_history,
            last_turn_information=last_turn_information
        )
        
        return ret

    def get_prompt_our_method_w_two_steps(self, task, actions, observations, new_observation, model_name):
        
        pure_actions, _ = self.projection_f(actions[:-2], self.envs.get_admissible_commands)
        # each_action_history = ", ".join(pure_actions)
        each_action_history = "\n"
        for idx, action in enumerate(pure_actions, 1):
            each_action_history += f"Step {idx}: {action}\n"

        action_idx = len(actions)
        last_two_turn_information = ''
        if len(actions) > 1:
            last_two_turn_information += f'Step {action_idx-1}: ' + actions[-2] + '\n' + observations[-1] + '\n' + f'Step {action_idx}: ' + actions[-1] + '\n' + new_observation + '\n'
        elif len(actions) == 1:
            last_two_turn_information += f'Step {action_idx}: ' + actions[-1] + '\n' + new_observation + '\n'
        else:
            last_two_turn_information += '\n' + new_observation[:new_observation.find('Your task is to: ')]
        
        each_action_history += last_two_turn_information

        prompt = ALFWORLD_TEMPLATE_OUR_METHOD_w_two_steps.format(
            task_goal=task,
            action_history=each_action_history,
        )

        if model_name == 'llama':
            pass
        elif model_name == 'qwen':
            ret = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            ret += '<|im_start|>user' + '\n' + prompt + '<|im_end|>' + '\n'
            ret += '<|im_start|>assistant' + '\n'
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return ret

    def get_fastchat_prompt(self, observations, actions, new_observation, model_name):
        # # 读取instruction
        # with open('/code/EUV/other_env/eval_agent/prompt/instructions/alfworld_inst.txt') as f:
        #     alfworld_instruction = f.read()
        
        if model_name == 'llama':
            alfworld_messages = []
            alfworld_messages.append(['user', self.alfworld_instruction])
            alfworld_messages.append(['assistant', 'OK'])

            for action, observation in zip(actions, observations):
                alfworld_messages.append(['user', observation])
                alfworld_messages.append(['assistant', action])

            alfworld_messages.append(['user', new_observation])
            alfworld_messages.append(['assistant', None])

            ret = "<|begin_of_text|>"
            for i, (role, message) in enumerate(alfworld_messages):
                if message:
                    ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
                    ret += f"{message.strip()}<|eot_id|>"
                else:
                    ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
                    
            return ret
        elif model_name == 'qwen':
            alfworld_messages = []
            alfworld_messages.append(['<|im_start|>user', self.alfworld_instruction])
            alfworld_messages.append(['<|im_start|>assistant', 'OK'])

            # for message in self.alfworld_icl:
            #     if message['role'] == 'user':
            #         alfworld_messages.append(['<|im_start|>user', message['content']])
            #     elif message['role'] == 'assistant':
            #         alfworld_messages.append(['<|im_start|>assistant', message['content']])

            for action, observation in zip(actions, observations):
                alfworld_messages.append(['<|im_start|>user', observation])
                alfworld_messages.append(['<|im_start|>assistant', action])

            alfworld_messages.append(['<|im_start|>user', new_observation])
            alfworld_messages.append(['<|im_start|>assistant', None])

            ret = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"

            for role, message in alfworld_messages:
                if message:
                    ret += role + "\n" + message + "<|im_end|>" + "\n"
                else:
                    ret += role + "\n"
            
            return ret
        elif model_name == 'gemma':
            alfworld_messages = []
            alfworld_messages.append(['<start_of_turn>user', self.alfworld_instruction])
            alfworld_messages.append(['<start_of_turn>model', 'OK'])

            for action, observation in zip(actions, observations):
                alfworld_messages.append(['<start_of_turn>user', observation])
                alfworld_messages.append(['<start_of_turn>model', action])

            alfworld_messages.append(['<start_of_turn>user', new_observation])
            alfworld_messages.append(['<start_of_turn>model', None])

            ret = ""

            for role, message in alfworld_messages:
                if message:
                    ret += role + "\n" + message + "<end_of_turn>\n"
                else:
                    ret += role + "\n"
            
            return ret
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def prompting_method_get_fastchat_prompt(self, observations, actions, new_observation, model_name):
        # # 读取instruction
        # with open('/code/EUV/other_env/eval_agent/prompt/instructions/alfworld_inst.txt') as f:
        #     alfworld_instruction = f.read()
        last_turn_information = ''
        for i in range(len(observations)):
            last_turn_information += f"{observations[i]}\n{actions[i]}\n"
        last_turn_information += f"{new_observation}\n"
        
        ret = prompting_method_ALFWORLD_TEMPLATE_BASELINE.format(
            last_turn_information=last_turn_information
        )
        
        return ret
        

    def get_prompt_baseline(self, task, actions, observations, new_observation, model_name):
        
        each_action_history = ""
        if len(observations) > 0:
            each_action_history += f"{observations[0]}\n"
        for idx, action in enumerate(actions[:-1], 1):
            each_action_history += f"Step {idx}: {action}\n{observations[idx]}\n"
        if len(actions) > 0:
            each_action_history += f"Step {len(actions)}: {actions[-1]}\n{new_observation}\n"
        else:
            each_action_history += f'{new_observation}\nOK'
        
        prompt = ALFWORLD_TEMPLATE_OUR_METHOD_BASELINE.format(
            action_history=each_action_history,
        )

        if model_name == 'llama':
            pass
        elif model_name == 'qwen':
            ret = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            ret += '<|im_start|>user' + '\n' + prompt + '<|im_end|>' + '\n'
            ret += '<|im_start|>assistant' + '\n'
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return ret

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break


class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    def __init__(self, envs, projection_f, config):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        obs, infos = self.envs.reset()
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs,   
                'anchor': obs
            }
        else:
            self.pre_text_obs = obs
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        self.memory.store({'text_obs': self.pre_text_obs, 'action': [self.ACTION_LOOKUP[act] for act in actions]})
        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs,
                'anchor': next_obs 
            }
        else:
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(infos)):
            if init or self.config.env.history_length <= 0:
                obs = SOKOBAN_VISUAL_TEMPLATE if self.is_multi_modal \
                 else SOKOBAN_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                )
            else:
                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs


class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos


    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if 'ezpoints' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif 'points24' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif 'numberline' in self.config.env.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.config.env.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(obs, infos, init=True), 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or self.config.env.history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return

class VirtualHomeEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.memory.reset(batch_size = len(infos))
        self.tasks = self.extract_task(obs)
        observations = {'text': self.build_text_obs(obs, infos, init=True), 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        return observations, infos

    def step(self, text_actions: List[str]):
        self.memory.store({'text_obs': self.pre_text_obs, 'action': text_actions})
        actions, valids = self.projection_f(text_actions, None)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            # Extract task from observation like "The task is task_name(task_description)."
            if "The task is " in obs:
                task_start = obs.find("The task is ") + len("The task is ")
                task_end = obs.find("(", task_start)
                if task_end == -1:
                    task_end = len(obs) - 1
                task_name = obs[task_start:task_end].strip()
                tasks.append(task_name)
            else:
                tasks.append("Unknown task")
        return tasks
            
    def build_text_obs(self, text_obs: List[str], infos: List[Dict], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        # if not init and self.config.env.history_length > 0:
        #     memory_contexts, valid_lens = self.memory.fetch(
        #             self.config.env.history_length,
        #             obs_key="text_obs",
        #             action_key="action")

        memory_contexts = []
        valid_lens = []
        for i in range(len(self.memory._data)):
            each_env_actions = []
            each_env_observations = []
            for j in range(len(self.memory._data[i])):
                each_env_actions.append(self.memory._data[i][j]['action'])
                each_env_observations.append("Observation: " + self.memory._data[i][j]['text_obs'])
            memory_contexts.append([each_env_actions, each_env_observations])
            
        for i in range(len(text_obs)):
            task_description = self.tasks[i]
            current_observation = text_obs[i]
            
            # if init or self.config.env.history_length <= 0:
            #     obs = VIRTUALHOME_TEMPLATE_NO_HIS.format(
            #         task_description=task_description,
            #         current_observation=current_observation
            #     )
            # else:
            #     obs = VIRTUALHOME_TEMPLATE.format(
            #         task_description=task_description,
            #         step_count=len(self.memory[i]),
            #         history_length=valid_lens[i],
            #         action_history=memory_contexts[i],
            #         current_step=len(self.memory[i]) + 1,
            #         current_observation=current_observation
            #     )
            #     if len(obs) > 13000:
            #         print(f"Warning len(obs)={len(obs)} is too long")
            #         obs = VIRTUALHOME_TEMPLATE_NO_HIS.format(
            #             task_description=task_description,
            #             current_observation=current_observation
            #         )

            prompt_method = self.config.env.virtualhome.baseline
            model_name = self.config.env.model_name
            if prompt_method == 'our_method':
                # obs = VIRTUALHOME_TEMPLATE_OUR_METHOD.format(
                #     task_description=task_description,
                #     action_history=memory_contexts[i],
                #     current_observation=current_observation
                # )
                obs = self.get_prompt_our_method(task_description, memory_contexts[i][0], current_observation, model_name)
            elif prompt_method == 'baseline':
                obs = self.get_fastchat_prompt(memory_contexts[i][0], memory_contexts[i][1], current_observation, model_name)
            elif prompt_method == 'prompting_baseline':
                obs = self.prompting_method_get_fastchat_prompt(memory_contexts[i][1], memory_contexts[i][0], current_observation, model_name)
            elif prompt_method == 'prompting_our_method':
                obs = self.prompting_method_get_prompt_our_method(task_description, memory_contexts[i][0], current_observation, model_name)
            else:
                raise ValueError(f"Unknown prompt method: {prompt_method}")

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def get_fastchat_prompt(self, actions, observations, new_observation, model_name):
        # 读取instruction
        with open('/code/EUV/other_env/eval_agent/prompt/instructions/virtualhome_inst.txt') as f:
            virtualhome_instruction = f.read()
        
        if model_name == 'llama':
            virtualhome_messages = []
            virtualhome_messages.append(['user', virtualhome_instruction])
            virtualhome_messages.append(['assistant', 'OK'])

            for action, observation in zip(actions, observations):
                virtualhome_messages.append(['user', observation])
                virtualhome_messages.append(['assistant', action])

            virtualhome_messages.append(['user', new_observation])
            virtualhome_messages.append(['assistant', None])

            ret = "<|begin_of_text|>"
            for i, (role, message) in enumerate(virtualhome_messages):
                if message:
                    ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
                    ret += f"{message.strip()}<|eot_id|>"
                else:
                    ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
                    
            return ret
        elif model_name == 'qwen':
            virtualhome_messages = []
            virtualhome_messages.append(['<|im_start|>user', virtualhome_instruction])
            virtualhome_messages.append(['<|im_start|>assistant', 'OK'])

            # for message in self.alfworld_icl:
            #     if message['role'] == 'user':
            #         alfworld_messages.append(['<|im_start|>user', message['content']])
            #     elif message['role'] == 'assistant':
            #         alfworld_messages.append(['<|im_start|>assistant', message['content']])

            for action, observation in zip(actions, observations):
                virtualhome_messages.append(['<|im_start|>user', observation])
                virtualhome_messages.append(['<|im_start|>assistant', action])

            virtualhome_messages.append(['<|im_start|>user', new_observation])
            virtualhome_messages.append(['<|im_start|>assistant', None])

            ret = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"

            for role, message in virtualhome_messages:
                if message:
                    ret += role + "\n" + message + "<|im_end|>" + "\n"
                else:
                    ret += role + "\n"
            
            return ret
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def prompting_method_get_fastchat_prompt(self, observations, actions, new_observation, model_name):
        # # 读取instruction
        # with open('/code/EUV/other_env/eval_agent/prompt/instructions/alfworld_inst.txt') as f:
        #     alfworld_instruction = f.read()
        last_turn_information = ''
        for i in range(len(observations)):
            last_turn_information += f"{observations[i]}\n{actions[i]}\n"
        last_turn_information += f"{new_observation}\n"
        
        ret = PROMPTING_METHOD_VIRTUALHOME_TEMPLATE_BASELINE.format(
            last_turn_information=last_turn_information
        )
        
        return ret

    def get_prompt_our_method(self, task, actions, new_observation, model_name):
        
        pure_actions, _ = self.projection_f(actions[:-1], None)
        # print('actions: ', actions)
        # print('pure_actions: ', pure_actions)
        each_action_history = ", ".join(pure_actions)
        # each_action_history = "\n"
        # for idx, action in enumerate(pure_actions, 1):
        #     each_action_history += f"Step {idx}: {action}\n"

        last_turn_information = ''
        if len(actions) > 0:
            last_turn_information += '\n' + actions[-1] + '\nObservation: ' + new_observation
        else:
            last_turn_information += '\n' + new_observation[:new_observation.find('Your task is to: ')]
        
        prompt = VH_TEMPLATE_OUR_METHOD.format(
            task_goal=task,
            action_history=each_action_history,
            last_turn_information=last_turn_information
        )

        if model_name == 'llama':
            ret = "<|begin_of_text|>"
            ret += "<|start_header_id|>user<|end_header_id|>\n\n" + prompt + "<|eot_id|>"
            ret += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        elif model_name == 'qwen':
            ret = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            ret += '<|im_start|>user' + '\n' + prompt + '<|im_end|>' + '\n'
            ret += '<|im_start|>assistant' + '\n'
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return ret

    def prompting_method_get_prompt_our_method(self, task, actions, new_observation, model_name):
        
        pure_actions, _ = self.projection_f(actions[:-1],None)
        each_action_history = ", ".join(pure_actions)
        # each_action_history = "\n"
        # for idx, action in enumerate(pure_actions, 1):
        #     each_action_history += f"Step {idx}: {action}\n"

        last_turn_information = ''
        if len(actions) > 0:
            last_turn_information += '\n' + actions[-1] + '\nObservation: ' + new_observation
        else:
            last_turn_information += '\n' + new_observation[:new_observation.find('Your task is to: ')]
        
        ret = PROMPTING_METHOD_VIRTUALHOME_TEMPLATE_OUR_METHOD.format(
            task_goal=task,
            action_history=each_action_history,
            last_turn_information=last_turn_information
        )
        
        return ret

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                return

class ScienceWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.memory.reset(batch_size = len(infos))
        self.tasks = self.extract_task(obs, infos)
        observations = {'text': self.build_text_obs(obs, infos, init=True), 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        return observations, infos

    def step(self, text_actions: List[str]):
        self.memory.store({'text_obs': self.pre_text_obs, 'action': text_actions})
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str], infos: List[Dict]):
        tasks = []
        for i, obs in enumerate(text_obs):
            # Extract task from observation or info
            if "Task:" in obs:
                task_start = obs.find("Task:") + len("Task:")
                task_end = obs.find("\n\n", task_start)
                if task_end == -1:
                    task_end = len(obs)
                task_desc = obs[task_start:task_end].strip()
                tasks.append(task_desc)
            elif infos[i] and 'task_description' in infos[i]:
                tasks.append(infos[i]['task_description'])
            else:
                tasks.append("Unknown task")
        return tasks
            
    def build_text_obs(self, text_obs: List[str], infos: List[Dict], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        memory_contexts = []
        valid_lens = []
        for i in range(len(self.memory._data)):
            each_env_actions = []
            each_env_observations = []
            for j in range(len(self.memory._data[i])):
                each_env_actions.append(self.memory._data[i][j]['action'])
                each_env_observations.append("Observation: " + self.memory._data[i][j]['text_obs'])
            memory_contexts.append([each_env_actions, each_env_observations])
            
        for i in range(len(text_obs)):
            task_description = self.tasks[i]
            current_observation = text_obs[i]

            prompt_method = getattr(self.config.env.scienceworld, 'baseline', 'baseline') if hasattr(self.config.env, 'scienceworld') else 'baseline'
            model_name = self.config.env.model_name
            if prompt_method == 'baseline':
                obs = self.get_fastchat_prompt(memory_contexts[i][0], memory_contexts[i][1], current_observation, model_name)
            elif prompt_method == "our_method":
                obs = self.get_prompt_our_method(task_description, memory_contexts[i][0], current_observation, model_name)
            elif prompt_method == 'variant_only_belief':
                obs = self.variat_only_belief(task_description, memory_contexts[i][0], current_observation, model_name)
            elif prompt_method == 'variant_repeat_o':
                obs = self.variant_repeat_o(task_description, memory_contexts[i][0], current_observation, model_name)
            elif prompt_method == 'prompting_baseline':
                obs = self.prompting_method_get_fastchat_prompt(memory_contexts[i][1], memory_contexts[i][0], current_observation, model_name)
            elif prompt_method == 'prompting_our_method':
                obs = self.prompting_method_get_prompt_our_method(task_description, memory_contexts[i][0], current_observation, model_name)
            elif prompt_method == 'prompting_variant_only_belief':
                obs = self.prompting_method_variant_only_belief(task_description, memory_contexts[i][0], current_observation, model_name)
            elif prompt_method == 'prompting_variant_repeat_o':
                obs = self.prompting_method_variant_repeat_o(task_description, memory_contexts[i][0], current_observation, model_name)
            elif prompt_method == 'prompting_vagen':
                obs = self.prompting_method_vagen(memory_contexts[i][1], memory_contexts[i][0], current_observation, model_name)
            elif prompt_method == 'prompting_reflact':
                obs = self.prompting_method_reflact(memory_contexts[i][1], memory_contexts[i][0], current_observation, model_name)
            else:
                raise ValueError(f"Unknown prompt method: {prompt_method}")

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def get_fastchat_prompt(self, actions, observations, new_observation, model_name):
        # Read instruction
        instruction_path = getattr(self.config.env.scienceworld, 'instruction_path', None) if hasattr(self.config.env, 'scienceworld') else None
        if instruction_path and os.path.exists(instruction_path):
            with open(instruction_path, 'r') as f:
                scienceworld_instruction = f.read()
        else:
            # Default instruction
            scienceworld_instruction = "You are a helpful assistant to do some scientific experiment in an environment."
        
        if model_name == 'llama':
            scienceworld_messages = []
            scienceworld_messages.append(['user', scienceworld_instruction])
            scienceworld_messages.append(['assistant', 'OK'])

            for action, observation in zip(actions, observations):
                scienceworld_messages.append(['user', observation])
                scienceworld_messages.append(['assistant', action])

            scienceworld_messages.append(['user', new_observation])
            scienceworld_messages.append(['assistant', None])

            ret = "<|begin_of_text|>"
            for i, (role, message) in enumerate(scienceworld_messages):
                if message:
                    ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
                    ret += f"{message.strip()}<|eot_id|>"
                else:
                    ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
                    
            return ret
        elif model_name == 'qwen':
            scienceworld_messages = []
            scienceworld_messages.append(['<|im_start|>user', scienceworld_instruction])
            scienceworld_messages.append(['<|im_start|>assistant', 'OK'])

            for action, observation in zip(actions, observations):
                scienceworld_messages.append(['<|im_start|>user', observation])
                scienceworld_messages.append(['<|im_start|>assistant', action])

            scienceworld_messages.append(['<|im_start|>user', new_observation])
            scienceworld_messages.append(['<|im_start|>assistant', None])

            ret = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"

            for role, message in scienceworld_messages:
                if message:
                    ret += role + "\n" + message + "<|im_end|>" + "\n"
                else:
                    ret += role + "\n"
            
            return ret
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def prompting_method_get_fastchat_prompt(self, observations, actions, new_observation, model_name):
        # # 读取instruction
        # with open('/code/EUV/other_env/eval_agent/prompt/instructions/alfworld_inst.txt') as f:
        #     alfworld_instruction = f.read()
        last_turn_information = ''
        for i in range(len(observations)):
            last_turn_information += f"{observations[i]}\n{actions[i]}\n"
        last_turn_information += f"{new_observation}\n"
        
        ret = PROMPTING_METHOD_SCIENCEWORLD_TEMPLATE_BASELINE.format(
            last_turn_information=last_turn_information
        )
        
        return ret

    def prompting_method_vagen(self, observations, actions, new_observation, model_name):
        # # 读取instruction
        # with open('/code/EUV/other_env/eval_agent/prompt/instructions/alfworld_inst.txt') as f:
        #     alfworld_instruction = f.read()
        last_turn_information = ''
        for i in range(len(observations)):
            last_turn_information += f"{observations[i]}\n{actions[i]}\n"
        last_turn_information += f"{new_observation}\n"
        
        ret = VAGEN_PROMPTING_METHOD_SCIENCEWORLD_TEMPLATE_BASELINE.format(
            last_turn_information=last_turn_information
        )
        
        return ret

    def prompting_method_reflact(self, observations, actions, new_observation, model_name):
        # # 读取instruction
        # with open('/code/EUV/other_env/eval_agent/prompt/instructions/alfworld_inst.txt') as f:
        #     alfworld_instruction = f.read()
        last_turn_information = ''
        for i in range(len(observations)):
            last_turn_information += f"{observations[i]}\n{actions[i]}\n"
        last_turn_information += f"{new_observation}\n"
        
        ret = REFLACT_PROMPTING_METHOD_SCIENCEWORLD_TEMPLATE_BASELINE.format(
            last_turn_information=last_turn_information
        )
        
        return ret

    def get_prompt_our_method(self, task, actions, new_observation, model_name):
        
        pure_actions, _ = self.projection_f(actions[:-1])
        # print('actions: ', actions)
        # print('pure_actions: ', pure_actions)
        each_action_history = ", ".join(pure_actions)
        # each_action_history = "\n"
        # for idx, action in enumerate(pure_actions, 1):
        #     each_action_history += f"Step {idx}: {action}\n"

        last_turn_information = ''
        if len(actions) > 0:
            last_turn_information += '\n' + actions[-1] + '\nObservation: ' + new_observation
        else:
            last_turn_information += '\n' + new_observation[:new_observation.find('Your task is to: ')]
        
        prompt = SCIENCEWORLD_TEMPLATE_OUR_METHOD.format(
            task_goal=task,
            action_history=each_action_history,
            last_turn_information=last_turn_information
        )

        if model_name == 'llama':
            ret = "<|begin_of_text|>"
            ret += "<|start_header_id|>user<|end_header_id|>\n\n" + prompt + "<|eot_id|>"
            ret += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        elif model_name == 'qwen':
            ret = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            ret += '<|im_start|>user' + '\n' + prompt + '<|im_end|>' + '\n'
            ret += '<|im_start|>assistant' + '\n'
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return ret

    def prompting_method_get_prompt_our_method(self, task, actions, new_observation, model_name):
        
        pure_actions, _ = self.projection_f(actions[:-1])
        each_action_history = ", ".join(pure_actions)
        # each_action_history = "\n"
        # for idx, action in enumerate(pure_actions, 1):
        #     each_action_history += f"Step {idx}: {action}\n"

        last_turn_information = ''
        if len(actions) > 0:
            last_turn_information += '\n' + actions[-1] + '\nObservation: ' + new_observation
        else:
            last_turn_information += '\n' + new_observation[:new_observation.find('Your task is to: ')]
        
        ret = PROMPTING_METHOD_SCIENCEWORLD_TEMPLATE_OUR_METHOD.format(
            task_goal=task,
            action_history=each_action_history,
            last_turn_information=last_turn_information
        )
        
        return ret

    def prompting_method_variant_only_belief(self, task, actions, new_observation, model_name):
        
        pure_actions, _ = self.projection_f(actions[:-1])
        each_action_history = ", ".join(pure_actions)
        # each_action_history = "\n"
        # for idx, action in enumerate(pure_actions, 1):
        #     each_action_history += f"Step {idx}: {action}\n"

        last_turn_information = ''
        if len(actions) > 0:
            last_turn_information += '\n' + actions[-1] + '\nObservation: ' + new_observation
        else:
            last_turn_information += '\n' + new_observation[:new_observation.find('Your task is to: ')]
        
        ret = PROMPTING_VARIANR_SCIENCEWORLD_TEMPLATE_ONLY_BELIEF_THOUGHT_ACTION.format(
            task_goal=task,
            action_history=each_action_history,
            last_turn_information=last_turn_information
        )
        
        return ret

    def prompting_method_variant_repeat_o(self, task, actions, new_observation, model_name):
        
        pure_actions, _ = self.projection_f(actions[:-1])
        each_action_history = ", ".join(pure_actions)
        # each_action_history = "\n"
        # for idx, action in enumerate(pure_actions, 1):
        #     each_action_history += f"Step {idx}: {action}\n"

        last_turn_information = ''
        if len(actions) > 0:
            last_turn_information += '\n' + actions[-1] + '\nObservation: ' + new_observation
        else:
            last_turn_information += '\n' + new_observation[:new_observation.find('Your task is to: ')]
        
        ret = PROMPTING_VARIANR_SCIENCEWORLD_TEMPLATE_REPEAT_OBSERVATION_THOUGHT_ACTION.format(
            task_goal=task,
            action_history=each_action_history,
            last_turn_information=last_turn_information
        )
        
        return ret

    def variat_only_belief(self, task, actions, new_observation, model_name):
        
        pure_actions, _ = self.projection_f(actions[:-1])
        # print('actions: ', actions)
        # print('pure_actions: ', pure_actions)
        each_action_history = ", ".join(pure_actions)
        # each_action_history = "\n"
        # for idx, action in enumerate(pure_actions, 1):
        #     each_action_history += f"Step {idx}: {action}\n"

        last_turn_information = ''
        if len(actions) > 0:
            last_turn_information += '\n' + actions[-1] + '\nObservation: ' + new_observation
        else:
            last_turn_information += '\n' + new_observation[:new_observation.find('Your task is to: ')]
        
        prompt = VARIANR_SCIENCEWORLD_TEMPLATE_ONLY_BELIEF_THOUGHT_ACTION.format(
            task_goal=task,
            action_history=each_action_history,
            last_turn_information=last_turn_information
        )

        if model_name == 'llama':
            ret = "<|begin_of_text|>"
            ret += "<|start_header_id|>user<|end_header_id|>\n\n" + prompt + "<|eot_id|>"
            ret += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        elif model_name == 'qwen':
            ret = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            ret += '<|im_start|>user' + '\n' + prompt + '<|im_end|>' + '\n'
            ret += '<|im_start|>assistant' + '\n'
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return ret

    def variant_repeat_o(self, task, actions, new_observation, model_name):
        
        pure_actions, _ = self.projection_f(actions[:-1])
        # print('actions: ', actions)
        # print('pure_actions: ', pure_actions)
        each_action_history = ", ".join(pure_actions)
        # each_action_history = "\n"
        # for idx, action in enumerate(pure_actions, 1):
        #     each_action_history += f"Step {idx}: {action}\n"

        last_turn_information = ''
        if len(actions) > 0:
            last_turn_information += '\n' + actions[-1] + '\nObservation: ' + new_observation
        else:
            last_turn_information += '\n' + new_observation[:new_observation.find('Your task is to: ')]
        
        prompt = VARIANR_SCIENCEWORLD_TEMPLATE_REPEAT_OBSERVATION_THOUGHT_ACTION.format(
            task_goal=task,
            action_history=each_action_history,
            last_turn_information=last_turn_information
        )

        if model_name == 'llama':
            ret = "<|begin_of_text|>"
            ret += "<|start_header_id|>user<|end_header_id|>\n\n" + prompt + "<|eot_id|>"
            ret += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        elif model_name == 'qwen':
            ret = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            ret += '<|im_start|>user' + '\n' + prompt + '<|im_end|>' + '\n'
            ret += '<|im_start|>assistant' + '\n'
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return ret

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                return

class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"
                
                if len(action_history) > 10000:
                    action_history = "... " + action_history[-10000:]

                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs

def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        projection_f = partial(search_projection)
        envs = SearchEnvironmentManager(_envs, projection_f, config)
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, resources_per_worker=resources_per_worker)
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, resources_per_worker=resources_per_worker)
        
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': config.env.alfworld.eval_dataset, # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth
        }
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0, resources_per_worker=resources_per_worker)
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n, resources_per_worker=resources_per_worker)
        
        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "virtualhome" in config.env.env_name.lower():
        from agent_system.environments.env_package.virtualhome import build_virtualhome_envs, virtualhome_projection
        import json
        
        def load_task_data_list(data_path):
            """Helper function to load task data list from json/jsonl file"""
            task_data_list = []
            if data_path and os.path.exists(data_path):
                with open(data_path, 'r') as f:
                    # Check if it's a JSON array or JSONL (line-delimited JSON)
                    first_char = f.read(1)
                    f.seek(0)
                    if first_char == '[':
                        # JSON array format
                        task_data_list = json.load(f)
                    else:
                        # JSONL format (one JSON object per line)
                        for line in f:
                            if line.strip():
                                task_data_list.append(json.loads(line))
            return task_data_list
        
        # Load training task data list
        train_task_data_list = []
        train_data_path = None
        # Priority: train_data_path > task_data_path > default
        if hasattr(config.env, 'virtualhome') and hasattr(config.env.virtualhome, 'train_data_path'):
            train_data_path = config.env.virtualhome.train_data_path
        elif hasattr(config.env, 'virtualhome') and hasattr(config.env.virtualhome, 'task_data_path'):
            train_data_path = config.env.virtualhome.task_data_path
        else:
            # Default training data path
            train_data_path = '/code/EUV/other_env/eval_agent/sft/data/new_train_sft.json'
        
        train_task_data_list = load_task_data_list(train_data_path)
        if len(train_task_data_list) > 0:
            print(f"Loaded {len(train_task_data_list)} training tasks from {train_data_path}")
        else:
            print(f"Warning: Training data list is empty from {train_data_path}")
        
        # Load validation task data list (seen or unseen)
        val_task_data_list = []
        val_data_path = None
        
        # Check if val_data_path is explicitly configured
        if hasattr(config.env, 'virtualhome') and hasattr(config.env.virtualhome, 'val_data_path'):
            val_data_path = config.env.virtualhome.val_data_path
        else:
            # Auto-detect based on val_split
            val_split = config.env.virtualhome.get('val_split', 'seen') if hasattr(config.env, 'virtualhome') else 'seen'
            if val_split == 'seen':
                val_data_path = '/code/EUV/other_env/eval_agent/sft/data/new_seen_test.jsonl'
            elif val_split == 'unseen':
                val_data_path = '/code/EUV/other_env/eval_agent/sft/data/new_unseen_test.jsonl'
            else:
                # Fallback to train if val_split is not seen or unseen
                val_data_path = config.env.virtualhome.task_data_path if hasattr(config.env, 'virtualhome') and hasattr(config.env.virtualhome, 'task_data_path') else None
        
        val_task_data_list = load_task_data_list(val_data_path)
        
        # If validation data list is empty, fallback to training data
        if len(val_task_data_list) == 0:
            print(f"Warning: Validation data list is empty, falling back to training data")
            val_task_data_list = train_task_data_list
        else:
            print(f"Loaded {len(val_task_data_list)} validation tasks from {val_data_path}")
        
        env_kwargs = {
            'max_steps': config.env.max_steps if hasattr(config.env, 'max_steps') else 50,
        }
        if hasattr(config.env, 'virtualhome'):
            if hasattr(config.env.virtualhome, 'max_steps'):
                env_kwargs['max_steps'] = config.env.virtualhome.max_steps
        
        _envs = build_virtualhome_envs(
            seed=config.env.seed, 
            env_num=config.data.train_batch_size, 
            group_n=group_n, 
            is_train=True, 
            env_kwargs=env_kwargs, 
            task_data_list=train_task_data_list,
            resources_per_worker=resources_per_worker
        )
        _val_envs = build_virtualhome_envs(
            seed=config.env.seed + 1000, 
            env_num=config.data.val_batch_size, 
            group_n=1, 
            is_train=False, 
            env_kwargs=env_kwargs, 
            task_data_list=val_task_data_list,
            resources_per_worker=resources_per_worker
        )

        projection_f = partial(virtualhome_projection)
        envs = VirtualHomeEnvironmentManager(_envs, projection_f, config)
        val_envs = VirtualHomeEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "scienceworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.scienceworld import build_scienceworld_envs, scienceworld_projection
        import json
        
        def load_task_data_list(indices_path, config):
            """Helper function to load task data list from indices file"""
            task_data_list = []
            if hasattr(config.env, 'scienceworld') and hasattr(config.env.scienceworld, 'task_data_path'):
                task_data_path = config.env.scienceworld.task_data_path
                if os.path.exists(task_data_path):
                    with open(task_data_path, 'r') as f:
                        for line in f:
                            if line.strip():
                                task_data_list.append(json.loads(line))
            elif indices_path and os.path.exists(indices_path):
                # Load from indices file (similar to eval_agent format)
                with open(indices_path, 'r') as f:
                    task_indices = json.load(f)
                taskname2id_path = config.env.scienceworld.get('taskname2id_path', None)
                taskname2id = {}
                if taskname2id_path and os.path.exists(taskname2id_path):
                    with open(taskname2id_path, 'r') as f:
                        taskname2id = json.load(f)
                for item in task_indices:
                    task_name = item[0]
                    variation_idx = item[1]
                    task_data_list.append({
                        'sub_task_name': task_name,
                        'variation_idx': variation_idx,
                        'simplification_str': config.env.scienceworld.get('simplification_str', 'easy'),
                        'generate_gold_path': config.env.scienceworld.get('generate_gold_path', False)
                    })
            return task_data_list
        
        # Load training task data list
        train_task_data_list = []
        if hasattr(config.env, 'scienceworld') and hasattr(config.env.scienceworld, 'task_indices_path'):
            train_indices_path = config.env.scienceworld.task_indices_path
            train_task_data_list = load_task_data_list(train_indices_path, config)
        
        # Load validation task data list (dev or test)
        val_task_data_list = []
        val_indices_path = None
        
        # Check if val_indices_path is explicitly configured
        if hasattr(config.env, 'scienceworld') and hasattr(config.env.scienceworld, 'val_indices_path'):
            val_indices_path = config.env.scienceworld.val_indices_path
        else:
            # Auto-detect based on val_split
            val_split = config.env.scienceworld.get('val_split', 'dev') if hasattr(config.env, 'scienceworld') else 'dev'
            if val_split == 'dev':
                val_indices_path = '/code/EUV/other_env/dev_indices_filtered.json'
            elif val_split == 'test':
                val_indices_path = '/code/EUV/other_env/test_indices_filtered.json'
            else:
                # Fallback to train if val_split is not dev or test
                val_indices_path = config.env.scienceworld.task_indices_path if hasattr(config.env, 'scienceworld') and hasattr(config.env.scienceworld, 'task_indices_path') else None
        
        val_task_data_list = load_task_data_list(val_indices_path, config)
        
        # If validation data list is empty, fallback to training data
        if len(val_task_data_list) == 0:
            print(f"Warning: Validation data list is empty, falling back to training data")
            val_task_data_list = train_task_data_list
        else:
            print(f"Loaded {len(val_task_data_list)} validation tasks from {val_indices_path}")
        
        # Prepare environment kwargs
        env_kwargs = {
            'max_steps': config.env.max_steps if hasattr(config.env, 'max_steps') else 200,
        }
        if hasattr(config.env, 'scienceworld'):
            if hasattr(config.env.scienceworld, 'max_steps'):
                env_kwargs['max_steps'] = config.env.scienceworld.max_steps
            if hasattr(config.env.scienceworld, 'server_path'):
                env_kwargs['server_path'] = config.env.scienceworld.server_path
            if hasattr(config.env.scienceworld, 'max_steps_path'):
                env_kwargs['max_steps_path'] = config.env.scienceworld.max_steps_path
            env_kwargs['apply_monkey_patch'] = config.env.scienceworld.get('apply_monkey_patch', True)
        
        _envs = build_scienceworld_envs(
            seed=config.env.seed, 
            env_num=config.data.train_batch_size, 
            group_n=group_n, 
            is_train=True, 
            env_kwargs=env_kwargs, 
            task_data_list=train_task_data_list,
            resources_per_worker=resources_per_worker
        )
        _val_envs = build_scienceworld_envs(
            seed=config.env.seed + 1000, 
            env_num=config.data.val_batch_size, 
            group_n=1, 
            is_train=False, 
            env_kwargs=env_kwargs, 
            task_data_list=val_task_data_list,
            resources_per_worker=resources_per_worker
        )

        projection_f = partial(scienceworld_projection)
        envs = ScienceWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = ScienceWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)