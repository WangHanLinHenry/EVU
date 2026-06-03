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

import ray
import gym
import numpy as np
import json
import os
import sys

# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class ScienceWorldWorker:
    """Ray remote actor that hosts a ScienceWorld environment instance."""
    
    def __init__(self, seed, env_kwargs):
        # Lazy import avoids initialization issues
        from scienceworld import ScienceWorldEnv
        
        self.env_kwargs = env_kwargs
        self.seed = seed
        self.env = None
        self.max_steps_dict = None
        self.current_task = None
        self.current_variation_idx = None
        self.steps = 0
        self.max_steps = 200
        self.last_step_score = 0
        
        # Load max steps dictionary
        max_steps_path = env_kwargs.get('max_steps_path', None)
        if max_steps_path and os.path.exists(max_steps_path):
            with open(max_steps_path, 'r') as f:
                self.max_steps_dict = json.load(f)
        
        # Initialize environment
        server_path = env_kwargs.get('server_path', None)
        if server_path:
            self.env = ScienceWorldEnv("", serverPath=server_path, envStepLimit=self.max_steps)
        else:
            self.env = ScienceWorldEnv("", envStepLimit=self.max_steps)
        
        # Apply monkey patch for reward calculation if needed
        if env_kwargs.get('apply_monkey_patch', True):
            self._apply_monkey_patch()
    
    def _apply_monkey_patch(self):
        """Apply monkey patch to ScienceWorldEnv.step for reward calculation"""
        # Store reference to original step method
        original_step = self.env.__class__.step
        
        def patched_step(env_instance, inputStr: str):
            observation = env_instance.server.step(inputStr)
            raw_score = env_instance.server.getScore()
            score = int(round(100 * raw_score))
            isCompleted = env_instance.server.getCompleted()
            numMoves = env_instance.getNumMoves()
            
            # Calculate reward (delta score)
            reward = score - self.last_step_score
            self.last_step_score = score
            
            # Check step limit
            if numMoves > env_instance.envStepLimit:
                isCompleted = True
            
            # If score is less than zero, set completed flag
            if score < 0:
                isCompleted = True
            
            infos = {
                'moves': numMoves,
                'raw_score': raw_score,
                'score': score,
                'reward': reward,
                'look': env_instance.look(),
                'inv': env_instance.inventory(),
                'taskDesc': env_instance.taskdescription(),
                'valid': env_instance.getValidActionObjectCombinations(),
                'variationIdx': env_instance.variationIdx,
                'taskName': env_instance.taskName,
                'simplificationStr': env_instance.simplificationStr,
            }
            
            return observation, reward, isCompleted, infos
        
        # Bind the patched method to the instance
        import types
        self.env.step = types.MethodType(patched_step, self.env)
    
    def reset(self, task_data):
        """Reset the environment with given task data"""
        if task_data is None:
            return "No task available.", {"won": False, "task": "unknown"}
        
        self.current_task = task_data.get('sub_task_name')
        self.current_variation_idx = task_data.get('variation_idx', 0)
        self.steps = 0
        self.last_step_score = 0
        
        # Set max steps based on task
        if self.max_steps_dict and self.current_task:
            self.max_steps = self.max_steps_dict.get(self.current_task, 200)
            self.env.envStepLimit = self.max_steps
        
        # Load task
        simplification_str = task_data.get('simplification_str', 'easy')
        generate_gold_path = task_data.get('generate_gold_path', False)
        
        try:
            self.env.load(
                self.current_task, 
                self.current_variation_idx, 
                simplificationStr=simplification_str,
                generateGoldPath=generate_gold_path
            )
            obs, info = self.env.reset()
            
            task_desc = info.get('taskDesc', '')
            obs = f"Task: {task_desc}\n\n{obs}"
            
            info['won'] = False
            info['task'] = self.current_task
            info['task_description'] = task_desc
            
            return obs, info
        except Exception as e:
            return f"Error loading task: {str(e)}", {"won": False, "task": self.current_task}
    
    def step(self, action):
        """Execute a step in the environment"""
        self.steps += 1
        
        try:
            observation, reward, done, info = self.env.step(action)
            
            # Update info
            info['won'] = done
            info['task'] = self.current_task
            info['task_description'] = info.get('taskDesc', '')
            
            # Check if max steps exceeded
            if self.steps >= self.max_steps and not done:
                done = True
                info['won'] = False
            
            # Convert reward to success-based reward (10.0 for success, 0.0 otherwise)
            if done and info['won']:
                final_reward = 10.0
            else:
                final_reward = 0.0
            
            return observation, final_reward, done, info
            
        except Exception as e:
            observation = f"Error: {str(e)}"
            reward = 0.0
            done = False
            info = {
                'won': False,
                'task': self.current_task,
                'task_description': 'unknown',
                'error': str(e)
            }
            return observation, reward, done, info
    
    def close(self):
        """Close the environment"""
        if self.env:
            try:
                self.env.close()
            except:
                pass


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class ScienceWorldMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around ScienceWorld environment."""
    
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
        task_data_list: list = None,
    ) -> None:
        super().__init__()

        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train:
            assert group_n == 1

        self._rng = np.random.RandomState(seed)
        self._env_kwargs = env_kwargs if env_kwargs is not None else {}
        self.task_data_list = task_data_list if task_data_list is not None else []

        # Ray actors setup
        env_worker = ray.remote(**resources_per_worker)(ScienceWorldWorker)
        self._workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(seed + (i // self.group_n), self._env_kwargs)
            self._workers.append(worker)

    def step(self, actions: list):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        futures = []
        for worker, action in zip(self._workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        if len(self.task_data_list) > 0:
            # Use provided task data
            # idx = self._rng.choice(len(self.task_data_list), size=self.env_num, replace=True)
            idx = np.arange(min(self.env_num, len(self.task_data_list)))
            idx = np.repeat(idx, self.group_n).tolist()
            task_data_to_use = [self.task_data_list[i] for i in idx]
        else:
            # If no task data provided, create empty tasks
            task_data_to_use = [None] * self.num_processes

        futures = []
        for worker, task_data in zip(self._workers, task_data_to_use):
            if task_data is not None:
                future = worker.reset.remote(task_data)
                futures.append(future)
            else:
                # Return empty observation if no task data
                futures.append(ray.put(("No task available.", {"won": False})))

        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    def close(self):
        if getattr(self, '_closed', False):
            return

        close_futures = []
        for worker in self._workers:
            future = worker.close.remote()
            close_futures.append(future)
        
        ray.get(close_futures)
        
        for worker in self._workers:
            ray.kill(worker)
            
        self._closed = True

    def __del__(self):
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_scienceworld_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
    task_data_list: list = None,
):
    """Build ScienceWorld multi-process environments."""
    return ScienceWorldMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
        task_data_list=task_data_list,
    )

