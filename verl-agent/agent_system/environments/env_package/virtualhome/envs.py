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
import re
import random
import sys
from enum import Enum
from collections import OrderedDict
import parse

# Add virtualhome path
sys.path.append('/code/STeCa/virtualhome_master')
from virtualhome.simulation.evolving_graph import utils
from virtualhome.simulation.evolving_graph.scripts import parse_script_line, Script
from virtualhome.simulation.evolving_graph.execution import ScriptExecutor
from virtualhome.simulation.evolving_graph.environment import EnvironmentGraph, EnvironmentState

# Load class name equivalence
with open("/code/STeCa/class_name_equivalence.json", 'r') as f:
    abstract2detail = json.load(f)

detail2abstract = dict()
def merge_add(d, k, v):
    if k == v:
        return
    if k in d:
        prev_v = d[k]
        merge_add(d, v, prev_v)
    else:
        d[k] = v

for abstract, details in abstract2detail.items():
    for detail in details:
        merge_add(detail2abstract, detail, abstract)

def process_format(arg):
    arg = arg.replace(' ', '_')
    return arg

def remove_duplicate_edge(input_dict):
    Edges = input_dict['edges']
    for edge in Edges:
        fgledge = {'from_id':edge['to_id'], 'relation_type': 'INSIDE', 'to_id': edge['from_id']}
        if fgledge in Edges:
            if edge == fgledge:
                Edges.remove(edge)
            else:
                Edges.remove(fgledge)
                Edges.remove(edge)
    input_dict['edges'] = Edges
    return input_dict

class EvolveGraphAction(Enum):
    """All supported actions"""
    CLOSE = ("Close", 1, 'close {}')
    DRINK = ("Drink", 1, 'drink {}')
    FIND = ("Find", 1, 'find {}')
    WALK = ("Walk", 1, 'walk to {}')
    GRAB = ("Grab", 1, 'grab {}')
    LOOKAT = ("Look at", 1, 'look at {}')
    OPEN = ("Open", 1, 'open {}')
    POINTAT = ("Point at", 1, 'point at {}')
    PUTBACK = ("Put", 2, 'put {} on {}')
    PUTIN = ("Put in", 2, 'put {} in {}')
    PUTOBJBACK = ("Put back", 1, 'put back {}')
    RUN = ("Run", 1, 'run to {}')
    SIT = ("Sit", 1, 'sit on {}')
    STANDUP = ("Stand up", 0, 'stand up')
    SWITCHOFF = ("Switch off", 1, 'switch off {}')
    SWITCHON = ("Switch on", 1, 'switch on {}')
    TOUCH = ("Touch", 1, 'touch {}')
    TURNTO = ("Turn to", 1, 'turn to {}')
    WATCH = ("Watch", 1, 'watch {}')
    WIPE = ("Wipe", 1, 'wipe {}')
    PUTON = ("PutOn", 1, 'put on {}')
    PUTOFF = ("PutOff", 1, 'take off {}')
    GREET = ("Greet", 1, 'greet {}')
    DROP = ("Drop", 1, 'drop {}')
    READ = ("Read", 1, 'read {}')
    LIE = ("Lie", 1, 'lie on {}')
    POUR = ("Pour", 2, 'pour {} into {}')
    TYPE = ("Type", 1, 'type on {}')
    PUSH = ("Push", 1, 'push {}')
    PULL = ("Pull", 1, 'pull {}')
    MOVE = ("Move", 1, 'move {}')
    WASH = ("Wash", 1, 'wash {}')
    RINSE = ("Rinse", 1, 'rinse {}')
    SCRUB = ("Scrub", 1, 'scrub {}')
    SQUEEZE = ("Squeeze", 1, 'squeeze {}')
    PLUGIN = ("PlugIn", 1, 'plug in {}')
    PLUGOUT = ("PlugOut", 1, 'plug out {}')
    CUT = ("Cut", 1, 'cut {}')
    EAT = ("Eat", 1, 'eat {}') 
    SLEEP = ("Sleep", 0, 'sleep')
    WAKEUP = ("WakeUp", 0, 'wake up')
    RELEASE = ("Release", 1, 'release')

def check_action_format(program_text):
    action = re.findall(r'\[(.*?)\]', program_text)[0]
    num_para = EvolveGraphAction[action].value[1]
    action_para = program_text.count('<')
    if num_para == action_para:
        return program_text
    else:
        return action + " needs " + str(num_para) + " parameters. But there are " + str(action_para) + " parameters."

def change_obj_index(graph, program, id, specific_objects, last_obj_id):
    graph_dict = graph.to_dict()
    agent_has_objid = [n['to_id'] for n in graph_dict["edges"] if n['from_id'] == id and "HOLD" in n["relation_type"]]

    obj_id_dict = {}
    obj_ids_close = [n['to_id'] for n in graph_dict["edges"] if n['from_id'] == id and  n["relation_type"]=="CLOSE"]
    obj_ids_close_two = [n['from_id'] for n in graph_dict["edges"] if n['to_id'] == id and  n["relation_type"]=="CLOSE"]
    obj_ids_close.extend(obj_ids_close_two)
    obj_ids_close = list(set(obj_ids_close))
    obj = []
    for i in range(len(obj_ids_close)):
        obj.append([node['class_name'] for node in graph_dict['nodes'] if node['id']==obj_ids_close[i]][0])

    if last_obj_id != -1:
        last_obj_ids_close = [n['to_id'] for n in graph_dict["edges"] if n['from_id'] == last_obj_id and  n["relation_type"]=="CLOSE"]
        last_obj_ids_close_two = [n['from_id'] for n in graph_dict["edges"] if n['to_id'] == last_obj_id and  n["relation_type"]=="CLOSE"]
        last_obj_ids_close.extend(last_obj_ids_close_two)
        last_obj_ids_close = list(set(last_obj_ids_close))

        last_obj_ids_inside = [n['to_id'] for n in graph_dict["edges"] if n['from_id'] == last_obj_id and  n["relation_type"]=="INSIDE"]
        last_obj_ids_inside_two = [n['from_id'] for n in graph_dict["edges"] if n['to_id'] == last_obj_id and  n["relation_type"]=="INSIDE"]
        last_obj_ids_inside.extend(last_obj_ids_inside_two)
        last_obj_ids_inside = list(set(last_obj_ids_inside))   
        last_obj_ids_close.extend(last_obj_ids_inside)     

        last_obj = []
        for i in range(len(last_obj_ids_close)):
            last_obj.append([node['class_name'] for node in graph_dict['nodes'] if node['id']==last_obj_ids_close[i]][0])
    else:
        last_obj_ids_close = []
        last_obj = []

    if program.count('<') == 0:
        return program, specific_objects, last_obj_id
    
    if program.count('<') == 1:
        def extract_text(input_string):
            pattern = r'\[([^]]+)\]|\<([^>]+)\>|\(([^)]+)\)'
            matches = re.findall(pattern, input_string)
            extracted_text = [match[0] or match[1] or match[2] for match in matches]
            return extracted_text
        
        extracted_text = extract_text(program)

        for i in range(len(obj_ids_close)):
            if obj[i] == extracted_text[1]:
                obj_id_dict[obj[i]] = obj_ids_close[i]

        for i in range(len(last_obj_ids_close)):
            if last_obj[i] == extracted_text[1]:
                obj_id_dict[last_obj[i]] = last_obj_ids_close[i]

        if extracted_text[0] not in ['FIND', 'WALK']:
            obj_id1 = [node['id'] for node in graph_dict['nodes'] if node['class_name'] == extracted_text[1]]

            if extracted_text[1] in list(specific_objects.keys()):
                id1 = specific_objects[extracted_text[1]]
            elif extracted_text[1] in list(obj_id_dict.keys()):
                id1 = obj_id_dict[extracted_text[1]]
                specific_objects[extracted_text[1]] = id1
            elif len(obj_id1) == 0:
                return extracted_text[1] + " isn't available in the environment.", specific_objects, last_obj_id
            else:
                id1 = random.choice(obj_id1)
                specific_objects[extracted_text[1]] = id1
            pattern = r'\d+'
            replaced_string = re.sub(pattern, str(id1), program)     
            return replaced_string, specific_objects, id1   
        else:
            obj_id1 = [node['id'] for node in graph_dict['nodes'] if node['class_name'] == extracted_text[1]]
            if len(obj_id1)==0:
                return extracted_text[1] + " isn't available in the environment.", specific_objects, last_obj_id

            if extracted_text[1] in list(specific_objects.keys()):
                id1 = specific_objects[extracted_text[1]]
            elif extracted_text[1] in list(obj_id_dict.keys()):
                id1 = obj_id_dict[extracted_text[1]]
                specific_objects[extracted_text[1]] = id1
            else:
                id1 = random.choice(obj_id1)
                specific_objects[extracted_text[1]] = id1
            
            pattern = r'\d+'
            replaced_string = re.sub(pattern, str(id1), program)
            return replaced_string, specific_objects, id1  
            
    if program.count('<') == 2:
        ori_specific_objects = specific_objects

        def parse_content(input_string):
            pattern = r'\[(.*?)\]|\<(.*?)\>|\((.*?)\)'
            matches = re.findall(pattern, input_string)
            parsed_content = [group for match in matches for group in match if group]
            return parsed_content
        
        content = parse_content(program)
        obj_id1 = [node['id'] for node in graph_dict['nodes'] if node['class_name'] == content[1] and node['id'] in agent_has_objid]
        obj_id2 = [node['id'] for node in graph_dict['nodes'] if node['class_name'] == content[3]]
        
        for i in range(len(obj_ids_close)):
            if obj[i] == content[1]:
                obj_id_dict[obj[i]] = obj_ids_close[i]
            if obj[i] == content[3]:
                obj_id_dict[obj[i]] = obj_ids_close[i]

        for i in range(len(last_obj_ids_close)):
            if last_obj[i] == content[1]:
                obj_id_dict[last_obj[i]] = last_obj_ids_close[i]
            if last_obj[i] == content[3]:
                obj_id_dict[last_obj[i]] = last_obj_ids_close[i]

        if len(obj_id1) == 0:
            return content[1] + " not in hand. Robot agent should hold " + content[1] + " firstly.", specific_objects, last_obj_id

        id1 = random.choice(obj_id1)
        specific_objects[content[1]] = id1

        if len(obj_id2) == 0:
            return content[3] + " isn't available in the environment.", specific_objects, last_obj_id
        elif content[3] in list(specific_objects.keys()):
            id2 = specific_objects[content[3]]
        elif content[3] in list(obj_id_dict.keys()):
            id2 = obj_id_dict[content[3]]
            specific_objects[content[3]] = id2
        else:
            id2 = random.choice(obj_id2)
            specific_objects[content[3]] = id2

        if id1 == id2:
            return content[1] + " can't be put or pour into itself.", ori_specific_objects, last_obj_id

        program_list = list(program)
        positions = [index for index, element in enumerate(program_list) if element == ')']
        qian_program = program[:positions[0]+1]
        hou_program = program[positions[0]+1:]
        qian_program = re.sub(r'\((\d+)\)', '('+str(id1)+')', qian_program, count=1)
        hou_program = re.sub(r'\((\d+)\)', '('+str(id2)+')', hou_program, count=1)
        program = qian_program + hou_program

        return program, specific_objects, id2

def str2program_list(program_lines):
    def _format_arg(arg):
        arg = arg.lower().strip().replace(' ', '_')
        if arg in detail2abstract:
            return detail2abstract[arg]
        return arg

    info = dict()
    info['parsing_error'] = []
    pl = program_lines
    parsed_lines = []
    success_count = 0
    for i, line in enumerate(pl):
        line = line.lower().strip()
        if len(line) == 0:
            continue
        if ':' in line:
            line = line[line.index(':') + 1:].strip()
        try:
            possible_parsed = OrderedDict()
            for action in EvolveGraphAction:
                action_template = action.value[2]
                expected_num_args = action.value[1]
                parsed = parse.parse(action_template, line)
                if parsed is not None:
                    assert action.name not in possible_parsed
                    if len(parsed.fixed) == expected_num_args:
                        possible_parsed[action.name] = parsed
            assert len(possible_parsed) == 1, f'possible_parsed: {possible_parsed} does not equal to 1'
            parsed_action = list(possible_parsed.keys())[0]
            parsed_args = possible_parsed[parsed_action]
            if len(parsed_args.fixed) == 0:
                pl_str = '[{}]'
                pl_str = pl_str.format(parsed_action)
            elif len(parsed_args.fixed) == 1:
                pl_str = '[{}] <{}> (1)'
                pl_str = pl_str.format(parsed_action, process_format(parsed_args[0]))
            elif len(parsed_args.fixed) == 2:
                pl_str = '[{}] <{}> (1) <{}> (1)'
                pl_str = pl_str.format(parsed_action, process_format(parsed_args[0]), process_format(parsed_args[1]))
            else:
                raise NotImplementedError
            parsed_lines.append(pl_str)
            success_count += 1
        except AssertionError as e:
            message = "| {} | {} | '{}'".format(e.__class__.__name__, e, line)
            info['parsing_error'].append(message)
            line = pl[i]
            if ':' in line:
                line = line[line.index(':') + 1:].strip()
            if len(line) > 0:
                words = line.split(' ')
                if len(words) == 1:
                    pl_str = '[{}]'.format(words[0].upper())
                elif len(words) == 2:
                    pl_str = '[{}] <{}> (1)'.format(words[0].upper(), words[1])
                elif len(words) == 3:
                    pl_str = '[{}] <{}> (1) <{}> (1)'.format(words[0].upper(), words[1], words[2])
                else:
                    pl_str = '[{}] <{}> (1)'.format(words[0].upper(), '_'.join(words[1:]))
            else:
                pl_str = '[EMPTY]'
            parsed_lines.append(pl_str)
    info['num_parsed_lines'] = len(parsed_lines)
    info['num_total_lines'] = len(pl)
    if len(pl) != 0:
        info['parsibility'] = success_count / len(pl)
    else:
        info['parsibility'] = 0
    return parsed_lines, info

def check_env(now_state, revised_graph):
    r_nodes = revised_graph['nodes']
    r_remove_edges = revised_graph['revised_nodes']
    r_add_edges = revised_graph['revised_add_edges']

    n_nodes = now_state['nodes']
    n_edges = now_state['edges']

    mode = True
    for each_node in r_nodes:
        each_node_id = each_node['id']
        for each_n in n_nodes:
            if each_n['id'] == each_node_id:
                n_node = each_n

        for each in each_node.keys():
            if type(each_node[each]) == str:
                if each_node[each] != n_node[each]:
                    mode = False
            elif type(each_node[each]) == list:
                if set(each_node[each]) != set(n_node[each]):
                    mode = False
            else:
                if each_node[each] != n_node[each]:
                    mode = False

    for each_edge in r_remove_edges:
        if each_edge in n_edges:
            mode = False

    for each_edge in r_add_edges:
        if each_edge not in n_edges:
            mode = False

    return mode

# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class VirtualHomeWorker:
    """Ray remote actor that hosts a VirtualHome environment instance."""
    
    def __init__(self, seed, env_kwargs):
        self.env_kwargs = env_kwargs
        self.seed = seed
        self.task_path = None
        self.executor = None
        self.this_state = None
        self.agent_id = None
        self.specific_objects = {}
        self.last_obj_id = -1
        self.max_steps = env_kwargs.get('max_steps', 50)
        self.steps = 0
    
    def reset(self, task_data):
        """Reset the environment with given task data"""
        self.task_path = task_data
        self.steps = 0
        
        scene_path = "/code/STeCa/init_and_final_graphs/" + self.task_path['path'][151:-4] + ".json"
        with open(scene_path) as f:
            Tdata = json.load(f)
        Tdata = Tdata['init_graph']
        Tdata = remove_duplicate_edge(Tdata)
        env_graph = EnvironmentGraph(Tdata)
        self.agent_id = [n['id'] for n in Tdata["nodes"] if n['class_name'] == 'character'][0]
        name_equivalence = utils.load_name_equivalence()
        self.executor = ScriptExecutor(env_graph, name_equivalence)
        self.this_state = EnvironmentState(env_graph, name_equivalence, instance_selection=True)
        self.specific_objects = self.task_path.get('object_id_dict', {})
        self.last_obj_id = -1
        
        task_name = self.task_path['task']
        task_description = self.task_path['description']
        obs = "The task is " + task_name + "(" + task_description + ")."
        
        info = dict()
        info['won'] = False
        info['task'] = task_name
        info['task_description'] = task_description
        
        return obs, info
    
    def step(self, action):
        """Execute a step in the environment"""
        self.steps += 1
        
        try:
            parsed_action = str2program_list([action])[0][0]
            parsed_action, self.specific_objects, self.last_obj_id = change_obj_index(
                self.this_state, parsed_action, self.agent_id, self.specific_objects, self.last_obj_id
            )

            mode = False
            if '[' in parsed_action:
                matches_action = re.findall(r'\[(.*?)\]', parsed_action)[0]
                if matches_action in dir(EvolveGraphAction):
                    parsed_action = check_action_format(parsed_action)
                    if '[' in parsed_action:
                        script = parse_script_line(parsed_action, 0)
                        success, self.this_state = self.executor.execute_one_step(
                            Script([script]), self.this_state
                        ) 
                        mode = success

            if mode:
                temp_total_graph = self.this_state.to_dict()
                partial_graph = utils.get_visible_nodes(temp_total_graph, agent_id=self.agent_id)

                agent_has_objid = [n['to_id'] for n in temp_total_graph["edges"] 
                                 if n['from_id'] == self.agent_id and "HOLD" in n["relation_type"]]
                agent_has_obj = [n['class_name'] for n in temp_total_graph["nodes"] 
                               if n['id'] in agent_has_objid]
                
                obj_ids_close = [n['to_id'] for n in temp_total_graph["edges"] 
                               if n['from_id'] == self.agent_id and n["relation_type"]=="CLOSE"]
                obj = [node['class_name'] for node in partial_graph['nodes'] 
                      if node["id"] in obj_ids_close]
                obj_ids = dict([(node['id'], node['class_name']) 
                              for node in temp_total_graph['nodes'] 
                              if node["id"] in obj_ids_close and node['class_name'] in obj])
                relations = list(set([obj_ids[n['from_id']] +' '+ n["relation_type"] +' '+ obj_ids[n['to_id']] 
                                    for n in temp_total_graph["edges"] 
                                    if n['from_id'] in obj_ids and n['to_id'] in obj_ids 
                                    and n["relation_type"] not in ["CLOSE","FACING", "INSIDE", "HOLDS_LH", "HOLDS_RH"]]))
                
                obj_states = [(node['class_name'], node['states']) 
                            for node in temp_total_graph['nodes'] 
                            if node['class_name'] in obj]
                objs = ""
                
                for ob_states in obj_states:
                    if len(ob_states[1])>0:
                        objs = objs + ob_states[0] + ' is ' + ' and '.join(ob_states[1]) + ', '
                    else:
                        objs = objs + ob_states[0] + ', '
                objs = list(set(objs.split(', ')))
                objs = [ob for ob in objs if len(ob)>0]

                if len(objs) == 0:
                    if len(relations) != 0:
                        objs = ', '.join(relations) + '. '
                    else:
                        objs = ""
                else:
                    if len(relations) == 0:
                        objs = ', '.join(objs) + '. '
                    else:
                        objs = ', '.join(objs) + ', ' + ', '.join(relations)  + '. '

                if len(agent_has_obj)>0:
                    agent_has_obj = ', '.join(agent_has_obj)
                    objs += f"You have {agent_has_obj}. "
            else:
                objs = "Nothing happens."

            state_dict = self.this_state.to_dict()
            done = check_env(state_dict, self.task_path['revised_graph'])
            
            reward = 10.0 if done else 0.0
            
            info = dict()
            info['won'] = done
            info['task'] = self.task_path['task']
            info['task_description'] = self.task_path['description']
            
            if self.steps >= self.max_steps:
                done = True
                if not info['won']:
                    reward = 0.0
            
            return objs, reward, done, info
            
        except Exception as e:
            objs = f"Error: {str(e)}"
            reward = 0.0
            done = False
            info = dict()
            info['won'] = False
            info['task'] = self.task_path.get('task', 'unknown')
            info['task_description'] = self.task_path.get('description', 'unknown')
            return objs, reward, done, info
    
    def close(self):
        """Close the environment"""
        pass


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class VirtualHomeMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around VirtualHome environment."""
    
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
        env_worker = ray.remote(**resources_per_worker)(VirtualHomeWorker)
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
            idx = self._rng.choice(len(self.task_data_list), size=self.env_num, replace=True)
            idx = np.repeat(idx, self.group_n).tolist()
            task_data_to_use = [self.task_data_list[i] for i in idx]
        else:
            # If no task data provided, create empty tasks (will need to be set externally)
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

def build_virtualhome_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
    task_data_list: list = None,
):
    """Build VirtualHome multi-process environments."""
    return VirtualHomeMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
        task_data_list=task_data_list,
    )
