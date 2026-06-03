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

# --------------------- ALFWorld --------------------- #
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE_OUR_METHOD_BASELINE = """
Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. 

For each of your turn, you should first think about the current condition and plan for your future actions, and then output your action in this turn.

You need to process the information in a specific order:
1. **Thought**: Plan your future actions based on the updated belief.
2. **Action**: Output your next action.

The available actions are:
1. go to (recep)
2. task (obj) from (recep)
3. put (obj) in/on (recep)
4. open (recep)
5. close (recep)
6. toggle (obj) (recep)
7. clean (obj) with (recep)
8. heat (obj) with (recep)
9. cool (obj) with (recep)
where (obj) and (recep) correspond to objects and receptacles.
After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.

Your response should use the following format:

Thought: <your thoughts>
Action: <your next action>

Below is the action history:
Action History: {action_history}"""

ALFWORLD_TEMPLATE_OUR_METHOD = """
Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. 

At each step, you will be given task goal, action history and the last turn's information (Reason, Belief State, Thought, and Action).

You need to process the information in a specific order:
1. **Reason**: Analyze the last action and the observation in one or two concise sentences. What did you expect to see? What did you actually see? Does this confirm or contradict your previous belief?
2. **Belief State**: State where the agent is, what it is holding, and the known status of goal-related objects. Do NOT list irrelevant objects.
3. **Thought**: Plan your future actions based on the updated belief.
4. **Action**: Output your next action.

The available actions are:
1. go to (recep)
2. task (obj) from (recep)
3. put (obj) in/on (recep)
4. open (recep)
5. close (recep)
6. toggle (obj) (recep)
7. clean (obj) with (recep)
8. heat (obj) with (recep)
9. cool (obj) with (recep)
where (obj) and (recep) correspond to objects and receptacles.
After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.

Your response should use the following format:

Reason: <Analyze expectation vs. actual observation to update your understanding>
Belief State: <your belief state>
Thought: <your thoughts>
Action: <your next action>

Your task is to complete the task goal: {task_goal}

Below is the action history and the last turn's information:

Action History: {action_history}

Last Turn's Information: {last_turn_information}"""


ALFWORLD_TEMPLATE_OUR_METHOD_w_two_steps = """
Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. 

At each step, you will be given task goal, action history and the last turn's information (Reason, Belief State, Thought, and Action).

You need to process the information in a specific order:
1. **Reason**: Analyze the last action and the observation in one or two concise sentences. What did you expect to see? What did you actually see? Does this confirm or contradict your previous belief?
2. **Belief State**: State where the agent is, what it is holding, and the known status of goal-related objects. Do NOT list irrelevant objects.
3. **Thought**: Plan your future actions based on the updated belief.
4. **Action**: Output your next action.

The available actions are:
1. go to (recep)
2. task (obj) from (recep)
3. put (obj) in/on (recep)
4. open (recep)
5. close (recep)
6. toggle (obj) (recep)
7. clean (obj) with (recep)
8. heat (obj) with (recep)
9. cool (obj) with (recep)
where (obj) and (recep) correspond to objects and receptacles.
After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.

Your response should use the following format:

Reason: <Analyze expectation vs. actual observation to update your understanding>
Belief State: <your belief state>
Thought: <your thoughts>
Action: <your next action>

Your task is to complete the task goal: {task_goal}

Below is the action history and the last turn's information:

Action History: {action_history}"""




# 这个是prompting method中的react的模版
prompting_method_ALFWORLD_TEMPLATE_OUR_METHOD = """
Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. 

At each step, you will be given task goal, action history and the last turn's information (Reason, Belief State, Thought, and Action).

You need to process the information in a specific order:
1. **Reason**: Analyze the last action and the observation in one or two concise sentences. What did you expect to see? What did you actually see? Does this confirm or contradict your previous belief?
2. **Belief State**: State where the agent is, what it is holding, and the known status of goal-related objects. Do NOT list irrelevant objects.
3. **Thought**: Plan your future actions based on the updated belief.
4. **Action**: Output your next action.

The available actions are (Your action should strictly follow this format and only replace the (recep) or (obj) with the actual object with id or receptacle with id):
1. go to (recep)
2. task (obj) from (recep)
3. put (obj) in/on (recep)
4. open (recep)
5. close (recep)
6. toggle (obj) (recep)
7. clean (obj) with (recep)
8. heat (obj) with (recep)
9. cool (obj) with (recep)
where (obj) and (recep) correspond to objects and receptacles.
After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.

Your response should use the following format:

Reason: <Analyze expectation vs. actual observation to update your understanding>
Belief State: <your belief state>
Thought: <your thoughts>
Action: <your next action>

Here is an one-shot example:
Your task is to complete the task goal: put a toiletpaper in toiletpaperhanger.\n\nBelow is the action history and the last turn's information:\nAction History: \nLast Turn's Information: \nYou are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 1, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.
Reasoning: The agent has not yet taken any action, so there is no expected versus actual outcome to analyze.  \nBelief State: The agent is in the middle of a bathroom. The agent is not holding anything. The toiletpaperhanger is present in the room, but the toiletpaper has not been located yet.\nThought: I need to place a toilet paper roll on the toilet paper hanger. First, I should check if there's already a roll on the hanger or if it's empty.\nAction: go to toiletpaperhanger 1

Your task is to complete the task goal: put a toiletpaper in toiletpaperhanger.\n\nBelow is the action history and the last turn's information:\nAction History: go to toiletpaperhanger 1\nLast Turn's Information: \nReasoning: The agent has not yet taken any action, so there is no expected versus actual outcome to analyze.  \nBelief State: The agent is in the middle of a bathroom. The agent is not holding anything. The toiletpaperhanger is present in the room, but the toiletpaper has not been located yet.\nThought: I need to place a toilet paper roll on the toilet paper hanger. First, I should check if there's already a roll on the hanger or if it's empty.\nAction: go to toiletpaperhanger 1\nObservation: On the toiletpaperhanger 1, you see nothing.
Reasoning: The agent moved to the toiletpaperhanger expecting to see if a toiletpaper was already there; the observation confirmed it was empty, which is consistent with the goal of placing one there.  \nBelief State: The agent is at the toiletpaperhanger 1, not holding any toiletpaper. The toiletpaperhanger is empty, and the toiletpaper's location is still unknown.\nThought: The toilet paper hanger is empty, so I need to find a new roll of toilet paper to put on it.\nAction: go to toilet 1

Your task is to complete the task goal: put a toiletpaper in toiletpaperhanger.\n\nBelow is the action history and the last turn's information:\nAction History: go to toiletpaperhanger 1, go to toilet 1\nLast Turn's Information: \nReasoning: The agent moved to the toiletpaperhanger expecting to see if a toiletpaper was already there; the observation confirmed it was empty, which is consistent with the goal of placing one there.  \nBelief State: The agent is at the toiletpaperhanger 1, not holding any toiletpaper. The toiletpaperhanger is empty, and the toiletpaper's location is still unknown.\nThought: The toilet paper hanger is empty, so I need to find a new roll of toilet paper to put on it.\nAction: go to toilet 1\nObservation: On the toilet 1, you see a soapbottle 1, and a toiletpaper 1.
Reasoning: The agent moved to the toilet expecting to find a toilet paper roll there; the observation confirmed a toiletpaper is present on the toilet, matching the expectation and providing the needed object.  \nBelief State: The agent is currently at the toilet and is not holding anything. A toiletpaper is located on the toilet. The toiletpaperhanger is empty.\nThought: I've found a toilet paper roll on the toilet. I should pick it up so I can place it on the hanger.\nAction: take toiletpaper 1 from toilet 1

Your task is to complete the task goal: put a toiletpaper in toiletpaperhanger.\n\nBelow is the action history and the last turn's information:\nAction History: go to toiletpaperhanger 1, go to toilet 1, take toiletpaper 1 from toilet 1\nLast Turn's Information: \nReasoning: The agent moved to the toilet expecting to find a toilet paper roll there; the observation confirmed a toiletpaper is present on the toilet, matching the expectation and providing the needed object.  \nBelief State: The agent is currently at the toilet and is not holding anything. A toiletpaper is located on the toilet. The toiletpaperhanger is empty.\nThought: I've found a toilet paper roll on the toilet. I should pick it up so I can place it on the hanger.\nAction: take toiletpaper 1 from toilet 1\nObservation: You pick up the toiletpaper 1 from the toilet 1
Reasoning: The agent took the toilet paper from the toilet expecting to successfully pick it up; the observation confirmed the action succeeded, so the agent now holds the toilet paper and it is no longer on the toilet.  \nBelief State: The agent is currently at the toilet and is holding the toilet paper. The toilet paper hanger is empty and ready. The toilet paper is not on the toilet anymore.\nThought: Now that I have the toilet paper roll, I need to return to the toilet paper hanger to install it.\nAction: go to toiletpaperhanger 1

Your task is to complete the task goal: put a toiletpaper in toiletpaperhanger.\n\nBelow is the action history and the last turn's information:\nAction History: go to toiletpaperhanger 1, go to toilet 1, take toiletpaper 1 from toilet 1, go to toiletpaperhanger 1\nLast Turn's Information: \nReasoning: The agent took the toilet paper from the toilet expecting to successfully pick it up; the observation confirmed the action succeeded, so the agent now holds the toilet paper and it is no longer on the toilet.  \nBelief State: The agent is currently at the toilet and is holding the toilet paper. The toilet paper hanger is empty and ready. The toilet paper is not on the toilet anymore.\nThought: Now that I have the toilet paper roll, I need to return to the toilet paper hanger to install it.\nAction: go to toiletpaperhanger 1\nObservation: On the toiletpaperhanger 1, you see nothing.
Reasoning: The agent went to the toiletpaperhanger 1 expecting it to still be empty, ready to place the held toilet paper; the observation confirmed it was empty, matching the expectation.  \nBelief State: The agent is at the toiletpaperhanger 1, holding the toiletpaper 1. The toiletpaperhanger 1 is empty and ready to receive the toilet paper.\nThought: I'm back at the toilet paper hanger with the roll in hand. It's time to put the toilet paper on the hanger.\nAction: put toiletpaper 1 in/on toiletpaperhanger 1

Your task is to complete the task goal: {task_goal}

Below is the action history and the last turn's information:

Action History: {action_history}

Last Turn's Information: {last_turn_information}"""

prompting_method_ALFWORLD_TEMPLATE_BASELINE = """Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. 
For each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought: your thoughts.\nAction: your next action".

The available actions are (Your action should strictly follow this format and only replace the (recep) or (obj) with the actual object with id or receptacle with id):
1. go to (recep)
2. task (obj) from (recep)
3. put (obj) in/on (recep)
4. open (recep)
5. close (recep)
6. toggle (obj) (recep)
7. clean (obj) with (recep)
8. heat (obj) with (recep)
9. cool (obj) with (recep)
where (obj) and (recep) correspond to objects and receptacles.
After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.

Your response should use the following format:

Thought: <your thoughts>
Action: <your next action>

Here is an one-shot example:
You are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 1, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.\n\nYour task is to: put a toiletpaper in toiletpaperhanger.
Thought: I need to place a toilet paper roll on the toilet paper hanger. First, I should check if there's already a roll on the hanger or if it's empty.\nAction: go to toiletpaperhanger 1
Observation: On the toiletpaperhanger 1, you see nothing.
Thought: The toilet paper hanger is empty, so I need to find a new roll of toilet paper to put on it.\nAction: go to toilet 1
Observation: On the toilet 1, you see a soapbottle 1, and a toiletpaper 1.
Thought: I've found a toilet paper roll on the toilet. I should pick it up so I can place it on the hanger.\nAction: take toiletpaper 1 from toilet 1
Observation: You pick up the toiletpaper 1 from the toilet 1.
Thought: Now that I have the toilet paper roll, I need to return to the toilet paper hanger to install it.\nAction: go to toiletpaperhanger 1
Observation: On the toiletpaperhanger 1, you see nothing.
Thought: I'm back at the toilet paper hanger with the roll in hand. It's time to put the toilet paper on the hanger.\nAction: put toiletpaper 1 in/on toiletpaperhanger 1

Below is the interaction history: {last_turn_information}"""




# 这是prompting method中的no thinking的模版
# prompting_method_ALFWORLD_TEMPLATE_OUR_METHOD = """
# Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. 

# At each step, you will be given task goal, action history and the last turn's information (Reason, Belief State, and Action).

# You need to process the information in a specific order:
# 1. **Reason**: Analyze the last action and the observation in one or two concise sentences. What did you expect to see? What did you actually see? Does this confirm or contradict your previous belief?
# 2. **Belief State**: State where the agent is, what it is holding, and the known status of goal-related objects. Do NOT list irrelevant objects.
# 3. **Action**: Output your next action.

# The available actions are (Your action should strictly follow this format and only replace the (recep) or (obj) with the actual object with id or receptacle with id):
# 1. go to (recep)
# 2. task (obj) from (recep)
# 3. put (obj) in/on (recep)
# 4. open (recep)
# 5. close (recep)
# 6. toggle (obj) (recep)
# 7. clean (obj) with (recep)
# 8. heat (obj) with (recep)
# 9. cool (obj) with (recep)
# where (obj) and (recep) correspond to objects and receptacles.
# After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.

# Your response should use the following format:

# Reason: <Analyze expectation vs. actual observation to update your understanding>
# Belief State: <your belief state>
# Action: <your next action>

# Here is an one-shot example:
# Your task is to complete the task goal: put a toiletpaper in toiletpaperhanger.\n\nBelow is the action history and the last turn's information:\nAction History: \nLast Turn's Information: \nYou are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 1, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.
# Reasoning: The agent has not yet taken any action, so there is no expected versus actual outcome to analyze.  \nBelief State: The agent is in the middle of a bathroom. The agent is not holding anything. The toiletpaperhanger is present in the room, but the toiletpaper has not been located yet.\nAction: go to toiletpaperhanger 1

# Your task is to complete the task goal: put a toiletpaper in toiletpaperhanger.\n\nBelow is the action history and the last turn's information:\nAction History: go to toiletpaperhanger 1\nLast Turn's Information: \nReasoning: The agent has not yet taken any action, so there is no expected versus actual outcome to analyze.  \nBelief State: The agent is in the middle of a bathroom. The agent is not holding anything. The toiletpaperhanger is present in the room, but the toiletpaper has not been located yet.\nAction: go to toiletpaperhanger 1\nObservation: On the toiletpaperhanger 1, you see nothing.
# Reasoning: The agent moved to the toiletpaperhanger expecting to see if a toiletpaper was already there; the observation confirmed it was empty, which is consistent with the goal of placing one there.  \nBelief State: The agent is at the toiletpaperhanger 1, not holding any toiletpaper. The toiletpaperhanger is empty, and the toiletpaper's location is still unknown.\nAction: go to toilet 1

# Your task is to complete the task goal: put a toiletpaper in toiletpaperhanger.\n\nBelow is the action history and the last turn's information:\nAction History: go to toiletpaperhanger 1, go to toilet 1\nLast Turn's Information: \nReasoning: The agent moved to the toiletpaperhanger expecting to see if a toiletpaper was already there; the observation confirmed it was empty, which is consistent with the goal of placing one there.  \nBelief State: The agent is at the toiletpaperhanger 1, not holding any toiletpaper. The toiletpaperhanger is empty, and the toiletpaper's location is still unknown.\nAction: go to toilet 1\nObservation: On the toilet 1, you see a soapbottle 1, and a toiletpaper 1.
# Reasoning: The agent moved to the toilet expecting to find a toilet paper roll there; the observation confirmed a toiletpaper is present on the toilet, matching the expectation and providing the needed object.  \nBelief State: The agent is currently at the toilet and is not holding anything. A toiletpaper is located on the toilet. The toiletpaperhanger is empty.\nAction: take toiletpaper 1 from toilet 1

# Your task is to complete the task goal: put a toiletpaper in toiletpaperhanger.\n\nBelow is the action history and the last turn's information:\nAction History: go to toiletpaperhanger 1, go to toilet 1, take toiletpaper 1 from toilet 1\nLast Turn's Information: \nReasoning: The agent moved to the toilet expecting to find a toilet paper roll there; the observation confirmed a toiletpaper is present on the toilet, matching the expectation and providing the needed object.  \nBelief State: The agent is currently at the toilet and is not holding anything. A toiletpaper is located on the toilet. The toiletpaperhanger is empty.\nAction: take toiletpaper 1 from toilet 1\nObservation: You pick up the toiletpaper 1 from the toilet 1
# Reasoning: The agent took the toilet paper from the toilet expecting to successfully pick it up; the observation confirmed the action succeeded, so the agent now holds the toilet paper and it is no longer on the toilet.  \nBelief State: The agent is currently at the toilet and is holding the toilet paper. The toilet paper hanger is empty and ready. The toilet paper is not on the toilet anymore.\nAction: go to toiletpaperhanger 1

# Your task is to complete the task goal: put a toiletpaper in toiletpaperhanger.\n\nBelow is the action history and the last turn's information:\nAction History: go to toiletpaperhanger 1, go to toilet 1, take toiletpaper 1 from toilet 1, go to toiletpaperhanger 1\nLast Turn's Information: \nReasoning: The agent took the toilet paper from the toilet expecting to successfully pick it up; the observation confirmed the action succeeded, so the agent now holds the toilet paper and it is no longer on the toilet.  \nBelief State: The agent is currently at the toilet and is holding the toilet paper. The toilet paper hanger is empty and ready. The toilet paper is not on the toilet anymore.\nAction: go to toiletpaperhanger 1\nObservation: On the toiletpaperhanger 1, you see nothing.
# Reasoning: The agent went to the toiletpaperhanger 1 expecting it to still be empty, ready to place the held toilet paper; the observation confirmed it was empty, matching the expectation.  \nBelief State: The agent is at the toiletpaperhanger 1, holding the toiletpaper 1. The toiletpaperhanger 1 is empty and ready to receive the toilet paper.\nAction: put toiletpaper 1 in/on toiletpaperhanger 1

# Your task is to complete the task goal: {task_goal}

# Below is the action history and the last turn's information:

# Action History: {action_history}

# Last Turn's Information: {last_turn_information}"""

# prompting_method_ALFWORLD_TEMPLATE_BASELINE = """Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. 
# For each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought: your thoughts.\nAction: your next action".

# The available actions are (Your action should strictly follow this format and only replace the (recep) or (obj) with the actual object with id or receptacle with id):
# 1. go to (recep)
# 2. task (obj) from (recep)
# 3. put (obj) in/on (recep)
# 4. open (recep)
# 5. close (recep)
# 6. toggle (obj) (recep)
# 7. clean (obj) with (recep)
# 8. heat (obj) with (recep)
# 9. cool (obj) with (recep)
# where (obj) and (recep) correspond to objects and receptacles.
# After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.

# Your response should use the following format:

# Action: <your next action>

# Here is an one-shot example:
# You are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 1, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.\n\nYour task is to: put a toiletpaper in toiletpaperhanger.
# Action: go to toiletpaperhanger 1
# Observation: On the toiletpaperhanger 1, you see nothing.
# Action: go to toilet 1
# Observation: On the toilet 1, you see a soapbottle 1, and a toiletpaper 1.
# Action: take toiletpaper 1 from toilet 1
# Observation: You pick up the toiletpaper 1 from the toilet 1.
# Action: go to toiletpaperhanger 1
# Observation: On the toiletpaperhanger 1, you see nothing.
# Action: put toiletpaper 1 in/on toiletpaperhanger 1

# Below is the interaction history: {last_turn_information}"""

