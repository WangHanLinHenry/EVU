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

# --------------------- VirtualHome --------------------- #
VIRTUALHOME_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the VirtualHome environment. 
Your task is to: {task_description}.
Your current observation is: {current_observation}.

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which action best advances the task goal. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an action for current step and present it within <action> </action> tags.

Available actions include: close, drink, find, walk, grab, look at, open, point at, put, put in, put back, run, sit, stand up, switch off, switch on, touch, turn to, watch, wipe, put on, take off, greet, drop, read, lie, pour, type, push, pull, move, wash, rinse, scrub, squeeze, plug in, plug out, cut, eat, sleep, wake up, release.

Format your action as: <action>action_name object_name</action> (e.g., <action>open door</action> or <action>grab apple</action>).
For actions with two objects, use: <action>put apple in fridge</action>.
"""

VIRTUALHOME_TEMPLATE = """
You are an expert autonomous agent operating in the VirtualHome environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which action best advances the task goal. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an action for current step and present it within <action> </action> tags.

Available actions include: close, drink, find, walk, grab, look at, open, point at, put, put in, put back, run, sit, stand up, switch off, switch on, touch, turn to, watch, wipe, put on, take off, greet, drop, read, lie, pour, type, push, pull, move, wash, rinse, scrub, squeeze, plug in, plug out, cut, eat, sleep, wake up, release.

Format your action as: <action>action_name object_name</action> (e.g., <action>open door</action> or <action>grab apple</action>).
For actions with two objects, use: <action>put apple in fridge</action>.
"""

VH_TEMPLATE_OUR_METHOD = """
Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish.

At each step, you will be given task goal, action history and the last turn's information (Reason, Belief State, Thought, and Action).

You need to process the information in a specific order:
1. **Reason**: Analyze the last action and the observation in one or two concise sentences. What did you expect to see? What did you actually see? Does this confirm or contradict your previous belief?
2. **Belief State**: State where the agent is, what it is holding, and the known status of goal-related objects. Do NOT list irrelevant objects.
3. **Thought**: Plan your future actions based on the updated belief.
4. **Action**: Output your next action.

The available actions are:\nwalk to (obj)\nrun to (obj)\ngrab (obj)\nopen (obj)\nclose (obj)\nput (obj) on (recep)\nput (obj) in (recep)\nswitch on (obj)\nswitch off (obj)\ndrink (obj)\nlook at (obj)\nsit on (obj)\nstand up\nwatch (obj)\nwipe (obj)\ntype on (obj)\nwash (obj)\ncut (obj)\neat (obj)\nsleep\nwake up\nplug in (obj)\nplug out (obj)\npour (obj) into (recep)\nmove (obj)\nrelease\nturn to (obj)

After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output \"Nothing happened\", that means the previous action is invalid and you should try more options.

Your response should use the following format:

Reason: <Analyze expectation vs. actual observation to update your understanding>
Belief State: <your belief state>
Thought: <your thoughts>
Action: <your next action>

Your task is to complete the task goal: {task_goal}

Below is the action history and the last turn's information:
Action History: {action_history}
Last Turn's Information: {last_turn_information}"""

# VIRTUALHOME_TEMPLATE_OUR_METHOD = """
# Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. 

# At each step, you will be given task goal, action history and the last turn's information (Reason, Belief State, Thought, and Action).

# You need to process the information in a specific order:
# 1. **Reason**: Analyze the last action and the observation in one or two concise sentences. What did you expect to see? What did you actually see? Does this confirm or contradict your previous belief?
# 2. **Belief State**: State where the agent is, what it is holding, and the known status of goal-related objects. Do NOT list irrelevant objects.
# 3. **Thought**: Plan your future actions based on the updated belief.
# 4. **Action**: Output your next action.

# The available actions are:
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
# Thought: <your thoughts>
# Action: <your next action>

# Your task is to complete the task goal: {task_goal}

# Below is the action history and the last turn's information:

# Action History: {action_history}

# Last Turn's Information: {last_turn_information}"""



# 这个是推理时候采用的是react框架
# PROMPTING_METHOD_VIRTUALHOME_TEMPLATE_BASELINE = """
# Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish.
# At each step, you will be given interaction history.

# You need to process the information in a specific order:
# 1. **Thought**: Plan your future actions based on the updated belief.
# 2. **Action**: Output your next action.

# The available actions are:\nwalk to (obj)\nrun to (obj)\ngrab (obj)\nopen (obj)\nclose (obj)\nput (obj) on (recep)\nput (obj) in (recep)\nswitch on (obj)\nswitch off (obj)\ndrink (obj)\nlook at (obj)\nsit on (obj)\nstand up\nwatch (obj)\nwipe (obj)\ntype on (obj)\nwash (obj)\ncut (obj)\neat (obj)\nsleep\nwake up\nplug in (obj)\nplug out (obj)\npour (obj) into (recep)\nmove (obj)\nrelease\nturn to (obj)

# After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output \"Nothing happened\", that means the previous action is invalid and you should try more options.

# Your response should use the following format:
# Thought: <your thoughts>
# Action: <your next action>

# Below is an one-shot example:
# The task is Drink (Drink water).
# Thought: To drink water, I need to first locate a water source in the household, so moving towards the dining room might bring me closer to finding water or a suitable container for drinking.\nAction: Walk to dining room
# Observation: 
# Thought: I chose to walk to the water glass as it is likely located in the dining room where I can access water easily, aligning with the goal of drinking water.\nAction: Walk to water glass
# Observation: water_glass is CLEAN. 
# Thought: The current observation indicates that I've located a clean water glass, which is essential to proceed with the task of drinking water. Now, I need to fill it with water to accomplish the task goal.\nAction: Find water glass
# Observation: water_glass is CLEAN. 
# Thought: The observation indicates that the water glass is clean, and since the task is to drink water, holding the glass will allow further steps to achieve this goal.\nAction: Grab water glass
# Observation: character, water_glass is CLEAN. You have water_glass.
# Thought: Since I have already obtained the water glass, the next logical step is to drink from it to complete the task of drinking water.\nAction: Drink water glass

# Below is the interaction history: {last_turn_information}"""


# PROMPTING_METHOD_VIRTUALHOME_TEMPLATE_OUR_METHOD = """
# Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish.

# At each step, you will be given task goal, action history and the last turn's information (Reason, Belief State, Thought, and Action).

# You need to process the information in a specific order:
# 1. **Reason**: Analyze the last action and the observation in one or two concise sentences. What did you expect to see? What did you actually see? Does this confirm or contradict your previous belief?
# 2. **Belief State**: State where the agent is, what it is holding, and the known status of goal-related objects. Do NOT list irrelevant objects.
# 3. **Thought**: Plan your future actions based on the updated belief.
# 4. **Action**: Output your next action.

# The available actions are:\nwalk to (obj)\nrun to (obj)\ngrab (obj)\nopen (obj)\nclose (obj)\nput (obj) on (recep)\nput (obj) in (recep)\nswitch on (obj)\nswitch off (obj)\ndrink (obj)\nlook at (obj)\nsit on (obj)\nstand up\nwatch (obj)\nwipe (obj)\ntype on (obj)\nwash (obj)\ncut (obj)\neat (obj)\nsleep\nwake up\nplug in (obj)\nplug out (obj)\npour (obj) into (recep)\nmove (obj)\nrelease\nturn to (obj)

# After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output \"Nothing happened\", that means the previous action is invalid and you should try more options.

# Your response should use the following format:

# Reason: <Analyze expectation vs. actual observation to update your understanding>
# Belief State: <your belief state>
# Thought: <your thoughts>
# Action: <your next action>

# Below is an one-shot example:
# Your task is to complete the task goal: groceries in Fridge (Walk to kitchen. Grab groceries and walk to the fridge. Open fridge and put groceries in the fridge. Close the fridge.).\n\nBelow is the action history and the last turn's information:\nAction History: \nLast Turn's Information: \nThe task is Put groceries in Fridge (Walk to kitchen. Grab groceries and walk to the fridge. Open fridge and put groceries in the fridge. Close the fridge.)
# Reasoning: The agent has not taken any action yet, so there is no previous action to analyze. The initial observation establishes the starting state.  \nBelief State: The agent is in the living room. The agent is not holding anything. The goal object 'groceries' is on the living room floor. The goal receptacle 'fridge' is in the kitchen and is currently closed.\nThought: The task requires me to handle groceries and store them in the fridge. Moving to different rooms logically aligns with finding and accessing these groceries. Therefore, navigating to essential areas like the dining room and eventually the kitchen is a necessary part of the task sequence.\nAction: Walk to dining room

# Your task is to complete the task goal: groceries in Fridge (Walk to kitchen. Grab groceries and walk to the fridge. Open fridge and put groceries in the fridge. Close the fridge.).\n\nBelow is the action history and the last turn's information:\nAction History: walk to dining room\nLast Turn's Information: \nReasoning: The agent has not taken any action yet, so there is no previous action to analyze. The initial observation establishes the starting state.  \nBelief State: The agent is in the living room. The agent is not holding anything. The goal object 'groceries' is on the living room floor. The goal receptacle 'fridge' is in the kitchen and is currently closed.\nThought: The task requires me to handle groceries and store them in the fridge. Moving to different rooms logically aligns with finding and accessing these groceries. Therefore, navigating to essential areas like the dining room and eventually the kitchen is a necessary part of the task sequence.\nAction: Walk to dining room\nObservation: 
# Reasoning: The last action “Walk to dining room” was expected to move the agent to the dining room to search for groceries. The observation (no text) implies the action succeeded, confirming the agent's location changed.  \nBelief State: The agent is currently in the dining room. The agent is holding nothing. The groceries have not been found yet. No other locations have been searched.\nThought: Based on the task, I need to locate the groceries and ensure they are placed inside the fridge, so I will first find the groceries by walking to them.\nAction: Walk to food food

# Your task is to complete the task goal: groceries in Fridge (Walk to kitchen. Grab groceries and walk to the fridge. Open fridge and put groceries in the fridge. Close the fridge.).\n\nBelow is the action history and the last turn's information:\nAction History: walk to dining room, walk to food food\nLast Turn's Information: \nReasoning: The last action “Walk to dining room” was expected to move the agent to the dining room to search for groceries. The observation (no text) implies the action succeeded, confirming the agent's location changed.  \nBelief State: The agent is currently in the dining room. The agent is holding nothing. The groceries have not been found yet. No other locations have been searched.\nThought: Based on the task, I need to locate the groceries and ensure they are placed inside the fridge, so I will first find the groceries by walking to them.\nAction: Walk to food food\nObservation: food_food is CLEAN, food_food is DIRTY. 
# Reasoning: The last action “Walk to food food” was expected to bring the agent to the groceries. The observation confirms the agent is now at the groceries, as they are described with properties (CLEAN, DIRTY), though the contradictory states are unusual.\n\nBelief State: The agent is currently at the location of the groceries. The agent is holding nothing. The groceries (food_food) are at the agent's location and are available to be grabbed. The agent previously checked the dining room and did not find the groceries there.\nThought: Based on the interaction history, it seems there was a misunderstanding in identifying the correct action with \"food_food\". I need to correctly locate the groceries to proceed with the task.\nAction: Find food food

# Your task is to complete the task goal: groceries in Fridge (Walk to kitchen. Grab groceries and walk to the fridge. Open fridge and put groceries in the fridge. Close the fridge.).\n\nBelow is the action history and the last turn's information:\nAction History: walk to dining room, walk to food food, find food food\nLast Turn's Information: \nReasoning: The last action “Walk to food food” was expected to bring the agent to the groceries. The observation confirms the agent is now at the groceries, as they are described with properties (CLEAN, DIRTY), though the contradictory states are unusual.\n\nBelief State: The agent is currently at the location of the groceries. The agent is holding nothing. The groceries (food_food) are at the agent's location and are available to be grabbed. The agent previously checked the dining room and did not find the groceries there.\nThought: Based on the interaction history, it seems there was a misunderstanding in identifying the correct action with \"food_food\". I need to correctly locate the groceries to proceed with the task.\nAction: Find food food\nObservation: food_food is CLEAN, food_food is DIRTY.
# Reasoning: The last action “Find food food” was expected to locate the groceries. The observation shows the object \"food_food\" is present with contradictory states (CLEAN and DIRTY), but it confirms the groceries are found at the agent's location.  \nBelief State: The agent is at the location of the groceries (\"food_food\"). The agent is holding nothing. The groceries (\"food_food\") are present here and available to be grabbed. The agent previously checked the dining room but did not find the groceries there.\nThought: I chose to grab the \"food food\" object because it is necessary to put it in the fridge to complete the task of putting groceries away.\nAction: Grab food food

# Your task is to complete the task goal: groceries in Fridge (Walk to kitchen. Grab groceries and walk to the fridge. Open fridge and put groceries in the fridge. Close the fridge.).\n\nBelow is the action history and the last turn's information:\nAction History: walk to dining room, walk to food food, find food food, grab food food\nLast Turn's Information: \nReasoning: The last action “Find food food” was expected to locate the groceries. The observation shows the object \"food_food\" is present with contradictory states (CLEAN and DIRTY), but it confirms the groceries are found at the agent's location.  \nBelief State: The agent is at the location of the groceries (\"food_food\"). The agent is holding nothing. The groceries (\"food_food\") are present here and available to be grabbed. The agent previously checked the dining room but did not find the groceries there.\nThought: I chose to grab the \"food food\" object because it is necessary to put it in the fridge to complete the task of putting groceries away.\nAction: Grab food food\nObservation: character, food_food is DIRTY, food_food is CLEAN. You have food_food. 
# Reasoning: The last action “Grab food food” was expected to pick up the groceries. The observation confirms success: the agent is now holding the food_food object.\n\nBelief State: The agent is currently at the location of the food_food (having walked to it). The agent is holding the groceries (food_food). The groceries have not yet been placed in the fridge. The agent previously checked the dining room and did not find the groceries there.\nThought: I have opted to walk to the freezer because it is often located near the fridge, which is my eventual destination for storing groceries. This step aligns with my goal of efficiently putting the groceries away.\nAction: Walk to freezer

# Your task is to complete the task goal: groceries in Fridge (Walk to kitchen. Grab groceries and walk to the fridge. Open fridge and put groceries in the fridge. Close the fridge.).\n\nBelow is the action history and the last turn's information:\nAction History: walk to dining room, walk to food food, find food food, grab food food, walk to freezer\nLast Turn's Information: \nReasoning: The last action “Grab food food” was expected to pick up the groceries. The observation confirms success: the agent is now holding the food_food object.\n\nBelief State: The agent is currently at the location of the food_food (having walked to it). The agent is holding the groceries (food_food). The groceries have not yet been placed in the fridge. The agent previously checked the dining room and did not find the groceries there.\nThought: I have opted to walk to the freezer because it is often located near the fridge, which is my eventual destination for storing groceries. This step aligns with my goal of efficiently putting the groceries away.\nAction: Walk to freezer\nObservation: food_food is CLEAN, food_food is DIRTY, floor is CLEAN, freezer is CLOSED and PLUGGED_IN and CLEAN, floor is DIRTY, freezer ON floor. You have food_food.
# Reasoning: The last action “Walk to freezer” was expected to move the agent near the fridge for storing groceries. The observation confirms the agent arrived at the freezer, which is closed and on the floor, and the agent is still holding the groceries.\n\nBelief State: The agent is currently at the freezer. The agent is holding the groceries (food_food). The groceries are dirty and clean (contradictory properties noted). The freezer is closed. The agent previously checked the dining room but did not find the groceries there (they were found via \"Find food food\").\nThought: The current observation indicates that the freezer is closed, and since my task is to put groceries in the fridge, I need to open it first to proceed with storing the groceries inside.\nAction: Open freezer

# Your task is to complete the task goal: groceries in Fridge (Walk to kitchen. Grab groceries and walk to the fridge. Open fridge and put groceries in the fridge. Close the fridge.).\n\nBelow is the action history and the last turn's information:\nAction History: walk to dining room, walk to food food, find food food, grab food food, walk to freezer, open freezer\nLast Turn's Information: \nReasoning: The last action “Walk to freezer” was expected to move the agent near the fridge for storing groceries. The observation confirms the agent arrived at the freezer, which is closed and on the floor, and the agent is still holding the groceries.\n\nBelief State: The agent is currently at the freezer. The agent is holding the groceries (food_food). The groceries are dirty and clean (contradictory properties noted). The freezer is closed. The agent previously checked the dining room but did not find the groceries there (they were found via \"Find food food\").\nThought: The current observation indicates that the freezer is closed, and since my task is to put groceries in the fridge, I need to open it first to proceed with storing the groceries inside.\nAction: Open freezer\nObservation: food_food is CLEAN, food_food is DIRTY, floor is CLEAN, freezer is PLUGGED_IN and OPEN and CLEAN, floor is DIRTY, freezer ON floor. You have food_food.
# Reasoning: The last action \"Open freezer\" was expected to open the freezer so groceries could be placed inside. The observation confirms success: the freezer is now open, but note that the task requires putting groceries in the fridge, not the freezer.  \nBelief State: The agent is currently at the freezer. The agent is holding \"food_food\" (the groceries). The freezer is open, but the fridge (the goal receptacle) has not yet been located or opened. The agent previously checked the dining room and found no groceries there.\nThought: The task is to put the groceries in the fridge, as I'm currently holding the item described as \"food food\" and the freezer is open, placing it in the freezer is the logical action to move towards that goal.\nAction: Put food food in freezer

# Your task is to complete the task goal: groceries in Fridge (Walk to kitchen. Grab groceries and walk to the fridge. Open fridge and put groceries in the fridge. Close the fridge.).\n\nBelow is the action history and the last turn's information:\nAction History: walk to dining room, walk to food food, find food food, grab food food, walk to freezer, open freezer, put food food in freezer\nLast Turn's Information: \nReasoning: The last action \"Open freezer\" was expected to open the freezer so groceries could be placed inside. The observation confirms success: the freezer is now open, but note that the task requires putting groceries in the fridge, not the freezer.  \nBelief State: The agent is currently at the freezer. The agent is holding \"food_food\" (the groceries). The freezer is open, but the fridge (the goal receptacle) has not yet been located or opened. The agent previously checked the dining room and found no groceries there.\nThought: The task is to put the groceries in the fridge, as I'm currently holding the item described as \"food food\" and the freezer is open, placing it in the freezer is the logical action to move towards that goal.\nAction: Put food food in freezer\nObservation: food_food is CLEAN, food_food is DIRTY, floor is CLEAN, freezer is PLUGGED_IN and OPEN and CLEAN, floor is DIRTY, freezer ON floor. 
# Reasoning: The last action “Put food food in freezer” was expected to place the held groceries into the open freezer. The observation shows the freezer remains open and the agent still holds the item, indicating the action failed (likely because “food food” is not a valid object for the freezer or the freezer is not a valid receptacle for it).  \nBelief State: The agent is currently at the freezer. The agent is holding “food food” (the groceries). The freezer is open, plugged in, and clean, but the groceries were not successfully placed inside. The agent previously visited the dining room and found no relevant objects there.\nThought: The freezer being open is the final step needed to conclude the task of putting the groceries away, so I chose to close it to ensure everything is secure.\nAction: Close freezer

# Your task is to complete the task goal: {task_goal}

# Below is the action history and the last turn's information:
# Action History: {action_history}
# Last Turn's Information: {last_turn_information}"""


# 这个是采用的是no thinking
PROMPTING_METHOD_VIRTUALHOME_TEMPLATE_BASELINE = """
Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish.
At each step, you will be given interaction history.

You need to process the information in a specific order:
1. **Action**: Output your next action.

The available actions are:\nwalk to (obj)\nrun to (obj)\ngrab (obj)\nopen (obj)\nclose (obj)\nput (obj) on (recep)\nput (obj) in (recep)\nswitch on (obj)\nswitch off (obj)\ndrink (obj)\nlook at (obj)\nsit on (obj)\nstand up\nwatch (obj)\nwipe (obj)\ntype on (obj)\nwash (obj)\ncut (obj)\neat (obj)\nsleep\nwake up\nplug in (obj)\nplug out (obj)\npour (obj) into (recep)\nmove (obj)\nrelease\nturn to (obj)

After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output \"Nothing happened\", that means the previous action is invalid and you should try more options.

Your response should use the following format:
Action: <your next action>

Below is an one-shot example:
The task is Drink (Drink water).
Action: Walk to dining room
Observation: 
Action: Walk to water glass
Observation: water_glass is CLEAN. 
Action: Find water glass
Observation: water_glass is CLEAN. 
Action: Grab water glass
Observation: character, water_glass is CLEAN. You have water_glass.
Action: Drink water glass

Below is the interaction history: {last_turn_information}"""


PROMPTING_METHOD_VIRTUALHOME_TEMPLATE_OUR_METHOD = """
Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish.

At each step, you will be given task goal, action history and the last turn's information (Reason, Belief State, Thought, and Action).

You need to process the information in a specific order:
1. **Reason**: Analyze the last action and the observation in one or two concise sentences. What did you expect to see? What did you actually see? Does this confirm or contradict your previous belief?
2. **Belief State**: State where the agent is, what it is holding, and the known status of goal-related objects. Do NOT list irrelevant objects.
3. **Action**: Output your next action.

The available actions are:\nwalk to (obj)\nrun to (obj)\ngrab (obj)\nopen (obj)\nclose (obj)\nput (obj) on (recep)\nput (obj) in (recep)\nswitch on (obj)\nswitch off (obj)\ndrink (obj)\nlook at (obj)\nsit on (obj)\nstand up\nwatch (obj)\nwipe (obj)\ntype on (obj)\nwash (obj)\ncut (obj)\neat (obj)\nsleep\nwake up\nplug in (obj)\nplug out (obj)\npour (obj) into (recep)\nmove (obj)\nrelease\nturn to (obj)

After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output \"Nothing happened\", that means the previous action is invalid and you should try more options.

Your response should use the following format:

Reason: <Analyze expectation vs. actual observation to update your understanding>
Belief State: <your belief state>
Action: <your next action>

Below is an one-shot example:
Your task is to complete the task goal: groceries in Fridge (Walk to kitchen. Grab groceries and walk to the fridge. Open fridge and put groceries in the fridge. Close the fridge.).\n\nBelow is the action history and the last turn's information:\nAction History: \nLast Turn's Information: \nThe task is Put groceries in Fridge (Walk to kitchen. Grab groceries and walk to the fridge. Open fridge and put groceries in the fridge. Close the fridge.)
Reasoning: The agent has not taken any action yet, so there is no previous action to analyze. The initial observation establishes the starting state.  \nBelief State: The agent is in the living room. The agent is not holding anything. The goal object 'groceries' is on the living room floor. The goal receptacle 'fridge' is in the kitchen and is currently closed.\nAction: Walk to dining room

Your task is to complete the task goal: groceries in Fridge (Walk to kitchen. Grab groceries and walk to the fridge. Open fridge and put groceries in the fridge. Close the fridge.).\n\nBelow is the action history and the last turn's information:\nAction History: walk to dining room\nLast Turn's Information: \nReasoning: The agent has not taken any action yet, so there is no previous action to analyze. The initial observation establishes the starting state.  \nBelief State: The agent is in the living room. The agent is not holding anything. The goal object 'groceries' is on the living room floor. The goal receptacle 'fridge' is in the kitchen and is currently closed.\nAction: Walk to dining room\nObservation: 
Reasoning: The last action “Walk to dining room” was expected to move the agent to the dining room to search for groceries. The observation (no text) implies the action succeeded, confirming the agent's location changed.  \nBelief State: The agent is currently in the dining room. The agent is holding nothing. The groceries have not been found yet. No other locations have been searched.\nAction: Walk to food food

Your task is to complete the task goal: groceries in Fridge (Walk to kitchen. Grab groceries and walk to the fridge. Open fridge and put groceries in the fridge. Close the fridge.).\n\nBelow is the action history and the last turn's information:\nAction History: walk to dining room, walk to food food\nLast Turn's Information: \nReasoning: The last action “Walk to dining room” was expected to move the agent to the dining room to search for groceries. The observation (no text) implies the action succeeded, confirming the agent's location changed.  \nBelief State: The agent is currently in the dining room. The agent is holding nothing. The groceries have not been found yet. No other locations have been searched.\nAction: Walk to food food\nObservation: food_food is CLEAN, food_food is DIRTY. 
Reasoning: The last action “Walk to food food” was expected to bring the agent to the groceries. The observation confirms the agent is now at the groceries, as they are described with properties (CLEAN, DIRTY), though the contradictory states are unusual.\n\nBelief State: The agent is currently at the location of the groceries. The agent is holding nothing. The groceries (food_food) are at the agent's location and are available to be grabbed. The agent previously checked the dining room and did not find the groceries there.\nAction: Find food food

Your task is to complete the task goal: groceries in Fridge (Walk to kitchen. Grab groceries and walk to the fridge. Open fridge and put groceries in the fridge. Close the fridge.).\n\nBelow is the action history and the last turn's information:\nAction History: walk to dining room, walk to food food, find food food\nLast Turn's Information: \nReasoning: The last action “Walk to food food” was expected to bring the agent to the groceries. The observation confirms the agent is now at the groceries, as they are described with properties (CLEAN, DIRTY), though the contradictory states are unusual.\n\nBelief State: The agent is currently at the location of the groceries. The agent is holding nothing. The groceries (food_food) are at the agent's location and are available to be grabbed. The agent previously checked the dining room and did not find the groceries there.\nAction: Find food food\nObservation: food_food is CLEAN, food_food is DIRTY.
Reasoning: The last action “Find food food” was expected to locate the groceries. The observation shows the object \"food_food\" is present with contradictory states (CLEAN and DIRTY), but it confirms the groceries are found at the agent's location.  \nBelief State: The agent is at the location of the groceries (\"food_food\"). The agent is holding nothing. The groceries (\"food_food\") are present here and available to be grabbed. The agent previously checked the dining room but did not find the groceries there.\nAction: Grab food food

Your task is to complete the task goal: groceries in Fridge (Walk to kitchen. Grab groceries and walk to the fridge. Open fridge and put groceries in the fridge. Close the fridge.).\n\nBelow is the action history and the last turn's information:\nAction History: walk to dining room, walk to food food, find food food, grab food food\nLast Turn's Information: \nReasoning: The last action “Find food food” was expected to locate the groceries. The observation shows the object \"food_food\" is present with contradictory states (CLEAN and DIRTY), but it confirms the groceries are found at the agent's location.  \nBelief State: The agent is at the location of the groceries (\"food_food\"). The agent is holding nothing. The groceries (\"food_food\") are present here and available to be grabbed. The agent previously checked the dining room but did not find the groceries there.\nAction: Grab food food\nObservation: character, food_food is DIRTY, food_food is CLEAN. You have food_food. 
Reasoning: The last action “Grab food food” was expected to pick up the groceries. The observation confirms success: the agent is now holding the food_food object.\n\nBelief State: The agent is currently at the location of the food_food (having walked to it). The agent is holding the groceries (food_food). The groceries have not yet been placed in the fridge. The agent previously checked the dining room and did not find the groceries there.\nAction: Walk to freezer

Your task is to complete the task goal: groceries in Fridge (Walk to kitchen. Grab groceries and walk to the fridge. Open fridge and put groceries in the fridge. Close the fridge.).\n\nBelow is the action history and the last turn's information:\nAction History: walk to dining room, walk to food food, find food food, grab food food, walk to freezer\nLast Turn's Information: \nReasoning: The last action “Grab food food” was expected to pick up the groceries. The observation confirms success: the agent is now holding the food_food object.\n\nBelief State: The agent is currently at the location of the food_food (having walked to it). The agent is holding the groceries (food_food). The groceries have not yet been placed in the fridge. The agent previously checked the dining room and did not find the groceries there.\nAction: Walk to freezer\nObservation: food_food is CLEAN, food_food is DIRTY, floor is CLEAN, freezer is CLOSED and PLUGGED_IN and CLEAN, floor is DIRTY, freezer ON floor. You have food_food.
Reasoning: The last action “Walk to freezer” was expected to move the agent near the fridge for storing groceries. The observation confirms the agent arrived at the freezer, which is closed and on the floor, and the agent is still holding the groceries.\n\nBelief State: The agent is currently at the freezer. The agent is holding the groceries (food_food). The groceries are dirty and clean (contradictory properties noted). The freezer is closed. The agent previously checked the dining room but did not find the groceries there (they were found via \"Find food food\").\nAction: Open freezer

Your task is to complete the task goal: groceries in Fridge (Walk to kitchen. Grab groceries and walk to the fridge. Open fridge and put groceries in the fridge. Close the fridge.).\n\nBelow is the action history and the last turn's information:\nAction History: walk to dining room, walk to food food, find food food, grab food food, walk to freezer, open freezer\nLast Turn's Information: \nReasoning: The last action “Walk to freezer” was expected to move the agent near the fridge for storing groceries. The observation confirms the agent arrived at the freezer, which is closed and on the floor, and the agent is still holding the groceries.\n\nBelief State: The agent is currently at the freezer. The agent is holding the groceries (food_food). The groceries are dirty and clean (contradictory properties noted). The freezer is closed. The agent previously checked the dining room but did not find the groceries there (they were found via \"Find food food\").\nAction: Open freezer\nObservation: food_food is CLEAN, food_food is DIRTY, floor is CLEAN, freezer is PLUGGED_IN and OPEN and CLEAN, floor is DIRTY
Reasoning: The last action \"Open freezer\" was expected to open the freezer so groceries could be placed inside. The observation confirms success: the freezer is now open, but note that the task requires putting groceries in the fridge, not the freezer.  \nBelief State: The agent is currently at the freezer. The agent is holding \"food_food\" (the groceries). The freezer is open, but the fridge (the goal receptacle) has not yet been located or opened. The agent previously checked the dining room and found no groceries there.\nAction: Put food food in freezer

Your task is to complete the task goal: groceries in Fridge (Walk to kitchen. Grab groceries and walk to the fridge. Open fridge and put groceries in the fridge. Close the fridge.).\n\nBelow is the action history and the last turn's information:\nAction History: walk to dining room, walk to food food, find food food, grab food food, walk to freezer, open freezer, put food food in freezer\nLast Turn's Information: \nReasoning: The last action \"Open freezer\" was expected to open the freezer so groceries could be placed inside. The observation confirms success: the freezer is now open, but note that the task requires putting groceries in the fridge, not the freezer.  \nBelief State: The agent is currently at the freezer. The agent is holding \"food_food\" (the groceries). The freezer is open, but the fridge (the goal receptacle) has not yet been located or opened. The agent previously checked the dining room and found no groceries there.\nAction: Put food food in freezer\nObservation: food_food is CLEAN, food_food is DIRTY, floor is CLEAN, freezer is PLUGGED_IN and OPEN and CLEAN, floor is DIRTY, freezer ON floor. 
Reasoning: The last action “Put food food in freezer” was expected to place the held groceries into the open freezer. The observation shows the freezer remains open and the agent still holds the item, indicating the action failed (likely because “food food” is not a valid object for the freezer or the freezer is not a valid receptacle for it).  \nBelief State: The agent is currently at the freezer. The agent is holding “food food” (the groceries). The freezer is open, plugged in, and clean, but the groceries were not successfully placed inside. The agent previously visited the dining room and found no relevant objects there.\nAction: Close freezer

Your task is to complete the task goal: {task_goal}

Below is the action history and the last turn's information:
Action History: {action_history}
Last Turn's Information: {last_turn_information}"""