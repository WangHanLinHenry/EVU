SCIENCEWORLD_TEMPLATE_OUR_METHOD = """
You are a helpful assistant to do some scientific experiment in an environment.\nIn the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway\nYou should explore the environment and find the items you need to complete the experiment.\nYou can teleport to any room in one step.\nAll containers in the environment have already been opened, you can directly get items from the containers.

At each step, you will be given task goal, action history and the last turn's information (Reason, Belief State, Thought, and Action).

You need to process the information in a specific order:
1. **Reason**: Analyze the last action and the observation in one or two concise sentences. What did you expect to see? What did you actually see? Does this confirm or contradict your previous belief?
2. **Belief State**: State where the agent is, what it is holding, and the known status of goal-related objects. Do NOT list irrelevant objects.
3. **Thought**: Plan your future actions based on the updated belief.
4. **Action**: Output your next action.

The available actions are:\n    open OBJ: open a container\n    close OBJ: close a container\n    activate OBJ: activate a device\n    deactivate OBJ: deactivate a device\n    connect OBJ to OBJ: connect electrical components\n    disconnect OBJ: disconnect electrical components\n    use OBJ [on OBJ]: use a device/item\n    look around: describe the current room\n    examine OBJ: describe an object in detail\n    look at OBJ: describe a container's contents\n    read OBJ: read a note or book\n    move OBJ to OBJ: move an object to a container\n    pick up OBJ: move an object to the inventory\n    pour OBJ into OBJ: pour a liquid into a container\n    mix OBJ: chemically mix a container\n    teleport to LOC: teleport to a specific room\n    focus on OBJ: signal intent on a task object\n    wait: task no action for 10 steps\n    wait1: task no action for a step"

Your response should use the following format:

Reason: <Analyze expectation vs. actual observation to update your understanding>
Belief State: <your belief state>
Thought: <your thoughts>
Action: <your next action>

Your task is to complete the task goal: {task_goal}

Below is the action history and the last turn's information:
Action History: {action_history}
Last Turn's Information: {last_turn_information}"""

# 这个是variant中训练用的prompt
VARIANR_SCIENCEWORLD_TEMPLATE_ONLY_BELIEF_THOUGHT_ACTION = """
You are a helpful assistant to do some scientific experiment in an environment.\nIn the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway\nYou should explore the environment and find the items you need to complete the experiment.\nYou can teleport to any room in one step.\nAll containers in the environment have already been opened, you can directly get items from the containers.

At each step, you will be given task goal, action history and the last turn's information (Belief State, Thought, and Action).

You need to process the information in a specific order:
1. **Belief State**: State where the agent is, what it is holding, and the known status of goal-related objects. Do NOT list irrelevant objects.
2. **Thought**: Plan your future actions based on the updated belief.
3. **Action**: Output your next action.

The available actions are:\n    open OBJ: open a container\n    close OBJ: close a container\n    activate OBJ: activate a device\n    deactivate OBJ: deactivate a device\n    connect OBJ to OBJ: connect electrical components\n    disconnect OBJ: disconnect electrical components\n    use OBJ [on OBJ]: use a device/item\n    look around: describe the current room\n    examine OBJ: describe an object in detail\n    look at OBJ: describe a container's contents\n    read OBJ: read a note or book\n    move OBJ to OBJ: move an object to a container\n    pick up OBJ: move an object to the inventory\n    pour OBJ into OBJ: pour a liquid into a container\n    mix OBJ: chemically mix a container\n    teleport to LOC: teleport to a specific room\n    focus on OBJ: signal intent on a task object\n    wait: task no action for 10 steps\n    wait1: task no action for a step"

Your response should use the following format:

Belief State: <your belief state>
Thought: <your thoughts>
Action: <your next action>

Your task is to complete the task goal: {task_goal}

Below is the action history and the last turn's information:
Action History: {action_history}
Last Turn's Information: {last_turn_information}"""


VARIANR_SCIENCEWORLD_TEMPLATE_REPEAT_OBSERVATION_THOUGHT_ACTION = """
You are a helpful assistant to do some scientific experiment in an environment.\nIn the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway\nYou should explore the environment and find the items you need to complete the experiment.\nYou can teleport to any room in one step.\nAll containers in the environment have already been opened, you can directly get items from the containers.

At each step, you will be given task goal, action history and the last turn's information (Reason, Belief State, Thought, and Action).

You need to process the information in a specific order:
1. **Reason**: Repeat the last turn's observation in one or two concise sentences.
2. **Belief State**: State where the agent is, what it is holding, and the known status of goal-related objects. Do NOT list irrelevant objects.
3. **Thought**: Plan your future actions based on the updated belief.
4. **Action**: Output your next action.

The available actions are:\n    open OBJ: open a container\n    close OBJ: close a container\n    activate OBJ: activate a device\n    deactivate OBJ: deactivate a device\n    connect OBJ to OBJ: connect electrical components\n    disconnect OBJ: disconnect electrical components\n    use OBJ [on OBJ]: use a device/item\n    look around: describe the current room\n    examine OBJ: describe an object in detail\n    look at OBJ: describe a container's contents\n    read OBJ: read a note or book\n    move OBJ to OBJ: move an object to a container\n    pick up OBJ: move an object to the inventory\n    pour OBJ into OBJ: pour a liquid into a container\n    mix OBJ: chemically mix a container\n    teleport to LOC: teleport to a specific room\n    focus on OBJ: signal intent on a task object\n    wait: task no action for 10 steps\n    wait1: task no action for a step"

Your response should use the following format:

Reason: <Repeat the last turn's observation in one or two concise sentences.>
Belief State: <your belief state>
Thought: <your thoughts>
Action: <your next action>

Your task is to complete the task goal: {task_goal}

Below is the action history and the last turn's information:
Action History: {action_history}
Last Turn's Information: {last_turn_information}"""


# 这个是闭源大模型进行test用的prompt。variant
PROMPTING_VARIANR_SCIENCEWORLD_TEMPLATE_ONLY_BELIEF_THOUGHT_ACTION = """
You are a helpful assistant to do some scientific experiment in an environment.\nIn the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway\nYou should explore the environment and find the items you need to complete the experiment.\nYou can teleport to any room in one step.\nAll containers in the environment have already been opened, you can directly get items from the containers.

At each step, you will be given task goal, action history and the last turn's information (Belief State, Thought, and Action).

You need to process the information in a specific order:
1. **Belief State**: State where the agent is, what it is holding, and the known status of goal-related objects. Do NOT list irrelevant objects.
2. **Thought**: Plan your future actions based on the updated belief.
3. **Action**: Output your next action.

The available actions are:\n    open OBJ: open a container\n    close OBJ: close a container\n    activate OBJ: activate a device\n    deactivate OBJ: deactivate a device\n    connect OBJ to OBJ: connect electrical components\n    disconnect OBJ: disconnect electrical components\n    use OBJ [on OBJ]: use a device/item\n    look around: describe the current room\n    examine OBJ: describe an object in detail\n    look at OBJ: describe a container's contents\n    read OBJ: read a note or book\n    move OBJ to OBJ: move an object to a container\n    pick up OBJ: move an object to the inventory\n    pour OBJ into OBJ: pour a liquid into a container\n    mix OBJ: chemically mix a container\n    teleport to LOC: teleport to a specific room\n    focus on OBJ: signal intent on a task object\n    wait: task no action for 10 steps\n    wait1: task no action for a step"

Your response should use the following format:

Belief State: <your belief state>
Thought: <your thoughts>
Action: <your next action>

Here is an one-shot example:

Your task is to complete the task goal: find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.\n\nBelow is the action history and the last turn's information:\nAction History: \nLast Turn's Information: Your task is to find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen
Belief State: The task is to find an animal and move it to the red box in the kitchen. The agent is in an unknown starting location. No rooms have been searched, and no animals or the red box have been located yet.\nThought: I need to find an animal as per the task description. I should start by looking around the current location, which is the hallway, to identify any potential animals or clues to where one might be found.\nAction: look around

Your task is to complete the task goal: find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.\n\nBelow is the action history and the last turn's information:\nAction History: look around\nLast Turn's Information: Belief State: The task is to find an animal and move it to the red box in the kitchen. The agent is in an unknown starting location. No rooms have been searched, and no animals or the red box have been located yet.\nThought: I need to find an animal as per the task description. I should start by looking around the current location, which is the hallway, to identify any potential animals or clues to where one might be found.\nAction: look around\nObservation: This room is called the hallway. In it, you see: \n\tthe agent\n\ta substance called air\n\ta picture\nYou also see:\n\tA door to the art studio (that is open)\n\tA door to the bedroom (that is open)\n\tA door to the greenhouse (that is open)\n\tA door to the kitchen (that is open)\n\tA door to the living room (that is open)\n\tA door to the workshop (that is open)
Belief State: The agent is in the Hallway, which contains no animals. Doors are open to the art studio, bedroom, greenhouse, kitchen, living room, and workshop. The agent has not yet checked any other rooms for an animal.\nThought: After observing the hallway, I didn't find any animals here. There are multiple open doors leading to other rooms, but I should consider looking outside as animals are often found outdoors.\nAction: teleport to outside

Your task is to complete the task goal: find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.\n\nBelow is the action history and the last turn's information:\nAction History: look around, teleport to outside\nLast Turn's Information: Belief State: The agent is in the Hallway, which contains no animals. Doors are open to the art studio, bedroom, greenhouse, kitchen, living room, and workshop. The agent has not yet checked any other rooms for an animal.\nThought: After observing the hallway, I didn't find any animals here. There are multiple open doors leading to other rooms, but I should consider looking outside as animals are often found outdoors.\nAction: teleport to outside\nObservation: You teleport to the outside.
Belief State: The agent is now Outside. The hallway was checked and no animal was found there. The agent has not yet searched the outside area for animals.\nThought: Now that I'm outside, I should look around carefully to see if there are any animals or signs of animals in this area.\nAction: look around

Your task is to complete the task goal: find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.\n\nBelow is the action history and the last turn's information:\nAction History: look around, teleport to outside, look around\nLast Turn's Information: Belief State: The agent is now Outside. The hallway was checked and no animal was found there. The agent has not yet searched the outside area for animals.\nThought: Now that I'm outside, I should look around carefully to see if there are any animals or signs of animals in this area.\nAction: look around\nObservation: This outside location is called the outside. Here you see: \n\tthe agent\n\ta substance called air\n\tan axe\n\ta blue jay egg\n\ta butterfly egg\n\ta dove egg\n\ta fire pit (containing nothing)\n\ta fountain (containing a substance called water)\n\tthe ground\n\ta substance called wood\nYou also see:\n\tA door to the foundry (that is open)\n\tA door to the greenhouse (that is open)\n\tA door to the kitchen (that is open)
Belief State: The agent is currently in Outside. The agent has checked the hallway and found no animal. In Outside, visible potential animals are eggs (blue jay egg, butterfly egg, dove egg) on the ground. Also present are a fountain (containing water), an empty fire pit, and tools/materials (axe, wood). No adult animals are currently observed.\nThought: I see various eggs here, which are potential animals in early stages of development. The dove egg could be the animal I need to find for the task. I should focus on it to ensure I can interact with it properly.\nAction: focus on dove

Your task is to complete the task goal: find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.\n\nBelow is the action history and the last turn's information:\nAction History: look around, teleport to outside, look around, focus on dove\nLast Turn's Information: Belief State: The agent is currently in Outside. The agent has checked the hallway and found no animal. In Outside, visible potential animals are eggs (blue jay egg, butterfly egg, dove egg) on the ground. Also present are a fountain (containing water), an empty fire pit, and tools/materials (axe, wood). No adult animals are currently observed.\nThought: I see various eggs here, which are potential animals in early stages of development. The dove egg could be the animal I need to find for the task. I should focus on it to ensure I can interact with it properly.\nAction: focus on dove\nObservation: You focus on the dove egg.
Belief State: The agent is currently Outside, with the dove egg focused. Nearby are other potential animal eggs (blue jay egg, butterfly egg), an empty fire pit, a fountain (containing water), and tools/materials (axe, wood). The agent previously checked the hallway and found no animals there.\nThought: The dove egg is now my focus, and I need to move it to the red box in the kitchen as per the task. I should pick it up and add it to my inventory to transport it.\nAction: pick up dove

Your task is to complete the task goal: find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.\n\nBelow is the action history and the last turn's information:\nAction History: look around, teleport to outside, look around, focus on dove, pick up dove\nLast Turn's Information: Belief State: The agent is currently Outside, with the dove egg focused. Nearby are other potential animal eggs (blue jay egg, butterfly egg), an empty fire pit, a fountain (containing water), and tools/materials (axe, wood). The agent previously checked the hallway and found no animals there.\nThought: The dove egg is now my focus, and I need to move it to the red box in the kitchen as per the task. I should pick it up and add it to my inventory to transport it.\nAction: pick up dove\nObservation: You move the dove to the inventory.
Belief State: The agent is currently Outside, holding the dove egg in inventory. Nearby are other eggs (blue jay, butterfly) and objects (axe, fountain with water, empty fire pit, wood). The agent previously checked the hallway and found no animals. The next step is to teleport to the kitchen and locate the red box.\nThought: With the dove egg in my inventory, I need to go to the kitchen to complete the task. Teleporting there will be the quickest way.\nAction: teleport to kitchen

Your task is to complete the task goal: find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.\n\nBelow is the action history and the last turn's information:\nAction History: look around, teleport to outside, look around, focus on dove, pick up dove, teleport to kitchen\nLast Turn's Information: Belief State: The agent is currently Outside, holding the dove egg in inventory. Nearby are other eggs (blue jay, butterfly) and objects (axe, fountain with water, empty fire pit, wood). The agent previously checked the hallway and found no animals. The next step is to teleport to the kitchen and locate the red box.\nThought: With the dove egg in my inventory, I need to go to the kitchen to complete the task. Teleporting there will be the quickest way.\nAction: teleport to kitchen\nObservation: You teleport to the kitchen.
Belief State: The agent is in the Kitchen holding the dove egg (the focused animal). The red box (target location) is presumably present but not yet examined. Previously checked the Hallway (no animals found) and Outside (found dove, blue jay, and butterfly eggs, plus fountain with water and empty fire pit).\nThought: I'm now in the kitchen, and I must place the dove egg into the red box to finish the task. I should move it from my inventory to the red box carefully.\nAction: move egg dove egg in inventory to red box

Your task is to complete the task goal: {task_goal}

Below is the action history and the last turn's information:
Action History: {action_history}
Last Turn's Information: {last_turn_information}"""


PROMPTING_VARIANR_SCIENCEWORLD_TEMPLATE_REPEAT_OBSERVATION_THOUGHT_ACTION = """
You are a helpful assistant to do some scientific experiment in an environment.\nIn the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway\nYou should explore the environment and find the items you need to complete the experiment.\nYou can teleport to any room in one step.\nAll containers in the environment have already been opened, you can directly get items from the containers.

At each step, you will be given task goal, action history and the last turn's information (Reason, Belief State, Thought, and Action).

You need to process the information in a specific order:
1. **Reason**: Repeat the last turn's observation in one or two concise sentences.
2. **Belief State**: State where the agent is, what it is holding, and the known status of goal-related objects. Do NOT list irrelevant objects.
3. **Thought**: Plan your future actions based on the updated belief.
4. **Action**: Output your next action.

The available actions are:\n    open OBJ: open a container\n    close OBJ: close a container\n    activate OBJ: activate a device\n    deactivate OBJ: deactivate a device\n    connect OBJ to OBJ: connect electrical components\n    disconnect OBJ: disconnect electrical components\n    use OBJ [on OBJ]: use a device/item\n    look around: describe the current room\n    examine OBJ: describe an object in detail\n    look at OBJ: describe a container's contents\n    read OBJ: read a note or book\n    move OBJ to OBJ: move an object to a container\n    pick up OBJ: move an object to the inventory\n    pour OBJ into OBJ: pour a liquid into a container\n    mix OBJ: chemically mix a container\n    teleport to LOC: teleport to a specific room\n    focus on OBJ: signal intent on a task object\n    wait: task no action for 10 steps\n    wait1: task no action for a step"

Your response should use the following format:

Reason: <Repeat the last turn's observation in one or two concise sentences.>
Belief State: <your belief state>
Thought: <your thoughts>
Action: <your next action>

Below is an one-shot example:
Your task is to complete the task goal: find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.\n\nBelow is the action history and the last turn's information:\nAction History: \nLast Turn's Information: Your task is to find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen
Reasoning: None\n\nBelief State: The task is to find an animal and move it to the red box in the kitchen. The agent is in an unknown starting location. No rooms have been searched, and no animals or the red box have been located yet.\nThought: I need to find an animal as per the task description. I should start by looking around the current location, which is the hallway, to identify any potential animals or clues to where one might be found.\nAction: look around

Your task is to complete the task goal: find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.\n\nBelow is the action history and the last turn's information:\nAction History: look around\nLast Turn's Information: Reasoning: None\n\nBelief State: The task is to find an animal and move it to the red box in the kitchen. The agent is in an unknown starting location. No rooms have been searched, and no animals or the red box have been located yet.\nThought: I need to find an animal as per the task description. I should start by looking around the current location, which is the hallway, to identify any potential animals or clues to where one might be found.\nAction: look around\nObservation: This room is called the hallway. In it, you see: \n\tthe agent\n\ta substance called air\n\ta picture\nYou also see:\n\tA door to the art studio (that is open)\n\tA door to the bedroom (that is open)\n\tA door to the greenhouse (that is open)\n\tA door to the kitchen (that is open)\n\tA door to the living room (that is open)\n\tA door to the workshop (that is open)
Reasoning: This room is called the hallway. In it, you see: \n\tthe agent\n\ta substance called air\n\ta picture\nYou also see:\n\tA door to the art studio (that is open)\n\tA door to the bedroom (that is open)\n\tA door to the greenhouse (that is open)\n\tA door to the kitchen (that is open)\n\tA door to the living room (that is open)\n\tA door to the workshop (that is open)\n\nBelief State: The agent is in the Hallway, which contains no animals. Doors are open to the art studio, bedroom, greenhouse, kitchen, living room, and workshop. The agent has not yet checked any other rooms for an animal.\nThought: After observing the hallway, I didn't find any animals here. There are multiple open doors leading to other rooms, but I should consider looking outside as animals are often found outdoors.\nAction: teleport to outside

Your task is to complete the task goal: find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.\n\nBelow is the action history and the last turn's information:\nAction History: look around, teleport to outside\nLast Turn's Information: Reasoning: This room is called the hallway. In it, you see: \n\tthe agent\n\ta substance called air\n\ta picture\nYou also see:\n\tA door to the art studio (that is open)\n\tA door to the bedroom (that is open)\n\tA door to the greenhouse (that is open)\n\tA door to the kitchen (that is open)\n\tA door to the living room (that is open)\n\tA door to the workshop (that is open)\n\nBelief State: The agent is in the Hallway, which contains no animals. Doors are open to the art studio, bedroom, greenhouse, kitchen, living room, and workshop. The agent has not yet checked any other rooms for an animal.\nThought: After observing the hallway, I didn't find any animals here. There are multiple open doors leading to other rooms, but I should consider looking outside as animals are often found outdoors.\nAction: teleport to outside\nObservation: You teleport to the outside.
Reasoning: The last action (“teleport to outside”) was intended to move the agent to a location where animals are more likely to be found. The observation confirms the teleport succeeded instantly, with no delayed process or waiting required.  \nBelief State: The agent is now Outside. The hallway was checked and no animal was found there. The agent has not yet searched the outside area for animals.\nThought: Now that I'm outside, I should look around carefully to see if there are any animals or signs of animals in this area.\nAction: look around

Your task is to complete the task goal: find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.\n\nBelow is the action history and the last turn's information:\nAction History: look around, teleport to outside, look around\nLast Turn's Information: Reasoning: The last action (“teleport to outside”) was intended to move the agent to a location where animals are more likely to be found. The observation confirms the teleport succeeded instantly, with no delayed process or waiting required.  \nBelief State: The agent is now Outside. The hallway was checked and no animal was found there. The agent has not yet searched the outside area for animals.\nThought: Now that I'm outside, I should look around carefully to see if there are any animals or signs of animals in this area.\nAction: look around\nObservation: This outside location is called the outside. Here you see: \n\tthe agent\n\ta substance called air\n\tan axe\n\ta blue jay egg\n\ta butterfly egg\n\ta dove egg\n\ta fire pit (containing nothing)\n\ta fountain (containing a substance called water)\n\tthe ground\n\ta substance called wood\nYou also see:\n\tA door to the foundry (that is open)\n\tA door to the greenhouse (that is open)\n\tA door to the kitchen (that is open)
Reasoning: This outside location is called the outside. Here you see: \n\tthe agent\n\ta substance called air\n\tan axe\n\ta blue jay egg\n\ta butterfly egg\n\ta dove egg\n\ta fire pit (containing nothing)\n\ta fountain (containing a substance called water)\n\tthe ground\n\ta substance called wood\nYou also see:\n\tA door to the foundry (that is open)\n\tA door to the greenhouse (that is open)\n\tA door to the kitchen (that is open)\n\nBelief State: The agent is currently in Outside. The agent has checked the hallway and found no animal. In Outside, visible potential animals are eggs (blue jay egg, butterfly egg, dove egg) on the ground. Also present are a fountain (containing water), an empty fire pit, and tools/materials (axe, wood). No adult animals are currently observed.\nThought: I see various eggs here, which are potential animals in early stages of development. The dove egg could be the animal I need to find for the task. I should focus on it to ensure I can interact with it properly.\nAction: focus on dove

Your task is to complete the task goal: find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.\n\nBelow is the action history and the last turn's information:\nAction History: look around, teleport to outside, look around, focus on dove\nLast Turn's Information: Reasoning: This outside location is called the outside. Here you see: \n\tthe agent\n\ta substance called air\n\tan axe\n\ta blue jay egg\n\ta butterfly egg\n\ta dove egg\n\ta fire pit (containing nothing)\n\ta fountain (containing a substance called water)\n\tthe ground\n\ta substance called wood\nYou also see:\n\tA door to the foundry (that is open)\n\tA door to the greenhouse (that is open)\n\tA door to the kitchen (that is open)\n\nBelief State: The agent is currently in Outside. The agent has checked the hallway and found no animal. In Outside, visible potential animals are eggs (blue jay egg, butterfly egg, dove egg) on the ground. Also present are a fountain (containing water), an empty fire pit, and tools/materials (axe, wood). No adult animals are currently observed.\nThought: I see various eggs here, which are potential animals in early stages of development. The dove egg could be the animal I need to find for the task. I should focus on it to ensure I can interact with it properly.\nAction: focus on dove\nObservation: You focus on the dove egg.
Reasoning: The last action (“focus on dove”) was intended to select the dove egg as the target animal for the task. The observation confirms success: the dove egg is now focused, with no delayed process or waiting required.  \nBelief State: The agent is currently Outside, with the dove egg focused. Nearby are other potential animal eggs (blue jay egg, butterfly egg), an empty fire pit, a fountain (containing water), and tools/materials (axe, wood). The agent previously checked the hallway and found no animals there.\nThought: The dove egg is now my focus, and I need to move it to the red box in the kitchen as per the task. I should pick it up and add it to my inventory to transport it.\nAction: pick up dove

Your task is to complete the task goal: find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.\n\nBelow is the action history and the last turn's information:\nAction History: look around, teleport to outside, look around, focus on dove, pick up dove\nLast Turn's Information: Reasoning: The last action (“focus on dove”) was intended to select the dove egg as the target animal for the task. The observation confirms success: the dove egg is now focused, with no delayed process or waiting required.  \nBelief State: The agent is currently Outside, with the dove egg focused. Nearby are other potential animal eggs (blue jay egg, butterfly egg), an empty fire pit, a fountain (containing water), and tools/materials (axe, wood). The agent previously checked the hallway and found no animals there.\nThought: The dove egg is now my focus, and I need to move it to the red box in the kitchen as per the task. I should pick it up and add it to my inventory to transport it.\nAction: pick up dove\nObservation: You move the dove to the inventory.
Reasoning: You move the dove to the inventory.\n\nBelief State: The agent is currently Outside, holding the dove egg in inventory. Nearby are other eggs (blue jay, butterfly) and objects (axe, fountain with water, empty fire pit, wood). The agent previously checked the hallway and found no animals. The next step is to teleport to the kitchen and locate the red box.\nThought: With the dove egg in my inventory, I need to go to the kitchen to complete the task. Teleporting there will be the quickest way.\nAction: teleport to kitchen

Your task is to complete the task goal: find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.\n\nBelow is the action history and the last turn's information:\nAction History: look around, teleport to outside, look around, focus on dove, pick up dove, teleport to kitchen\nLast Turn's Information: Reasoning: You move the dove to the inventory.\n\nBelief State: The agent is currently Outside, holding the dove egg in inventory. Nearby are other eggs (blue jay, butterfly) and objects (axe, fountain with water, empty fire pit, wood). The agent previously checked the hallway and found no animals. The next step is to teleport to the kitchen and locate the red box.\nThought: With the dove egg in my inventory, I need to go to the kitchen to complete the task. Teleporting there will be the quickest way.\nAction: teleport to kitchen\nObservation: You teleport to the kitchen.
Reasoning: You teleport to the kitchen.\n\nBelief State: The agent is in the Kitchen holding the dove egg (the focused animal). The red box (target location) is presumably present but not yet examined. Previously checked the Hallway (no animals found) and Outside (found dove, blue jay, and butterfly eggs, plus fountain with water and empty fire pit).\nThought: I'm now in the kitchen, and I must place the dove egg into the red box to finish the task. I should move it from my inventory to the red box carefully.\nAction: move egg dove egg in inventory to red box

Your task is to complete the task goal: {task_goal}

Below is the action history and the last turn's information:
Action History: {action_history}
Last Turn's Information: {last_turn_information}"""


# 下面这个是给react的
# PROMPTING_METHOD_SCIENCEWORLD_TEMPLATE_BASELINE = """
# You are a helpful assistant to do some scientific experiment in an environment.\nIn the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway\nYou should explore the environment and find the items you need to complete the experiment.\nYou can teleport to any room in one step.\nAll containers in the environment have already been opened, you can directly get items from the containers.
# At each step, you will be given interaction history.

# You need to process the information in a specific order:
# 1. **Thought**: Plan your future actions based on the updated belief.
# 2. **Action**: Output your next action.

# The available actions are(Your action should strictly follow this format):\n    open OBJ: open a container\n    close OBJ: close a container\n    activate OBJ: activate a device\n    deactivate OBJ: deactivate a device\n    connect OBJ to OBJ: connect electrical components\n    disconnect OBJ: disconnect electrical components\n    use OBJ [on OBJ]: use a device/item\n    look around: describe the current room\n    examine OBJ: describe an object in detail\n    look at OBJ: describe a container's contents\n    read OBJ: read a note or book\n    move OBJ to OBJ: move an object to a container\n    pick up OBJ: move an object to the inventory\n    pour OBJ into OBJ: pour a liquid into a container\n    mix OBJ: chemically mix a container\n    teleport to LOC: teleport to a specific room\n    focus on OBJ: signal intent on a task object\n    wait: task no action for 10 steps\n    wait1: task no action for a step"

# Your response should use the following format:
# Thought: <your thoughts>
# Action: <your next action>

# Below is an one-shot example:
# Your task is to find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.
# Thought: I need to find an animal as per the task description. I should start by looking around the current location, which is the hallway, to identify any potential animals or clues to where one might be found.\nAction: look around
# Observation: This room is called the hallway. In it, you see: \n\tthe agent\n\ta substance called air\n\ta picture\nYou also see:\n\tA door to the art studio (that is open)\n\tA door to the bedroom (that is open)\n\tA door to the greenhouse (that is open)\n\tA door to the kitchen (that is open)\n\tA door to the living room (that is open)\n\tA door to the workshop (that is open)
# Thought: After observing the hallway, I didn't find any animals here. There are multiple open doors leading to other rooms, but I should consider looking outside as animals are often found outdoors.\nAction: teleport to outside
# Observation: You teleport to the outside.
# Thought: Now that I'm outside, I should look around carefully to see if there are any animals or signs of animals in this area.\nAction: look around
# Observation: This outside location is called the outside. Here you see: \n\tthe agent\n\ta substance called air\n\tan axe\n\ta blue jay egg\n\ta butterfly egg\n\ta dove egg\n\ta fire pit (containing nothing)\n\ta fountain (containing a substance called water)\n\tthe ground\n\ta substance called wood\nYou also see:\n\tA door to the foundry (that is open)\n\tA door to the greenhouse (that is open)\n\tA door to the kitchen (that is open)
# Thought: I see various eggs here, which are potential animals in early stages of development. The dove egg could be the animal I need to find for the task. I should focus on it to ensure I can interact with it properly.\nAction: focus on dove
# Observation: You focus on the dove egg.
# Thought: The dove egg is now my focus, and I need to move it to the red box in the kitchen as per the task. I should pick it up and add it to my inventory to transport it.\nAction: pick up dove
# Observation: You move the dove to the inventory.
# Thought: With the dove egg in my inventory, I need to go to the kitchen to complete the task. Teleporting there will be the quickest way.\nAction: teleport to kitchen
# Observation: You teleport to the kitchen.
# Thought: I'm now in the kitchen, and I must place the dove egg into the red box to finish the task. I should move it from my inventory to the red box carefully.\nAction: move egg dove egg in inventory to red box

# Below is the interaction history: {last_turn_information}"""


# PROMPTING_METHOD_SCIENCEWORLD_TEMPLATE_OUR_METHOD = """
# You are a helpful assistant to do some scientific experiment in an environment.\nIn the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway\nYou should explore the environment and find the items you need to complete the experiment.\nYou can teleport to any room in one step.\nAll containers in the environment have already been opened, you can directly get items from the containers.

# At each step, you will be given task goal, action history and the last turn's information (Reason, Belief State, Thought, and Action).

# You need to process the information in a specific order:
# 1. **Reason**: Analyze the last action and the observation in one or two concise sentences. What did you expect to see? What did you actually see? Does this confirm or contradict your previous belief?
# 2. **Belief State**: State where the agent is, what it is holding, and the known status of goal-related objects. Do NOT list irrelevant objects.
# 3. **Thought**: Plan your future actions based on the updated belief.
# 4. **Action**: Output your next action.

# The available actions are(Your action should strictly follow this format):\n    open OBJ: open a container\n    close OBJ: close a container\n    activate OBJ: activate a device\n    deactivate OBJ: deactivate a device\n    connect OBJ to OBJ: connect electrical components\n    disconnect OBJ: disconnect electrical components\n    use OBJ [on OBJ]: use a device/item\n    look around: describe the current room\n    examine OBJ: describe an object in detail\n    look at OBJ: describe a container's contents\n    read OBJ: read a note or book\n    move OBJ to OBJ: move an object to a container\n    pick up OBJ: move an object to the inventory\n    pour OBJ into OBJ: pour a liquid into a container\n    mix OBJ: chemically mix a container\n    teleport to LOC: teleport to a specific room\n    focus on OBJ: signal intent on a task object\n    wait: task no action for 10 steps\n    wait1: task no action for a step"

# Your response should use the following format:

# Reason: <Analyze expectation vs. actual observation to update your understanding>
# Belief State: <your belief state>
# Thought: <your thoughts>
# Action: <your next action>

# Below is an one-shot example:

# Your task is to complete the task goal: find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.\n\nBelow is the action history and the last turn's information:\nAction History: \nLast Turn's Information: \nYour task is to find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen
# Reasoning: The agent has not yet taken any action, so there is no expectation vs. reality discrepancy to analyze. The initial observation simply presents the task.\n\nBelief State: The task is to find an animal and move it to the red box in the kitchen. The agent is in an unknown starting location. No rooms have been searched, and no animals or the red box have been located yet.\nThought: I need to find an animal as per the task description. I should start by looking around the current location, which is the hallway, to identify any potential animals or clues to where one might be found.\nAction: look around

# Your task is to complete the task goal: find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.\n\nBelow is the action history and the last turn's information:\nAction History: look around\nLast Turn's Information: \nReasoning: The agent has not yet taken any action, so there is no expectation vs. reality discrepancy to analyze. The initial observation simply presents the task.\n\nBelief State: The task is to find an animal and move it to the red box in the kitchen. The agent is in an unknown starting location. No rooms have been searched, and no animals or the red box have been located yet.\nThought: I need to find an animal as per the task description. I should start by looking around the current location, which is the hallway, to identify any potential animals or clues to where one might be found.\nAction: look around\nObservation: This room is called the hallway. In it, you see: \n\tthe agent\n\ta substance called air\n\ta picture\nYou also see:\n\tA door to the art studio (that is open)\n\tA door to the bedroom (that is open)\n\tA door to the greenhouse (that is open)\n\tA door to the kitchen (that is open)\n\tA door to the living room (that is open)\n\tA door to the workshop (that is open)
# Reasoning: The last action (“look around”) was expected to reveal potential animals or clues in the hallway. The observation shows only the agent, air, and a picture, with no animals present. The action succeeded in surveying the room, but no animal was found, so the agent must search other rooms.\n\nBelief State: The agent is in the Hallway, which contains no animals. Doors are open to the art studio, bedroom, greenhouse, kitchen, living room, and workshop. The agent has not yet checked any other rooms for an animal.\nThought: After observing the hallway, I didn't find any animals here. There are multiple open doors leading to other rooms, but I should consider looking outside as animals are often found outdoors.\nAction: teleport to outside

# Your task is to complete the task goal: find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.\n\nBelow is the action history and the last turn's information:\nAction History: look around, teleport to outside\nLast Turn's Information: \nReasoning: The last action (“look around”) was expected to reveal potential animals or clues in the hallway. The observation shows only the agent, air, and a picture, with no animals present. The action succeeded in surveying the room, but no animal was found, so the agent must search other rooms.\n\nBelief State: The agent is in the Hallway, which contains no animals. Doors are open to the art studio, bedroom, greenhouse, kitchen, living room, and workshop. The agent has not yet checked any other rooms for an animal.\nThought: After observing the hallway, I didn't find any animals here. There are multiple open doors leading to other rooms, but I should consider looking outside as animals are often found outdoors.\nAction: teleport to outside\nObservation: You teleport to the outside.
# Reasoning: The last action (“teleport to outside”) was intended to move the agent to a location where animals are more likely to be found. The observation confirms the teleport succeeded instantly, with no delayed process or waiting required.  \nBelief State: The agent is now Outside. The hallway was checked and no animal was found there. The agent has not yet searched the outside area for animals.\nThought: Now that I'm outside, I should look around carefully to see if there are any animals or signs of animals in this area.\nAction: look around

# Your task is to complete the task goal: find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.\n\nBelow is the action history and the last turn's information:\nAction History: look around, teleport to outside, look around\nLast Turn's Information: \nReasoning: The last action (“teleport to outside”) was intended to move the agent to a location where animals are more likely to be found. The observation confirms the teleport succeeded instantly, with no delayed process or waiting required.  \nBelief State: The agent is now Outside. The hallway was checked and no animal was found there. The agent has not yet searched the outside area for animals.\nThought: Now that I'm outside, I should look around carefully to see if there are any animals or signs of animals in this area.\nAction: look around\nObservation: This outside location is called the outside. Here you see: \n\tthe agent\n\ta substance called air\n\tan axe\n\ta blue jay egg\n\ta butterfly egg\n\ta dove egg\n\ta fire pit (containing nothing)\n\ta fountain (containing a substance called water)\n\tthe ground\n\ta substance called wood\nYou also see:\n\tA door to the foundry (that is open)\n\tA door to the greenhouse (that is open)\n\tA door to the kitchen (that is open)
# Reasoning: The last action (“teleport to outside”) was expected to move the agent to the outside location to search for an animal. The observation confirms success: the agent is now outside, and the teleportation was instantaneous with no delayed process.\n\nBelief State: The agent is currently in Outside. The agent has checked the hallway and found no animal. In Outside, visible potential animals are eggs (blue jay egg, butterfly egg, dove egg) on the ground. Also present are a fountain (containing water), an empty fire pit, and tools/materials (axe, wood). No adult animals are currently observed.\nThought: I see various eggs here, which are potential animals in early stages of development. The dove egg could be the animal I need to find for the task. I should focus on it to ensure I can interact with it properly.\nAction: focus on dove

# Your task is to complete the task goal: find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.\n\nBelow is the action history and the last turn's information:\nAction History: look around, teleport to outside, look around, focus on dove\nLast Turn's Information: \nReasoning: The last action (“teleport to outside”) was expected to move the agent to the outside location to search for an animal. The observation confirms success: the agent is now outside, and the teleportation was instantaneous with no delayed process.\n\nBelief State: The agent is currently in Outside. The agent has checked the hallway and found no animal. In Outside, visible potential animals are eggs (blue jay egg, butterfly egg, dove egg) on the ground. Also present are a fountain (containing water), an empty fire pit, and tools/materials (axe, wood). No adult animals are currently observed.\nThought: I see various eggs here, which are potential animals in early stages of development. The dove egg could be the animal I need to find for the task. I should focus on it to ensure I can interact with it properly.\nAction: focus on dove\nObservation: You focus on the dove egg.
# Reasoning: The last action (“focus on dove”) was intended to select the dove egg as the target animal for the task. The observation confirms success: the dove egg is now focused, with no delayed process or waiting required.  \nBelief State: The agent is currently Outside, with the dove egg focused. Nearby are other potential animal eggs (blue jay egg, butterfly egg), an empty fire pit, a fountain (containing water), and tools/materials (axe, wood). The agent previously checked the hallway and found no animals there.\nThought: The dove egg is now my focus, and I need to move it to the red box in the kitchen as per the task. I should pick it up and add it to my inventory to transport it.\nAction: pick up dove

# Your task is to complete the task goal: find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.\n\nBelow is the action history and the last turn's information:\nAction History: look around, teleport to outside, look around, focus on dove, pick up dove\nLast Turn's Information: \nReasoning: The last action (“focus on dove”) was intended to select the dove egg as the target animal for the task. The observation confirms success: the dove egg is now focused, with no delayed process or waiting required.  \nBelief State: The agent is currently Outside, with the dove egg focused. Nearby are other potential animal eggs (blue jay egg, butterfly egg), an empty fire pit, a fountain (containing water), and tools/materials (axe, wood). The agent previously checked the hallway and found no animals there.\nThought: The dove egg is now my focus, and I need to move it to the red box in the kitchen as per the task. I should pick it up and add it to my inventory to transport it.\nAction: pick up dove\nObservation: You move the dove to the inventory.
# Reasoning: The last action (“pick up dove”) was intended to collect the focused animal (the dove egg) for transport to the red box in the kitchen. The observation confirms success: the dove egg is now in the agent’s inventory, with no delayed process or waiting required.\n\nBelief State: The agent is currently Outside, holding the dove egg in inventory. Nearby are other eggs (blue jay, butterfly) and objects (axe, fountain with water, empty fire pit, wood). The agent previously checked the hallway and found no animals. The next step is to teleport to the kitchen and locate the red box.\nThought: With the dove egg in my inventory, I need to go to the kitchen to complete the task. Teleporting there will be the quickest way.\nAction: teleport to kitchen

# Your task is to complete the task goal: find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.\n\nBelow is the action history and the last turn's information:\nAction History: look around, teleport to outside, look around, focus on dove, pick up dove, teleport to kitchen\nLast Turn's Information: \nReasoning: The last action (“pick up dove”) was intended to collect the focused animal (the dove egg) for transport to the red box in the kitchen. The observation confirms success: the dove egg is now in the agent’s inventory, with no delayed process or waiting required.\n\nBelief State: The agent is currently Outside, holding the dove egg in inventory. Nearby are other eggs (blue jay, butterfly) and objects (axe, fountain with water, empty fire pit, wood). The agent previously checked the hallway and found no animals. The next step is to teleport to the kitchen and locate the red box.\nThought: With the dove egg in my inventory, I need to go to the kitchen to complete the task. Teleporting there will be the quickest way.\nAction: teleport to kitchen\nObservation: You teleport to the kitchen.
# Reasoning: The last action (“teleport to kitchen”) was expected to instantly move the agent to the kitchen to proceed with placing the animal in the red box. The observation confirms success: the agent is now in the kitchen, with no delayed process.\n\nBelief State: The agent is in the Kitchen holding the dove egg (the focused animal). The red box (target location) is presumably present but not yet examined. Previously checked the Hallway (no animals found) and Outside (found dove, blue jay, and butterfly eggs, plus fountain with water and empty fire pit).\nThought: I'm now in the kitchen, and I must place the dove egg into the red box to finish the task. I should move it from my inventory to the red box carefully.\nAction: move egg dove egg in inventory to red box
    
# Your task is to complete the task goal: {task_goal}

# Below is the action history and the last turn's information:
# Action History: {action_history}
# Last Turn's Information: {last_turn_information}"""


# no thinking的
PROMPTING_METHOD_SCIENCEWORLD_TEMPLATE_BASELINE = """
You are a helpful assistant to do some scientific experiment in an environment.\nIn the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway\nYou should explore the environment and find the items you need to complete the experiment.\nYou can teleport to any room in one step.\nAll containers in the environment have already been opened, you can directly get items from the containers.
At each step, you will be given interaction history.

You need to process the information in a specific order:
1. **Action**: Output your next action.

The available actions are(Your action should strictly follow this format):\n    open OBJ: open a container\n    close OBJ: close a container\n    activate OBJ: activate a device\n    deactivate OBJ: deactivate a device\n    connect OBJ to OBJ: connect electrical components\n    disconnect OBJ: disconnect electrical components\n    use OBJ [on OBJ]: use a device/item\n    look around: describe the current room\n    examine OBJ: describe an object in detail\n    look at OBJ: describe a container's contents\n    read OBJ: read a note or book\n    move OBJ to OBJ: move an object to a container\n    pick up OBJ: move an object to the inventory\n    pour OBJ into OBJ: pour a liquid into a container\n    mix OBJ: chemically mix a container\n    teleport to LOC: teleport to a specific room\n    focus on OBJ: signal intent on a task object\n    wait: task no action for 10 steps\n    wait1: task no action for a step"

Your response should use the following format:
Action: <your next action>

Below is an one-shot example:
Your task is to find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.
Action: look around
Observation: This room is called the hallway. In it, you see: \n\tthe agent\n\ta substance called air\n\ta picture\nYou also see:\n\tA door to the art studio (that is open)\n\tA door to the bedroom (that is open)\n\tA door to the greenhouse (that is open)\n\tA door to the kitchen (that is open)\n\tA door to the living room (that is open)\n\tA door to the workshop (that is open)
Action: teleport to outside
Observation: You teleport to the outside.
Action: look around
Observation: This outside location is called the outside. Here you see: \n\tthe agent\n\ta substance called air\n\tan axe\n\ta blue jay egg\n\ta butterfly egg\n\ta dove egg\n\ta fire pit (containing nothing)\n\ta fountain (containing a substance called water)\n\tthe ground\n\ta substance called wood\nYou also see:\n\tA door to the foundry (that is open)\n\tA door to the greenhouse (that is open)\n\tA door to the kitchen (that is open)
Action: focus on dove
Observation: You focus on the dove egg.
Action: pick up dove
Observation: You move the dove to the inventory.
Action: teleport to kitchen
Observation: You teleport to the kitchen.
Action: move egg dove egg in inventory to red box

Below is the interaction history: {last_turn_information}"""


PROMPTING_METHOD_SCIENCEWORLD_TEMPLATE_OUR_METHOD = """
You are a helpful assistant to do some scientific experiment in an environment.\nIn the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway\nYou should explore the environment and find the items you need to complete the experiment.\nYou can teleport to any room in one step.\nAll containers in the environment have already been opened, you can directly get items from the containers.

At each step, you will be given task goal, action history and the last turn's information (Reason, Belief State, Thought, and Action).

You need to process the information in a specific order:
1. **Reason**: Analyze the last action and the observation in one or two concise sentences. What did you expect to see? What did you actually see? Does this confirm or contradict your previous belief?
2. **Belief State**: State where the agent is, what it is holding, and the known status of goal-related objects. Do NOT list irrelevant objects.
3. **Action**: Output your next action.

The available actions are(Your action should strictly follow this format):\n    open OBJ: open a container\n    close OBJ: close a container\n    activate OBJ: activate a device\n    deactivate OBJ: deactivate a device\n    connect OBJ to OBJ: connect electrical components\n    disconnect OBJ: disconnect electrical components\n    use OBJ [on OBJ]: use a device/item\n    look around: describe the current room\n    examine OBJ: describe an object in detail\n    look at OBJ: describe a container's contents\n    read OBJ: read a note or book\n    move OBJ to OBJ: move an object to a container\n    pick up OBJ: move an object to the inventory\n    pour OBJ into OBJ: pour a liquid into a container\n    mix OBJ: chemically mix a container\n    teleport to LOC: teleport to a specific room\n    focus on OBJ: signal intent on a task object\n    wait: task no action for 10 steps\n    wait1: task no action for a step"

Your response should use the following format:

Reason: <Analyze expectation vs. actual observation to update your understanding>
Belief State: <your belief state>
Action: <your next action>

Below is an one-shot example:

Your task is to complete the task goal: find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.\n\nBelow is the action history and the last turn's information:\nAction History: \nLast Turn's Information: \nYour task is to find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen
Reasoning: The agent has not yet taken any action, so there is no expectation vs. reality discrepancy to analyze. The initial observation simply presents the task.\n\nBelief State: The task is to find an animal and move it to the red box in the kitchen. The agent is in an unknown starting location. No rooms have been searched, and no animals or the red box have been located yet.\nAction: look around

Your task is to complete the task goal: find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.\n\nBelow is the action history and the last turn's information:\nAction History: look around\nLast Turn's Information: \nReasoning: The agent has not yet taken any action, so there is no expectation vs. reality discrepancy to analyze. The initial observation simply presents the task.\n\nBelief State: The task is to find an animal and move it to the red box in the kitchen. The agent is in an unknown starting location. No rooms have been searched, and no animals or the red box have been located yet.\nAction: look around\nObservation: This room is called the hallway. In it, you see: \n\tthe agent\n\ta substance called air\n\ta picture\nYou also see:\n\tA door to the art studio (that is open)\n\tA door to the bedroom (that is open)\n\tA door to the greenhouse (that is open)\n\tA door to the kitchen (that is open)\n\tA door to the living room (that is open)\n\tA door to the workshop (that is open)
Reasoning: The last action (“look around”) was expected to reveal potential animals or clues in the hallway. The observation shows only the agent, air, and a picture, with no animals present. The action succeeded in surveying the room, but no animal was found, so the agent must search other rooms.\n\nBelief State: The agent is in the Hallway, which contains no animals. Doors are open to the art studio, bedroom, greenhouse, kitchen, living room, and workshop. The agent has not yet checked any other rooms for an animal.\nAction: teleport to outside

Your task is to complete the task goal: find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.\n\nBelow is the action history and the last turn's information:\nAction History: look around, teleport to outside\nLast Turn's Information: \nReasoning: The last action (“look around”) was expected to reveal potential animals or clues in the hallway. The observation shows only the agent, air, and a picture, with no animals present. The action succeeded in surveying the room, but no animal was found, so the agent must search other rooms.\n\nBelief State: The agent is in the Hallway, which contains no animals. Doors are open to the art studio, bedroom, greenhouse, kitchen, living room, and workshop. The agent has not yet checked any other rooms for an animal.\nAction: teleport to outside\nObservation: You teleport to the outside.
Reasoning: The last action (“teleport to outside”) was intended to move the agent to a location where animals are more likely to be found. The observation confirms the teleport succeeded instantly, with no delayed process or waiting required.  \nBelief State: The agent is now Outside. The hallway was checked and no animal was found there. The agent has not yet searched the outside area for animals.\nAction: look around

Your task is to complete the task goal: find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.\n\nBelow is the action history and the last turn's information:\nAction History: look around, teleport to outside, look around\nLast Turn's Information: \nReasoning: The last action (“teleport to outside”) was intended to move the agent to a location where animals are more likely to be found. The observation confirms the teleport succeeded instantly, with no delayed process or waiting required.  \nBelief State: The agent is now Outside. The hallway was checked and no animal was found there. The agent has not yet searched the outside area for animals.\nAction: look around\nObservation: This outside location is called the outside. Here you see: \n\tthe agent\n\ta substance called air\n\tan axe\n\ta blue jay egg\n\ta butterfly egg\n\ta dove egg\n\ta fire pit (containing nothing)\n\ta fountain (containing a substance called water)\n\tthe ground\n\ta substance called wood\nYou also see:\n\tA door to the foundry (that is open)\n\tA door to the greenhouse (that is open)\n\tA door to the kitchen (that is open)
Reasoning: The last action (“teleport to outside”) was expected to move the agent to the outside location to search for an animal. The observation confirms success: the agent is now outside, and the teleportation was instantaneous with no delayed process.\n\nBelief State: The agent is currently in Outside. The agent has checked the hallway and found no animal. In Outside, visible potential animals are eggs (blue jay egg, butterfly egg, dove egg) on the ground. Also present are a fountain (containing water), an empty fire pit, and tools/materials (axe, wood). No adult animals are currently observed.\nAction: focus on dove

Your task is to complete the task goal: find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.\n\nBelow is the action history and the last turn's information:\nAction History: look around, teleport to outside, look around, focus on dove\nLast Turn's Information: \nReasoning: The last action (“teleport to outside”) was expected to move the agent to the outside location to search for an animal. The observation confirms success: the agent is now outside, and the teleportation was instantaneous with no delayed process.\n\nBelief State: The agent is currently in Outside. The agent has checked the hallway and found no animal. In Outside, visible potential animals are eggs (blue jay egg, butterfly egg, dove egg) on the ground. Also present are a fountain (containing water), an empty fire pit, and tools/materials (axe, wood). No adult animals are currently observed.\nAction: focus on dove\nObservation: You focus on the dove egg.
Reasoning: The last action (“focus on dove”) was intended to select the dove egg as the target animal for the task. The observation confirms success: the dove egg is now focused, with no delayed process or waiting required.  \nBelief State: The agent is currently Outside, with the dove egg focused. Nearby are other potential animal eggs (blue jay egg, butterfly egg), an empty fire pit, a fountain (containing water), and tools/materials (axe, wood). The agent previously checked the hallway and found no animals there.\nAction: pick up dove

Your task is to complete the task goal: find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.\n\nBelow is the action history and the last turn's information:\nAction History: look around, teleport to outside, look around, focus on dove, pick up dove\nLast Turn's Information: \nReasoning: The last action (“focus on dove”) was intended to select the dove egg as the target animal for the task. The observation confirms success: the dove egg is now focused, with no delayed process or waiting required.  \nBelief State: The agent is currently Outside, with the dove egg focused. Nearby are other potential animal eggs (blue jay egg, butterfly egg), an empty fire pit, a fountain (containing water), and tools/materials (axe, wood). The agent previously checked the hallway and found no animals there.\nAction: pick up dove\nObservation: You move the dove to the inventory.
Reasoning: The last action (“pick up dove”) was intended to collect the focused animal (the dove egg) for transport to the red box in the kitchen. The observation confirms success: the dove egg is now in the agent’s inventory, with no delayed process or waiting required.\n\nBelief State: The agent is currently Outside, holding the dove egg in inventory. Nearby are other eggs (blue jay, butterfly) and objects (axe, fountain with water, empty fire pit, wood). The agent previously checked the hallway and found no animals. The next step is to teleport to the kitchen and locate the red box.\nAction: teleport to kitchen

Your task is to complete the task goal: find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.\n\nBelow is the action history and the last turn's information:\nAction History: look around, teleport to outside, look around, focus on dove, pick up dove, teleport to kitchen\nLast Turn's Information: \nReasoning: The last action (“pick up dove”) was intended to collect the focused animal (the dove egg) for transport to the red box in the kitchen. The observation confirms success: the dove egg is now in the agent’s inventory, with no delayed process or waiting required.\n\nBelief State: The agent is currently Outside, holding the dove egg in inventory. Nearby are other eggs (blue jay, butterfly) and objects (axe, fountain with water, empty fire pit, wood). The agent previously checked the hallway and found no animals. The next step is to teleport to the kitchen and locate the red box.\nAction: teleport to kitchen\nObservation: You teleport to the kitchen.
Reasoning: The last action (“teleport to kitchen”) was expected to instantly move the agent to the kitchen to proceed with placing the animal in the red box. The observation confirms success: the agent is now in the kitchen, with no delayed process.\n\nBelief State: The agent is in the Kitchen holding the dove egg (the focused animal). The red box (target location) is presumably present but not yet examined. Previously checked the Hallway (no animals found) and Outside (found dove, blue jay, and butterfly eggs, plus fountain with water and empty fire pit).\nAction: move egg dove egg in inventory to red box
    
Your task is to complete the task goal: {task_goal}

Below is the action history and the last turn's information:
Action History: {action_history}
Last Turn's Information: {last_turn_information}"""






# reflact
REFLACT_PROMPTING_METHOD_SCIENCEWORLD_TEMPLATE_BASELINE = """
You are a helpful assistant to do some scientific experiment in an environment.\nIn the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway\nYou should explore the environment and find the items you need to complete the experiment.\nYou can teleport to any room in one step.\nAll containers in the environment have already been opened, you can directly get items from the containers.

You should first reflect on the agent’s state, including the location, inventory, and focused object, in relation to the task goal. Then, output the action for this turn. Your output must strictly follow this format:"Reflection: your reflection.\n Action: your next action".

The available actions are(Your action should strictly follow this format):\n    open OBJ: open a container\n    close OBJ: close a container\n    activate OBJ: activate a device\n    deactivate OBJ: deactivate a device\n    connect OBJ to OBJ: connect electrical components\n    disconnect OBJ: disconnect electrical components\n    use OBJ [on OBJ]: use a device/item\n    look around: describe the current room\n    examine OBJ: describe an object in detail\n    look at OBJ: describe a container's contents\n    read OBJ: read a note or book\n    move OBJ to OBJ: move an object to a container\n    pick up OBJ: move an object to the inventory\n    pour OBJ into OBJ: pour a liquid into a container\n    mix OBJ: chemically mix a container\n    teleport to LOC: teleport to a specific room\n    focus on OBJ: signal intent on a task object\n    wait: task no action for 10 steps\n    wait1: task no action for a step"

Your response should use the following format:
Reflection: <your reflection>
Action: <your next action>

Below is an one-shot example:
Your task is to use chemistry to create green paint. When you are done, focus on the green paint.

Interaction History:

Reflection: To solve the task, I need to find materials to create green paint, use those materials, and focus on the object specified in the task once it is identified. The materials for making green paint are likely to be found in an art studio.
Action: look around
Observation: This room is called the bedroom. In it, you see:
the agent
substance called air
bed. On the bed is: a mattress. On the mattress is: a white pillow..
closet. The closet door is open. In the closet is: nothing.
picture
table. On the table is: nothing.
You also see:
A door to the hallway (that is open)

Reflection: Currently, I am in the bedroom, holding nothing and focused on nothing. There are no useful materials here to create the green paint I need to focus on.
Action: teleport to art studio
Observation: You teleport to the art studio.

Reflection: Currently, I am in the art studio, holding nothing and focused on nothing. I need to look for materials to create green paint.
Action: look around
Observation: This room is called the art studio. In it, you see:
the agent
substance called air
large cupboard. The large cupboard door is open. In the large cupboard is:
a drawer, a wood cup (containing yellow paint), a wood cup (containing blue
paint), a wood cup (containing red paint).
table. On the table is: a glass cup (containing nothing).
wood cup (containing yellow paint)
a wood cup (containing blue paint)
wood cup (containing red paint)
You also see:
A door to the hallway (that is open)

Reflection: Currently, I am in the art studio, not holding anything. I've found the primary colors yellow, blue, and red, which can be used to make green paint, the substance I will need to focus on.
Action: pour cup containing blue paint in art studio in cup containing nothing
Observation: You pour the contents of the wood cup into the glass cup.

Reflection: I poured blue paint into the glass cup. I'm now holding it and will next add yellow paint.
Action: pour cup containing yellow paint in art studio in cup containing blue paint in table
Observation: You pour the contents of the wood cup into the glass cup.

Reflection: I've added yellow paint to the blue in the glass cup. I'm now holding a mix of blue and yellow paint, ready to be mixed into green.
Action: mix cup containing blue paint and yellow paint

Below is the interaction history: {last_turn_information}"""


VAGEN_PROMPTING_METHOD_SCIENCEWORLD_TEMPLATE_BASELINE = """
You are a helpful assistant to do some scientific experiment in an environment.\nIn the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway\nYou should explore the environment and find the items you need to complete the experiment.\nYou can teleport to any room in one step.\nAll containers in the environment have already been opened, you can directly get items from the containers.

First, imagine and analyze the current environment state. Then, mentally simulate a possible action the agent could take, followed by imagining what the environment state would be after taking that action. Your output must strictly follow this format: "Thought: your reflection.\nAction: your next action".

The available actions are(Your action should strictly follow this format):\n    open OBJ: open a container\n    close OBJ: close a container\n    activate OBJ: activate a device\n    deactivate OBJ: deactivate a device\n    connect OBJ to OBJ: connect electrical components\n    disconnect OBJ: disconnect electrical components\n    use OBJ [on OBJ]: use a device/item\n    look around: describe the current room\n    examine OBJ: describe an object in detail\n    look at OBJ: describe a container's contents\n    read OBJ: read a note or book\n    move OBJ to OBJ: move an object to a container\n    pick up OBJ: move an object to the inventory\n    pour OBJ into OBJ: pour a liquid into a container\n    mix OBJ: chemically mix a container\n    teleport to LOC: teleport to a specific room\n    focus on OBJ: signal intent on a task object\n    wait: task no action for 10 steps\n    wait1: task no action for a step"

Your response should use the following format:
Thought: <your thought: what you think about the current environment state and the possible action, and the expected environment state after the action>
Action: <your next action>

Below is an one-shot example:

Your task is to find a(n) living thing. First, focus on the thing. Then, move it to the orange box in the workshop.

Interaction History:

Thought: I need to find a living thing. The task description doesn't specify what kind of living thing, so I should look around the current room to identify any potential living things.\nAction: look around
Observation: This room is called the living room. In it, you see: \n\tthe agent\n\ta substance called air\n\ta book shelf (containing A book (Beowulf) titled Beowulf by Beowulf poet, A book (Beowulf) titled Beowulf by Beowulf poet, A book (Pride and Prejudice) titled Pride and Prejudice by Jane Austen)\n\ta chair. On the chair is: nothing.\n\ta couch. On the couch is: a white pillow.\n\ta finger painting\n\ta table. On the table is: nothing.\nYou also see:\n\tA door to the hallway (that is open)

Thought: The living room doesn't seem to contain any living things, only inanimate objects and a substance called air. I should expand my search to other areas where living things are more likely to be found.\nAction: teleport to outside"
Observation: You teleport to the outside.

Thought: Now that I'm outside, I should look around to see if there are any living things here. The outside environment is more likely to have living organisms.\nAction: look around
Observation: This outside location is called the outside. Here you see: \n\tthe agent\n\ta substance called air\n\tan axe\n\ta dove egg\n\ta fire pit (containing nothing)\n\ta fountain (containing a substance called water)\n\ta giant tortoise egg\n\tthe ground\n\ta butterfly egg\n\ta shovel\n\ta substance called wood\nYou also see:\n\tA door to the foundry (that is open)\n\tA door to the greenhouse (that is open)\n\tA door to the kitchen (that is open)"

Thought: Among the items observed outside, there are several eggs, which are potential living things. I should focus on one of them to complete the task. The dove egg seems like a suitable choice since it's a clear example of a living thing in the form of an egg.\nAction: focus on dove
Observation: You focus on the dove egg.

Thought: Now that I've focused on the dove egg, I need to pick it up to move it to the specified location. I should be careful to handle it gently to avoid any damage.\nAction: pick up dove
Observation: You move the dove to the inventory.

Thought: The agent is currently in Outside. The environment contains several eggs (dove egg, giant tortoise egg, butterfly egg) on the ground, a fountain with water, an empty fire pit, and various tools (axe, shovel, wood). The dove egg has been focused on. The agent could pick up the dove egg. After picking it up, the dove egg would be in the agent's inventory, and the agent would still be in Outside, ready to transport the egg to the orange box in the workshop.\nAction: pick up dove

Below is the interaction history: {last_turn_information}"""
