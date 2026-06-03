# 基于vllm推理重写
import os
import json
from tqdm import tqdm

import sys
sys.path.append("/code/EUV/other_env")
import eval_agent.tasks as tasks
import eval_agent.envs as envs
from eval_agent.utils.datatypes import State

from fastchat.model.model_adapter import get_conversation_template

from vllm import LLM, SamplingParams

from typing import List, Dict, Any, Mapping

class VLLMAgent:
    """
    Agent that uses vllm for fast inference and follows the FastChat conversation template.
    """

    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        self.stop_words = ["\nObservation:", "\nTask:", "\n---"]

        self.model_path = config["model_name"]
        self.temperature = config.get("temperature", 0.8)
        self.max_new_tokens = config.get("max_new_tokens", 512)
        self.top_p = config.get("top_p", 1)
        self.do_sample = config.get("do_sample", False)
        self.device = config.get("device", "cuda:0")  # vllm can auto-detect device

        # 加载vllm LLM实例（一般显存充足时建议在脚本开头仅实例化一次）
        self.llm = LLM(
            model=self.model_path,
            trust_remote_code=True,
            dtype="auto"
        )

    def _make_prompt(self, messages: List[dict]) -> str:
        conv = get_conversation_template(self.model_path)
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "user":
                conv.append_message(conv.roles[0], content)
            elif role == "assistant":
                conv.append_message(conv.roles[1], content)
            else:
                raise ValueError(f"invalid role: {role}")
        conv.append_message(conv.roles[1], None)  # assistant to reply
        return conv.get_prompt(), conv

    def __call__(self, messages: List[dict]) -> str:
        prompt, conv = self._make_prompt(messages)
        stop_tokens = set(self.stop_words)
        if isinstance(conv.stop_str, str):
            stop_tokens.add(conv.stop_str)
        elif conv.stop_str is not None:
            stop_tokens.update(conv.stop_str)
        # 配置vllm采样参数
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
            stop=list(stop_tokens),
            skip_special_tokens=True,
            do_sample=self.do_sample
        )
        outputs = self.llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text.strip()
        # 提取回答（针对分隔符、stop标志进一步裁剪）
        for stop in stop_tokens:
            if stop in response:
                response = response[:response.index(stop)]
        # 进一步规范输出
        return response.strip()

# ---- 配置部分 ----
with open("/code/EUV/other_env/eval_agent/configs/task/alfworld.json") as f:
    exp_config = json.load(f)

with open("/code/EUV/other_env/eval_agent/configs/model/fastchat.json") as f:
    agent_config_dict = json.load(f)

# 强制覆盖模型路径，如果你想用config文件里配置的可以注释掉下面几行
agent_config_dict["model_name"] = "/code/EUV/checkpoints/global_step_33"
agent_config_dict["temperature"] = 0.8
agent_config_dict["max_new_tokens"] = 512
agent_config_dict["top_p"] = 1
agent_config_dict["do_sample"] = False

# 初始化vllm agent
agent = VLLMAgent(agent_config_dict)

env_config = exp_config["env_config"]
task_config = exp_config["task"]
task_class = getattr(tasks, task_config["task_class"])

output_path = '/code/EUV/evaluation/verl_alfworld_unseen_llama_3b_sft_another_inference2'
split = 'test'
part_num = 1
part_idx = 0
all_tasks, n_tasks = task_class.load_tasks(split, part_num, part_idx)

# 只运行未完成的任务
state_list = []
done_task_id = []
if os.path.exists(output_path):
    for file in os.listdir(output_path):
        if not file.endswith('json'):
            continue
        done_task_id.append(file.split('.')[0])

def interactive_loop(task, agent, env_config):
    env = getattr(envs, env_config["env_class"])(task, **env_config)
    observation, state = env.reset()

    cur_step = 1
    while not state.finished:
        cur_step += 1
        try:
            llm_output: str = agent(state.history)

            number_n = llm_output.count('\n')
            if number_n >= 2:
                first_n = llm_output.find('\n')
                second_n = llm_output.find('\n', first_n + 1)
                llm_output = llm_output[:second_n]

        except Exception as e:
            state.success = False
            state.finished = True
            state.terminate_reason = f"vllm error: {e}"
            break

        observation, state = env.step(llm_output)

    if state.reward is not None:
        print(
            f"Task finished in {state.steps} steps. Success: {state.success}. Reward: {state.reward}"
        )
    else:
        print(
            f"Task finished in {state.steps} steps. Success: {state.success}"
        )

    return state

os.makedirs(output_path, exist_ok=True)

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