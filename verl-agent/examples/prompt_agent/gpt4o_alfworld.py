import os
import numpy as np
import time
import logging
from datetime import datetime
from collections import defaultdict
from agent_system.environments.env_manager import *
from openai import OpenAI
from tqdm import tqdm

from omegaconf import OmegaConf

# 创建一个config对象
config = OmegaConf.create({
    'env': {
        'alfworld': {
            'baseline': 'baseline',  # 或 'our_method'
        },
    }
})

def build_env(env_name, env_num=1):
    group_n = 1
    if env_name == "alfworld":
        # Test AlfWorldEnvironmentManager
        from agent_system.environments.env_package.alfworld import alfworld_projection
        from agent_system.environments.env_package.alfworld import build_alfworld_envs
        alf_config_path = os.path.join(os.path.dirname(__file__), '../../agent_system/environments/env_package/alfworld/configs/config_tw.yaml')
        env_kwargs = {
            'eval_dataset': "eval_in_distribution", # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        resources_per_worker = {"num_cpus": 0.1, "num_gpus": 0.0}
        envs = build_alfworld_envs(alf_config_path, seed=1, env_num=env_num, group_n=group_n, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        # env_manager = AlfWorldEnvironmentManager(envs, alfworld_projection, 'alfworld/AlfredThorEnv')
        env_manager = AlfWorldEnvironmentManager(envs, alfworld_projection, config)
    else:
        raise ValueError(f"Unsupported environment name: {env_name}")
    
    return env_manager

# class Agent:
#     def __init__(self, model_name="gpt-4o"):
#         self.model_name = model_name
#         self.client = OpenAI(
#             api_key=os.environ['OPENAI_API_KEY'],
#         )
        
#     def get_action_from_gpt(self, obs):
#         response = self.client.chat.completions.create(
#             model=self.model_name,
#             messages=[
#                 {
#                     "role": "user", 
#                     "content": obs
#                 }
#             ],
#             temperature=0.4,
#             n=1,
#             stop=None
#         )
#         action = response.choices[0].message.content.strip()
#         return action

class Agent:
    def __init__(self, model_name="deepseek-chat"):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.environ.get("API_KEY"),
            base_url="https://api.deepseek.com"
        )
        
    def get_action_from_gpt(self, obs, max_retries=3, wait_time=5):
        """
        从 GPT (深度寻求聊天) 获取动作，带有重试机制。
        Args:
            obs: 输入的观测字符串（prompt）
            max_retries: 最大重试次数，默认3
            wait_time: 两次失败重试的等待（单位秒）
        Returns:
            action: 模型返回的字符串动作
        """
        import time
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user", 
                            "content": obs
                        }
                    ],
                    temperature=0.0,
                    n=1,
                    stop=None
                )
                action = response.choices[0].message.content.strip()
                return action
            except Exception as e:
                print(f"\n[警告] 第 {attempt + 1}/{max_retries} 次尝试失败")
                print(f"错误类型: {type(e).__name__}")
                print(f"错误信息: {str(e)[:200]}...")  # 只显示前200字符
                if attempt < max_retries - 1:
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"已达到最大重试次数 ({max_retries})，放弃此次请求")
                    raise  # 抛给调用者



if __name__ == "__main__":

    # -------- logging ----------
    os.makedirs("logs/alfworld", exist_ok=True)
    log_fp = os.path.join(
        "logs/alfworld", f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_fp, encoding="utf-8"), logging.StreamHandler()],
    )

    # -------- Parameters ----------
    max_steps = 40
    env_num = 3
    test_times = 1
    env_name = "alfworld" 

    # Keywords for 6 subtasks
    TASKS = [
        "pick_and_place",
        "pick_two_obj_and_place",
        "look_at_obj_in_light",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_clean_then_place_in_recep",
    ]

    # -------- Environment and agent setup ----------
    env_manager = build_env(env_name, env_num)
    agent = Agent()

    # Accumulated statistics
    overall_success_rates = []         # Overall success per round
    task_success_history = defaultdict(list)  # Subtask success per round

    # ======================= Main Loop =======================
    for test_idx in tqdm(range(test_times), desc="Tests"):
        logging.info(f"\n========== Start test {test_idx} ==========")
        start_time = time.time()
        
        kwargs = {}
        obs, infos = env_manager.reset(kwargs)
        env_dones = [False] * env_num

        # Statistics for single round
        overall_success_this_round = np.zeros(env_num, dtype=bool)
        task_success_cnt = defaultdict(int)
        task_total_cnt = defaultdict(int)

        for step_idx in tqdm(range(max_steps), desc="Steps"):
            logging.info(f"Step {step_idx}; Dones ({np.array(env_dones).sum().item()}/{env_num}); SR {overall_success_this_round.mean().item()}")

            # --- Assemble actions ---
            actions = []
            for i in range(env_num):
                if env_dones[i]:
                    actions.append("None")
                else:
                    actions.append(agent.get_action_from_gpt(obs["text"][i]))
                    print('action: ', actions[-1])

            # --- Environment stepping ---
            obs, rewards, dones, infos = env_manager.step(actions)

            # --- Determine endings and successes ---
            for i in range(env_num):
                if env_dones[i]:
                    continue

                if dones[i]:
                    env_dones[i] = True
                    won = bool(infos[i].get("won", False))
                    overall_success_this_round[i] = won

                    # Parse task type
                    gamefile = infos[i].get("extra.gamefile", "")
                    matched = False
                    for task in TASKS:
                        if task in gamefile:
                            task_total_cnt[task] += 1
                            if won:
                                task_success_cnt[task] += 1
                            matched = True
                            break
                    if not matched:
                        # Unrecognized tasks are also counted in total
                        task_total_cnt["other"] += 1
                        if won:
                            task_success_cnt["other"] += 1

            if all(env_dones):
                logging.info("All environments finished early!")
                break

        # -------- Single round results --------
        round_success_rate = overall_success_this_round.mean()
        overall_success_rates.append(round_success_rate)

        logging.info(f"Test {test_idx} overall success: {round_success_rate:.4f}")

        for task in TASKS + ["other"]:
            if task_total_cnt.get(task, 0) > 0:
                rate = task_success_cnt[task] / task_total_cnt[task]
                task_success_history[task].append(rate)
                logging.info(
                    f"    {task:<35s}: {rate:.4f} "
                    f"({task_success_cnt[task]}/{task_total_cnt[task]})"
                )

        logging.info(
            f"Test {test_idx} time elapsed: {time.time() - start_time:.2f}s\n"
        )

    # ======================= Final Summary =======================
    logging.info("=============== Final Summary ===============")
    logging.info(
        f"Total tests: {test_times} | Envs / test: {env_num} | Total envs: {env_num * test_times}"
    )
    logging.info(
        f"Overall success avg ± std: "
        f"{np.mean(overall_success_rates):.4f} ± {np.std(overall_success_rates):.4f}"
    )

    for task in TASKS + ["other"]:
        if task_success_history.get(task):
            logging.info(
                f"{task:<35s}: "
                f"{np.mean(task_success_history[task]):.4f} ± "
                f"{np.std(task_success_history[task]):.4f}"
            )
