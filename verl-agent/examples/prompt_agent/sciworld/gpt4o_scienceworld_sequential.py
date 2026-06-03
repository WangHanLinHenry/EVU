import os
import numpy as np
import time
import logging
from datetime import datetime
from collections import defaultdict
import json
from agent_system.environments.env_manager import *
from openai import OpenAI
from tqdm import tqdm

from omegaconf import OmegaConf

# 从环境变量读取配置，如果没有设置则使用默认值
PROMPTING_METHOD = os.getenv('PROMPTING_METHOD', 'baseline')  # baseline 或 our_method

# 创建一个config对象
config = OmegaConf.create({
    'env': {
        'scienceworld': {
            'baseline': PROMPTING_METHOD,
        },
        'model_name': 'qwen',  # 模型名称，用于决定 prompt 格式 (llama, qwen 等)
    }
})

def compute_reward(info):
    """计算奖励"""
    reward = 10.0 * float(info['won'])
    return reward

class DirectScienceWorldEnv:
    """
    直接使用 ScienceWorld 环境，不使用 Ray
    """
    def __init__(self, seed, env_kwargs={}, task_data_list=None):
        from scienceworld import ScienceWorldEnv
        
        self.env_kwargs = env_kwargs
        self.seed = seed
        self.task_data_list = task_data_list if task_data_list is not None else []
        self.current_task_idx = 0
        self.max_steps_dict = None
        self.taskname2id = None
        self.current_task = None
        self.current_variation_idx = None
        self.steps = 0
        self.max_steps = 200
        self.last_step_score = 0
        
        # Load taskname2id mapping
        taskname2id_path = env_kwargs.get('taskname2id_path', None)
        if taskname2id_path and os.path.exists(taskname2id_path):
            with open(taskname2id_path, 'r') as f:
                self.taskname2id = json.load(f)
        
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
    
    def reset(self):
        """重置环境"""
        if len(self.task_data_list) == 0:
            return ["No task available."], [{"won": False, "task": "unknown"}]
        
        # 获取当前任务数据
        if self.current_task_idx >= len(self.task_data_list):
            self.current_task_idx = 0
        
        task_data = self.task_data_list[self.current_task_idx]
        self.current_task_idx += 1
        
        if task_data is None:
            return ["No task available."], [{"won": False, "task": "unknown"}]
        
        # 如果 task_data 是列表格式 [task_name, variation_idx]，转换为字典
        if isinstance(task_data, list) and len(task_data) >= 2:
            task_data = {
                'sub_task_name': task_data[0],
                'variation_idx': task_data[1]
            }
        
        # 确保 task_data 是字典
        if not isinstance(task_data, dict):
            logging.warning(f"Invalid task_data format: {type(task_data)}, expected dict or list")
            return ["Invalid task data format."], [{"won": False, "task": "unknown"}]
        
        self.current_task = task_data.get('sub_task_name')
        self.current_variation_idx = task_data.get('variation_idx', 0)
        self.steps = 0
        self.last_step_score = 0
        
        # Set max steps based on task
        # Try to find max_steps using task name first, then try task id if taskname2id is available
        if self.max_steps_dict and self.current_task:
            # First try using task name directly
            self.max_steps = self.max_steps_dict.get(self.current_task, None)
            
            # If not found and taskname2id is available, try using task id
            if self.max_steps is None and self.taskname2id:
                task_id = self.taskname2id.get(self.current_task, None)
                if task_id is not None:
                    # Try to find max_steps using task id as key (try both string and int)
                    self.max_steps = self.max_steps_dict.get(str(task_id), None)
                    if self.max_steps is None:
                        self.max_steps = self.max_steps_dict.get(int(task_id), None)
            
            # If still not found, use default
            if self.max_steps is None:
                self.max_steps = 200
                logging.warning(f"Max steps not found for task '{self.current_task}', using default: 200")
            else:
                logging.info(f"Max steps for task '{self.current_task}': {self.max_steps}")
            
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
            
            return [obs], [info]
        except Exception as e:
            return [f"Error loading task: {str(e)}"], [{"won": False, "task": self.current_task}]
    
    def step(self, actions):
        """执行一步
        Args:
            actions: 动作列表，对于单个环境，应该是包含一个动作的列表
        """
        # 确保 actions 是列表
        if not isinstance(actions, list):
            actions = [actions]
        
        # 对于单个环境，取第一个动作
        action = actions[0] if len(actions) > 0 else "pass"
        
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
            
            return [observation], [final_reward], [done], [info]
            
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
            return [observation], [reward], [done], [info]

def load_task_data_list(data_path):
    """加载任务数据列表"""
    if not data_path or not os.path.exists(data_path):
        return []
    
    task_data_list = []
    try:
        with open(data_path, 'r') as f:
            if data_path.endswith('.jsonl'):
                for line in f:
                    if line.strip():
                        task_data = json.loads(line)
                        # 如果是列表格式 [task_name, variation_idx]，转换为字典
                        if isinstance(task_data, list) and len(task_data) >= 2:
                            task_data = {
                                'sub_task_name': task_data[0],
                                'variation_idx': task_data[1]
                            }
                        task_data_list.append(task_data)
            elif data_path.endswith('.json'):
                raw_data = json.load(f)
                # 如果整个文件是一个列表
                if isinstance(raw_data, list):
                    for item in raw_data:
                        # 如果列表项是列表格式 [task_name, variation_idx]，转换为字典
                        if isinstance(item, list) and len(item) >= 2:
                            task_data = {
                                'sub_task_name': item[0],
                                'variation_idx': item[1]
                            }
                        # 如果已经是字典格式，直接使用
                        elif isinstance(item, dict):
                            task_data = item
                        else:
                            continue
                        task_data_list.append(task_data)
                # 如果整个文件是一个字典，包装成列表
                elif isinstance(raw_data, dict):
                    task_data_list.append(raw_data)
    except Exception as e:
        logging.warning(f"Failed to load task data from {data_path}: {e}")
    
    return task_data_list

def build_env(env_name, env_num=1):
    """不使用 Ray 的环境构建函数"""
    if env_name == "scienceworld":
        from agent_system.environments.env_package.scienceworld import scienceworld_projection
        
        # 从环境变量读取任务数据路径
        task_data_path = os.getenv('TASK_DATA_PATH', None)
        task_data_list = []
        if task_data_path:
            task_data_list = load_task_data_list(task_data_path)
            logging.info(f"Loaded {len(task_data_list)} tasks from {task_data_path}")
        
        # 从环境变量读取其他配置
        env_kwargs = {
            'server_path': os.getenv('SERVER_PATH', None),
            'max_steps_path': os.getenv('MAX_STEPS_PATH', None),
            'taskname2id_path': os.getenv('TASKNAME2ID_PATH', None),
            'apply_monkey_patch': True,
        }
        
        # 直接创建环境，不使用 Ray
        direct_env = DirectScienceWorldEnv(seed=1, env_kwargs=env_kwargs, task_data_list=task_data_list)
        
        # 创建环境管理器
        env_manager = ScienceWorldEnvironmentManager(direct_env, scienceworld_projection, config)
        return env_manager
    else:
        raise ValueError(f"Unsupported environment name: {env_name}")

class Agent:
    def __init__(self, model_name="deepseek-chat"):
        self.model_name = model_name
        # 从环境变量读取 API key，如果没有设置则使用默认值
        api_key = os.environ.get("API_KEY")
        self.client = OpenAI(
            api_key=api_key,
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
    os.makedirs("logs/scienceworld", exist_ok=True)
    log_fp = os.path.join(
        "logs/scienceworld", f"run_log_sequential_no_ray_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_fp, encoding="utf-8"), logging.StreamHandler()],
    )

    # -------- Parameters ----------
    max_steps = 200
    total_test_cases = int(os.getenv('TOTAL_TEST_CASES', '100'))  # 总共要测试的案例数量
    env_name = "scienceworld" 

    # -------- Environment and agent setup (不使用 Ray) ----------
    logging.info("开始创建环境（不使用 Ray）...")
    try:
        env_manager = build_env(env_name, env_num=1)
        logging.info("环境创建完成")
    except Exception as e:
        logging.error(f"环境创建失败: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise
    
    agent = Agent()
    logging.info("Agent 初始化完成")

    # Accumulated statistics
    overall_success_list = []         # 每个案例的成功情况
    task_success_cnt = defaultdict(int)  # 各任务类型的成功计数
    task_total_cnt = defaultdict(int)    # 各任务类型的总数

    # ======================= Main Loop =======================
    # 串行测试：一个案例一个案例地测试
    for case_idx in tqdm(range(total_test_cases), desc="测试案例"):
        logging.info(f"\n========== 开始测试案例 {case_idx + 1}/{total_test_cases} ==========")
        start_time = time.time()
        
        # 重置环境，获取新的测试案例
        kwargs = {}
        obs, infos = env_manager.reset(kwargs)
        env_done = False
        success = False
        
        # 从环境中获取实际的 max_steps（每个任务可能不同）
        current_max_steps = max_steps  # 默认值
        if hasattr(env_manager, 'envs') and hasattr(env_manager.envs, 'max_steps'):
            current_max_steps = env_manager.envs.max_steps
            if current_max_steps != max_steps:
                logging.info(f"当前任务的最大步数: {current_max_steps} (不同于默认值 {max_steps})")

        # 单个案例的测试循环
        for step_idx in tqdm(range(current_max_steps), desc=f"案例 {case_idx + 1} 步骤", leave=False):
            if env_done:
                break
                
            logging.info(f"案例 {case_idx + 1} - 步骤 {step_idx + 1}/{current_max_steps}")

            # --- 获取动作 ---
            # 因为只有一个环境，直接使用 obs["text"][0]
            action = agent.get_action_from_gpt(obs["text"][0])
            print('观测：', obs["text"][0][:200] + '...' if len(obs["text"][0]) > 200 else obs["text"][0])
            print('动作：', action)

            # --- 环境步进 ---
            obs, rewards, dones, infos = env_manager.step([action])
            
            # 调试：检查动作映射结果
            if len(infos) > 0:
                action_valid = infos[0].get('is_action_valid', None)
                logging.info(f"案例 {case_idx + 1} - 步骤 {step_idx + 1} - 动作有效性: {action_valid}")
                if action_valid == 0 or action_valid is False:
                    logging.warning(f"⚠️ 案例 {case_idx + 1} - 步骤 {step_idx + 1} - 动作被标记为无效！原始动作: {action}")
                else:
                    logging.info(f"✓ 案例 {case_idx + 1} - 步骤 {step_idx + 1} - 动作有效")

            # --- 检查是否完成 ---
            if dones[0]:
                env_done = True
                success = bool(infos[0].get("won", False))
                
                # 解析任务类型
                task_name = infos[0].get("task", "unknown")
                task_total_cnt[task_name] += 1
                if success:
                    task_success_cnt[task_name] += 1
                
                logging.info(f"任务类型: {task_name}")
                logging.info(f"案例 {case_idx + 1} 完成 - 成功: {success}")

        # 如果达到最大步数仍未完成
        if not env_done:
            logging.info(f"案例 {case_idx + 1} 达到最大步数 ({current_max_steps}) 未完成")
            success = False
            # 尝试从infos中获取任务类型
            task_name = infos[0].get("task", "unknown")
            task_total_cnt[task_name] += 1

        # 记录结果
        overall_success_list.append(success)
        elapsed_time = time.time() - start_time
        logging.info(f"案例 {case_idx + 1} 耗时: {elapsed_time:.2f}秒, 成功: {success}")

    # ======================= Final Summary =======================
    logging.info("\n=============== 最终统计 ===============")
    total_cases = len(overall_success_list)
    success_count = sum(overall_success_list)
    success_rate = success_count / total_cases if total_cases > 0 else 0.0
    
    logging.info(f"总测试案例数: {total_cases}")
    logging.info(f"成功案例数: {success_count}")
    logging.info(f"总体成功率: {success_rate:.4f} ({success_count}/{total_cases})")
    logging.info("\n各任务类型统计:")
    
    for task in sorted(task_total_cnt.keys()):
        total = task_total_cnt.get(task, 0)
        success = task_success_cnt.get(task, 0)
        if total > 0:
            rate = success / total
            logging.info(
                f"  {task:<35s}: {rate:.4f} ({success}/{total})"
            )

    # ======================= 将结果写入txt文件 =======================
    # 从环境变量读取保存文件路径，如果没有设置则使用默认路径（目录+时间戳文件名）
    result_file = os.getenv('RESULT_FILE', None)
    if result_file:
        # 如果指定了完整文件路径，直接使用
        result_fp = result_file
        # 确保文件所在目录存在
        result_dir = os.path.dirname(result_fp)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
    else:
        # 如果没有指定，使用默认目录+时间戳文件名
        result_dir = os.getenv('RESULT_DIR', '/code/EUV/outputs/results/scienceworld')
        os.makedirs(result_dir, exist_ok=True)
        result_fp = os.path.join(
            result_dir, f"result_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
    
    with open(result_fp, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("ScienceWorld 测试结果汇总\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总测试案例数: {total_cases}\n")
        f.write(f"成功案例数: {success_count}\n")
        f.write(f"总体成功率: {success_rate:.4f} ({success_count}/{total_cases})\n\n")
        
        f.write("各任务类型统计:\n")
        f.write("-" * 50 + "\n")
        for task in sorted(task_total_cnt.keys()):
            total = task_total_cnt.get(task, 0)
            success = task_success_cnt.get(task, 0)
            if total > 0:
                rate = success / total
                f.write(f"{task:<35s}: {rate:.4f} ({success}/{total})\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("详细案例结果:\n")
        f.write("-" * 50 + "\n")
        for idx, success in enumerate(overall_success_list):
            f.write(f"案例 {idx + 1}: {'成功' if success else '失败'}\n")
    
    logging.info(f"\n结果已保存到文件: {result_fp}")

