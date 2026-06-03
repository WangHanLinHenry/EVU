import os
import numpy as np
import time
import logging
from datetime import datetime
from collections import defaultdict
import yaml
from agent_system.environments.env_manager import *
from openai import OpenAI
from tqdm import tqdm

from omegaconf import OmegaConf

# 从环境变量读取配置，如果没有设置则使用默认值
PROMPTING_METHOD = os.getenv('PROMPTING_METHOD', 'prompting_baseline')  # prompting_baseline 或 prompting_our_method

# 创建一个config对象
config = OmegaConf.create({
    'env': {
        'alfworld': {
            'baseline': PROMPTING_METHOD,
        },
        'model_name': 'qwen',  # 模型名称，用于决定 prompt 格式 (llama, qwen 等)
    }
})

def load_config_file(path):
    """加载配置文件"""
    assert os.path.exists(path), f"Invalid config file: {path}"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def compute_reward(info, multi_modal=False):
    """计算奖励"""
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

class DirectAlfworldEnv:
    """
    直接使用 AlfWorld 环境，不使用 Ray
    """
    def __init__(self, alf_config_path, seed, env_kwargs={}):
        from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment
        
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        
        base_env = get_environment(env_type)(config, train_eval=eval_dataset)
        self.env = base_env.init_env(batch_size=1)
        self.env.seed(seed)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.prev_admissible_commands = None
    
    def _extract_value(self, value, default):
        """辅助函数：从元组或列表中提取值"""
        if isinstance(value, tuple):
            return value[0] if len(value) > 0 else default
        elif isinstance(value, list):
            return value[0] if len(value) > 0 else default
        return value if value is not None else default
        
    def reset(self):
        """重置环境"""
        result = self.env.reset()
        
        # 处理返回格式：可能是 (obs, infos) 元组
        if isinstance(result, tuple) and len(result) == 2:
            obs, infos = result
        else:
            # 如果格式不对，尝试其他方式
            obs = result if not isinstance(result, (tuple, list)) else (result[0] if len(result) > 0 else "")
            infos = {}
        
        # 确保 obs 是字符串
        obs = self._extract_value(obs, "")
        if not isinstance(obs, str):
            obs = str(obs) if obs is not None else ""
        
        # 确保 infos 是字典
        infos = self._extract_value(infos, {})
        if not isinstance(infos, dict):
            infos = {}
        
        # 保存 admissible_commands
        self.prev_admissible_commands = infos.get('admissible_commands', [])
        if isinstance(self.prev_admissible_commands, list) and len(self.prev_admissible_commands) > 0:
            if isinstance(self.prev_admissible_commands[0], list):
                self.prev_admissible_commands = self.prev_admissible_commands[0]
        
        # 返回格式：text_obs_list, image_obs_list, info_list
        # text_obs_list 必须是字符串列表
        return [obs], None, [infos]
    
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
        
        # textworld 的 step 方法期望接收列表（batch_size=1）
        result = self.env.step([action])
        
        # 处理返回格式：可能是 (obs, scores, dones, infos) 元组
        if isinstance(result, tuple) and len(result) >= 4:
            obs, scores, dones, infos = result[0], result[1], result[2], result[3]
        elif isinstance(result, tuple) and len(result) >= 3:
            obs, scores, dones = result[0], result[1], result[2]
            infos = {}
        else:
            obs, scores, dones, infos = "", 0.0, False, {}
        
        # 确保 obs 是字符串
        obs = self._extract_value(obs, "")
        if not isinstance(obs, str):
            obs = str(obs) if obs is not None else ""
        
        # 确保 scores 是数字
        scores = self._extract_value(scores, 0.0)
        if not isinstance(scores, (int, float)):
            scores = 0.0
        
        # 确保 dones 是布尔值
        dones = self._extract_value(dones, False)
        if not isinstance(dones, bool):
            dones = bool(dones) if dones is not None else False
        
        # 确保 infos 是字典
        infos = self._extract_value(infos, {})
        if not isinstance(infos, dict):
            infos = {}
        
        # 处理 infos 中的值：如果值是列表，取第一个元素
        processed_infos = {}
        for key, value in infos.items():
            if isinstance(value, list):
                if len(value) > 0:
                    processed_infos[key] = value[0]
                else:
                    processed_infos[key] = None
            elif isinstance(value, tuple):
                if len(value) > 0:
                    processed_infos[key] = value[0]
                else:
                    processed_infos[key] = None
            else:
                processed_infos[key] = value
        infos = processed_infos
            
        # 保存 admissible_commands
        self.prev_admissible_commands = infos.get('admissible_commands', [])
        if isinstance(self.prev_admissible_commands, list) and len(self.prev_admissible_commands) > 0:
            if isinstance(self.prev_admissible_commands[0], list):
                self.prev_admissible_commands = self.prev_admissible_commands[0]
        
        reward = compute_reward(infos, self.multi_modal)
        
        return [obs], None, [reward], [dones], [infos]
    
    @property
    def get_admissible_commands(self):
        """获取可执行命令（作为属性，可以直接访问）"""
        if self.prev_admissible_commands is None:
            return []
        # 返回格式应该是一个列表，每个元素是一个环境的命令列表
        # 由于我们只有一个环境，返回包含一个列表的列表
        if isinstance(self.prev_admissible_commands, list):
            # 如果已经是列表，确保返回格式正确
            if len(self.prev_admissible_commands) > 0 and isinstance(self.prev_admissible_commands[0], list):
                # 如果已经是嵌套列表，直接返回
                return self.prev_admissible_commands
            else:
                # 如果是单层列表，包装成嵌套列表
                return [self.prev_admissible_commands]
        else:
            return [[self.prev_admissible_commands]]

def build_env(env_name, env_num=1):
    """不使用 Ray 的环境构建函数"""
    if env_name == "alfworld":
        from agent_system.environments.env_package.alfworld import alfworld_projection
        
        alf_config_path = os.path.join(
            os.path.dirname(__file__), 
            '/code/EUV/verl-agent/agent_system/environments/env_package/alfworld/configs/config_tw.yaml'
        )
        # 从环境变量读取 eval_dataset，默认为 eval_in_distribution
        eval_dataset = os.getenv('EVAL_DATASET', 'eval_in_distribution')
        env_kwargs = {
            'eval_dataset': eval_dataset,
        }
        
        # 直接创建环境，不使用 Ray
        direct_env = DirectAlfworldEnv(alf_config_path, seed=1, env_kwargs=env_kwargs)
        
        # 创建环境管理器
        env_manager = AlfWorldEnvironmentManager(direct_env, alfworld_projection, config)
        return env_manager
    else:
        raise ValueError(f"Unsupported environment name: {env_name}")

class Agent:
    # def __init__(self, model_name="deepseek-chat"):
    def __init__(self, model_name="qwen3-max"):
        self.model_name = model_name
        # 从环境变量读取 API key，如果没有设置则使用默认值
        api_key = os.environ.get("API_KEY")
        self.client = OpenAI(
            api_key=api_key,
            # base_url="https://api.deepseek.com"
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
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
        "logs/alfworld", f"run_log_sequential_no_ray_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_fp, encoding="utf-8"), logging.StreamHandler()],
    )

    # -------- Parameters ----------
    max_steps = 40
    total_test_cases = 50  # 总共要测试的案例数量
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
        # INSERT_YOUR_CODE
        # 显示当前已测试案例数和成功数
        logging.info(f"当前已成功案例数: {sum(overall_success_list)} / 已测试案例数: {case_idx}")

        start_time = time.time()
        
        # 重置环境，获取新的测试案例
        kwargs = {}
        obs, infos = env_manager.reset(kwargs)
        env_done = False
        success = False

        # 单个案例的测试循环
        for step_idx in tqdm(range(max_steps), desc=f"案例 {case_idx + 1} 步骤", leave=False):
            if env_done:
                break
                
            logging.info(f"案例 {case_idx + 1} - 步骤 {step_idx + 1}/{max_steps}")

            # --- 获取动作 ---
            # 因为只有一个环境，直接使用 obs["text"][0]
            action = agent.get_action_from_gpt(obs["text"][0])
            print('观测：', obs["text"][0])
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
                gamefile = infos[0].get("extra.gamefile", "")
                matched = False
                for task in TASKS:
                    if task in gamefile:
                        task_total_cnt[task] += 1
                        if success:
                            task_success_cnt[task] += 1
                        matched = True
                        logging.info(f"任务类型: {task}")
                        break
                if not matched:
                    task_total_cnt["other"] += 1
                    if success:
                        task_success_cnt["other"] += 1
                    logging.info(f"任务类型: other (未识别)")
                
                logging.info(f"案例 {case_idx + 1} 完成 - 成功: {success}")

        # 如果达到最大步数仍未完成
        if not env_done:
            logging.info(f"案例 {case_idx + 1} 达到最大步数 ({max_steps}) 未完成")
            success = False
            # 尝试从infos中获取任务类型
            gamefile = infos[0].get("extra.gamefile", "")
            matched = False
            for task in TASKS:
                if task in gamefile:
                    task_total_cnt[task] += 1
                    matched = True
                    break
            if not matched:
                task_total_cnt["other"] += 1

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
    
    for task in TASKS + ["other"]:
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
        result_dir = os.getenv('RESULT_DIR', '/code/EUV/outputs/results/alfworld')
        os.makedirs(result_dir, exist_ok=True)
        result_fp = os.path.join(
            result_dir, f"result_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
    
    with open(result_fp, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("AlfWorld 测试结果汇总\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总测试案例数: {total_cases}\n")
        f.write(f"成功案例数: {success_count}\n")
        f.write(f"总体成功率: {success_rate:.4f} ({success_count}/{total_cases})\n\n")
        
        f.write("各任务类型统计:\n")
        f.write("-" * 50 + "\n")
        for task in TASKS + ["other"]:
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