import os
from vllm import LLM, SamplingParams

# 设置使用0号GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 设置模型路径
model_path = "/code/models/Qwen2.5-3B-Instruct"

# 初始化 LLM
llm = LLM(model=model_path)

# 生成示例 prompt
prompt = "hello, who are you?"

# 采样参数
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=128)

# 推理
outputs = llm.generate([prompt], sampling_params)

# 打印输出
for output in outputs:
    print("Prompt:", output.prompt)
    print("Generated:", output.outputs[0].text)
