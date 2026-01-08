"""
4-bit量化模型测试
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import warnings
warnings.filterwarnings("ignore")

MODEL_PATH = r"GLM-Z1-9B"  # 修改为实际路径

print("=" * 60)
print("GLM-Z1-9B 4-bit量化模型测试")
print("=" * 60)

# 检查CUDA
print(f"\n环境检查:")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")

print("\n正在加载模型（4-bit量化）...")

try:
    # 4-bit量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16  # 改为float16
    )

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    # 加载模型 - 修改device_map设置
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},  # 直接指定GPU 0
        trust_remote_code=True,
        torch_dtype=torch.float16,  # 改为float16
        low_cpu_mem_usage=True
    )

    print("模型加载成功！")

    # 显示内存使用
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"\n显存使用:")
    print(f"  已分配: {allocated:.2f}GB")
    print(f"  已预留: {reserved:.2f}GB")
    print(f"  剩余: {8 - reserved:.2f}GB")

except Exception as e:
    print(f"\n模型加载失败: {e}")
    print("\n可能的解决方案:")
    print("1. 确保bitsandbytes版本 >= 0.43.2")
    print("2. 尝试重新安装: pip install bitsandbytes==0.43.3")
    print("3. 检查CUDA和PyTorch版本兼容性")
    import sys
    sys.exit(1)

# 测试生成功能
print("\n" + "=" * 60)
print("测试文本生成")
print("=" * 60)

test_cases = [
    "人工智能的发展",
    "Python是一种",
    "今天天气",
]

for i, prompt in enumerate(test_cases, 1):
    print(f"\n测试 {i}: {prompt}")
    print("-" * 40)

    try:
        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt")

        # 将输入移到GPU
        inputs = {k: v.cuda() for k, v in inputs.items()}

        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # 解码输出
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"生成结果: {generated_text}")

    except Exception as e:
        print(f"生成失败: {e}")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)