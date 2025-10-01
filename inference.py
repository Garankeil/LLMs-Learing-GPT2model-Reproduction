import tiktoken
import torch
from model.Transformers import GPT, GPTconfig


def question2tensor(question, config_q):
    enco = tiktoken.get_encoding("gpt2")
    encoded_text = enco.encode(question)
    full_encoded = encoded_text + [config_q.eos_token]

    if len(full_encoded) >= config.max_seq_len:
        raise ValueError(f"Input length {len(full_encoded)} exceeds max_length {config.max_seq_len}")

    # 填充至固定长度 max_length
    padded = full_encoded + [config.eos_token] * (config.max_seq_len - len(full_encoded))

    # 转为 Tensor（增加 batch 维度）
    return torch.tensor([padded], dtype=torch.long)  # 形状: (1, max_length)


enc = tiktoken.get_encoding("gpt2")
config = GPTconfig()

model = GPT(config)  # 你的模型定义
# # 检测可用GPU数量
if torch.cuda.device_count() >= 2:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)  # 包装为DataParallel

# 将模型放到设备（默认会复制到所有GPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 验证设备
print(f"Model is on: {next(model.parameters()).device}")
print(f"All devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")

# 加载 checkpoint
checkpoint_path = '/home/jnu/jiananfu/project/GPT2/model_save/model_epoch_1.pt'  # 替换为你要加载的 checkpoint 路径
checkpoint = torch.load(checkpoint_path)

# 恢复模型的状态
model.load_state_dict(checkpoint['model_state_dict'])
model = model.module
# 将模型设置为评估模式（如果你只是进行推理）
model.eval()

while True:
    ques = input("请输入问题(输入quit退出)：")
    if ques == "quit":
        print("GPT-2 期待着再次见到你！")
        break
    x = question2tensor(ques, config).to("cuda")
    out = model.generate(x)
    output = enc.decode(out.detach().cpu().numpy()[0])
    print("GPT-2:", output)
