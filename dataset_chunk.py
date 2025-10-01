import json
import numpy as np
import tiktoken
import os
from tqdm import tqdm


def preprocess_json_with_progress(
        input_path: str,
        output_dir: str,
        max_seq_len: int = 512,
        lines_per_file: int = 100000,
        max_lines: int = None,
        show_progress: bool = True
):
    """
    按行分块预处理JSON，增强进度条显示

    Args:
        input_path: JSON文件路径（必须每行一个{"text": "..."}）
        output_dir: 输出目录（自动创建）
        max_seq_len: 每个chunk的token长度（默认512）
        lines_per_file: 每个.npy文件对应的行数（默认10万行）
        max_lines: 最大处理行数（测试用）
        show_progress: 是否显示进度条
    """
    os.makedirs(output_dir, exist_ok=True)
    enc = tiktoken.get_encoding('gpt2')
    eot_token = enc.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]

    current_tokens = []
    all_chunks = []
    file_counter = 0
    line_counter = 0
    chunk_counter = 0  # 新增：统计总chunk数

    # 第一阶段：计算总行数（用于进度条）
    total_lines = 0
    if max_lines is not None:
        total_lines = max_lines
    else:
        if show_progress:
            with open(input_path, 'r', encoding='utf-8') as f:
                for _ in tqdm(f, desc="Counting lines"):
                    if max_lines and total_lines >= max_lines:
                        break
                    total_lines += 1

    # 第二阶段：处理数据
    with open(input_path, 'r', encoding='utf-8') as f:
        pbar = tqdm(
            f,
            total=total_lines,
            desc=f"Processing lines (File 0)",
            disable=not show_progress
        )

        for line in pbar:
            if max_lines and line_counter >= max_lines:
                break

            try:
                text = json.loads(line.strip())['text']
                tokens = enc.encode(text) + [eot_token]
                current_tokens.extend(tokens)

                # 分割chunks
                while len(current_tokens) >= max_seq_len + 1:
                    chunk = current_tokens[:max_seq_len + 1]
                    all_chunks.append(chunk)
                    current_tokens = current_tokens[max_seq_len + 1:]
                    chunk_counter += 1

                line_counter += 1
                pbar.set_description(
                    f"Processing lines (File {file_counter}) | "
                    f"Lines: {line_counter}/{total_lines} | "
                    f"Chunks: {chunk_counter}"
                )

                # 每处理够lines_per_file行，保存一个文件
                if line_counter % lines_per_file == 0:
                    np.save(
                        os.path.join(output_dir, f'2e5lines_{file_counter}.npy'),
                        np.array(all_chunks, dtype=np.int32).copy()  # 确保可写
                    )
                    file_counter += 1
                    all_chunks = []
                    pbar.set_description(f"Processing lines (File {file_counter})")

            except json.JSONDecodeError:
                continue

        # 保存剩余数据
        if all_chunks:
            np.save(
                os.path.join(output_dir, f'2e5lines_{file_counter}.npy'),
                np.array(all_chunks, dtype=np.int32).copy()
            )


# 示例调用（显示进度条）(10000行约35000个chunk，35000*512个token)
preprocess_json_with_progress(
    input_path='dataset/mobvoi_seq_monkey_general_open_corpus.jsonl',
    output_dir="dataset/dataset_chunk_small",
    lines_per_file=10000,
    max_lines=1e6,
    show_progress=True
)