# import json
# import numpy as np
# from transformers import AutoTokenizer

# # ================= 配置区域 =================
# # 1. 替换为你的 jsonl 文件路径
# file_path = '/disk0/wanzhenjie/lt-eval/results_prompt_token_cnt/__disk0__wanzhenjie__LLMs__Qwen2.5-7B-Instruct/samples_gpqa_diamond_generative_n_shot_2026-01-30T02-00-05.423363.jsonl' 

# # 2. 替换为你使用的模型名称或本地路径
# # 如果是为了估算 Llama/Qwen 等模型的 token，建议使用对应的 tokenizer
# # 这里的 "gpt2" 只是一个通用的示例
# model_name = "/disk0/wanzhenjie/LLMs/Qwen2.5-7B-Instruct" 
# # ===========================================

# def calculate_template_tokens(file_path, model_name):
#     print(f"Loading tokenizer: {model_name}...")
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#     except Exception as e:
#         print(f"Error loading tokenizer: {e}")
#         return

#     token_counts = []
    
#     print(f"Processing {file_path}...")
    
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line_num, line in enumerate(f):
#             try:
#                 data = json.loads(line)
                
#                 # 1. 获取完整的 Prompt (arg_0)
#                 # 注意：这里根据你的 json 结构路径提取
#                 full_prompt = data.get('arguments', {}).get('gen_args_0', {}).get('arg_0', "")
                
#                 # 2. 获取该样本特定的 Question 内容
#                 question_text = data.get('doc', {}).get('question', "")
#                 if not question_text:
#                     question_text = data.get('doc', {}).get('Pre-Revision Question', "")
#                 if not question_text:
#                     question_text = data.get('doc', {}).get('input', "")            
                
#                 if not full_prompt or not question_text:
#                     continue

#                 # 3. 核心步骤：从 Prompt 中移除具体的 Question 文本
#                 # 这样剩下的就是 Few-shot examples + System prompt + 格式字符
#                 # 使用 replace 将具体问题替换为空字符串
#                 template_text = full_prompt.replace(question_text, "")
                
#                 # 4. 计算 Token 数量
#                 tokens = tokenizer.encode(template_text, add_special_tokens=False)
#                 token_counts.append(len(tokens))

#             except json.JSONDecodeError:
#                 print(f"Skipping invalid JSON at line {line_num}")
#             except Exception as e:
#                 print(f"Error at line {line_num}: {e}")

#     if token_counts:
#         avg_tokens = np.mean(token_counts)
#         print(f"\n======== 统计结果 ========")
#         print(f"处理样本数: {len(token_counts)}")
#         print(f"平均 Template Token 数: {avg_tokens:.2f}")
#         print(f"最大 Template Token 数: {np.max(token_counts)}")
#         print(f"最小 Template Token 数: {np.min(token_counts)}")
#     else:
#         print("未找到有效数据。")

# # 运行函数
# if __name__ == "__main__":
#     calculate_template_tokens(file_path, model_name)


import os
import json
import re
import numpy as np
from transformers import AutoTokenizer
from collections import defaultdict
from prettytable import PrettyTable  # 用于打印漂亮的表格，如果没有安装可以用简单的print替代

# ================= 配置区域 =================
# 1. 你的数据目录路径 (假设当前目录下，如果是其他目录请修改)
DATA_DIR = "/disk0/wanzhenjie/lt-eval/results_prompt_token_cnt/__disk0__wanzhenjie__LLMs__Qwen2.5-7B-Instruct" 

# 2. 模型 Tokenizer (用于计算长度)
MODEL_NAME = "/disk0/wanzhenjie/LLMs/Qwen2.5-7B-Instruct" 
# ===========================================

def parse_file_info(filename):
    """
    根据文件名解析 Dataset 和 Subset。
    基于提供的文件列表编写的正则规则。
    """
    # 1. GSM8K
    if "gsm8k" in filename:
        return "GSM8K", "Main"
    
    # 2. GPQA
    if "gpqa" in filename:
        # 提取 'diamond' 或其他
        # pattern: samples_gpqa_diamond_generative...
        match = re.search(r"samples_gpqa_(.*?)_generative", filename)
        subset = match.group(1) if match else "Main"
        return "GPQA", subset

    # 3. BBH
    # pattern: samples_bbh_ltthinker_fewshot_[subset]_[timestamp].jsonl
    if "bbh" in filename:
        # 提取 fewshot_ 和 _2026 之间的内容
        match = re.search(r"fewshot_(.*?)_\d{4}-\d{2}", filename)
        subset = match.group(1) if match else "Unknown"
        return "BBH", subset

    # 4. MMLU
    # pattern: samples_mmlu_[subset]_generative_[timestamp].jsonl
    if "mmlu" in filename:
        # 提取 mmlu_ 和 _generative 之间的内容
        match = re.search(r"samples_mmlu_(.*?)_generative", filename)
        subset = match.group(1) if match else "Unknown"
        return "MMLU", subset

    return "Unknown", filename

def get_template_token_counts(file_path, tokenizer):
    """
    读取单个文件，计算所有样本的 (Prompt - Question) token 长度列表
    """
    counts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                
                # 获取路径
                args = data.get('arguments', {})
                # 兼容可能的 key 变化，通常是 gen_args_0
                gen_args = args.get('gen_args_0', {}) or args.get('gen_args_1', {})
                full_prompt = gen_args.get('arg_0', "")
                                
                question_text = data.get('doc', {}).get('question', "")
                if not question_text:
                    question_text = data.get('doc', {}).get('Pre-Revision Question', "")
                if not question_text:
                    question_text = data.get('doc', {}).get('input', "")   
                
                if not full_prompt or not question_text:
                    continue

                # 剔除 Question 内容
                template_text = full_prompt.replace(question_text, "")
                
                # 计算 Token
                tokens = tokenizer.encode(template_text, add_special_tokens=False)
                counts.append(len(tokens))
                
            except Exception:
                continue
    return counts

def main():
    print(f"Loading tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # 数据结构: data_store[dataset][subset] = [list of counts]
    data_store = defaultdict(lambda: defaultdict(list))
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".jsonl") and f.startswith("samples_")]
    files.sort()
    
    print(f"Found {len(files)} files. Start processing...\n")

    total_files = len(files)
    for i, filename in enumerate(files):
        dataset, subset = parse_file_info(filename)
        file_path = os.path.join(DATA_DIR, filename)
        
        # print(f"[{i+1}/{total_files}] Processing {dataset} / {subset} ...")
        
        counts = get_template_token_counts(file_path, tokenizer)
        if counts:
            data_store[dataset][subset].extend(counts)

    # ================= 统计与输出 =================
    
    # 1. 汇总所有数据
    global_counts = []
    dataset_stats = {} # 存储每个数据集的汇总信息

    # 准备表格
    table = PrettyTable()
    table.field_names = ["Dataset", "Subset", "Samples", "Avg Tokens (Template Only)"]
    table.align["Subset"] = "l" # 左对齐
    
    print("\n" + "="*60)
    print("STATISTICS REPORT (Prompt Length excluding Question)")
    print("="*60)

    for dataset in sorted(data_store.keys()):
        dataset_total_counts = []
        
        # 遍历该数据集下的所有 Subset
        for subset in sorted(data_store[dataset].keys()):
            counts = data_store[dataset][subset]
            avg = np.mean(counts)
            
            # 添加到表格
            table.add_row([dataset, subset, len(counts), f"{avg:.2f}"])
            
            # 收集用于上层聚合
            dataset_total_counts.extend(counts)
            global_counts.extend(counts)
        
        # 计算该 Dataset 的整体平均
        if dataset_total_counts:
            ds_avg = np.mean(dataset_total_counts)
            dataset_stats[dataset] = ds_avg
            # 添加一个分隔行或者汇总行
            table.add_row([f"--- {dataset} Total ---", "ALL", len(dataset_total_counts), f"** {ds_avg:.2f} **"])
            table.add_row(["", "", "", ""]) # 空行分隔

    print(table)

    print("\n" + "="*30)
    print("SUMMARY")
    print("="*30)
    
    # 打印各数据集总览
    for ds, avg in dataset_stats.items():
        print(f"Dataset: {ds:<10} | Avg Template Tokens: {avg:.2f}")

    # 打印全局总览
    if global_counts:
        grand_avg = np.mean(global_counts)
        print("-" * 30)
        print(f"GLOBAL AVERAGE     | {grand_avg:.2f}")
        print(f"TOTAL SAMPLES      | {len(global_counts)}")
    else:
        print("No valid data found.")

if __name__ == "__main__":
    main()