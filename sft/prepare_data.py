from datasets import load_dataset
import re

# Load dataset
ds = load_dataset('ElonTusk2001/rstar_sft', split='train')

def change_format(example):
    """
    Process response format conversion:
    1. <code>...<end_of_code> 中的 <end_of_step> 去掉，并换成 ```python ... ```
    2. <output>...<end_of_output> 换成 Code execution result: ...
    3. <answer>...<end_of_answer> 外面的标签去掉，只保留内容
    """
    if 'response' not in example or not example['response']:
        return example
    
    response = example['response']
    
    # 1. 处理 <code>...</code> 部分：去掉 <end_of_step>，替换为 ```python ... ```
    def process_code(match):
        code_content = match.group(1)
        # 去掉所有的 <end_of_step>
        code_content = code_content.replace('<end_of_step>\n', '').replace('<end_of_step>\n\n', '\n')
        # 替换为 markdown 代码块格式
        return f'```python\n{code_content}\n```'
    
    response = re.sub(r'<code>\s*(.*?)\s*<end_of_code>', process_code, response, flags=re.DOTALL)
    
    # 2. 处理 <output>...</output> 部分：替换为 Code execution result: ...
    def process_output(match):
        output_content = match.group(1)
        return f'Code execution result: {output_content}'
    
    response = re.sub(r'<output>\s*(.*?)\s*<end_of_output>', process_output, response, flags=re.DOTALL)
    
    # 3. 处理 <answer>...</answer> 部分：去掉标签，只保留内容
    def process_answer(match):
        answer_content = match.group(1)
        return answer_content.strip()
    
    response = re.sub(r'<answer>\s*(.*?)\s*<end_of_answer>', process_answer, response, flags=re.DOTALL)
    
    example['response'] = response
    return example


# Convert data format: from query/response to messages format
def convert_to_messages(example):
    """
    Convert query/response format to messages format required by axolotl
    格式: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    messages = []
    
    # 添加 user 消息（从 query 字段）
    if 'query' in example and example['query']:
        messages.append({
            "role": "user",
            "content": example['query']
        })
    
    # 添加 assistant 消息（从 response 字段）
    if 'response' in example and example['response']:
        messages.append({
            "role": "assistant",
            "content": example['response']
        })
    
    return {"messages": messages}

# 先处理 response 格式转换
ds = ds.map(change_format)

# 然后转换为 messages 格式
ds = ds.map(convert_to_messages, remove_columns=ds.column_names)

# 保存为 JSON Lines 格式
ds.to_json('sft_data.jsonl')

print(f"数据集已转换并保存到: sft_data.jsonl")
print(f"数据集大小: {len(ds)} 条记录")
print(f"示例数据格式:")
print(ds[0])