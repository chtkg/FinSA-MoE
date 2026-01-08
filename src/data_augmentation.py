import pandas as pd
import random
import requests
import json
import re


input_csv_path = r"../data/raw/forum/stock_comments.csv"    
output_csv_path = r"../data/augmented/stock_comments.csv"  

df = pd.read_csv(input_csv_path)

print(f"Loaded {len(df)} entries from the original dataset.")

random.seed(42)  
total_count = len(df)
syn_count = int(total_count * 0.2)
syn_indices = set(random.sample(range(total_count), syn_count))
llm_indices = [i for i in range(total_count) if i not in syn_indices]

df_syn = df.iloc[list(syn_indices)].copy()  
df_llm = df.iloc[llm_indices].copy()       

print(f"Synonym augmentation for {len(df_syn)} entries, LLM augmentation for {len(df_llm)} entries.")


synonym_dict = {
    # 价格走势 (Price Trend)
    "上涨": ["攀升", "走高", "上扬", "拉升", "上行", "飙升", "大涨"],
    "下跌": ["下挫", "走低", "回落", "下滑", "下行", "暴跌", "大跌"],
    "股价": ["股票价格", "市价", "股值", "价格"],
    "震荡": ["波动", "振荡", "起伏", "整理"],
    "反弹": ["回升", "反攻", "回暖", "复苏"],
    "调整": ["回调", "修正", "整理", "盘整"],
    "突破": ["冲破", "站上", "攻克", "突破"],
    # 业绩相关 (Performance Related)
    "业绩": ["经营业绩", "财务表现", "营收表现", "盈利"],
    "增长": ["提升", "上升", "增加", "攀升", "提高"],
    "下滑": ["下降", "减少", "降低", "萎缩"],
    "盈利": ["获利", "赚钱", "营利", "收益"],
    "亏损": ["赔钱", "损失", "负收益"],
    # 消息面 (News/Signals)
    "利好": ["正面消息", "积极因素", "利好消息", "好消息"],
    "利空": ["负面消息", "利空因素", "消极信号", "坏消息"],
    # 市场状态 (Market Condition)
    "强势": ["强劲", "有力", "给力", "凶猛"],
    "疲软": ["弱势", "乏力", "低迷", "萎靡"],
    "活跃": ["火热", "热络", "热闹"],
    "清淡": ["冷清", "低迷", "惨淡"],
    # 投资者行为 (Investor Action)
    "买入": ["建仓", "抄底", "进场", "买进"],
    "卖出": ["出货", "离场", "抛售", "卖掉"],
    "持有": ["拿着", "持仓", "守住"],
    # 常用形容词 (Common Adjectives)
    "很好": ["不错", "优秀", "出色", "棒"],
    "很差": ["糟糕", "差劲", "不行", "烂"],
    "可能": ["或许", "也许", "大概", "估计"],
    "应该": ["应当", "理应", "该当"],
    "看好": ["看涨", "乐观", "看多"],
    "看空": ["看跌", "悲观", "不看好"]
}

for word, syn_list in synonym_dict.items():
    synonym_dict[word] = [syn for syn in syn_list if syn != word]

def synonym_replace_text(text):
    new_text = text
    for word, syn_list in synonym_dict.items():
        if word in new_text:

            if syn_list:  
                replacement = random.choice(syn_list)
                new_text = new_text.replace(word, replacement)
    return new_text


augmented_records = []  

max_id = df['id'].max() if df['id'].dtype != object else None
next_id = int(max_id) + 1 if max_id is not None else len(df) + 1  

for idx, row in df_syn.iterrows():
    orig_id = row['id']
    orig_text = str(row['text'])
    orig_label = row['label']
    
    for j in range(3):
        new_text = synonym_replace_text(orig_text)
        
        attempts = 0
        while new_text == orig_text and attempts < 3:
            new_text = synonym_replace_text(orig_text)
            attempts += 1
        if new_text == orig_text:
            continue
        new_id = next_id
        next_id += 1
        augmented_records.append((new_id, new_text, orig_label))


ollama_url = "http://localhost:11434/api/generate"

for idx, row in df_llm.iterrows():
    orig_text = str(row['text'])
    orig_label = row['label']
    prompt = (
        f"请将以下股票评论用不同的措辞改写，同时确保情绪保持为「{orig_label}」。"
        f"请直接给出改写后的文本，不要输出任何解释或思考过程。\n原文：{orig_text}\n改写："
    )
    for j in range(3):
        payload = {"model": "deepseek-r1:32b", "prompt": prompt}
        try:
            response = requests.post(ollama_url, json=payload)
        except Exception as e:
            print(f"Error connecting to Ollama server: {e}")
            break  
        if response.status_code != 200:
            print(f"Ollama API returned error {response.status_code}: {response.text}")
            break
        lines = response.text.splitlines()
        output = ""
        for line in lines:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  
            if "response" in obj:
                output += obj["response"]
        cleaned_output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL)
        cleaned_output = cleaned_output.strip()
        if cleaned_output.startswith("回答"):
            cleaned_output = cleaned_output.lstrip("回答：:").lstrip()  
        if cleaned_output == "":
            continue
        new_id = next_id
        next_id += 1
        augmented_records.append((new_id, cleaned_output, orig_label))

aug_df = pd.DataFrame(augmented_records, columns=["id", "text", "label"])
combined_df = pd.concat([df, aug_df], ignore_index=True)
print(f"Augmented dataset size: {len(combined_df)} (including original and new data)")

combined_df.to_csv(output_csv_path, index=False)
print(f"Augmented dataset saved to {output_csv_path}")
