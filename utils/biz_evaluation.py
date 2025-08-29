"""
The evaluation code is from BizFinBench. 
https://github.com/HiThink-Research/BizFinBench/blob/main/benchmark_code/BizFinBench/eval_financial_description.py
"""
import re
import json
import argparse
import time
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from openai import OpenAI

def parse_output_biz(ans_judge):
  try:
    score = int(ans_judge['评分分数'])
  except:
      print('no score----')
      print(ans_judge)
      score=0
  return score

prompt_template = '''你是一个金融内容评测专家，正在进行金融数据描述准确性的评估。

    你的打分需要考虑两个方面：

    1. 数据错用：<answer>中的指标数字应该和<question>中的对应上，不应该出现指标错用、时间错用等情况，例如：从55.32增长到59.14描述成从55.24增长到58.32。

    2. 数据描述: 只需判断<answer>中是否存在描述与具体数据相背的情况，如果有则得0分。例如：一连串数值越来越大，描述却是递减、两两比较错误，或最大、最小值判断错误、涨跌幅大于零说成下跌、主力资金小于零说成资金流入。当<question>中未取到数或取到的数据为空时，<answer>中回答不能说该数据为0，如果有则得0分。

    | 分数    | 描述                                                         |
    | ------- | ------------------------------------------------------------ | 
    | **100** | 完全正确。趋势描述和数据描述的均完全正确，且语言流程，无幻觉。 |
    | **60** | 部分错误。数据趋势描述正确，但数据值描述错误，例如从55.32增长到59.14描述成从55.24增长到58.32。 |
    | **0** | 错误较多。数据趋势描述错误即不得分，例如数据趋势是越来越大，描述是递减。       |

    ### 以下是你需要评分的案例：
    <question>
      <参考材料>
    </question>

    <answer>
    <回复答案替换>
    </answer>

    ### 要求：
    返回结果以json格式展示，参考：
    {"评分分数":"xx","描述": "xxxx"}

    ### 回答如下：
'''.strip()

parser = argparse.ArgumentParser(description="An example of using argparse.")
parser.add_argument('--input_file',type=str,default='')
parser.add_argument('--scores_file',type=str,default='')

args = parser.parse_args()
input_file = args.input_file 
scores_file = args.scores_file
right_num = 0
all_num = 0
data1 = []
total_scores = 0
with open(input_file,'r',encoding='utf-8',errors='ignore') as f2:
    contents = f2.readlines()
    # i = 0
    for _ in tqdm(contents):
        row = json.loads(_.strip())

        if 'output' in row:
            row['response'] = row['output']
            row['messages'] = [{},{'content':row['prompt']}]
        ref = row['messages'][1]['content']
        if "response" in row:
            ans = row['response']
        else:
            ans = row['messages'][2]['content']
            row['response'] = ans
        answer = ans.replace('\n\n\n','\n').replace('\n\n','\n')
        answer = answer.strip()
        try:
            answer = answer.split('</think>')[1]
        except:
            answer = answer

        answer = answer.replace('<think>','').replace('</think>','').replace('<answer>','').replace('</answer>','').replace('\n\n','\n').replace('\n需','')
        def remove_table(text):
            return re.sub(r'<table>.*?</table>', '', text, flags=re.DOTALL)
        answer = remove_table(answer)

        prompt_ref = prompt_template.replace('<参考材料>',ref).replace('<br/>','\n').replace('\n\n\n','\n')
        prompt_answer = prompt_ref.replace('<回复答案替换>',answer).replace('<br/>','\n').replace('\n\n\n','\n')
        messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": prompt_answer
                }
            ]
      
        client  = OpenAI(api_key="",base_url="")
        data = []
        temp_output = ''
        temp_score = 0
        try:
          completion = client.chat.completions.create(
          model="",
          messages=messages,
          temperature=0,
          max_tokens=3000,
          )
          message_content = completion.choices[0].message.content
          content = json.loads(message_content)
          temp_output = content
          score = parse_output_biz(content)
          temp_score = score
        except Exception as e:
          print (e)
          continue
        total_scores += temp_score
        print ('分数为', temp_score)
        data1.append({'query': ref, 'response': answer, 'score': temp_score, 'gpt_output': temp_output})
        new_data1 = pd.DataFrame(data1)
        new_data1.to_csv(scores_file)
print(f'平均分数是：{round(total_scores/len(data1),2)}')     

