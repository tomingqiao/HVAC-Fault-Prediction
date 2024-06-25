import openai
openai.api_base = "http://localhost:13000/v1"
openai.api_key = "sk-96GeVW4uPRlNGwO56eC90d799a664d79Ba624dFfC04a75Bc"

# 使用流式回复的请求
# for chunk in openai.ChatCompletion.create(
#     model="SparkDesk",
#     messages=[
#         {"role": "user", "content": "原神好玩吗"}
#     ],
#     stream=True
#     # 流式输出的自定义stopwords功能尚未支持，正在开发中
# ):
#     if hasattr(chunk.choices[0].delta, "content"):
#         print(chunk.choices[0].delta.content, end="", flush=True)

# 不使用流式回复的请求
response = openai.ChatCompletion.create(
    model="SparkDesk",
    messages=[
        {"role": "user", "content": 
         """给你一个数学问题，请从选项中选出最优的答案


question: 一辆汽车正以匀速直线行驶，驶向一座垂直塔楼的底部。从汽车上观察塔顶，在此过程中，仰角从45°变为60°需要10分钟。这辆车还要多久才能到达塔的底部？

choices: 
A)5(√3 + 1)
B)6(√3 + √2)
C)7(√3 – 1)
D)8(√3 – 2)
E)没有答案

答案必须以json格式结束: {Answer: one of options[A,B,C,D,E]}."""}
    ],
    stream=False,
    stop=[] # 在此处添加自定义的stop words 例如ReAct prompting时需要增加： stop=["Observation:"]。
)
print(response.choices[0].message.content)
