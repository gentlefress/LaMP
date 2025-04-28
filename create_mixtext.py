import time
import openai
import requests
# def openai_chatgpt_function():
#     question="西游记是谁写的？"
#     print("问题:{}".format(question))
#     url="https://api.openai.com/v1"   #可以替换为任何代理的接口
#     OPENAI_API_KEY="sk-LECc8U0BcVAQWx1VE23aB8B5595f4963929d799b712382E3"  # openai官网获取key
#     openai.api_key = OPENAI_API_KEY
#     openai.api_base = url
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",messages=[{"role": "user", "content": question}],stream=False)
#     print("完整的响应结果:{}".format(response))
#     answer=response.choices[0].message.content
#     print("答案:{}".format(answer))
# if __name__ == "__main__":
#     openai_chatgpt_function() # 通用方法：利用requests 正常请求调用
# import openai

def chat_with_gpt(prompt):
    openai.api_key = '您的API密钥'

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content']
    except Exception as e:
        return str(e)

def main():
    user_prompt = input("请输入您的问题或话题: ")
    response = chat_with_gpt(user_prompt)
    print("ChatGPT 回复:", response)

if __name__ == "__main__":
    main()