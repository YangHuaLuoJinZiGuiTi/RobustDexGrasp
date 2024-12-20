import os
from openai import OpenAI
import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')




client = OpenAI(

    api_key="sk-ab18ab097de14e1f8851347bccb7f663", 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

image_base64 = image_to_base64("apple.jpg")

completion = client.chat.completions.create(
    # 一个月以内随便用，注册送了100w token，一个月内免费
    model="qwen-vl-max-0809", 
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': [
                {'type': 'text', 'text': 'Please identify the object in the image, with one word only: '},
                {'type': 'image_url', 'image_url': f"data:image/jpeg;base64,{image_base64}"}
            ]
        }
            ],
    )
    
print(completion.model_dump_json())