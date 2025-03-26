from openai import OpenAI
import re
import json_repair
import base64
import json
from PIL import Image, ImageDraw

def parse_json(json_output):
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            break
    return json_output


def extract_list(text_output: str) -> list:
    pattern = r'\[(.*?)\]'
    match = re.search(pattern, text_output, re.DOTALL)
    if not match:
        raise ValueError(f"No list format content found: {text_output}")

    items_pattern = r'"([^"]*)"'
    items_list = re.findall(items_pattern, match.group(1))

    if not items_list:
        items_pattern = r"'([^']*)'"
        items_list = re.findall(items_pattern, match.group(1))
        
    return items_list

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

max_token = 218
model = "qwen-vl-max-2025-01-25"
client = OpenAI(api_key="sk-ab18ab097de14e1f8851347bccb7f663",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                )

frame_path = "/home/ubuntu/hand/github/vision_dex/raisimGymTorch/raisimGymTorch/env/hardware/test/vlm_planning/cluster_dense.png"

instruction = "give me a green vegetable"

prompt = f"""
            Analyze the image and identify the best matching object with the description: {instruction}.
            Instructions for object analysis:
            1. Select ONE object that best matches the description
            2. For the selected object, provide:
            - Accurate bbox coordinates

            Required JSON format with an example:
            ```json
            {{
                "bbox_2d": [x1, y1, x2, y2],
            }}
            ```

            Critical requirements:
            - Return EXACTLY ONE object
            - Use single JSON object format, not an array
            - Ensure bbox coordinates are within image boundaries
            """
            

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}],
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": prompt}]
    }
]
if frame_path is not None:
    messages[1]["content"].append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{encode_image(frame_path)}",
        },
    })

chat_completion = client.chat.completions.create(
    model=model,
    max_completion_tokens=max_token,
    messages=messages
)

response = chat_completion.choices[0].message.content
response_lower = response.lower()

bbox_str = parse_json(response)
bbox_json = json_repair.loads(bbox_str)

print(response_lower)
print(",,,,,,")
print(bbox_json)



bbox_2d = bbox_json['bbox_2d']
image = Image.open(frame_path)
draw = ImageDraw.Draw(image)
# 绘制矩形框
x1, y1, x2, y2 = bbox_2d
draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
image.show()
image.save("/home/ubuntu/hand/github/vision_dex/raisimGymTorch/raisimGymTorch/env/hardware/test/vlm_planning/cluster_dense.png")