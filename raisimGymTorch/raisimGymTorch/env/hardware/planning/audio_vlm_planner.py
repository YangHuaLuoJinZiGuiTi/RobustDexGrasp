from openai import OpenAI
import re
import json_repair
import base64
import json
from PIL import Image, ImageDraw

import speech_recognition as sr
from pynput import keyboard
import time
# from googletrans import Translator

class audio_vlm_planner:
    def __init__(self):
        self.model = "qwen-vl-max-2025-01-25"
        self.client = OpenAI(api_key="sk-ab18ab097de14e1f8851347bccb7f663",
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                        )

        # 初始化识别器和翻译器
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 4000  # 设定音量阈值，数值越大要求音量越高
        # self.translator = Translator()

        # 标志位和语言设置
        self.recording = False
        self.language = None
        self.get_text = None

    def parse_json(self, json_output):
        lines = json_output.splitlines()
        for i, line in enumerate(lines):
            if line == "```json":
                json_output = "\n".join(lines[i+1:])
                json_output = json_output.split("```")[0]
                break
        return json_output


    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


    def request_task(self,
            frame_path: str = None,
            instruction: str = None,
            max_token: int = 218
    ) -> str:

        prompt = f"""
        Analyze the image and identify the best matching object with the description: {instruction}.
        Instructions for object analysis:
        1. Select ONE object that best matches the description
        2. For the selected object, provide accurate bbox coordinates

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
                    "url": f"data:image/jpeg;base64,{self.encode_image(frame_path)}",
                },
            })

        chat_completion = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=max_token,
            messages=messages
        )

        response = chat_completion.choices[0].message.content
        response_lower = response.lower()

        bbox_str = self.parse_json(response)
        bbox_json = json_repair.loads(bbox_str)
        return bbox_json

    def on_press(self, key):
        if self.recording is False:
            try:
                if key.char == 't':
                    self.recording = True
                    self.language = 'zh-CN'  # 中文
                    print("按下 't' ...")
                    self.record_audio()
            except AttributeError:
                pass

    def on_release(self, key):
        try:
            if key.char == 't':  # 如果松开 't' 键
                self.recording = False
                print("松开 't' 键 ...")
                return False  # 停止监听器
        except AttributeError:
            pass
            
    def record_audio(self):
        with sr.Microphone() as source:
            try:
                # 调整为适应环境噪声
                self.recognizer.adjust_for_ambient_noise(source, duration=1)

                print("开始录音...")
                t1 = time.time()
                # 设置timeout为0.5秒，phrase_time_limit限制每次录音最大时间
                audio_data = self.recognizer.listen(source)
                print("识别完成...")
                t2 = time.time()
                print(f"----------- 监听耗时 = {t2 - t1}")
                # 根据选择的语言进行识别
                text = self.recognizer.recognize_google(audio_data, language=self.language)
                t3 = time.time()
                print(f"----------- 识别耗时 = {t3 - t2}")
                print(f"识别结果: {text}")
                self.get_text = text
                # 如果是中文输入，翻译为英文
                # if self.language == 'zh-CN':
                #     translated_text = self.translator.translate(text, dest='en').text
                #     print(f"翻译结果: {translated_text}")
                #     self.get_text = translated_text
            except sr.UnknownValueError:
                print("无法理解音频")
            except sr.RequestError:
                print("无法请求结果")
            finally:
                pass

    def start_detection(self):
        # 监听键盘事件
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            print("按住 't' 开始录音， 松开 't' 结束录音 ")
            listener.join()  # 等待监听器结束
            print("监听结束！")
        
        return self.get_text

def main() -> None:

    vlm = audio_vlm_planner()
    
    for i in range (5):
        object_cmd = vlm.start_detection()

    exit(0)
    frame_path = "/home/ubuntu/Downloads/Demo/test.png"
    bbox_2d = vlm.request_task(frame_path, object_cmd)
    image = Image.open(frame_path)
    draw = ImageDraw.Draw(image)
    # 绘制矩形框
    x1, y1, x2, y2 = bbox_2d['bbox_2d']
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    image.show()
    image.save("/home/ubuntu/Downloads/Demo/testnew.png")

if __name__ == "__main__":
    main()