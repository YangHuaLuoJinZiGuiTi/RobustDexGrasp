from openai import OpenAI
import re
import json_repair
import base64
import json
from PIL import Image, ImageDraw

import speech_recognition as sr
from pynput import keyboard
from googletrans import Translator

class audio_vlm_planner:
    def __init__(self):
        self.model = "qwen-vl-max-2025-01-25"
        self.client = OpenAI(api_key="sk-ab18ab097de14e1f8851347bccb7f663",
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                        )

        # 初始化识别器和翻译器
        self.recognizer = sr.Recognizer()
        self.translator = Translator()

        # 标志位和语言设置
        self.recording = False
        self.language = None
        self.get_text = None

    def extract_list(self, text_output: str) -> list:
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
            task_name: str,
            frame_path: str = None,
            instruction: str = None,
            max_token: int = 218
    ) -> str:
        if task_name == "classify_user_prompt":
            prompt = f"""
            Analyze the following user prompt: {instruction}

            User prompt types:
            - Type I (return True): User prompts with any specific descriptions
            Examples: 
            * Color-based: "green objects"
            * Position-based: "objects from the right"
            * Property-based: "all cups"
            * Combination: "the red cup on the left"

            - Type II (return False): Abstract prompts without any object descriptions
            Examples: "clear the table", "clean up", "remove everything"

            Please determine:
            - Is this a Type I prompt? (True/False)
            - Provide your reasoning

            Return format:
            True/False: your reasoning

            Examples:
            - "grab the green cup" -> True: Contains specific object (cup) and property (green)
            - "clear the table" -> False: No specific object characteristics mentioned
            """

        elif task_name == "decompose_user_prompt":
            prompt = f"""
            For user prompt: {instruction}
            Process:
            1. Analyze the user prompt and image together:
            - Match user prompt descriptions with visible objects in the image
            - If a description (e.g., "green objects") matches multiple objects, include all matching objects
            - Verify each mentioned object actually exists in the image

            2. Based on the robot arm's position (top of the screen) and table layout
            3. Determine the most efficient grasping sequence
            4. Generate a reordered list of objects to grasp
            
            Requirements:
            - Only include objects mentioned in the original user prompt
            - Keep position information for each object
            - Return as a list, ordered by grasping sequence

            Expected output format:
            ["object with position 1", "object with position 2", ...]
            """

        elif task_name == "generate_instruction":
            prompt = f"""
            Analyze the current desktop layout and select the most suitable object to grasp, considering the following factors:

            Grasping Strategy:
            1. The robotic arm is positioned at the top of the picture (outside the frame)
            2. Grasping Priority Order:
               - Prioritize objects on the right to avoid knocking over other objects during later operations
               - Then consider objects in the middle
               - Finally, consider objects on the left
            3. Accessibility Analysis:
               - Relative positions between objects
               - Potential obstacles
               - Whether the grasping path might interfere with other objects

            Please provide your response in the following JSON format:
            {{
                "analysis": {{
                    "priority_consideration": "Explanation of why this object has priority",
                    "accessibility": "Analysis of object's accessibility",
                    "risk_assessment": "Potential risks in grasping this object"
                }},
                "target": "A comprehensive description of the target object 
                (e.g., 'the blue cube on the far right of the desktop, next to the red cylinder')"
            }}

            Ensure the output is in valid JSON format.
            Note: The 'target' field should ONLY contain the object's color, shape, and position in a natural, flowing sentence. Do not include any analysis or reasoning in this field.
            """

        elif task_name == "mark_bounding_box":
            prompt = f"""
            Analyze the image and identify the best matching object with the description: {instruction}.
            Instructions for object analysis:
            1. Select ONE object that best matches the description
            2. For the selected object, provide:
            - A concise label, object name (3-4 words max)
            - A detailed description (position, color, shape, context)
            - Accurate bbox coordinates

            Required JSON format with an example:
            ```json
            {{
                "bbox_2d": [x1, y1, x2, y2],
                "label": "green cup",  # Keep this very brief (3-4 words)
                "description": "A cylindrical green ceramic cup located on the right side of the wooden table, next to the laptop"  # Detailed description
            }}
            ```

            Critical requirements:
            - Return EXACTLY ONE object
            - "label": Must be brief (3-4 words) for quick reference
            - "description": Must be detailed and include spatial context
            - Use single JSON object format, not an array
            - Ensure bbox coordinates are within image boundaries
            """

        elif task_name == "check_grasp_success":
            prompt = f"""
            Analyze the image and determine if the robotic arm has successfully grasped an object:
            1. Observe the spatial relationship between the robotic hand and the object
            2. Output format: explain your reasoning, then conclude with a boolean value (True=grasped, False=not grasped)
            """
        
        elif task_name == "check_instruction_complete":
            prompt = f"""
            Please check whether {instruction} exists on the desktop. If it does not exist, output True; otherwise, output False.
            """
        
        elif task_name == "check_user_prompt_complete":
            prompt = """
            Please analyze the table in the image:

            Requirements:
            - Only detect physical objects with noticeable height/thickness (3D objects)
            - Exclude from consideration:
            * Flat items (papers, tablecloths, mats)
            * Light projections
            * Shadows
            * Surface patterns or textures

            Return format:
            - True: if the table is empty of 3D objects
            - False: if there are any 3D objects, followed by their names

            Example responses:
            True  (for empty table)
            False: cup, bottle, plate  (for table with objects)
            """

        else:
            raise ValueError(f"The task_name {task_name} is not a valid task name.")

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

        if task_name == "classify_user_prompt":
            if 'true' in response_lower:
                return "TypeI"
            elif 'false' in response_lower:
                return "TypeII"
            else:
                raise ValueError(f"The output text {response} is in the wrong format.")
        elif task_name == "decompose_user_prompt":
            list_output = self.extract_list(response)
            if type(list_output) == list:
                return list_output
            else:
                raise ValueError(f"The output text {list_output} is not a valid list.")
        elif task_name == "generate_instruction":
            generate_task_str = self.parse_json(response)
            generate_task_json = json_repair.loads(generate_task_str)
            generate_task = generate_task_json['target']
            if type(generate_task) == str:
                return generate_task
            else:
                raise ValueError(f"The output text {generate_task} is not a valid string.")
        elif task_name == "mark_bounding_box":
            bbox_str = self.parse_json(response)
            bbox_json = json_repair.loads(bbox_str)
            return bbox_json
        else:
            if 'true' in response_lower:
                return True
            elif 'false' in response_lower:
                return False
            else:
                raise ValueError(f"The output text {response} does not contain a valid boolean value.")

    def on_press(self, key):
        try:
            if key.char == 't':
                self.recording = True
                self.language = 'zh-CN'  # 中文
                print("开始录制中文...")
                self.record_audio()
            elif key.char == 'u':
                self.recording = True
                self.language = 'en-US'  # 英文
                print("开始录制英文...")
                self.record_audio()
            elif key.char == 'e':
                self.recording = False
                print("录音结束")
                return False  # 停止监听器
        except AttributeError:
            pass

    def record_audio(self):
        with sr.Microphone() as source:
            while self.recording:
                print("正在识别...")
                audio_data = self.recognizer.listen(source)
                try:
                    # 根据选择的语言进行识别
                    text = self.recognizer.recognize_google(audio_data, language=self.language)
                    print(f"识别结果: {text}")
                    self.get_text = text
                    # 如果是中文输入，翻译为英文
                    if self.language == 'zh-CN':
                        translated_text = self.translator.translate(text, dest='en').text
                        print(f"翻译结果: {translated_text}")
                        self.get_text = translated_text
                except sr.UnknownValueError:
                    print("无法理解音频")
                except sr.RequestError:
                    print("无法请求结果")
                finally:
                    self.recording = False

    def start_detection(self):
        # 监听键盘事件
        with keyboard.Listener(on_press=self.on_press) as listener:
            print("请按 't' 开始录制中文，按 'u' 开始录制英文，按 'e' 停止录制")
            listener.join()  # 等待监听器结束
            print("监听结束！")
        
        return self.get_text

def main() -> None:

    vlm = audio_vlm_planner()
    
    object_cmd = vlm.start_detection()

    frame_path = "/home/ubuntu/Downloads/Demo/test.png"
    bbox_2d = vlm.request_task("mark_bounding_box", frame_path, object_cmd)
    image = Image.open(frame_path)
    draw = ImageDraw.Draw(image)
    # 绘制矩形框
    x1, y1, x2, y2 = bbox_2d['bbox_2d']
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    image.show()
    image.save("/home/ubuntu/Downloads/Demo/testnew.png")

if __name__ == "__main__":
    main()