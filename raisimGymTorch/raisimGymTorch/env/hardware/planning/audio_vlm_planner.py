from openai import OpenAI
import json_repair
import base64
from PIL import Image, ImageDraw
import time
import signal  # for keyboard events handling (press "Ctrl+C" to terminate recording and translation)
import sys

import dashscope
import pyaudio
from dashscope.audio.asr import *

import re

def remove_target_string(text, target):
    if text is None:
        return False, None, 0

    # 忽略大小写的正则表达式, 检查目标字符串是否存在
    pattern = re.compile(re.escape(target), re.IGNORECASE)
    match = pattern.search(text)
    
    if match:
        # 去除匹配的目标字符串
        modified_text = pattern.sub('', text)
        # 统计字母字符的数量
        letter_count = len(re.findall(r'[a-zA-Z]', modified_text))
        
        return True, modified_text, letter_count
    else:
        return False, text, 0

# from googletrans import Translator
mic = None
stream = None
end_flag = False
text_buf = None
origin_text_buf = None
text_stash_buf = None
origin_text_stash_buf = None

class Callback(TranslationRecognizerCallback):
    def on_open(self) -> None:
        global mic
        global stream
        mic = pyaudio.PyAudio()
        stream = mic.open(format=pyaudio.paInt16,
                          channels=1,
                          rate=16000,
                          input=True)
        print(' ************ mic open ok *******************')

    def on_close(self) -> None:
        global mic
        global stream
        global text_buf
        global origin_text_buf
        global text_stash_buf
        global origin_text_stash_buf
        print('---------------------- audio close.')
        stream.stop_stream()
        stream.close()
        mic.terminate()
        stream = None
        mic = None
        text_buf = None
        origin_text_buf = None
        text_stash_buf = None
        origin_text_stash_buf = None

    def on_complete(self) -> None:
        print(' ~~~~~~~~~~~~~~~~~~~~~~ Translation completed.')  # translation completed

    def on_error(self, message) -> None:
        print(' ----------------------------- audio  error: ', message.message)
        # Stop and close the audio stream if it is running
        if 'stream' in globals() and stream.active:
            stream.stop()
            stream.close()
        # Forcefully exit the program
        sys.exit(1)

    def on_event(
        self,
        request_id,
        transcription_result: TranscriptionResult,
        translation_result: TranslationResult,
        usage,
    ) -> None:
        global end_flag
        global text_buf
        global origin_text_buf
        global text_stash_buf
        global origin_text_stash_buf
        if transcription_result is not None:
            origin_text_buf = transcription_result.text
            if transcription_result.stash is not None:
                origin_text_stash_buf = transcription_result.stash.text
            else:
                origin_text_stash_buf = None
        if translation_result is not None:
            translation = translation_result.get_translation('en')
            if translation.is_sentence_end:
                if translation.stash is not None:
                    text_stash_buf = translation.stash.text
                text_buf = translation.text
                end_flag = True

class audio_vlm_planner:
    def __init__(self):
        self.model = "qwen-vl-max-2025-01-25"
        self.client = OpenAI(api_key="sk-ab18ab097de14e1f8851347bccb7f663",
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                        )
        
        dashscope.api_key = 'sk-ab18ab097de14e1f8851347bccb7f663'

        # Create the translation callback
        self.callback = Callback()


        # my_vocabulary = [
        #     {"text": "小迪", "lang": "zh", "target_lang": "en", "translation": "xiaodi"}
        # ]
        # service = VocabularyService()
        # vocabulary_id = service.create_vocabulary(
        #     prefix='prefix',
        #     target_model="gummy-realtime-v1",
        #     vocabulary=my_vocabulary)


        # Call recognition service by async mode, you can customize the recognition parameters, like model, format,
        # sample_rate For more information, please refer to https://help.aliyun.com/document_detail/2712536.html
        self.translator = TranslationRecognizerRealtime(
            model='gummy-realtime-v1',
            format='pcm',
            sample_rate=16000,
            transcription_enabled=False,
            translation_enabled=True,
            # vocabulary_id=vocabulary_id,
            source_language='zh',
            translation_target_languages=['en'],
            callback=self.callback,
        )
        #self.translator.start()

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
        Analyze the image and identify the best matching object in the command: {instruction}.
        Instructions for object analysis:
        1. Select ONE object that best matches in the command
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

        # 3. For the selected object, provide the similarity between the object and the command described from 0 to 1
        #     "confidence": c0,
        # - the confidence set 0 is totally different from the description and the confidence set 1 is totally the same as the description
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
    
        
    def signal_handler(self, sig, frame):
        print('Ctrl+C pressed, stop translation ...')
        # Stop translation
        self.translator.stop()
        print('Translation stopped.')
        print(
            '[Metric] requestId: {}, first package delay ms: {}, last package delay ms: {}'
            .format(
                self.translator.get_last_request_id(),
                self.translator.get_first_package_delay(),
                self.translator.get_last_package_delay(),
            ))
        # Forcefully exit the program
        sys.exit(0)


    def start_detection(self):
        global end_flag
        global text_buf
        global origin_text_buf
        global text_stash_buf
        global origin_text_stash_buf
        self.translator.start()
        while True:
            if stream:
                data = stream.read(3200, exception_on_overflow=False)
                self.translator.send_audio_frame(data)
            else:
                print('******************************* no stream end')
                break

            if end_flag is True:
                end_flag = False
                break
        
        # print(f' ******************* origin_text_buf: {origin_text_buf} ***************************** ')
        # print(f' ************* origin_text_stash_buf: {origin_text_stash_buf} ***************************** ')
        # print(f' ************************** text_buf: {text_buf} ***************************** ')
        # print(f' ******************** text_stash_buf: {text_stash_buf} ***************************** ')
        flag, ret, len = remove_target_string(text_buf, "xiaodi")
        if flag is False:
            flag, ret, len = remove_target_string(text_buf, "xiao di")
        self.translator.stop()
        return flag, ret, len
    
def main() -> None:

    vlm = audio_vlm_planner()
    
    for i in range(5):
        while True:
            flag, object_cmd, len = vlm.start_detection()
            if flag is True and len > 8:
                print(f"get the audio text: {object_cmd}, len = {len}")
                break
            else:
                print(f" !!!!!!!!!!!!!!! {object_cmd} can not awake, Please say 小迪, 小迪 at frist")

        frame_path = "/home/ubuntu/hand/github/vision_dex/raisimGymTorch/raisimGymTorch/env/hardware/realsense/rgb.png"
        bbox_2d = vlm.request_task(frame_path, object_cmd)
        image = Image.open(frame_path)
        draw = ImageDraw.Draw(image)
        # 绘制矩形框
        x1, y1, x2, y2 = bbox_2d['bbox_2d']
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        image.show()
        image.save("/home/ubuntu/hand/github/vision_dex/raisimGymTorch/raisimGymTorch/env/hardware/realsense/testnew.png")

if __name__ == "__main__":
    main()