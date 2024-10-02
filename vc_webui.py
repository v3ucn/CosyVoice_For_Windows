import os,sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio

from pydub import AudioSegment

cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-25Hz')

# 设置HF_ENDPOINT环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 设置HF_HOME环境变量为当前目录下的hf_download文件夹
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hf_download")
import gradio as gr

import numpy as np
from pydub import AudioSegment


initial_md = """

整合包制作:刘悦的技术博客 https://space.bilibili.com/3031494

"""


def convert_wav(input_file,output):
    # 加载音频文件
    audio = AudioSegment.from_wav(input_file)
    
    # 转换为单声道
    audio = audio.set_channels(1)
    
    # 设置采样率为16kHz
    audio = audio.set_frame_rate(16000)
    
    # 设置采样位数为16位
    audio = audio.set_sample_width(2)  # 2 bytes = 16 bits
    
    # 导出处理后的音频
    audio.export(output, format="wav")

    return output


# 请求接口
def request_api(audio1,audio2):

    convert_wav(audio1,"audio1.wav")
    convert_wav(audio2,"audio2.wav")

    prompt_speech_16k = load_wav("audio2.wav", 16000)
    source_speech_16k = load_wav("audio1.wav", 16000)
    
    for i, j in enumerate(cosyvoice.inference_vc(source_speech_16k, prompt_speech_16k, stream=False)):
        torchaudio.save('vc_test.wav'.format(i), j['tts_speech'], 22050)

    return "vc_test.wav"

with gr.Blocks() as app:
    gr.Markdown(initial_md)


    with gr.Accordion("音频处理"):
        with gr.Row():


            uploaded_audio1 = gr.Audio(type="filepath", label="目标音频")

            uploaded_audio2 = gr.Audio(type="filepath", label="音色参考音频")

            
            api_button = gr.Button("点击转换")
            

        with gr.Row():

            output_audio = gr.Audio(type="filepath", label="音频输出")
            
            api_button.click(fn=request_api, inputs=[uploaded_audio1,uploaded_audio2], outputs=[output_audio])


    
if __name__ == '__main__':
    app.queue()
    app.launch(inbrowser=True)

    


