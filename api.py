
import time
import io, os, sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/AcademiCodec'.format(ROOT_DIR))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

import requests
from pydub import AudioSegment

import numpy as np
from flask import Flask, request, Response,send_from_directory
import torch
import torchaudio

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import ffmpeg

from flask_cors import CORS
from flask import make_response

import shutil

import json

cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-25Hz')

default_voices = ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']

spk_new = []

for name in os.listdir(f"{ROOT_DIR}/voices/"):
    print(name.replace(".py",""))
    spk_new.append(name.replace(".py",""))

print("默认音色",cosyvoice.list_avaliable_spks())
print("自定义音色",spk_new)

app = Flask(__name__)

CORS(app, cors_allowed_origins="*")

CORS(app, supports_credentials=True)


def download_and_convert(mp3_url, wav_filename):
    """Downloads an MP3 file and converts it to WAV.

    Args:
        mp3_url: The URL of the MP3 file.
        wav_filename: The desired filename for the WAV file (including .wav extension).
    """
    try:
        response = requests.get(mp3_url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        with open("temp.mp3", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Convert MP3 to WAV using pydub
        sound = AudioSegment.from_mp3("temp.mp3")
        sound.export(wav_filename, format="wav")

        print(f"音频已成功下载并转换为 {wav_filename}")

    except requests.exceptions.RequestException as e:
        print(f"下载音频时出错: {e}")
    except Exception as e:
        print(f"转换音频时出错: {e}")
    finally:
        # Clean up the temporary MP3 file
        import os
        try:
            os.remove("temp.mp3")
        except OSError as e:
            print(f"删除临时文件时出错: {e}")
def speed_change(input_audio: np.ndarray, speed: float, sr: int):
    # 检查输入数据类型和声道数
    if input_audio.dtype != np.int16:
        raise ValueError("输入音频数据类型必须为 np.int16")


    # 转换为字节流
    raw_audio = input_audio.astype(np.int16).tobytes()

    # 设置 ffmpeg 输入流
    input_stream = ffmpeg.input('pipe:', format='s16le', acodec='pcm_s16le', ar=str(sr), ac=1)

    # 变速处理
    output_stream = input_stream.filter('atempo', speed)

    # 输出流到管道
    out, _ = (
        output_stream.output('pipe:', format='s16le', acodec='pcm_s16le')
        .run(input=raw_audio, capture_stdout=True, capture_stderr=True)
    )

    # 将管道输出解码为 NumPy 数组
    processed_audio = np.frombuffer(out, np.int16)

    return processed_audio

@app.route("/", methods=['POST'])
def sft_post():
    question_data = request.get_json()

    text = question_data.get('text')
    speaker = question_data.get('speaker')
    streaming = question_data.get('streaming',0)

    speed = request.args.get('speed',1.0)
    speed = float(speed)
    

    if not text:
        return {"error": "文本不能为空"}, 400

    if not speaker:
        return {"error": "角色名不能为空"}, 400

    # 非流式
    if streaming == 0:

        buffer = io.BytesIO()

        tts_speeches = []

        for i, j in enumerate(cosyvoice.inference_sft(text,speaker,stream=False,speed=speed,new_dropdown="无")):
            # torchaudio.save('sft_{}.wav'.format(i), j['tts_speech'], 22050)
            tts_speeches.append(j['tts_speech'])
        
        audio_data = torch.concat(tts_speeches, dim=1)
        torchaudio.save(buffer,audio_data, 22050, format="wav")
        buffer.seek(0)
        return Response(buffer.read(), mimetype="audio/wav")

    # 流式模式
    else:

        def generate():

            for i, j in enumerate(cosyvoice.inference_sft(text,speaker,stream=True,speed=speed,new_dropdown="无")):

                tts_speeches = []
                buffer = io.BytesIO()
                tts_speeches.append(j['tts_speech'])
                audio_data = torch.concat(tts_speeches, dim=1)
                torchaudio.save(buffer,audio_data, 22050, format="ogg")
                buffer.seek(0)

                yield buffer.read()

        response = make_response(generate())
        response.headers['Content-Type'] = 'audio/ogg'
        response.headers['Content-Disposition'] = 'attachment; filename=sound.ogg'
        return response

@app.route("/save_voice", methods=['GET'])
def save_voice():

    text = request.args.get('text')
    audio = request.args.get('audio')
    voice_name = request.args.get('voice_name')


    download_and_convert(audio,"zero_test.wav")

    prompt_speech_16k = load_wav('zero_test.wav', 16000)
    tts_speeches = []
    for i, j in enumerate(cosyvoice.inference_zero_shot(text,text, prompt_speech_16k, stream=False)):
        # torchaudio.save('sft_{}.wav'.format(i), j['tts_speech'], 22050)
        tts_speeches.append(j['tts_speech'])
    
    audio_data = torch.concat(tts_speeches, dim=1)
    torchaudio.save('zero_shot.wav',audio_data, 22050, format="wav")

    shutil.copyfile(f"{ROOT_DIR}/output.pt",f"{ROOT_DIR}/voices/{voice_name}.pt")

    response = app.response_class(
        response=json.dumps({"voice_name":voice_name}),
        status=200,
        mimetype='application/json'
    )
    return response





@app.route("/", methods=['GET'])
def sft_get():

    text = request.args.get('text')
    speaker = request.args.get('speaker')
    new = request.args.get('new',0)
    streaming = request.args.get('streaming',0)
    speed = request.args.get('speed',1.0)
    speed = float(speed)

    if not text:
        return {"error": "文本不能为空"}, 400

    if not speaker:
        return {"error": "角色名不能为空"}, 400

    # 非流式
    if streaming == 0:

        buffer = io.BytesIO()

        tts_speeches = []

        for i, j in enumerate(cosyvoice.inference_sft(text,speaker,stream=False,speed=speed,new_dropdown="无")):
            # torchaudio.save('sft_{}.wav'.format(i), j['tts_speech'], 22050)
            tts_speeches.append(j['tts_speech'])
        
        audio_data = torch.concat(tts_speeches, dim=1)
        torchaudio.save(buffer,audio_data, 22050, format="wav")
        buffer.seek(0)
        return Response(buffer.read(), mimetype="audio/wav")

    # 流式模式
    else:

        
        def generate():

            for i, j in enumerate(cosyvoice.inference_sft(text,speaker,stream=True,speed=speed,new_dropdown="无")):

                tts_speeches = []
                buffer = io.BytesIO()
                tts_speeches.append(j['tts_speech'])
                audio_data = torch.concat(tts_speeches, dim=1)
                torchaudio.save(buffer,audio_data, 22050, format="ogg")
                buffer.seek(0)

                yield buffer.read()

        response = make_response(generate())
        response.headers['Content-Type'] = 'audio/ogg'
        response.headers['Content-Disposition'] = 'attachment; filename=sound.ogg'
        return response
        
        # return Response(generate(), mimetype='audio/x-wav')

                





@app.route("/tts_to_audio/", methods=['POST'])
def tts_to_audio():

    import speaker_config
    
    question_data = request.get_json()

    text = question_data.get('text')
    speaker = speaker_config.speaker

    speed = speaker_config.speed
    

    if not text:
        return {"error": "文本不能为空"}, 400

    if not speaker:
        return {"error": "角色名不能为空"}, 400

    buffer = io.BytesIO()

    tts_speeches = []

    for i, j in enumerate(cosyvoice.inference_sft(text,speaker,stream=False,speed=speed,new_dropdown="无")):
        # torchaudio.save('sft_{}.wav'.format(i), j['tts_speech'], 22050)
        tts_speeches.append(j['tts_speech'])
    
    audio_data = torch.concat(tts_speeches, dim=1)
    torchaudio.save(buffer,audio_data, 22050, format="wav")
    buffer.seek(0)
    return Response(buffer.read(), mimetype="audio/wav")



@app.route("/speakers", methods=['GET'])
def speakers():

    voices = []

    for x in default_voices:
        voices.append({"name":x,"voice_id":x})

    for name in os.listdir("voices"):
        name = name.replace(".pt","")
        voices.append({"name":name,"voice_id":name})

    response = app.response_class(
        response=json.dumps(voices),
        status=200,
        mimetype='application/json'
    )
    return response


@app.route("/speakers_list", methods=['GET'])
def speakers_list():

    response = app.response_class(
        response=json.dumps(["female_calm","female","male"]),
        status=200,
        mimetype='application/json'
    )
    return response


@app.route('/file/<filename>')
def uploaded_file(filename):
    return send_from_directory("音频输出", filename)
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9880)
