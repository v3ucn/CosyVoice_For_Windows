
import time
import io, os, sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/AcademiCodec'.format(ROOT_DIR))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

import numpy as np
from flask import Flask, request, Response
import torch
import torchaudio

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio

from flask_cors import CORS

import json

cosyvoice = CosyVoice('./pretrained_models/CosyVoice-300M-SFT')

spk_new = []

for name in os.listdir("./voices/"):
    print(name.replace(".py",""))
    spk_new.append(name.replace(".py",""))

print("默认音色",cosyvoice.list_avaliable_spks())
print("自定义音色",spk_new)

app = Flask(__name__)

CORS(app, cors_allowed_origins="*")

CORS(app, supports_credentials=True)

@app.route("/", methods=['POST'])
def sft_post():
    question_data = request.get_json()

    text = question_data.get('text')
    speaker = question_data.get('speaker')
    new = question_data.get('new',0)
    

    if not text:
        return {"error": "文本不能为空"}, 400

    if not speaker:
        return {"error": "角色名不能为空"}, 400

    start = time.process_time()
    if not new:
        output = cosyvoice.inference_sft(text,speaker,"无")
    else:
        output = cosyvoice.inference_sft(text,speaker,speaker)
    end = time.process_time()
    print("infer time:", end - start)
    buffer = io.BytesIO()
    torchaudio.save(buffer, output['tts_speech'], 22050, format="wav")
    buffer.seek(0)
    return Response(buffer.read(), mimetype="audio/wav")


@app.route("/", methods=['GET'])
def sft_get():

    text = request.args.get('text')
    speaker = request.args.get('speaker')
    new = request.args.get('new',0)

    if not text:
        return {"error": "文本不能为空"}, 400

    if not speaker:
        return {"error": "角色名不能为空"}, 400

    start = time.process_time()
    if not new:
        output = cosyvoice.inference_sft(text,speaker,"无")
    else:
        output = cosyvoice.inference_sft(text,speaker,speaker)
    end = time.process_time()
    print("infer time:", end - start)
    buffer = io.BytesIO()
    torchaudio.save(buffer, output['tts_speech'], 22050, format="wav")
    buffer.seek(0)
    return Response(buffer.read(), mimetype="audio/wav")




@app.route("/tts_to_audio/", methods=['POST'])
def tts_to_audio():

    import speaker_config
    
    question_data = request.get_json()

    text = question_data.get('text')
    speaker = speaker_config.speaker
    new = speaker_config.new
    

    if not text:
        return {"error": "文本不能为空"}, 400

    if not speaker:
        return {"error": "角色名不能为空"}, 400

    start = time.process_time()
    if not new:
        output = cosyvoice.inference_sft(text,speaker,"无")
    else:
        output = cosyvoice.inference_sft(text,speaker,speaker)
    end = time.process_time()
    print("infer time:", end - start)
    buffer = io.BytesIO()
    torchaudio.save(buffer, output['tts_speech'], 22050, format="wav")
    buffer.seek(0)
    return Response(buffer.read(), mimetype="audio/wav")



@app.route("/speakers", methods=['GET'])
def speakers():

    response = app.response_class(
        response=json.dumps([{"name":"default","vid":1}]),
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
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9880)
