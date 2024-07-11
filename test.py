import os
# os.environ['MODELSCOPE_CACHE'] ='./.cache/modelscope'
# os.environ['TORCH_HOME'] = './.cache/torch'  #设置torch的缓存目录
# os.environ["HF_HOME"] = "./.cache/huggingface" #设置transformer的缓存目录
# os.environ['XDG_CACHE_HOME']="./.cache"
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/AcademiCodec'.format(ROOT_DIR))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio

cosyvoice = CosyVoice('./pretrained_models/CosyVoice-300M-SFT')
# sft usage
print(cosyvoice.list_avaliable_spks())
output = cosyvoice.inference_sft('你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？', '中文女',"无")
torchaudio.save('sft.wav', output['tts_speech'], 22050)