# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import torch
import torchaudio
torchaudio.set_audio_backend('soundfile')
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel
import time


def time_it(func):
  """
  这是一个装饰器，用来计算类方法运行的时长，单位秒.
  """
  def wrapper(self, *args, **kwargs):
    start_time = time.time()
    result = func(self, *args, **kwargs)
    end_time = time.time()
    duration = end_time - start_time
    print(f"推理方法 {func.__name__} 运行时长: {duration:.4f} 秒")
    return result
  return wrapper

class CosyVoice:

    def __init__(self, model_dir):
        instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v1.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          instruct,
                                          configs['allowed_special'])
        self.model = CosyVoiceModel(configs['llm'], configs['flow'], configs['hift'])
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        del configs

    def list_avaliable_spks(self):
        # print(self.frontend.spk2info)
        # with open(r'./spk2info.txt', 'a',encoding='utf-8') as f:
        #     f.write(str(self.frontend.spk2info))
        spks = list(self.frontend.spk2info.keys())
        return spks

    def inference_sft_stream(self, tts_text, spk_id,new_dropdown):
        if new_dropdown != "无":
            spk_id = "中文女"
        tts_speeches = []

        joblist = self.frontend.text_normalize_stream(tts_text, split=True)
        
        for i in joblist:
            
            # model_input = self.frontend.frontend_sft(i, spk_id)
            #print(model_input)
            print(i)
            # with open(r'srt_model_input.txt', 'a',encoding='utf-8') as f:
            #     f.write(str(model_input))
            # if new_dropdown != "无":
            #     # 加载数据
            #     print(new_dropdown)
            #     print("读取pt")
            #     newspk = torch.load(f'./voices/{new_dropdown}.pt')
            #     # with open(f'./voices/{new_dropdown}.py','r',encoding='utf-8') as f:
            #     #     newspk = f.read()
            #     #     newspk = eval(newspk)
            #     model_input["flow_embedding"] = newspk["flow_embedding"]
            #     model_input["llm_embedding"] = newspk["llm_embedding"]

            #     model_input["llm_prompt_speech_token"] = newspk["llm_prompt_speech_token"]
            #     model_input["llm_prompt_speech_token_len"] = newspk["llm_prompt_speech_token_len"]

            #     model_input["flow_prompt_speech_token"] = newspk["flow_prompt_speech_token"]
            #     model_input["flow_prompt_speech_token_len"] = newspk["flow_prompt_speech_token_len"]

            #     model_input["prompt_speech_feat_len"] = newspk["prompt_speech_feat_len"]
            #     model_input["prompt_speech_feat"] = newspk["prompt_speech_feat"]
            #     model_input["prompt_text"] = newspk["prompt_text"]
            #     model_input["prompt_text_len"] = newspk["prompt_text_len"]

            # model_output = next(self.model.inference_stream(**model_input))
            # # print(model_input)
            # tts_speeches.append(model_output['tts_speech'])
            # print(tts_speeches)
            yield i
            # try:
            #     model_output = next(self.model.inference_stream(**model_input))
            #     print(f"Model output: {model_output}")
            #     tts_speeches.append(model_output['tts_speech'])
            #     # yield torch.concat(tts_speeches, dim=1)
            # except StopIteration:
            #     print("Inference stream ended.")
            #     break
            # except Exception as e:
            #     print(f"Error during inference: {e}")
            #     break


    @time_it
    def inference_sft(self, tts_text, spk_id,new_dropdown):
        if new_dropdown != "无":
            spk_id = "中文女"
        tts_speeches = []
        for i in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_sft(i, spk_id)
            #print(model_input)
            print(i)
            # with open(r'srt_model_input.txt', 'a',encoding='utf-8') as f:
            #     f.write(str(model_input))
            if new_dropdown != "无":
                # 加载数据
                print(new_dropdown)
                print("读取pt")
                newspk = torch.load(f'./voices/{new_dropdown}.pt')
                # with open(f'./voices/{new_dropdown}.py','r',encoding='utf-8') as f:
                #     newspk = f.read()
                #     newspk = eval(newspk)
                model_input["flow_embedding"] = newspk["flow_embedding"]
                model_input["llm_embedding"] = newspk["llm_embedding"]

                model_input["llm_prompt_speech_token"] = newspk["llm_prompt_speech_token"]
                model_input["llm_prompt_speech_token_len"] = newspk["llm_prompt_speech_token_len"]

                model_input["flow_prompt_speech_token"] = newspk["flow_prompt_speech_token"]
                model_input["flow_prompt_speech_token_len"] = newspk["flow_prompt_speech_token_len"]

                model_input["prompt_speech_feat_len"] = newspk["prompt_speech_feat_len"]
                model_input["prompt_speech_feat"] = newspk["prompt_speech_feat"]
                model_input["prompt_text"] = newspk["prompt_text"]
                model_input["prompt_text_len"] = newspk["prompt_text_len"]

            model_output = self.model.inference(**model_input)
            # print(model_input)
            
            tts_speeches.append(model_output['tts_speech'])

        return {'tts_speech': torch.concat(tts_speeches, dim=1)}

    def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False)
        tts_speeches = []
        for i in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k)
            # save_input = {}
            # save_input["flow_embedding"] = model_input["flow_embedding"]
            # save_input["llm_embedding"] = model_input["llm_embedding"]
            # with open(r'./output.py', 'w',encoding='utf-8') as f:
            #     f.write(str(model_input))

            # 保存数据
            torch.save(model_input, 'output.pt') 
            model_output = self.model.inference(**model_input)
            tts_speeches.append(model_output['tts_speech'])

        return {'tts_speech': torch.concat(tts_speeches, dim=1)}

    def inference_cross_lingual(self, tts_text, prompt_speech_16k):
        if self.frontend.instruct is True:
            raise ValueError('{} do not support cross_lingual inference'.format(self.model_dir))
        tts_speeches = []
        for i in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_cross_lingual(i, prompt_speech_16k)
            model_output = self.model.inference(**model_input)
            tts_speeches.append(model_output['tts_speech'])

        return {'tts_speech': torch.concat(tts_speeches, dim=1)}

    def inference_instruct(self, tts_text, spk_id, instruct_text,new_dropdown):

        if new_dropdown != "无":
            spk_id = "中文女"

        if self.frontend.instruct is False:
            raise ValueError('{} do not support instruct inference'.format(self.model_dir))
        instruct_text = self.frontend.text_normalize_instruct(instruct_text, split=False)
        tts_speeches = []

    
        for i in self.frontend.text_normalize_instruct(tts_text, split=True):
            model_input = self.frontend.frontend_instruct(i, spk_id, instruct_text)

            if new_dropdown != "无":
                # 加载数据
                print(new_dropdown)
                print("读取pt")
                newspk = torch.load(f'./voices/{new_dropdown}.pt')
                # with open(f'./voices/{new_dropdown}.py','r',encoding='utf-8') as f:
                #     newspk = f.read()
                #     newspk = eval(newspk)
                model_input["flow_embedding"] = newspk["flow_embedding"]
                model_input["llm_embedding"] = newspk["llm_embedding"]

                model_input["llm_prompt_speech_token"] = newspk["llm_prompt_speech_token"]
                model_input["llm_prompt_speech_token_len"] = newspk["llm_prompt_speech_token_len"]

                model_input["flow_prompt_speech_token"] = newspk["flow_prompt_speech_token"]
                model_input["flow_prompt_speech_token_len"] = newspk["flow_prompt_speech_token_len"]

                model_input["prompt_speech_feat_len"] = newspk["prompt_speech_feat_len"]
                model_input["prompt_speech_feat"] = newspk["prompt_speech_feat"]
                model_input["prompt_text"] = newspk["prompt_text"]
                model_input["prompt_text_len"] = newspk["prompt_text_len"]
            model_output = self.model.inference(**model_input)
            
            tts_speeches.append(model_output['tts_speech'])

        return {'tts_speech': torch.concat(tts_speeches, dim=1)}
