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


def ms_to_srt_time(ms):
    N = int(ms)
    hours, remainder = divmod(N, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, milliseconds = divmod(remainder, 1000)
    timesrt = f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    # print(timesrt)
    return timesrt

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
    def inference_sft(self, tts_text, spk_id,new_dropdown,spk_mix="无",w1=0.5,w2=0.5,token_max_n=30,token_min_n=20,merge_len=15):

        default_voices = ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']

        # if new_dropdown != "无":
        #     spk_id = "中文女"


        tts_speeches = []
        audio_opt = []
        audio_samples = 0
        srtlines = []
        for i in self.frontend.text_normalize(tts_text,True,token_max_n,token_min_n,merge_len):
            if spk_id not in default_voices:
                back_spk_id = "中文女"
            else:
                back_spk_id = spk_id
                
            model_input = self.frontend.frontend_sft(i, back_spk_id)
            #print(model_input)
            print(i)
            # with open(r'srt_model_input.txt', 'a',encoding='utf-8') as f:
            #     f.write(str(model_input))
            if new_dropdown != "无" or spk_id not in default_voices:
                # 加载数据
                if spk_id not in default_voices:
                    new_dropdown = spk_id
                print(new_dropdown)
                print("读取pt")

                newspk = torch.load(f'./voices/{new_dropdown}.pt')

                

                if spk_mix != "无":

                    print("融合音色:",spk_mix)
                    
                    if spk_mix not in ["中文女","中文男","中文男","日语男","粤语女","粤语女","英文女","英文男","韩语女"]:

                        newspk_1 = torch.load(f'./voices/{spk_mix}.pt')
                    else:
                        newspk_1 = self.frontend.frontend_sft(i, spk_mix)



                    model_input["flow_embedding"] = (newspk["flow_embedding"] * w1) + (newspk_1["flow_embedding"] * w2)
                    # model_input["llm_embedding"] = (newspk["llm_embedding"] * w1) + (newspk_1["llm_embedding"] * w2)

                else:

                    model_input["flow_embedding"] = newspk["flow_embedding"] 
                    model_input["llm_embedding"] = newspk["llm_embedding"]


                # with open(f'./voices/{new_dropdown}.py','r',encoding='utf-8') as f:
                #     newspk = f.read()
                #     newspk = eval(newspk)
                # model_input["flow_embedding"] = newspk["flow_embedding"] * 0.1 + newspk_1["flow_embedding"] * 0.9
                # model_input["llm_embedding"] = newspk["llm_embedding"] * 0.1 + newspk_1["llm_embedding"] * 0.9

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

            print(model_output['tts_speech'])

            # 使用 .numpy() 方法将 tensor 转换为 numpy 数组
            numpy_array = model_output['tts_speech'].numpy()
            # 使用 np.ravel() 方法将多维数组展平成一维数组
            audio = numpy_array.ravel()
            print(audio)
            srtline_begin=ms_to_srt_time(audio_samples*1000.0 / 22050)
            audio_samples += audio.size
            srtline_end=ms_to_srt_time(audio_samples*1000.0 / 22050)
            audio_opt.append(audio)

            srtlines.append(f"{len(audio_opt):02d}\n")
            srtlines.append(srtline_begin+' --> '+srtline_end+"\n")

            srtlines.append(i.replace("、。","")+"\n\n")

            
            
            tts_speeches.append(model_output['tts_speech'])

        print(tts_speeches)
        audio_data = torch.concat(tts_speeches, dim=1)

        torchaudio.save("音频输出/output.wav", audio_data, 22050)
        with open('音频输出/output.srt', 'w', encoding='utf-8') as f:
            f.writelines(srtlines)

        return {'tts_speech':audio_data}

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
