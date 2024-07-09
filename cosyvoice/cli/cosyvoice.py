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

    def inference_sft(self, tts_text, spk_id,new_dropdown):
        if new_dropdown != "无":
            spk_id = "中文女"
        tts_speeches = []
        for i in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_sft(i, spk_id)
            print(i)
            if new_dropdown != "无":
                with open(f'./voices/{new_dropdown}.py','r',encoding='utf-8') as f:
                    newspk = f.read()
                    newspk = eval(newspk.replace("tensor","torch.tensor"))
                model_input["flow_embedding"] = newspk["flow_embedding"]
                model_input["llm_embedding"] = newspk["llm_embedding"]

            model_output = self.model.inference(**model_input)
            print(model_input)
            tts_speeches.append(model_output['tts_speech'])
        self.model.clear_cache()
        return {'tts_speech': torch.concat(tts_speeches, dim=1)}

    def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False)
        tts_speeches = []
        for i in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k)
            save_input = {}
            save_input["flow_embedding"] = model_input["flow_embedding"]
            save_input["llm_embedding"] = model_input["llm_embedding"]
            with open(r'./output.py', 'w',encoding='utf-8') as f:
                f.write(str(save_input))
            model_output = self.model.inference(**model_input)
            tts_speeches.append(model_output['tts_speech'])
        self.model.clear_cache()
        return {'tts_speech': torch.concat(tts_speeches, dim=1)}

    def inference_cross_lingual(self, tts_text, prompt_speech_16k):
        if self.frontend.instruct is True:
            raise ValueError('{} do not support cross_lingual inference'.format(self.model_dir))
        tts_speeches = []
        for i in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_cross_lingual(i, prompt_speech_16k)
            model_output = self.model.inference(**model_input)
            tts_speeches.append(model_output['tts_speech'])
        self.model.clear_cache()
        return {'tts_speech': torch.concat(tts_speeches, dim=1)}

    def inference_instruct(self, tts_text, spk_id, instruct_text):
        if self.frontend.instruct is False:
            raise ValueError('{} do not support instruct inference'.format(self.model_dir))
        instruct_text = self.frontend.text_normalize(instruct_text, split=False)
        tts_speeches = []
        for i in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_instruct(i, spk_id, instruct_text)
            model_output = self.model.inference(**model_input)
            tts_speeches.append(model_output['tts_speech'])
        self.model.clear_cache()
        return {'tts_speech': torch.concat(tts_speeches, dim=1)}
