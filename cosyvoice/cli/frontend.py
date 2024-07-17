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
from functools import partial
import onnxruntime
import torch
import numpy as np
import whisper
from typing import Callable
import torchaudio.compliance.kaldi as kaldi
import torchaudio
import os
import inflect
# import ttsfrd
from cosyvoice.utils.frontend_utils import contains_chinese, replace_blank, replace_corner_mark, remove_bracket, spell_out_number, split_paragraph
import re
import jieba.posseg as pseg

def text_normalize(text):
    """
    对文本进行归一化处理
    :param text:
    :return:
    """
    from .zh_normalization import TextNormalizer
    # ref: https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization
    tx = TextNormalizer()
    sentences = tx.normalize(text)
    # print(sentences)

    _txt = ''.join(sentences)
    # 替换掉除中文之外的所有字符
    # _txt = re.sub(
    #     r"[^\u4e00-\u9fa5，。！？、]+", "", _txt
    # )

    return _txt

def remove_chinese_punctuation(text):
    """
    移除文本中的中文标点符号 [：；！（），【】『』「」《》－‘“’”:,;!\(\)\[\]><\-] 替换为 ，
    :param text:
    :return:
    """
    chinese_punctuation_pattern = r"[：；！（），【】『』「」《》－‘“’”:,;!\(\)\[\]><\-·]"
    text = re.sub(chinese_punctuation_pattern, '，', text)
    # 使用正则表达式将多个连续的句号替换为一个句号
    text = re.sub(r'[。，]{2,}', '。', text)
    # 删除开头和结尾的 ， 号
    text = re.sub(r'^，|，$', '', text)
    return text

def normalize_zh(text):
    return process_ddd(text_normalize(remove_chinese_punctuation(text)))


def process_ddd(text):
    """
    处理“地”、“得” 字的使用，都替换为“的”
    依据：地、得的使用，主要是在动词和形容词前后，本方法没有严格按照语法替换，因为时常遇到用错的情况。
    另外受 jieba 分词准确率的影响，部分情况下可能会出漏掉。例如：小红帽疑惑地问
    :param text: 输入的文本
    :return: 处理后的文本
    """
    word_list = [(word, flag) for word, flag in pseg.cut(text, use_paddle=False)]
    # print(word_list)
    processed_words = []
    for i, (word, flag) in enumerate(word_list):
        if word in ["地", "得"]:
            # Check previous and next word's flag
            # prev_flag = word_list[i - 1][1] if i > 0 else None
            # next_flag = word_list[i + 1][1] if i + 1 < len(word_list) else None

            # if prev_flag in ['v', 'a'] or next_flag in ['v', 'a']:
            if flag in ['uv', 'ud']:
                processed_words.append("的")
            else:
                processed_words.append(word)
        else:
            processed_words.append(word)

    return ''.join(processed_words)


class CosyVoiceFrontEnd:

    def __init__(self,
                 get_tokenizer: Callable,
                 feat_extractor: Callable,
                 campplus_model: str,
                 speech_tokenizer_model: str,
                 spk2info: str = '',
                 instruct: bool = False,
                 allowed_special: str = 'all'):
        self.tokenizer = get_tokenizer()
        self.feat_extractor = feat_extractor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 4
        self.campplus_session = onnxruntime.InferenceSession(campplus_model, sess_options=option, providers=["CPUExecutionProvider"])
        self.speech_tokenizer_session = onnxruntime.InferenceSession(speech_tokenizer_model, sess_options=option, providers=["CUDAExecutionProvider"if torch.cuda.is_available() else "CPUExecutionProvider"])
        if os.path.exists(spk2info):
            self.spk2info = torch.load(spk2info, map_location=self.device)
        self.instruct = instruct
        self.allowed_special = allowed_special
        self.inflect_parser = inflect.engine()
        self.frd = None
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


    def _extract_text_token(self, text):
        text_token = self.tokenizer.encode(text, allowed_special=self.allowed_special)
        text_token = torch.tensor([text_token], dtype=torch.int32).to(self.device)
        text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(self.device)
        return text_token, text_token_len

    def _extract_speech_token(self, speech):
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        speech_token = self.speech_tokenizer_session.run(None, {self.speech_tokenizer_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                                                                self.speech_tokenizer_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()
        speech_token = torch.tensor([speech_token], dtype=torch.int32).to(self.device)
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(self.device)
        return speech_token, speech_token_len

    def _extract_spk_embedding(self, speech):
        feat = kaldi.fbank(speech,
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_session.run(None, {self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        embedding = torch.tensor([embedding]).to(self.device)
        return embedding

    def _extract_speech_feat(self, speech):
        speech_feat = self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(self.device)
        return speech_feat, speech_feat_len

    def text_normalize(self, text, split=True):
        text = text.strip()
        if contains_chinese(text):
            # text = self.frd.get_frd_extra_info(text, 'input').replace("\n", "")
            text += '.。'
            text = text.replace("\n", "")
            text = normalize_zh(text)
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = text.replace(".", "、")
            text = text.replace(" - ", "，")
            text = remove_bracket(text)
            texts = [i for i in split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "zh", token_max_n=30,
                                                token_min_n=20, merge_len=5,
                                                comma_split=False)]
        else:
            text += '.'
            text = spell_out_number(text, self.inflect_parser)
            texts = [i for i in split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "en", token_max_n=30,
                                                token_min_n=20, merge_len=5,
                                                comma_split=False)]
        if split is False:
            return text
        return texts

    def text_normalize_stream(self, text, split=True):
        text = text.strip()
        if contains_chinese(text):
            # text = self.frd.get_frd_extra_info(text, 'input').replace("\n", "")
            text += '.。'
            text = text.replace("\n", "")
            text = normalize_zh(text)
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = text.replace(".", "、")
            text = text.replace(" - ", "，")
            text = remove_bracket(text)
            texts = [i for i in split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "zh", token_max_n=30,
                                                token_min_n=20, merge_len=5,
                                                comma_split=True)]
        else:
            text += '.'
            text = spell_out_number(text, self.inflect_parser)
            texts = [i for i in split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "en", token_max_n=30,
                                                token_min_n=20, merge_len=5,
                                                comma_split=True)]
        if split is False:
            return text
        return texts

    def text_normalize_instruct(self, text, split=True):
        text = text.strip()
        if contains_chinese(text):
            # text = self.frd.get_frd_extra_info(text, 'input').replace("\n", "")
            text += '.。'
            text = text.replace("\n", "")
            # text = normalize_zh(text)
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = text.replace(".", "、")
            text = text.replace(" - ", "，")
            text = remove_bracket(text)
            texts = [i for i in split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "zh", token_max_n=30,
                                                token_min_n=20, merge_len=5,
                                                comma_split=False)]
        else:
            text += '.'
            text = spell_out_number(text, self.inflect_parser)
            texts = [i for i in split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "en", token_max_n=30,
                                                token_min_n=20, merge_len=5,
                                                comma_split=False)]
        if split is False:
            return text
        return texts

    def frontend_sft(self, tts_text, spk_id):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        embedding = self.spk2info[spk_id]['embedding']
        model_input = {'text': tts_text_token, 'text_len': tts_text_token_len, 'llm_embedding': embedding, 'flow_embedding': embedding}
        return model_input

    def frontend_zero_shot(self, tts_text, prompt_text, prompt_speech_16k):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
        prompt_speech_22050 = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)(prompt_speech_16k)
        speech_feat, speech_feat_len = self._extract_speech_feat(prompt_speech_22050)
        speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        model_input = {'text': tts_text_token, 'text_len': tts_text_token_len,
                       'prompt_text': prompt_text_token, 'prompt_text_len': prompt_text_token_len,
                       'llm_prompt_speech_token': speech_token, 'llm_prompt_speech_token_len': speech_token_len,
                       'flow_prompt_speech_token': speech_token, 'flow_prompt_speech_token_len': speech_token_len,
                       'prompt_speech_feat': speech_feat, 'prompt_speech_feat_len': speech_feat_len,
                       'llm_embedding': embedding, 'flow_embedding': embedding}
        return model_input

    def frontend_cross_lingual(self, tts_text, prompt_speech_16k):
        model_input = self.frontend_zero_shot(tts_text, '', prompt_speech_16k)
        # in cross lingual mode, we remove prompt in llm
        del model_input['prompt_text']
        del model_input['prompt_text_len']
        del model_input['llm_prompt_speech_token']
        del model_input['llm_prompt_speech_token_len']
        return model_input

    def frontend_instruct(self, tts_text, spk_id, instruct_text):
        model_input = self.frontend_sft(tts_text, spk_id)
        # in instruct mode, we remove spk_embedding in llm due to information leakage
        del model_input['llm_embedding']
        instruct_text_token, instruct_text_token_len = self._extract_text_token(instruct_text + '<endofprompt>')
        model_input['prompt_text'] = instruct_text_token
        model_input['prompt_text_len'] = instruct_text_token_len
        return model_input
