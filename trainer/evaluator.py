
import torch
import torch.nn as nn

import logging

from tqdm import tqdm
import sacrebleu

from datasets import chinese_tokenizer_load, english_tokenizer_load, beam_decoder, beam_search
from utils import config

def evaluate(data, model):
    """在data上用训练好的模型进行预测，打印模型翻译结果"""
    sp_chn = chinese_tokenizer_load()  # 加载中文分词器
    trg = []  # 存储目标句子（真实句子）
    res = []  # 存储模型翻译的结果
    with torch.no_grad():  # 禁用梯度计算，节省内存和计算
        # 在data的英文数据长度上遍历下标
        for batch in tqdm(data):  # 使用tqdm显示进度条
            cn_sent = batch.trg_text  # 获取当前批次的中文句子
            src = batch.src   # 获取当前批次的源语言（英文）句子
            src_mask = (src != 0).unsqueeze(-2)    # 为源语言句子创建mask，排除padding部分

            # 使用束搜索生成模型翻译结果
            decode_result, _ = beam_search(model, src, src_mask, config.max_len,
                                               config.padding_idx, config.bos_idx, config.eos_idx,
                                               config.beam_size, config.device)

            # `decode_result`是一个包含多个翻译结果的列表，取最优结果
            decode_result = [h[0] for h in decode_result]
            # 解码后的id转为中文句子
            translation = [sp_chn.decode_ids(_s) for _s in decode_result]
            trg.extend(cn_sent)  # 将当前批次的真实句子添加到`trg`中
            res.extend(translation)  # 将模型的翻译结果添加到`res`中

    # 计算BLEU分数，使用SacreBLEU工具库
    trg = [trg]  # 真实目标句子
    bleu = sacrebleu.corpus_bleu(res, trg, tokenize='zh')  # 计算BLEU分数
    return float(bleu.score)  # 返回BLEU分数