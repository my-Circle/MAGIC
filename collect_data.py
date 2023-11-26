import argparse
import os
import string
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from dataset import hierarchical_dataset, AlignCollate
# from Datasets import hierarchical_dataset, AlignCollate
# from seqda_model import Model
from utils import AttnLabelConverter, Averager, TokenLabelConverter
from utils import load_char_dict, compute_loss

import torch.nn.functional as F
from losses.CCLoss import CCLoss
from models import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load(model, saved_model):
    params = torch.load(saved_model)

    if 'model' not in params: #baseline
        model.load_state_dict(params) #,strict=False
    else:  #adapt
        model.load_state_dict(params['model']) #,strict=False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data', required=True, help='path to evaluation dataset')
    parser.add_argument('--benchmark_all_eval', action='store_true',
                        help='evaluate 10 benchmark evaluation datasets')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    parser.add_argument('--visualize', action='store_true', help='use rgb input')
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=128, help='the width of the input image') #default=100
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--char_dict', type=str, default=None,
                        help="path to char dict dataset/iam/char_dict.txt")
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz',
                        help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--ignore_special_char', action='store_true',
                        help='for evaluation mode, ignore special char')
    parser.add_argument('--ignore_case_sensitive', action='store_true',
                        help='for evaluation mode, ignore sensitive character')
    parser.add_argument('--PAD', action='store_true',
                        help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true',
                        help='for data_filtering_off mode')
    # """ Model Architecture """
    # parser.add_argument('--Transformation', type=str, required=True,
    #                     help='Transformation stage. None|TPS')
    # parser.add_argument('--FeatureExtraction', type=str, required=True,
    #                     help='FeatureExtraction stage. VGG|RCNN|ResNet')
    # parser.add_argument('--SequenceModeling', type=str, required=True,
    #                     help='SequenceModeling stage. None|BiLSTM')
    # parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    # parser.add_argument('--num_fiducial', type=int, default=20,
    #                     help='number of fiducial points of TPS-STN')
    # parser.add_argument('--input_channel', type=int, default=1,
    #                     help='the number of input channel of Feature extractor')
    # parser.add_argument('--output_channel', type=int, default=512,
    #                     help='the number of output channel of Feature extractor')
    # parser.add_argument('--hidden_size', type=int, default=256,
    #                     help='the size of the LSTM hidden state')
    """ Model Architecture """
    parser.add_argument('--Transformer', type=str, required=True, help='Transformer stage. mgp-str|char-str')

    choices = ["mgp_str_base_patch4_3_32_128", "mgp_str_tiny_patch4_3_32_128",
               "mgp_str_small_patch4_3_32_128", "char_str_base_patch4_3_32_128"]
    parser.add_argument('--TransformerModel', default='', help='Which mgp_str transformer model', choices=choices)
    parser.add_argument('--Transformation', type=str, default='', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='',
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='', help='Prediction stage. None|CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=3,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
    if opt.char_dict is not None:
        opt.character = load_char_dict(opt.char_dict)[3:-2]  # 去除Attention 和 CTC引入的一些特殊符号
    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    """ model configuration """
    converter = TokenLabelConverter(opt)
    opt.num_class = len(converter.character)  # 38

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)  # seqda_model
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel,
          opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation,
          opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    # model.load_state_dict(torch.load(opt.saved_model))
    load(model, opt.saved_model)

    """ evaluation """
    model.eval()
    AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW,
                                           keep_ratio_with_pad=opt.PAD)
    eval_data = hierarchical_dataset(root=opt.eval_data, opt=opt)
    evaluation_loader = torch.utils.data.DataLoader(
        eval_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_evaluation, pin_memory=True)
    data_embed_collect = []
    label_collect = []
    for i, (image_tensors, labels) in enumerate(evaluation_loader):  #192,1,32,100
        batch_size = image_tensors.size(0)
        image = image_tensors.to(device)
        # For max length prediction
        target = converter.encode(labels)  # 用tokenizer解析labels,进入utils.py,target(bs,max_seq_len)=(192,27)
        len_target, char_text = converter.char_encode(labels)
        bpe_text = converter.bpe_encode(labels)
        wp_text = converter.wp_encode(labels)

        start_time = time.time()

        #_, char_preds, bpe_preds, wp_preds = model(image, is_eval=True)

        char_preds, bpe_preds, wp_preds, global_feature, local_feature, local_feature_bpe, local_feature_wp = model(image)

        forward_time = time.time() - start_time

        # target = text_for_loss[:, 1:]
        # cost = criterion(char_preds.contiguous().view(-1, char_preds.shape[-1]),
        #                  target.contiguous().view(-1))  # char_preds:192*26,38  target:192*26
        label = wp_text.contiguous().view(-1) #B*T
        data = local_feature_wp.contiguous().view(-1, local_feature_wp.shape[-1]) #B*T,D
        data_embed_collect.append(data)
        label_collect.append(label)
        if i==4:
            break
    data_embed_npy = torch.cat(data_embed_collect, axis=0).cpu().detach().numpy()
    label_npu = torch.cat(label_collect, axis=0).cpu().numpy()
    # print(data_embed_npy.shape)
    # print(label_npu.shape)
    np.save("data_embed_npy6.npy", data_embed_npy)
    np.save("label_npu6.npy", label_npu)



# data_embed_collect = []
# label_collect = []
#
# for ......
#     # inputs.shape=[BS,C,H,W]
#     # embed_4096.shape=[BS,4096]
#     # output.shape=[BS,1000]
#     output, embed_4096 = model(inputs)
#
#     data_embed_collect.append(embed_4096)
#     label_collect.append(label)
#     ......
#
# # data_embed_collect.shape=[iters,BS,4096]
# # label_collect.shape=[iters,BS,]
#
# # 在这里，所有样本的4096特征都收集了，并且每个样本的标签也收集了
# # data_embed_npy.shape=[samples,4096]
# # label_npu.shape=[samples,]
# data_embed_npy = torch.cat(data_embed_collect, axis=0).cpu().numpy()
# label_npu = torch.cat(label_collect, axis=0).cpu().numpy()
#
# np.save("data_embed_npy.npy", data_embed_npy)
# np.save("label_npu.npy", label_npu).
