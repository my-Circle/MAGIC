# coding='utf-8'
"""t-SNE 对手写数字进行可视化"""
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE

import argparse
import os
import random
import string
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data

from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
# from Datasets import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
# import copy
from modules.domain_adapt import d_cls_inst
from modules.radam import AdamW, RAdam
# from seqda_model import Model
from test import validation
from utils import AttnLabelConverter, Averager, load_char_dict, TokenLabelConverter
from losses.CCLoss import CCLoss
from models import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_data():
    digits = datasets.load_digits(n_class=6)
    data = digits.data  #B,C=1083,64
    label = digits.target  #B,=1083
    n_samples, n_features = data.shape  #B=1083,C=64
    return data, label, n_samples, n_features


def plot_embedding(data1, data2, title):
    data = np.concatenate((data1, data2), axis=0)
    x_min, x_max = np.min(data, 0), np.max(data, 0)

    # x_min_1, x_max_1 = np.min(data1, 0), np.max(data1, 0)
    data1 = (data1 - x_min) / (x_max - x_min)

    # x_min_2, x_max_2 = np.min(data2, 0), np.max(data2, 0)
    data2 = (data2 - x_min) / (x_max - x_min)

    fig = plt.figure(frameon=False)
    ax = plt.subplot(111)
    for i in range(data1.shape[0]):
        plt.plot(data1[i, 0], data1[i, 1], marker='o', markersize=1, color='r')  #source
        plt.plot(data2[i, 0], data2[i, 1], marker='o', markersize=1, color='g')  #target
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

def load(model, saved_model):
    params = torch.load(saved_model)

    if 'model' not in params: #baseline
        model.load_state_dict(params) #,strict=False
    else:  #adapt
        model.load_state_dict(params['model']) #,strict=False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', help='Where to store logs and models')
    parser.add_argument('--src_train_data', required=True, help='path to training dataset')
    parser.add_argument('--tar_train_data', required=True, help='path to training dataset')
    parser.add_argument('--valid_data', help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000,
                        help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=500,
                        help='Interval between each validation')
    parser.add_argument('--saved_model', required=True, default='', help="path to saved_model to evaluation")
    parser.add_argument('--adam', action='store_true',
                        help='Whether to use adam (default is Adadelta)')

    """ Data processing """
    parser.add_argument('--src_select_data', type=str, default='MJ-ST',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--src_batch_ratio', type=str, default='0.5-0.5',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--tar_select_data', type=str, default='real_data',
                        help='select training data (default is real_data, which means MJ and ST used as training data)')
    parser.add_argument('--tar_batch_ratio', type=str, default='1',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=128, help='the width of the input image')  # default=100
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--char_dict', type=str, default=None,
                        help="path to char dict: dataset/iam/char_dict.txt")
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz',
                        help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--filtering_special_chars', action='store_true',
                        help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true',
                        help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true',
                        help='for data_filtering_off mode')

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
    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    # print('device count', opt.num_gpu)
    if opt.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        opt.workers = opt.workers * opt.num_gpu

        """ previous version
        print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', opt.batch_size)
        opt.batch_size = opt.batch_size * opt.num_gpu
        print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
        If you dont care about it, just commnet out these line.)
        opt.num_iter = int(opt.num_iter / opt.num_gpu)
        """

    opt.src_select_data = opt.src_select_data.split('-')
    opt.src_batch_ratio = opt.src_batch_ratio.split('-')
    opt.tar_select_data = opt.tar_select_data.split('-')
    opt.tar_batch_ratio = opt.tar_batch_ratio.split('-')

    # opt.valid_select_data = opt.valid_select_data.split('-')


    """ model configuration """
    converter = TokenLabelConverter(opt)
    opt.num_class = len(converter.character)

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
    src_train_data = opt.src_train_data
    src_select_data = opt.src_select_data
    src_batch_ratio = opt.src_batch_ratio
    src_dataset = Batch_Balanced_Dataset(opt, src_train_data, src_select_data,
                                               src_batch_ratio)

    tar_train_data = opt.tar_train_data
    tar_select_data = opt.tar_select_data
    tar_batch_ratio = opt.tar_batch_ratio
    tar_dataset = Batch_Balanced_Dataset(opt, tar_train_data, tar_select_data,
                                               tar_batch_ratio)
    model.eval()
    start_iter = 0
    source_bpe = torch.empty(opt.batch_size*27, 192) #384 768
    source_wp = torch.empty(opt.batch_size*27, 192)
    target_bpe = torch.empty(opt.batch_size*27, 192)
    target_wp = torch.empty(opt.batch_size*27, 192)
    for step in range(start_iter, opt.num_iter + 1):  #192,1,32,100
        src_image, src_labels = src_dataset.get_batch()
        src_image = src_image.to(device)
        # src_text, src_length = self.converter.encode(src_labels,
        #                                              batch_max_length=opt.batch_max_length)
        len_target, char_src_text = converter.char_encode(src_labels)
        bpe_src_text = converter.bpe_encode(src_labels)
        wp_src_text = converter.wp_encode(src_labels)

        tar_image, tar_labels = tar_dataset.get_batch()
        tar_image = tar_image.to(device)
        # tar_text, tar_length = self.converter.encode(tar_labels,
        #                                              batch_max_length=opt.batch_max_length)
        len_target, char_tar_text = converter.char_encode(tar_labels)
        bpe_tar_text = converter.bpe_encode(tar_labels)
        wp_tar_text = converter.wp_encode(tar_labels)

        start_time = time.time()
        src_char_preds, src_bpe_preds, src_wp_preds, src_global_feature, src_local_feature, src_local_feature_bpe, src_local_feature_wp = model(src_image)
        src_local_feature_bpe = src_local_feature_bpe.contiguous().view(-1, src_local_feature_bpe.shape[-1])
        src_local_feature_wp = src_local_feature_wp.contiguous().view(-1, src_local_feature_wp.shape[-1])

        tar_char_preds, tar_bpe_preds, tar_wp_preds, tar_global_feature, tar_local_feature, tar_local_feature_bpe, tar_local_feature_wp = model(tar_image)
        tar_local_feature_bpe = tar_local_feature_bpe.contiguous().view(-1, tar_local_feature_bpe.shape[-1])
        tar_local_feature_wp = tar_local_feature_wp.contiguous().view(-1, tar_local_feature_wp.shape[-1])

        forward_time = time.time() - start_time

        # cost = criterion(char_preds.contiguous().view(-1, char_preds.shape[-1]),
        #                  target.contiguous().view(-1))  # char_preds:192*26,38  target:192*26

        if step == 5:
            source_bpe = src_local_feature_bpe
            source_wp = src_local_feature_wp
            target_bpe = tar_local_feature_bpe
            target_wp = tar_local_feature_wp
        elif step == 100:
            break
        else:
            continue
            # label_a = torch.cat((label_a,label), dim=0)
            # data_a = torch.cat((data_a,data), dim=0)

    # data, label, n_samples, n_features = get_data()
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time.time()
    # result1 = tsne.fit_transform(source_bpe.cpu().detach().numpy()) #降维后 B,2=1083,2
    # result2 = tsne.fit_transform(target_bpe.cpu().detach().numpy())
    bpe_feature = torch.cat((source_bpe,target_bpe),dim=0)
    result = tsne.fit_transform(bpe_feature.cpu().detach().numpy()) #降维后 B,2=1083,2
    b = result.shape[0]
    b = b // 2
    result1 = result[:b, :]
    result2 = result[b:, :]
    fig1 = plot_embedding(result1, result2,
                         'BPE-Level Local Feature of Source&Target Domain')
    plt.show()
    # result3 = tsne.fit_transform(source_wp.cpu().detach().numpy())  # 降维后 B,2=1083,2
    # result4 = tsne.fit_transform(target_wp.cpu().detach().numpy())
    wp_feature = torch.cat((source_wp, target_wp), dim=0)
    result = tsne.fit_transform(wp_feature.cpu().detach().numpy())  # 降维后 B,2=1083,2
    b = result.shape[0]
    b = b // 2
    result3 = result[:b, :]
    result4 = result[b:, :]
    fig2 = plot_embedding(result3, result4,
                         'WP-Level Local Feature of Source&Target Domain')
    plt.show()


if __name__ == '__main__':
    main()