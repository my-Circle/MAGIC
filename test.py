import argparse
import os
import string
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from dataset import hierarchical_dataset, AlignCollate
from utils import AttnLabelConverter, Averager, TokenLabelConverter
from utils import load_char_dict, compute_loss

import torch.nn.functional as F
from losses.CCLoss import CCLoss
from models import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def benchmark_all_eval(model, criterion, converter, opt, calculate_infer_time=False):
    """ evaluation with 10 benchmark evaluation datasets """
    # The evaluation datasets, dataset order is same with Table 1 in our paper.
    eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857',
                      'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']
    # eval_data_list = ['IIIT5k_3000', 'SVT', 'IAM', 'IC13_857',
    #                   'IC15_1811', 'SVTP', 'CUTE80', 'WA_test']

    if calculate_infer_time:
        evaluation_batch_size = 1  # batch_size should be 1 to calculate the GPU inference time per image.
    else:
        evaluation_batch_size = opt.batch_size

    list_accuracy = []
    total_forward_time = 0
    total_evaluation_data_number = 0
    total_correct_number = 0
    print('-' * 80)
    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data)
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW,
                                               keep_ratio_with_pad=opt.PAD)
        eval_data = hierarchical_dataset(root=eval_data_path, opt=opt)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=evaluation_batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)

        _, accuracy_by_best_model, norm_ED_by_best_model, char_accuracy, bpe_accuracy, wp_accuracy, out_accuracy,_, _, infer_time, length_of_data = validation(
            model, criterion, evaluation_loader, converter, opt)
        list_accuracy.append(f'{accuracy_by_best_model:0.3f}')
        total_forward_time += infer_time
        total_evaluation_data_number += len(eval_data)
        total_correct_number += accuracy_by_best_model * length_of_data
        print('Acc %0.3f\t normalized_ED %0.3f' % (accuracy_by_best_model, norm_ED_by_best_model))
        print('-' * 80)

    averaged_forward_time = total_forward_time / total_evaluation_data_number * 1000
    total_accuracy = total_correct_number / total_evaluation_data_number
    params_num = sum([np.prod(p.size()) for p in model.parameters()])

    evaluation_log = 'accuracy: '
    for name, accuracy in zip(eval_data_list, list_accuracy):
        evaluation_log += f'{name}: {accuracy}\t'
    evaluation_log += f'total_accuracy: {total_accuracy:0.3f}\t'
    evaluation_log += f'averaged_infer_time: {averaged_forward_time:0.3f}\t# parameters: {params_num/1e6:0.3f}'
    print(evaluation_log)
    with open(f'./result/{opt.experiment_name}/log_all_evaluation.txt', 'a') as log:
        log.write(evaluation_log + '\n')

    return None


def validation(model, criterion, evaluation_loader, converter, opt):
    """ validation or evaluation """
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    char_n_correct = 0
    bpe_n_correct = 0
    wp_n_correct = 0
    out_n_correct = 0

    for i, (image_tensors, labels) in enumerate(evaluation_loader):  #192,1,32,100
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size  #192
        image = image_tensors.to(device)
        # For max length prediction
        target = converter.encode(labels)  #(192,27)



        start_time = time.time()

        _, char_preds, bpe_preds, wp_preds = model(image, is_eval=True)


        forward_time = time.time() - start_time




        cost = criterion(char_preds.contiguous().view(-1, char_preds.shape[-1]),
                         target.contiguous().view(-1))  # char_preds:192*26,38  target:192*26


        # char pred
        _, char_pred_index = char_preds.topk(1, dim=-1, largest=True, sorted=True)  # char_pred_index(192,27,1)
        char_pred_index = char_pred_index.view(-1, converter.batch_max_length)  # (192,27)
        length_for_pred = torch.IntTensor([converter.batch_max_length - 1] * batch_size).to(
            device)
        char_preds_str = converter.char_decode(char_pred_index[:, 1:],
                                               length_for_pred)
        char_pred_prob = F.softmax(char_preds, dim=2)
        char_pred_max_prob, _ = char_pred_prob.max(dim=2)
        char_preds_max_prob = char_pred_max_prob[:, 1:]

        # bpe pred
        _, bpe_preds_index = bpe_preds.topk(1, dim=-1, largest=True, sorted=True)
        bpe_preds_index = bpe_preds_index.view(-1, converter.batch_max_length)
        bpe_preds_str = converter.bpe_decode(bpe_preds_index[:, 1:],
                                             length_for_pred)
        bpe_preds_prob = F.softmax(bpe_preds, dim=2)  # 192,27,50257
        bpe_preds_max_prob, _ = bpe_preds_prob.max(dim=2)
        bpe_preds_max_prob = bpe_preds_max_prob[:, 1:]
        bpe_preds_index = bpe_preds_index[:, 1:]

        # wp pred
        _, wp_preds_index = wp_preds.topk(1, dim=-1, largest=True, sorted=True)
        wp_preds_index = wp_preds_index.view(-1, converter.batch_max_length)
        wp_preds_str = converter.wp_decode(wp_preds_index[:, 1:],
                                           length_for_pred)
        wp_preds_prob = F.softmax(wp_preds, dim=2)  # 192,27,30522
        wp_preds_max_prob, _ = wp_preds_prob.max(dim=2)
        wp_preds_max_prob = wp_preds_max_prob[:, 1:]
        wp_preds_index = wp_preds_index[:, 1:]

        infer_time += forward_time
        valid_loss_avg.add(cost)

        preds_str = []
        gts = []

        # calculate accuracy & confidence score
        confidence_score_list = []
        for index, gt in enumerate(labels):
            max_confidence_score = 0.0
            out_pred = None

            # preds_str = []
            # gts = []

            # char
            char_pred = char_preds_str[index]
            char_pred_max_prob = char_preds_max_prob[index]
            char_pred_EOS = char_pred.find('[s]')
            char_pred = char_pred[:char_pred_EOS]  # prune after "end of sentence" token ([s])
            if char_pred == gt:
                char_n_correct += 1
            char_pred_max_prob = char_pred_max_prob[:char_pred_EOS + 1]
            try:
                char_confidence_score = char_pred_max_prob.cumprod(dim=0)[-1]
            except:
                char_confidence_score = 0.0
            if char_confidence_score > max_confidence_score:
                max_confidence_score = char_confidence_score
                out_pred = char_pred

            # bpe
            bpe_pred = bpe_preds_str[index]
            bpe_pred_max_prob = bpe_preds_max_prob[index]
            bpe_pred_EOS = bpe_pred.find('#')
            bpe_pred = bpe_pred[:bpe_pred_EOS]
            if bpe_pred == gt:
                bpe_n_correct += 1
            bpe_pred_index = bpe_preds_index[index].cpu().tolist()
            try:
                bpe_pred_EOS_index = bpe_pred_index.index(2)
            except:
                bpe_pred_EOS_index = -1
            bpe_pred_max_prob = bpe_pred_max_prob[:bpe_pred_EOS_index + 1]
            try:
                bpe_confidence_score = bpe_pred_max_prob.cumprod(dim=0)[-1]
            except:
                bpe_confidence_score = 0.0
            if bpe_confidence_score > max_confidence_score:
                max_confidence_score = bpe_confidence_score
                out_pred = bpe_pred

            # wp
            wp_pred = wp_preds_str[index]
            wp_pred_max_prob = wp_preds_max_prob[index]
            wp_pred_EOS = wp_pred.find('[SEP]')
            wp_pred = wp_pred[:wp_pred_EOS]
            if wp_pred == gt:
                wp_n_correct += 1
            wp_pred_index = wp_preds_index[index].cpu().tolist()
            try:
                wp_pred_EOS_index = wp_pred_index.index(102)
            except:
                wp_pred_EOS_index = -1
            wp_pred_max_prob = wp_pred_max_prob[:wp_pred_EOS_index + 1]
            try:
                wp_confidence_score = wp_pred_max_prob.cumprod(dim=0)[-1]
            except:
                wp_confidence_score = 0.0
            if wp_confidence_score > max_confidence_score:
                max_confidence_score = wp_confidence_score
                out_pred = wp_pred

            if out_pred == None:
                out_pred = char_pred

            if out_pred == gt:
                out_n_correct += 1

            preds_str.append(out_pred)
            gts.append(gt)

            confidence_score_list.append(char_confidence_score)

        # calculate accuracy.
        batch_n_correct, batch_char_acc = compute_loss(preds_str, gts, opt)
        n_correct += batch_n_correct
        norm_ED += batch_char_acc

    accuracy = n_correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data) * 100

    char_accuracy = char_n_correct / float(length_of_data) * 100
    bpe_accuracy = bpe_n_correct / float(length_of_data) * 100
    wp_accuracy = wp_n_correct / float(length_of_data) * 100
    out_accuracy = out_n_correct / float(length_of_data) * 100

    return valid_loss_avg.val(), accuracy, norm_ED, char_accuracy, bpe_accuracy, wp_accuracy, out_accuracy, preds_str, gts, infer_time, length_of_data


def load(model, saved_model):
    params = torch.load(saved_model)

    if 'model' not in params: #baseline
        model.load_state_dict(params)
    else:  #adapt
        model.load_state_dict(params['model'])


def test(opt):
    """ model configuration """
    converter = TokenLabelConverter(opt)
    opt.num_class = len(converter.character) #38

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)  #seqda_model
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
    opt.experiment_name = '_'.join(opt.saved_model.split('/')[1:])
    # print(model)

    """ keep evaluation model and result logs """
    os.makedirs(f'./result/{opt.experiment_name}', exist_ok=True)
    os.system(f'cp {opt.saved_model} ./result/{opt.experiment_name}/')

    """ setup loss """
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(
        device)  # ignore [GO] token = ignore index 0
    # char_criterion = CCLoss(ignore_index=0).to(
    #     device)  # ignore [GO] token = ignore index 0

    """ evaluation """
    model.eval()
    with torch.no_grad():
        if opt.benchmark_all_eval:  # evaluation with 10 benchmark evaluation datasets
            benchmark_all_eval(model, criterion, converter, opt)
        else:
            AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW,
                                                   keep_ratio_with_pad=opt.PAD)
            eval_data = hierarchical_dataset(root=opt.eval_data, opt=opt)
            evaluation_loader = torch.utils.data.DataLoader(
                eval_data, batch_size=opt.batch_size,
                shuffle=False,
                num_workers=int(opt.workers),
                collate_fn=AlignCollate_evaluation, pin_memory=True)
            _, accuracy_by_best_model, char_acc_by_best_model, char_accuracy, bpe_accuracy, wp_accuracy, out_accuracy,_, _, _, _ = validation(
                model, criterion, evaluation_loader, converter, opt)

            print('accuracy:'+ str(accuracy_by_best_model))
            print('norm_ED:'+ str(char_acc_by_best_model))
            print('char_accuracy:' + str(char_accuracy))
            print('bpe_accuracy:' + str(bpe_accuracy))
            print('wp_accuracy:' + str(wp_accuracy))
            print('fused_accuracy:' + str(out_accuracy))
            with open('./result/{0}/log_evaluation.txt'.format(opt.experiment_name), 'a') as log:
                log.write('accuracy:' + str(accuracy_by_best_model) + '\n')
                log.write('norm_ED:' + str(char_acc_by_best_model) + '\n')
                log.write('char_accuracy:' + str(char_accuracy) + '\n')
                log.write('bpe_accuracy:' + str(bpe_accuracy) + '\n')
                log.write('wp_accuracy:' + str(wp_accuracy) + '\n')
                log.write('fused_accuracy:' + str(out_accuracy) + '\n')



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
        opt.character = load_char_dict(opt.char_dict)[3:-2]  
    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    test(opt)
