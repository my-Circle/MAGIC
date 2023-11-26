import os
import time
import string
import argparse
import re
import PIL
import math

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from matplotlib import pyplot as plt
from matplotlib import colors
import cv2
from torchvision import transforms
import torchvision.utils as vutils

from utils import TokenLabelConverter
from models import Model
# from utils import get_args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_model(image_tensors, model, converter, opt):
    image = image_tensors.to(device)
    batch_size = image.shape[0]

    attens, char_preds, bpe_preds, wp_preds = model(image, is_eval=True)  # final

    # char pred
    _, char_pred_index = char_preds.topk(1, dim=-1, largest=True, sorted=True)
    char_pred_index = char_pred_index.view(-1, converter.batch_max_length)
    length_for_pred = torch.IntTensor([converter.batch_max_length - 1] * batch_size).to(device)
    char_preds_str = converter.char_decode(char_pred_index[:, 1:], length_for_pred)
    char_pred_prob = F.softmax(char_preds, dim=2)
    char_pred_max_prob, _ = char_pred_prob.max(dim=2)
    char_preds_max_prob = char_pred_max_prob[:, 1:]

    # bpe pred
    _, bpe_preds_index = bpe_preds.topk(1, dim=-1, largest=True, sorted=True)
    bpe_preds_index = bpe_preds_index.view(-1, converter.batch_max_length)
    bpe_preds_str = converter.bpe_decode(bpe_preds_index[:, 1:], length_for_pred)
    bpe_preds_prob = F.softmax(bpe_preds, dim=2)
    bpe_preds_max_prob, _ = bpe_preds_prob.max(dim=2)
    bpe_preds_max_prob = bpe_preds_max_prob[:, 1:]
    bpe_preds_index = bpe_preds_index[:, 1:]

    # wp pred
    _, wp_preds_index = wp_preds.topk(1, dim=-1, largest=True, sorted=True)
    wp_preds_index = wp_preds_index.view(-1, converter.batch_max_length)
    wp_preds_str = converter.wp_decode(wp_preds_index[:, 1:], length_for_pred)
    wp_preds_prob = F.softmax(wp_preds, dim=2)
    wp_preds_max_prob, _ = wp_preds_prob.max(dim=2)
    wp_preds_max_prob = wp_preds_max_prob[:, 1:]
    wp_preds_index = wp_preds_index[:, 1:]

    # for index in range(image.shape[0]):
    index = 0

    # char
    char_pred = char_preds_str[index]
    char_pred_max_prob = char_preds_max_prob[index]
    char_pred_EOS = char_pred.find('[s]')
    char_pred = char_pred[:char_pred_EOS]  # prune after "end of sentence" token ([s])

    char_pred_max_prob = char_pred_max_prob[:char_pred_EOS + 1]
    try:
        char_confidence_score = char_pred_max_prob.cumprod(dim=0)[-1].cpu().tolist()
    except:
        char_confidence_score = 0.0
    print('char:', char_pred, char_confidence_score)

    # bpe
    bpe_pred = bpe_preds_str[index]
    bpe_pred_max_prob = bpe_preds_max_prob[index]
    bpe_pred_EOS = bpe_pred.find('#')
    bpe_pred = bpe_pred[:bpe_pred_EOS]

    bpe_pred_index = bpe_preds_index[index].cpu().tolist()
    try:
        bpe_pred_EOS_index = bpe_pred_index.index(2)
    except:
        bpe_pred_EOS_index = -1
    bpe_pred_max_prob = bpe_pred_max_prob[:bpe_pred_EOS_index + 1]
    try:
        bpe_confidence_score = bpe_pred_max_prob.cumprod(dim=0)[-1].cpu().tolist()
    except:
        bpe_confidence_score = 0.0
    print('bpe:', bpe_pred, bpe_confidence_score)

    # wp
    wp_pred = wp_preds_str[index]
    wp_pred_max_prob = wp_preds_max_prob[index]
    wp_pred_EOS = wp_pred.find('[SEP]')
    wp_pred = wp_pred[:wp_pred_EOS]

    wp_pred_index = wp_preds_index[index].cpu().tolist()
    try:
        wp_pred_EOS_index = wp_pred_index.index(102)
    except:
        wp_pred_EOS_index = -1
    wp_pred_max_prob = wp_pred_max_prob[:wp_pred_EOS_index + 1]
    try:
        wp_confidence_score = wp_pred_max_prob.cumprod(dim=0)[-1].cpu().tolist()
    except:
        wp_confidence_score = 0.0
    print('wp:', wp_pred, wp_confidence_score)

    # draw atten
    pil = transforms.ToPILImage()
    tensor = transforms.ToTensor()
    size = opt.imgH, opt.imgW
    resize = transforms.Resize(size=size, interpolation=0)
    char_atten = attens[0][index] #attens:3个（B,T,N+1)大小的Tensor构成的数组 attens[0]->(B,T,N+1):一个batch图片的所有字符注意力图  (B,T,N+1)[index]->(T,N+1):第index张图片的字符注意力图
    bpe_atten = attens[1][index] #同上
    wp_atten = attens[2][index] #同上
    char_atten = char_atten[:, 1:].view(-1, 8, 32) #char_atten[:, 1:] (T,N=256=8*32) 去掉了分类头[CLS]token
    char_atten = char_atten[1:char_pred_EOS + 1]
    bpe_atten = bpe_atten[:, 1:].view(-1, 8, 32)
    bpe_atten = bpe_atten[1:bpe_pred_EOS_index + 1]
    wp_atten = wp_atten[:, 1:].view(-1, 8, 32)
    wp_atten = wp_atten[1:wp_pred_EOS_index + 1]
    draw_atten(opt.demo_imgs, char_pred, char_atten, pil, tensor, resize, flag='char')


def load_img(img_path, opt):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((opt.imgW, opt.imgH), Image.BICUBIC)
    img_arr = np.array(img)
    img_tensor = transforms.ToTensor()(img)
    image_tensor = img_tensor.unsqueeze(0)
    return image_tensor


def draw_atten(img_path, pred, attn, pil, tensor, resize, flag=''):
    image = PIL.Image.open(img_path).convert('RGB')
    image = cv2.resize(np.array(image), (128, 32))

    image = tensor(image)
    image_np = np.array(pil(image))

    attn_pil = [pil(a) for a in attn[:, None, :, :]]
    attn = [tensor(resize(a)).repeat(3, 1, 1) for a in attn_pil]
    attn_sum = np.array([np.array(a) for a in attn_pil[:len(pred)]]).sum(axis=0)
    blended_sum = tensor(blend_mask(image_np, attn_sum))
    blended = [tensor(blend_mask(image_np, np.array(a))) for a in attn_pil]
    save_image = torch.stack([image] + attn + [blended_sum] + blended)
    save_image = save_image.view(2, -1, *save_image.shape[1:])
    save_image = save_image.permute(1, 0, 2, 3, 4).flatten(0, 1)

    gt = os.path.basename(img_path).split('.')[0]
    vutils.save_image(save_image, f'demo_imgs/attens/{gt}_{pred}_{flag}.jpg', nrow=2, normalize=True, scale_each=True)


def blend_mask(image, mask, alpha=0.5, cmap='jet', color='b', color_alpha=1.0):
    # normalize mask
    mask = (mask - mask.min()) / (mask.max() - mask.min() + np.finfo(float).eps)
    if mask.shape != image.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    # get color map
    color_map = plt.get_cmap(cmap)
    mask = color_map(mask)[:, :, :3]
    # convert float to uint8
    mask = (mask * 255).astype(dtype=np.uint8)

    # set the basic color
    basic_color = np.array(colors.to_rgb(color)) * 255
    basic_color = np.tile(basic_color, [image.shape[0], image.shape[1], 1])
    basic_color = basic_color.astype(dtype=np.uint8)
    # blend with basic color
    blended_img = cv2.addWeighted(image, color_alpha, basic_color, 1 - color_alpha, 0)
    # blend with mask
    blended_img = cv2.addWeighted(blended_img, alpha, mask, 1 - alpha, 0)

    return blended_img

def load(model, saved_model):
    params = torch.load(saved_model)

    if 'model' not in params: #baseline
        model.load_state_dict(params) #,strict=False
    else:  #adapt
        model.load_state_dict(params['model']) #,strict=False
def test(opt):
    """ model configuration """
    converter = TokenLabelConverter(opt)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)

    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    load(model, opt.saved_model)
    # model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # load img
    if os.path.isdir(opt.demo_imgs):
        imgs = [os.path.join(opt.demo_imgs, fname) for fname in os.listdir(opt.demo_imgs)]
        imgs = [img for img in imgs if img.endswith('.jpg') or img.endswith('.png')]
    else:
        imgs = [opt.demo_imgs]

    for img in imgs:
        opt.demo_imgs = img
        img_tensor = load_img(opt.demo_imgs, opt)
        print('imgs:', img)

        """ evaluation """
        model.eval()
        opt.eval = True
        with torch.no_grad():
            run_model(img_tensor, model, converter, opt)
        print('============================================================================')

def get_args(is_train=True):
    parser = argparse.ArgumentParser(description='STR')

    # for test
    parser.add_argument('--eval_data', help='path to evaluation dataset')
    parser.add_argument('--benchmark_all_eval', action='store_true', help='evaluate 10 benchmark evaluation datasets')
    parser.add_argument('--calculate_infer_time', action='store_true', help='calculate inference timing')
    parser.add_argument('--flops', action='store_true', help='calculates approx flops (may not work)')

    # for train
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--train_data', required=is_train, help='path to training dataset')
    parser.add_argument('--valid_data', required=is_train, help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers. Use -1 to use all cores.',
                        default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--saved_path', default='./saved_models', help="path to save")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--sgd', action='store_true', help='Whether to use SGD (default is Adadelta)')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='MJ-ST',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='0.5-0.5',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=128, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')

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

    # selective augmentation
    # can choose specific data augmentation
    parser.add_argument('--issel_aug', action='store_true', help='Select augs')
    parser.add_argument('--sel_prob', type=float, default=1., help='Probability of applying augmentation')
    parser.add_argument('--pattern', action='store_true', help='Pattern group')
    parser.add_argument('--warp', action='store_true', help='Warp group')
    parser.add_argument('--geometry', action='store_true', help='Geometry group')
    parser.add_argument('--weather', action='store_true', help='Weather group')
    parser.add_argument('--noise', action='store_true', help='Noise group')
    parser.add_argument('--blur', action='store_true', help='Blur group')
    parser.add_argument('--camera', action='store_true', help='Camera group')
    parser.add_argument('--process', action='store_true', help='Image processing routines')

    # use cosine learning rate decay
    parser.add_argument('--scheduler', action='store_true', help='Use lr scheduler')

    parser.add_argument('--intact_prob', type=float, default=0.5, help='Probability of not applying augmentation')
    parser.add_argument('--isrand_aug', action='store_true', help='Use RandAug')
    parser.add_argument('--augs_num', type=int, default=3, help='Number of data augment groups to apply. 1 to 8.')
    parser.add_argument('--augs_mag', type=int, default=None,
                        help='Magnitude of data augment groups to apply. None if random.')

    # for comparison to other augmentations
    parser.add_argument('--issemantic_aug', action='store_true', help='Use Semantic')
    parser.add_argument('--isrotation_aug', action='store_true', help='Use ')
    parser.add_argument('--isscatter_aug', action='store_true', help='Use ')
    parser.add_argument('--islearning_aug', action='store_true', help='Use ')

    # orig paper uses this for fast benchmarking
    parser.add_argument('--fast_acc', action='store_true', help='Fast average accuracy computation')

    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # mask train
    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    parser.add_argument("--patch_size", type=int, default=4)

    # for eval
    parser.add_argument('--eval_img', action='store_true', help='eval imgs dataset')
    parser.add_argument('--range', default=None, help="start-end for example(800-1000)")
    parser.add_argument('--model_dir', default='')
    parser.add_argument('--demo_imgs', default='')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    opt = get_args(is_train=False)

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    opt.saved_model = opt.model_dir
    test(opt)
