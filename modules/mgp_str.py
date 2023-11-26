'''
Implementation of MGP-STR based on ViTSTR.

Copyright 2022 Alibaba
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch 
import torch.nn as nn
import logging
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from copy import deepcopy
from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models import create_model
from .token_learner import TokenLearner

_logger = logging.getLogger(__name__)

__all__ = [
    'mgp_str_base_patch4_3_32_128',
    'mgp_str_tiny_patch4_3_32_128',
    'mgp_str_small_patch4_3_32_128',
]

def create_mgp_str(batch_max_length, num_tokens, model=None, checkpoint_path=''):
    mgp_str = create_model(
        model,
        pretrained=True,
        num_classes=num_tokens,
        checkpoint_path=checkpoint_path,
        batch_max_length=batch_max_length)  #进入mgp_str-base/small/tiny_patch4_3_32_128方法

    # might need to run to get zero init head for transfer learning
    mgp_str.reset_classifier(num_classes=num_tokens) #进入MGPSTR类的reset_classifier方法

    return mgp_str

class MGPSTR(VisionTransformer):

    def __init__(self, batch_max_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.batch_max_length = batch_max_length  #27
        self.char_tokenLearner = TokenLearner(self.embed_dim, self.batch_max_length)  #768,27 进入token_learner.py的init方法

        self.bpe_tokenLearner = TokenLearner(self.embed_dim, self.batch_max_length)  #768,27
        self.wp_tokenLearner = TokenLearner(self.embed_dim, self.batch_max_length)   #768,27

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.char_head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.bpe_head = nn.Linear(self.embed_dim, 50257) if num_classes > 0 else nn.Identity()
        self.wp_head = nn.Linear(self.embed_dim, 30522) if num_classes > 0 else nn.Identity()
        

    def forward_features(self, x):  #重点逻辑 192,3,32,128
        B = x.shape[0]
        x = self.patch_embed(x)  #192,3*4*4,8,32->192,48,8*32->192,256,48->192,256,768

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks  #cls_token(192,1,768)分类token
        x = torch.cat((cls_tokens, x), dim=1) #加上分类token 192,257,768
        x = x + self.pos_embed  #加上位置编码 pos_embed1,257,768,这里用来个广播机制
        x = self.pos_drop(x)

        for i,blk in enumerate(self.blocks):
            x = blk(x)  #attn中的qkv先用了个线性层768->768*3=2304然后拆成了q、k、v进行自注意力机制 x还是(192,257,768)

        #相对于mgp-str新的部分
        visual_feature = x #(192,257,768)
            
        attens = []

        # char
        char_attn, x_char = self.char_tokenLearner(x) #char_attn:(192,27,257) x_char:(192,27,768)
        # 相对于mgp-str新的部分
        global_feature = x_char
        x_char = self.char_head(x_char) #192,27,38
        char_out = x_char
        attens = [char_attn] 

        # bpe
        bpe_attn, x_bpe = self.bpe_tokenLearner(x)  #bpe_attn:(192,27,257) x_bpe:(192,27,768)
        global_feature_bpe = x_bpe
        bpe_out = self.bpe_head(x_bpe) #192,27,50257
        attens += [bpe_attn]

        # wp
        wp_attn, x_wp = self.wp_tokenLearner(x)  #同上
        global_feature_wp = x_wp
        wp_out = self.wp_head(x_wp)  #192,27,30522
        attens += [wp_attn]
        
        return attens, char_out, bpe_out, wp_out, visual_feature, global_feature, global_feature_bpe, global_feature_wp

    def forward(self, x, is_eval=False):  #192,3,32,128
        attn_scores, char_out, bpe_out, wp_out, visual_f, global_f, global_f_bpe, global_f_wp = self.forward_features(x) #进入forward_features方法
        if is_eval:
            return [attn_scores, char_out, bpe_out, wp_out]
        else:
            return [char_out, bpe_out, wp_out, visual_f, global_f, global_f_bpe, global_f_wp]

def load_pretrained(model, cfg=None, num_classes=1000, in_chans=1, filter_fn=None, strict=True):  #加载预训练模型
    '''
    Loads a pretrained checkpoint
    From an older version of timm
    '''
    if cfg is None:
        cfg = getattr(model, 'default_cfg')
    if cfg is None or 'url' not in cfg or not cfg['url']:
        _logger.warning("Pretrained model URL is invalid, using random initialization.")
        return

    state_dict = model_zoo.load_url(cfg['url'], progress=True, map_location='cpu')
    if "model" in state_dict.keys():
        state_dict = state_dict["model"]

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)  #进入_conv_filter函数

    print("in_chans",in_chans)
    if in_chans == 1:
        conv1_name = cfg['first_conv']
        _logger.info('Converting first conv (%s) pretrained weights from 3 to 1 channel' % conv1_name)
        key = conv1_name + '.weight'
        if key in state_dict.keys():
            _logger.info('(%s) key found in state_dict' % key)
            conv1_weight = state_dict[conv1_name + '.weight']
        else:
            _logger.info('(%s) key NOT found in state_dict' % key)
            return
        # Some weights are in torch.half, ensure it's float for sum on CPU
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I > 3:
            assert conv1_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        conv1_weight = conv1_weight.to(conv1_type)
        state_dict[conv1_name + '.weight'] = conv1_weight

    classifier_name = cfg['classifier']
    if num_classes == 1000 and cfg['num_classes'] == 1001:
        # special case for imagenet trained models with extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
    elif num_classes != cfg['num_classes']:   #删除分类头
        # completely discard fully connected for all other differences between pretrained and created model
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False

    print("Loading pre-trained vision transformer weights from %s ..." % cfg['url'])
    model.load_state_dict(state_dict, strict=strict)

def _conv_filter(state_dict):  #加载预训练模型权重时除去投影层
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if not 'patch_embed' in k and  not 'pos_embed' in k :
            out_dict[k] = v
        else:
            print("not load",k) 
    return out_dict


@register_model
def mgp_str_base_patch4_3_32_128(pretrained=False, **kwargs):
    kwargs['in_chans'] = 3
    model = MGPSTR(
        img_size=(32,128), patch_size=4, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs)  #进入MGPSTR类init方法
    model.default_cfg = _cfg(
            #url='https://github.com/roatienza/public/releases/download/v0.1-deit-base/deit_base_patch16_224-b5f2ef4d.pth'
            url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth'
    )
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter) #进入load_pretrained方法
    return model

# below is work in progress
@register_model
def mgp_str_tiny_patch4_3_32_128(pretrained=False, **kwargs):
    kwargs['in_chans'] = 3
    model = MGPSTR(
        img_size=(32,128), patch_size=4, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = _cfg(
            url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth'
    )
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model


@register_model
def mgp_str_small_patch4_3_32_128(pretrained=False, **kwargs):
    kwargs['in_chans'] = 3
    model = MGPSTR(
        img_size=(32,128), patch_size=4, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = _cfg(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth"
    )
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model