#  MAGIC in Text Recognition: A Generalizable Recognition Model based on Unsupervised Domain Adaptation 

## Introduction
The Code of Paper "MAGIC in Text Recognition: A Generalizable Recognition Model based on Unsupervised Domain Adaptation"




### Environment & Install

```cuda==11.6, python==3.7.16```

Install Requirements

```
pip install torch==1.12.0 pillow==8.1.0 torchvision==0.13.0 lmdb==1.2.1 nltk==3.6.2 natsort==7.1.1 torchaudio==0.12.0
pip install validators==0.20.0 timm==0.4.12 opencv-python==4.5.1.48 opencv-contrib-python==4.5.1.48 wand==0.6.7 transformers==4.2.1 strsimpy==0.2.1
```

### Dataset

- You can download lmdb dataset from [here(password:izsv)](https://pan.baidu.com/s/17i3ArCWhK4nBPe7de7fYHg). 
  - Source Datasets: Synthetic Text
    -  [MJSynth (MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/)
    -  [SynthText (ST)](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)
  - Target & Evaluation Datasets:
    -  Real Scene Text
       - The union of the training sets [IC13](http://rrc.cvc.uab.es/?ch=2), [IC15](http://rrc.cvc.uab.es/?ch=4), [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html), and [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset).
       - Benchmark evaluation scene text datasets : consist of [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html), [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset), [IC03](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions), [IC13](http://rrc.cvc.uab.es/?ch=2), [IC15](http://rrc.cvc.uab.es/?ch=4),
        [SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf), and [CUTE](http://cs-chan.com/downloads_CUTE80_dataset.html).
    - Handwritten Text
      - [IAM](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
    - Artistic Text
      -  [WordArt](https://github.com/xdxie/WordArt)
      -  Details of this dataset can be found in the paper [Toward Understanding WordArt: Corner-Guided Transformer for Scene Text Recognition](https://github.com/xdxie/WordArt)
      -  Use tools/create_lmdb_dataset.py to convert images into LMDB dataset

<div style='display: none'> 
- Synthetic scene text : [MJSynth (MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/) and [SynthText (ST)](http://www.robots.ox.ac.uk/~vgg/data/scenetext/) \
- Real scene text : the union of the training sets [IC13](http://rrc.cvc.uab.es/?ch=2), [IC15](http://rrc.cvc.uab.es/?ch=4), [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html), and [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset).\
- Benchmark evaluation scene text datasets : consist of [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html), [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset), [IC03](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions), [IC13](http://rrc.cvc.uab.es/?ch=2)[3], [IC15](http://rrc.cvc.uab.es/?ch=4),
 [SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf), and [CUTE](http://cs-chan.com/downloads_CUTE80_dataset.html).
 - The prepared handwritten text dataset can be downloaded from [here](https://www.dropbox.com/sh/4a9vrtnshozu929/AAAZucKLtEAUDuOufIRDVPOTa?dl=0)    
 - Handwritten text: [IAM](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
</div>

### Pretrained Model
You can download the pretrained models from [here(password:gsqt)](https://pan.baidu.com/s/1QQywykZbhwSVguNCmH61sQ) to the path 
``` pretrain_m/ ```.

### Training and Evaluation


- Training
    
    ```
    CUDA_VISIBLE_DEVICES=1 python train_last.py --Transformer mgp-str --TransformerModel=mgp_str_base_patch4_3_32_128 \
    --src_train_data ../../dataset/data_lmdb_release/training/ \
    --tar_train_data ../../dataset/IAM --tar_select_data IAM --tar_batch_ratio 1 --valid_data ../../dataset/IAM \
    --continue_model pretrain_m/base.pth \
    --batch_size 64 --lr 0.1 --experiment_name _synth2iam_base \
    --rgb --tar_lambda 0.1 
    ```
   - Please remember to change the parameter ```--tar_lambda``` when you change target training set.
   - Parameters ```--Transformer --TransformerModel``` need to be changed when replacing the version.
    
- Evaluation

    ```
    CUDA_VISIBLE_DEVICES=1 python test.py --Transformer mgp-str --TransformerModel=mgp_str_base_patch4_3_32_128 \
    --eval_data ../../dataset/IAM --saved_model <path_to/best_accuracy.pth> \
    --rgb --batch_size 64
    ```



## Citation
If you use this code for a paper please cite:

```

```


##  Acknowledgement

This implementation is based on [MGP-STR](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/OCR/MGP-STR) and [Seq2SeqAdapt](https://github.com/AprilYapingZhang/Seq2SeqAdapt).

