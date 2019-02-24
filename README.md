# [Semi-supervised Adversarial Learning to Generate Photorealistic Face Images of New Identities from 3D Morphable Model](http://openaccess.thecvf.com/content_ECCV_2018/papers/Baris_Gecer_Semi-supervised_Adversarial_Learning_ECCV_2018_paper.pdf)

 [Baris Gecer](http://barisgecer.github.io)<sup> 1</sup>, [Binod Bhattarai](https://sites.google.com/view/bbinod)<sup> 1</sup>, [Josef Kittler](https://www.surrey.ac.uk/people/josef-kittler)<sup> 2</sup>, & [Tae-Kyun Kim](https://www.imperial.ac.uk/people/tk.kim)<sup> 1</sup>
 <br/>
 <sup>1 </sup>Department of Electrical and Electronic Engineering, Imperial College London, UK
 <br/>
 <sup>2 </sup>Centre for Vision, Speech and Signal Processing, University of Surrey, UK

**This repository provides a Tensorflow implementation of our study where we propose a novel end-to-end semi-supervised adversarial framework to generate photorealistic face images of new identities with wide ranges of expressions, poses, and illuminations conditioned by a 3D morphable model.**

<br/>

<p align="center"><img width="100%" src="figures/images/randomsample.png" /></p>


<br/>
(This documentation is still under construction, please refer to our paper for more details)
<br/>

## Approach

<p align="center"><img width="100%" src="figures/images/overview.png" /></p>
Our approach aims to synthesize photorealistic images conditioned by a given synthetic image by 3DMM. It regularizes cycle consistency by introducing an additional adversarial game between the two generator networks in an unsupervised fashion. Thus the under-constraint cycle loss is supervised to have correct matching between the two domains by the help of a limited number of paired data. We also encourage the generator to preserve face identity by a set-based supervision through a pretrained classification network.

<br/>

## Dependencies
* [TensorFlow 1.2](https://www.tensorflow.org/)
* [Python 3.5](https://www.continuum.io/downloads)

## Data

* Generate synthetic images using any 3DMM model i.e. [LSFM](https://xip.uclb.com/i/healthcare_tools/LSFM.html) or [Basel Face Model](https://faces.cs.unibas.ch/bfm/) by running gen_syn_latent.m
* Align and crop all datasets using MTCNN to 108x108

## Usage

Train by the following script

```bash
$ python main.py    --log_dir [path2_logdir] --data_dir [path2_datadir] --syn_dataset [synthetic_dataset_name]
                    --dataset [real_dataset_name] --dataset_3dmm [300W-3D & AFLW2000_dirname] --input_scale_size 108
```
Add --load_path [paused_training_logdir] to continue a training

Generate realistic images after training by the following script
```bash
$ python main.py    --log_dir [path2_logdir] --data_dir [path2_datadir] --syn_dataset [synthetic_dataset_name]
                    --dataset [real_dataset_name] --dataset_3dmm [300W-3D & AFLW2000_dirname] --input_scale_size 108
                    --save_syn_dataset [saving_dir] --train_generator False --generate_dataset True --pretrained_gen [path2_logdir + /model.ckpt]
```

## Pretrained Model

You can download the [pretrained model](https://www.dropbox.com/s/n6g0saf772jx9w4/master_gen_0221_200317.zip?dl=0)

## More Results

<p align="center"><img width="100%" src="figures/images/results.png" /></p>

<br/>

## Citation
if you find this work is useful for your research, please cite our [paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Baris_Gecer_Semi-supervised_Adversarial_Learning_ECCV_2018_paper.pdf):

```
@inproceedings{gecer2018semi,
  title={Semi-supervised adversarial learning to generate photorealistic face images of new identities from 3D morphable model},
  author={Gecer, Baris and Bhattarai, Binod and Kittler, Josef and Kim, Tae-Kyun},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={217--234},
  year={2018}
}

```

<br/>

## Acknowledgement
This work was supported by the EPSRC Programme Grant ‘FACER2VM’ (EP/N007743/1). Baris Gecer is funded by the Turkish Ministry of National Education. This study is morally motivated to improve face recognition to help prediction of genetic disorders visible on human face in earlier stages.

Code borrows heavily from carpedm20's [BEGAN implementation](https://github.com/carpedm20/began-tensorflow).
