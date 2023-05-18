# CLearViD: Curriculum Learning for Video Description   
<!-- [[`arXiv`](www.google.com)][[`pdf`](www.google.com)] -->

Video description entails automatically generating coherent natural language sentences that narrate the content of a given video. We introduce CLearViD, a transformer-based model for video description generation that leverages curriculum learning to accomplish this task. In particular, we investigate two curriculum strategies: (1) progressively exposing the model to more challenging samples by gradually applying a Gaussian noise to the video data, and (2) gradually reducing the capacity of the network through dropout during the training process. These methods enable the model to learn more robust and generalizable features. Moreover, CLearViD leverages the Mish activation function, which provides non-linearity and non-monotonicity and helps alleviate the issue of vanishing gradients. Our extensive experiments and ablation studies demonstrate the effectiveness of the proposed model. The results on two datasets, namely ActivityNet Captions and YouCook2, show that CLearViD significantly outperforms existing state-of-the-art models in terms of both accuracy and diversity metrics. Finally, we discuss the limitations of the current work and outline the future plans.


## Environment Setup
1. Clone this repository
```bash 
git clone https://github.com/UARK-AICV/VLTinT.git
cd VLTinT
```


2. Prepare Conda environment 

```bash
conda env create -f environment.yml
conda activate pytorch
```


3. Add project root to `PYTHONPATH`
> Note that you need to do this each time you start a new session.

```bash
source setup.sh
```

## Data preparation

We assume to have following file structure after this preparation.
>If you want to change the file structure, please modify the `data_path` in `src/rtransformer/recursive_caption_dataset.py`
```
cache
  |- anet_vocab_clip.pt
  |- anet_word2idx.json
  |- yc2_vocab_clip.pt
  |_ yc2_word2idx.json
data
  |- anet
  |   |- c3d_env
  |   |- c3d_agent
  |   |_ clip_b16
  |       |- lang_feature
  |       |_ sent_feature
densevid_eval
preprocess
scripts
src
video_feature
  |- anet_duration_frame.csv
  |_ yc2_duration_frame.csv
```

### 
Our features extracted from rescaled videos of ActivityNet-1.3 can be downloaded below:
* Env features are [here](https://uark.box.com/s/01twnsrjxbf7d48wki5s5v43ri5p66vl).
* Agent features are [here](https://drive.google.com/file/d/1lOQG1FgDseRKDs3RNgpKd000OOZiag1s/view?usp=sharing).
* Lang features are [here](https://uark.box.com/s/un9t7vv2l61u1541krqfxqro1t9hfkm4).

You can use our preprocessed features above or process by yourself as follows:

<details>
<summary><b>1. Download data</b></summary>
<br>

1. Download raw videos of [ActivityNet](https://cs.stanford.edu/people/ranjaykrishna/densevid/) and [YouCook2](http://youcook2.eecs.umich.edu/download) and convert all the videos into `mp4` for the later process (you need `ffmpeg` for the script below).

    ```bash
    python preprocess/convert_to_mp4.py --video-root path/to/video/dir --output-root path/to/dir/*.mp4
    ```

1. Rescale each video into 1600 frames and extract the middle frame of every 16 frames (100 middle frames will be extracted). 

    ```
    python preprocess/rescale_video.py --video-root path/to/dir/*.mp4 --output-root path/to/dir/rescaled --frame-dir path/to/dir/middle_frames
    ```
</details>

<details>
<summary><b>2. Env feature extraction</b></summary>
<br>

1. To extract the visual features from the rescaled videos, we will use [this](https://github.com/vhvkhoa/SlowFast) repo.
    ```
    git clone https://github.com/vhvkhoa/SlowFast
    cd SlowFast
    python setup.py build develop
    ```
    Then, run the following command to extract the env features.
    ```
    python tools/run_net.py --cfg configs/Kinetics/SLOWONLY_8x8_R50.yaml --feature_extraction --num_features 100 --video_dir path/to/dir/rescaled --feat_dir path/to/data/[anet/yc2]/c3d_env TEST.CHECKPOINT_FILE_PATH models/SLOWONLY_8x8_R50.pkl NUM_GPUS 1 TEST.CHECKPOINT_TYPE caffe2 TEST.BATCH_SIZE 1 DATA.SAMPLING_RATE 1 DATA.NUM_FRAMES 16 DATA_LOADER.NUM_WORKERS 0
    ```
    
</details>

<details>
<summary><b>3. Agent feature extraction</b></summary>
<br>

### 
1. To extract the agent features, we will use [detectron](https://github.com/facebookresearch/detectron2) for bbox detection. 
    ```
    git clone https://github.com/vhvkhoa/detectron2
    python -m pip install -e detectron2
    wget https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl
    python tools/bbox_extract.py path/to/dir/rescaled path/to/dir/bbox --config-file configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml --sampling-rate 16 --target-frames 100 --opts MODEL.WEIGHTS model_final_f6e8b1.pkl
    ```
    Then follow the command below to extract the agent features.
    ```
    cd SlowFast
    python tools/run_net.py --cfg configs/Kinetics/SLOWONLY_8x8_R50.yaml --feature_extraction --num_features 100 --video_dir path/to/dir/rescaled --feat_dir path/to/data/[anet/yc2]/c3d_agent MODEL.NUM_CLASSES 200 TEST.CHECKPOINT_TYPE caffe2 TEST.CHECKPOINT_FILE_PATH models/SLOWONLY_8x8_R50.pkl NUM_GPUS 1 TEST.BATCH_SIZE 1 DATA.PATH_TO_BBOX_DIR path/to/dir/bbox DETECTION.ENABLE True DETECTION.SPATIAL_SCALE_FACTOR 32 DATA.SAMPLING_RATE 1 DATA.NUM_FRAMES 16 RESNET.SPATIAL_STRIDES [[1],[2],[2],[1]] RESNET.SPATIAL_DILATIONS [[1],[1],[1],[2]] DATA.PATH_TO_TMP_DIR /tmp/agent_0/
    ```
    
</details>

<details>
<summary><b>4. Lang feature extraction</b></summary>
<br>


1. To extract the linguistic features from those videos, run the following commands. Change `--dset_name` to `anet` or `yc2` to specify the dataset.

    ```
    python preprocess/build_lang_feat_vocab.py --dset_name [anet/yc2]
    python preprocess/extract_lang_feat.py --frame-root path/to/dir/middle_frames --output-root path/to/data/[anet/yc2]/clip_b16/lang_feature --dset_name [anet/yc2]
    python preprocess/extract_sent_feat.py --caption_root ./densevid_eval/[anet/yc2]_data/train.json --output_root path/to/data/[anet/yc2]/clip_b16/sent_features
   ```
   
</details>

### Build Vocabularies
Execute following command to create a vocablary for the model. Change `--dset_name` to `anet` or `yc2` to specify the dataset.
```
python preprocess/build_vocab.py --dset_name [anet/yc2] --min_word_count 1
python preprocess/extract_vocab_emb.py --dset_name [anet/yc2]
```

## Training

To train our best CLearViD model on ActivityNet Captions:

```
bash scripts/train_cl_noise.sh anet 0.3 25 0.3 25 --use_env --use_agent --use_lang --use_tint
```
To train our best CLearViD model on YouCook2 Captions:
```
bash scripts/train_cl_noise.sh yc2 0.35 25 0.35 25 --use_env --use_lang --use_tint
```

Training log and model will be saved at  `results/anet_re_*`.  
Once you have a trained model, you can follow the instructions below to generate captions. 

## Evaluation
1. Generate captions 
```
bash scripts/translate_greedy.sh anet_re_* [val/test]
```
Replace `anet_re_*` with your own model directory name. 
The generated captions are saved at `results/anet_re_*/greedy_pred_[val/test].json`


2. Evaluate generated captions
```
bash scripts/eval.sh anet [val/test] results/anet_re_*/greedy_pred_[val/test].json
```
The results should be comparable with the results of the paper. 

## Visualization
To visualize the result with the video (mp4):
```
python visualization/demo.py --input_mp4_folder /path/to/folder/contains/mp4 --output_mp4_folder /path/to/output/ --caption_file results/anet_re_*/greedy_pred_val.json --video_id v_5qsXmDi8d74
```

## Acknowledgement
We acknowledge the following open-source projects that we based on our work:

[VLTinT](https://github.com/UARK-AICV/VLTinT)

