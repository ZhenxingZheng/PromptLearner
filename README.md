# PromptLearner
An action contains rich multi-modal information, and current methods generally map the action class to a digital number as supervised information to train models. However, numerical labels cannot describe the semantic content contained in the action. This paper proposes PromptLearner-CLIP for action recognition, where the text pathway uses PromptLearner to automatically learn the text content of prompt as the input and calculates the semantic features of actions, and the vision pathway takes video data as the input to learn the visual features of actions. To strengthen the interaction between features of different modalities, this paper proposes a multi-modal information interaction module that utilizes Graph Neural Network(GNN) to process both the semantic features of text content and the visual features of a video. In addition, the single-modal video classification problem is transformed into a multi-modal video-text matching problem. Multi-modal contrastive learning is used to disclose the feature distance of the same but different modalities samples. The experimental results showed that PromptLearner-CLIP could utilize the textual semantic information to significantly improve the performance of various single-modal backbone networks on action recognition and achieved top-tier results on Kinetics400, UCF101, and HMDB51 datasets.

## Pipeline
![pipeline](https://github.com/ZhenxingZheng/PromptLearner/figs/pipeline.jpg)


## Model training
1. modify the hyper-parameters in config.
2. prepare your dataset.
3. run main.py on the console. 
>python train.py



## Bibtex
```bibtex

@InProceedings{PromptLearner,
  title={PromptLearner-CLIP: Contrastive Multi-Modal Action Representation Learning with Context Optimization},
  author={Zheng, Zhenxing and An, Gaoyun and Cao, Shan and Yang, Zhaoqilin and Ruan, Qiuqi},
  booktitle={Asian Conference on Computer Vision (ACCV)},
  year={2022}
}
```

## Acknowledgments
This code is based on [ActionCLIP](https://github.com/sallymmx/ActionCLIP), [SupContrast](https://github.com/HobbitLong/SupContrast), and [CoOp](https://github.com/KaiyangZhou/CoOp)


