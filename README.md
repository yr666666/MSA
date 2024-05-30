# MSA
Code of paper “Transcending Fusion: A Multi Scale Alignment Method for Remote Sensing Image Text Retrieval”
Arxiv:https://arxiv.org/pdf/2405.18959

**Preparation:**

0. Setup Environment
1. Update the data path in option/RSITMD_AMFMN.yaml to match your own data location.
2. Modify the path for the ResNet weights on line 250 of resnet.py. Download link for ResNet weights:
ResNet50: https://download.pytorch.org/models/resnet50-19c8e357.pth
3. You will need to modify the path for /bert-base-uncased/ on line 482 of retrieval.py and line 225 of model.py. /bert-base-uncased/ can be downloaded from the Hugging Face official website.
4. Set batch_size in option/RSITMD_AMFMN.yaml. Training epoch and learning rate can be modified on lines 480 and 481 of retrieval.py.

   
**Execution:**
Run the following command: python retrieval.py

This is running convergence curve:
![image](https://github.com/yr666666/MSA/assets/41632617/de2e546d-edde-4ae2-a6be-d0763169ddb7)
