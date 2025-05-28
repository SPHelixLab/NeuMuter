
# Decoupling Memories, Muting Neurons: Towards Practical Machine Unlearning for Large Language Models (NeuMuter)----ACL Findings 2025


## About The Project
**NeuMuter** has been accepted by *Findings of ACL 2025*, which is a machine unlearning (MU) method for large language models (LLMs).
It eliminates the influence of unlearned data from LLMs by muting the outputs of neurons responsible for memorizing the unlearned data with only availability of the unlearned data. 

<br>

## Getting Started
### File Structure 
```
NeuMuter-master
├── data
├── utils
│   ├── modeling_hardconcrete.py
│   ├── utils.py
│   ├── NeuMuter.py
└── main.py
```
There are several parts of the code:

- `modeling_hardconcrete.py`: This file mainly contains the structure and computation method for the learnable mask. 
- `utils.py`: This file mainly contains the preprocessing of the dataset and model loading. 
- `NeuMuter.py`: This file contains the funcations of the whole unlearning scheme, including neuron localization and memorization removal.
- `main.py`: The main function of **NeuMuter**. 

<br>

### Requirements

* python 3.8.19 
* [pytorch](https://pytorch.org/get-started/locally/) 1.13.1 & torchvision 0.14.1
* CUDA 11.7 and above are recommended (this is for GPU users)
* numpy 1.24.4
* transformers 4.39.3

Before running the project, make sure you have set up the right environment and installed the above required packages.

<br>

### Hyper-parameters 
The settings of **NeuMuter** are determined in the parameter **args** in **main.py**. Here, we mainly introduce the important hyper-parameters.
- device: which device to use (CPU or GPU). Default: cuda:0.
- start_mask_layer: the initial layer where the mask is injected. Default: 1.
- r: threshold for selected the target neurons. Default: 0.15.
- lambda_l1: the sparsity regularization coefficient. Default: 500.
- eta: the coefficient of the KL divergence optimization loss on neighbor samples. Default: 5.
- epoch: the training epochs to train the injected mask. Default: 200.
- per_nei_num: the number of neighbour samples generated for per unlearned data. Default: 1.
- mask_prob: the replacement radio of generated neighbor samples. Default: 1.
- lr: the learning rate to train the injected mask. Default: 0.1.

<br>

### Run
You could run `main.py` in your python IDE directly.
The example codes below show the workflow to perform a complete unlearning process, which is in `main.py`.

```python
def main(args):
    # load pretrained model and unlearned dataset
    seed_everything()
    tokenizer, model = load_pretrained(args)
    pretrained_model = copy.deepcopy(model)  
    dataset = MyCustomDataset(args, tokenizer)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # NeuMuter
    inject_mask(model, args)
    # Memorization Neuron Localization
    NeuMuter_localization(args, model, pretrained_model,tokenizer,dataset, data_loader, device)
    # Memorization Removal
    NeuMuter_removal(args, model)
```

You can also run main.py using the cmd command.

```python
$ python main.py --model_name "gpt-neo-1.3B" --lambda_l1 500 --eta 5 --epoch 200 --lr 0.1
```
<br>

## Note
- We use the unlearned samples from: https://github.com/joeljang/knowledge-unlearning.
- A GPU is not required, but we recommend using one to increase the running speed. 
<br>

## 📄 Cite Our Paper

```bibtex
@inproceedings{,
  title={Decoupling Memories, Muting Neurons: Towards Practical Machine Unlearning for Large Language Models},
  author={Hou, Lishuai and Wang, Zixiong and Liu, Gaoyang and Wang, Chen and Peng, Kai and Liu, Wei},
  year={2025},
  booktitle={Proceedings of ACL Findings}
}










