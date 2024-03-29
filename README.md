# Robust Federated Learning with Attack-Adaptive Aggregation
This is the code for our [paper](https://arxiv.org/abs/2102.05257).

- **8/21/2021** : Our paper won the [best paper award](http://federated-learning.org/fl-ijcai-2021/BestPaper.png) in FTL-IJCAI'21 :sunglasses:.
- **7/20/2021** : Our paper has been accepted by the [FTL-IJCAI'21](http://federated-learning.org/fl-ijcai-2021/) workshop for oral presentation.
- **7/20/2021** : We have released the latest codes for automating experiments in the [runx branch](https://github.com/cpwan/Attack-Adaptive-Aggregation/tree/runx).

## Abstract
Federated learning is vulnerable to various attacks, such as model poisoning and backdoor attacks, even if some existing defense strategies are used. To address this challenge, we propose an attack-adaptive aggregation strategy to defend against various attacks for robust federated learning. The proposed approach is based on training a neural network with an attention mechanism that learns the vulnerability of federated learning models from a set of plausible attacks. To the best of our knowledge, our aggregation strategy is the first one that can be adapted to defend against various attacks in a data-driven fashion. Our approach has achieved competitive performance in defending model poisoning
and backdoor attacks in federated learning tasks on image and text datasets.

<p align="center">
<img src="https://i.imgur.com/Aqjwra7.png" alt="overview" width="80%"/>
</p>

## Code structures
- `server.py, client,py, dataloader.py`: our FL framework
- `main.py`: entry point of the command line tool, it allocates resources and print out FL configs
- `parser.py`: a parser of the FL configs, also assigns possible attackers given the configs
- `_main.py`: the real controller of the FL experiments 
- `./aaa`: the architecture and the training configurations of our **A**ttack-**A**daptive **A**ggregation model
- `./rules`: previous rule-based approaches for defending adversarial attacks
- `./tasks`: the FL tasks, including the definition of the datasets and the DNN models involved
- `./utils`: helper functions for allocating resources, adapting and evaluating the defenses in our FL framework




## Running Federated Learning tasks
### Demos 
We provide a simple framework for simulating FL in a local machine. Take a look at the example in [FL.ipynb](FL.ipynb) for its usages. 

### Command line tools
```
python main.py --dataset cifar --AR attention --attacks "name_of_the_attack" --n_attacker_labelFlippingDirectional 1 --loader_type dirichlet --output_folder "subfolder_name_under_<logs>" --experiment_name "experiment 1" --epochs 30 --path_to_aggNet "./aaa/attention_cifar.pt --save_model_weights"
```
Check out `parser.py` for the use of the arguments, most of them are self-explanatory. 
- You may give an arbitrary name to `--attacks`. If the name contains "backdoor" (case insensitive), it will trigger the server to test on the backdoor samples.
- The `--output_folder` is under the `./logs` directory. 
- The `experiment_name` will be the name shown on the tensorboard. 
- If you choose the aggregation rule `--AR` to be `attention` or `mlp` , then you will also need to specify the location of the model parameters in `--path_to_aggNet`.
- If `--save_model_weights` is specified, the local models and their label (benign or malicious) will be saved under `./AggData` directory. This is how we can collect empirical update vectors.
## Training Attack-Adaptive Aggregation model
### Demos 
Check [AAA.ipynb](AAA.ipynb) for a demo of our Attack-Adaptive Aggregation on a synthetic dataset.

### Command line tools
To train on the empirical update vectors of FL, you need to locate the list of paths that contains the saved update vectors. The list of paths can be specified by concatenating `--path_prefix` and each line of the `--train_path`. Please check `./aaa/args_cifar_backdoor` for an example.  After that, run the following
```
python aaa/train_attention.py @aaa/args_cifar_backdoor --eps 0.005 --scale 0.5
```
where `--eps` and `--scale` are the hyperparameters of our model, please check out our paper for the definition.
