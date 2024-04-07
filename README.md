#  SFT_7B：在Agent调度任务上微调语言模型
这里我们验证语言模型在agent调度上的能力，即在agent调度任务进行有监督微调（SFT）。
## 1.环境依赖
开始训练前，请至少在Anaconda基础上安装如下依赖包：
```commandline
torch==2.2.1
transformers==4.38.2
accelerate==0.28.0
peft==0.9.0
jsonlines==4.0.0
deepspeed==0.14.0
```

## 2.数据准备
我们在 [gorilla](https://gorilla.cs.berkeley.edu/) 发布的数据集。该数据集的任务可以描述为使用语言模型来调用开源社区api。

具体来说，请将 [apibench](https://github.com/ShishirPatil/gorilla/tree/main/data) 中的数据下载，并放置于你的本地文件夹中。

```commandline
$你的文件夹路径/apibench/*
```
请使用以上路径替换`train.sh`中`data_path`参数的对应部分。

## 3.模型准备

参考 gpt2-base 模型的使用：

- 需要您去 [huggingface](https://huggingface.co/models)上搜索`模型名称`，如gpt2-base。然后在对应界面下载模型参数以及对应包括tokenizer、config等文件在内的文件。
并将其放置于你的本地文件夹中。

- 将模型对应 modeling.py 文件放置于 `modeling` 下您自己新建的对应该模型的文件夹中。

- 在`sft.py` 中，导入该模型并添加加载该模型的代码

- 在`train.sh`中， 添加该模型参数文件对应的文件路径，以及 tokenizer 对应的特殊 token.

本套代码理论上支持所有所有GPT形式的自回归语言模型模型,只需要大约几分钟的适配。

注意: 类似T5等基于编码器-解码器形式的模型，本套代码还尚未支持，按上述形式添加可能存在问题。

```commandline
$你的文件夹路径/gpt2_base/*
```
请使用以上路径替换`train.sh`中`model_name_or_path`参数的对应部分。

模型定义文件可以通过transformers中直接导入。在本项目中，为了方便日后需要对模型结构进行改动，我们创建了modeling文件夹。请将前面在下载模型参数界面中包含的modeling_gpt.py放到`modeling/gpt2`文件间中

如果想尝试训练其他模型，准备步骤参考如上。

## 3.训练模型
```commandline
sh train.sh
```
经验来讲，如果初始的loss在10以内，基本说明模型参数导入以及模型训练没问题。

## 4.关键参数
在`train.sh`中有一些关键的参数值得关注：
```commandline
only_api_call：是否只输出api_call这个结果
target_loss：是否只对输出的token计算loss
```


## 5.TODO LIST

- [x] Target Loss
- [x] Inferring script
- [ ] T5 Model
- [ ] A Platform for displaying the results of different models in a GSB way