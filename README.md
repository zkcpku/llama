# LLaMA 

## Eval ppl & next token accuracy

### 构造数据集
```shell
python construct_corpus.py --inp_path CSN_python_func_code_string_top1000_norm.json --out_path CSN_python_func_code_string_top1000_norm.save.json --text_key func_code_string
```

### 运行测试

```shell
cd /code/llama/;
torchrun --nproc_per_node 1 test_ppl_acc_for_corpus.py --ckpt_dir /code/llama_weight/7B --tokenizer_path /code/llama_weight/tokenizer.model --max_seq_len 2048 --max_batch_size 4 --corpus_path /code/llama/CSN_python_func_code_string_top1000_norm.json --output_path /model/CSN_pythontop1000_7B.ppl.json --not_consider_eos True 2>&1| tee /code/CSN_pythontop1000_7B.ppl.log;
```

### 计算指标

```shell
python cal_ppl.py --inp_path CSN_python_func_code_string_top1000_norm.save.json
# token_num:  250169
# sentence_num:  1000
# ====================================
# sentence_ppl:  4.3770499722145955
# corpus_ppl:  3.607500909948467
# sentence_acc:  0.6797184633111134
# corpus_acc:  0.7150166487454481
# sentence_loss:  1.4763749751448632
# corpus_loss:  1.2830152639038244
```



# Original README
This repository is intended as a minimal, hackable and readable example to load [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) ([arXiv](https://arxiv.org/abs/2302.13971v1)) models and run inference.
In order to download the checkpoints and tokenizer, fill this [google form](https://forms.gle/jk851eBVbX1m5TAv5)

## Setup

In a conda env with pytorch / cuda available, run:
```
pip install -r requirements.txt
```
Then in this repository:
```
pip install -e .
```

## Download

Once your request is approved, you will receive links to download the tokenizer and model files.
Edit the `download.sh` script with the signed url provided in the email to download the model weights and tokenizer.

## Inference

The provided `example.py` can be run on a single or multi-gpu node with `torchrun` and will output completions for two pre-defined prompts. Using `TARGET_FOLDER` as defined in `download.sh`:
```
torchrun --nproc_per_node MP example.py --ckpt_dir $TARGET_FOLDER/model_size --tokenizer_path $TARGET_FOLDER/tokenizer.model
```

Different models require different MP values:

|  Model | MP |
|--------|----|
| 7B     | 1  |
| 13B    | 2  |
| 33B    | 4  |
| 65B    | 8  |

## FAQ

- [1. The download.sh script doesn't work on default bash in MacOS X](FAQ.md#1)
- [2. Generations are bad!](FAQ.md#2)
- [3. CUDA Out of memory errors](FAQ.md#3)
- [4. Other languages](FAQ.md#4)

## Reference

LLaMA: Open and Efficient Foundation Language Models -- https://arxiv.org/abs/2302.13971

```
@article{touvron2023llama,
  title={LLaMA: Open and Efficient Foundation Language Models},
  author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and Martinet, Xavier and Lachaux, Marie-Anne and Lacroix, Timoth{\'e}e and Rozi{\`e}re, Baptiste and Goyal, Naman and Hambro, Eric and Azhar, Faisal and Rodriguez, Aurelien and Joulin, Armand and Grave, Edouard and Lample, Guillaume},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}
```

## Model Card
See [MODEL_CARD.md](MODEL_CARD.md)

## License
See the [LICENSE](LICENSE) file.
