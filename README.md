# <img alt="icon" src="figures/icon.png" height=30> MUSE: Machine Unlearning Six-Way Evaluation for Language Models

This repository provides the original implementation of *Machine Unlearning Six-Way Evaluation for Language Models* by Weijia Shi*, Jaechan Lee*, Yangsibo Huang*, Sadhika Malladi, Jieyu Zhao, Ari Holtzman, Daogao Liu, Luke Zettlemoyer, Noah A. Smith, and Chiyuan Zhang. (*Equal contribution)


[Paper](https://www.arxiv.org/abs/2407.06460) | [Website](https://muse-bench.github.io/) |  [Leaderboard](https://huggingface.co/spaces/muse-bench/MUSE-Leaderboard) | [MUSE-News Benchmark](https://huggingface.co/datasets/muse-bench/MUSE-News) | [MUSE-News Benchmark](https://huggingface.co/datasets/muse-bench/MUSE-News) |  [MUSE-Books Benchmark](https://huggingface.co/datasets/muse-bench/MUSE-Books) 

🎉 Happy to share that MUSE is now incorporated into [open-ulearning](https://github.com/locuslab/open-unlearning). Please use it for evaluation.


## Overview
**MUSE** is a comprehensive machine unlearning evaluation benchmark that assesses six desirable properties for unlearned models: (1) no verbatim memorization, (2) no knowledge memorization, (3) no privacy leakage, (4) utility preservation for non-removed data, (5) scalability with respect to removal requests, and (6) sustainability over sequential unlearning requests.

<p align="center">
  <img src="figures/main.png" width="80%" height="80%">
</p>

:star: If you find our implementation and paper helpful, please consider citing our work :star: :

```bibtex
@article{shi2024muse,
        title={MUSE: Machine Unlearning Six-Way Evaluation for Language Models},
        author={Weijia Shi and Jaechan Lee and Yangsibo Huang and Sadhika Malladi and Jieyu Zhao and Ari Holtzman and Daogao Liu and Luke Zettlemoyer and Noah A. Smith and Chiyuan Zhang},
        year={2024},
        eprint={2407.06460},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/2407.06460},
}
```

## Content
- [ MUSE: Machine Unlearning Six-Way Evaluation for Language Models](#-muse-machine-unlearning-six-way-evaluation-for-language-models)
  - [Overview](#overview)
  - [Content](#content)
  - [🛠️ Installa and \`Newstion](#️-installa-and-newstion)
    - [Conda Environment](#conda-environment)
  - [📘 Data \& Target Models](#-data--target-models)
  - [🚀 Run unlearning baselines](#-run-unlearning-baselines)
- [🔍 Evaluation of Unlearned Models](#-evaluation-of-unlearned-models)
    - [Example Command](#example-command)
  - [Customize Your Evaluation](#customize-your-evaluation)
    - [`eval_model` Function](#eval_model-function)
  - [➕ Add to the Leaderboard](#-add-to-the-leaderboard)

## 🛠️ Installa and `Newstion

### Conda Environment

To create a conda environment for Python 3.10, run:
```bash
conda env create -f environment.yml
conda activate py310
```

## 📘 Data & Target Models

Two corpora `News` and `Books` and the associated target models are available as follows:

| Domain | <div style="text-align: center">Target Model for Unlearning</div> | Dataset |
|----------|:------------------------------:|----------| 
| News | [Target model](https://huggingface.co/muse-bench/MUSE-News_target) | [Dataset](https://huggingface.co/datasets/muse-bench/MUSE-News) |
| Books | [Target model](https://huggingface.co/muse-bench/MUSE-Books_target) | [Dataset](https://huggingface.co/datasets/muse-bench/MUSE-Books) | 

Before proceeding, load all the data from HuggingFace to the root of this repostiory by running the following instruction:
```
python load_data.py
```

## 🚀 Run unlearning baselines

To unlearn the target model using our baseline method, run `unlearn.py` in the `baselines` folder. Example scripts `baselines/scripts/unlearn_news.sh` and `scripts/unlearn_books.sh` in the `baselines` folder demonstrate the usage of `unlearn.py`. Here is an example:
```bash
algo="ga"
CORPUS="news"

python unlearn.py \
        --algo $algo \
        --model_dir $TARGET_DIR --tokenizer_dir 'meta-llama/Llama-2-7b-hf' \
        --data_file $FORGET --retain_data_file $RETAIN \
        --out_dir "./ckpt/$CORPUS/$algo" \
        --max_len $MAX_LEN --epochs $EPOCHS --lr $LR \
        --per_device_batch_size $PER_DEVICE_BATCH_SIZE
```

- `algo`: Unlearning algorithm to run (`ga`, `ga_gdr`, `ga_klr`, `npo`, `npo_gdr`, `npo_klr`, or `tv`).
- `model_dir`: Directory of the target model.
- `tokenizer_dir`: Directory of the tokenizer.
- `data_file`: Forget set.
- `retain_data_file`: Retain set for GDR/KLR regularizations if required by the algorithm.
- `out_dir`: Directory to save the unlearned model (default: `ckpt`).
- `max_len`: Maximum input length (default: 2048).
- `per_device_batch_size`, `epochs`, `lr`: Hyperparameters.

----
**Resulting models are saved in the `ckpt` folder as shown:**
```
ckpt
├── news/
│   ├── ga/
│   │   ├── checkpoint-102
│   │   ├── checkpoint-204
│   │   ├── checkpoint-306
│   │   └── ...
│   └── npo/
│       └── ...
└── books/
    ├── ga
    └── ...
```
# 🔍 Evaluation of Unlearned Models

To evaluate your unlearned model(s), run `eval.py` from the root of this repository with the following command-line arguments:

- `--model_dirs`: A list of directories containing the unlearned models. These can be either HuggingFace model directories or local storage paths.
- `--names`: A unique name assigned to each unlearned model in `--model_dirs`. The length of `--names` should match the length of `--model_dirs`.
- `--corpus`: The corpus to use for evaluation. Options are `news` or `books`.
- `--out_file`: The name of the output file. The file will be in CSV format, with each row corresponding to an unlearning method from `--model_dirs`, and columns representing the metrics specified by `--metrics`.
- `--tokenizer_dir` (Optional): The directory of the tokenizer. Defaults to `meta-llama/Llama-2-7b-hf`, which is the default tokenizer for LLaMA.
- `--metrics` (Optional): The metrics to evaluate. Options are `verbmem_f` (VerbMem Forget), `privleak` (PrivLeak), `knowmem_f` (KnowMem Forget), and `knowmem_r` (Knowmem Retain, i.e., Utility). Defaults to evaluating all these metrics.
- `--temp_dir` (Optional): The directory for saving intermediate computations. Defaults to `temp`.

### Example Command

Run the following command with placeholder values:

```bash
python eval.py \
  --model_dirs "jaechan-repo/model1" "jaechan-repo/model2" \
  --names "model1" "model2" \
  --corpus books \
  --out_file "out.csv"
```

## Customize Your Evaluation

You may want to customize the evaluation for various reasons, such as:
- Your unlearning method is applied at test-time (e.g., interpolates the logits of multiple model outputs), so there is no saved checkpoint for your unlearned model.
- You want to use a different dataset for evaluation.
- You want to use an MIA metric other than the default one (`Min-40%`) for the PrivLeak calculation, such as `PPL` (perplexity).
- You want to change the number of tokens greedily decoded from the model when computing `VerbMem` or `KnowMem`.

<details>
<summary>Click here if interested in customization:</summary>

The maximum amount of customization that we support is through the `eval_model` function implemented in `eval.py`. This function runs the evaluation for a single unlearned model and outputs a Python dictionary that corresponds to one row in the aforementioned CSV. Any additional logic—such as loading the model from a local path, evaluating multiple models, or locally saving an output dictionary—is expected to be implemented by the client.

### `eval_model` Function

The function accepts the following arguments:

- `model`: An instance of `LlamaForCausalLM`. Any HuggingFace model with `self.forward` and `self.generate` implemented should suffice.
- `tokenizer`: An instance of `LlamaTokenizer`.
- `metrics` (Optional): Same as earlier.
- `corpus` (Optional): The corpus to run the evaluation on, either `news` or `books`. Defaults to `None`, which means the client must provide all the data required by the calculations of `metrics`.
- `privleak_auc_key` (Optional): The MIA metric to use for calculating PrivLeak. Defaults to `forget_holdout_Min-40%`. The first keyword (`forget`) corresponds to the non-member data (options: `forget`, `retain`, `holdout`), the second keyword (`holdout`) corresponds to the member data (options: `forget`, `retain`, `holdout`), and the last keyword (`Min-40%`) corresponds to the metric name (options: `Min-20%`, `Min-40%`, `Min-60%`, `PPL`, `PPL/lower`, `PPL/zlib`).
- `verbmem_agg_key` (Optional): The Rouge-like metric to use for calculating VerbMem. Defaults to `mean_rougeL`. Other options include `max_rougeL`, `mean_rouge1`, and `max_rouge1`.
- `verbmem_max_new_tokens` (Optional): The maximum number of new tokens for VerbMem evaluation. Defaults to 128.
- `knowmem_agg_key` (Optional): The Rouge-like metric to use for calculating KnowMem. Defaults to `mean_rougeL`.
- `knowmem_max_new_tokens` (Optional): The maximum number of new tokens for KnowMem evaluation. Defaults to 32.
- `verbmem_forget_file`, `privleak_forget_file`, `privleak_retain_file`, `privleak_holdout_file`, `knowmem_forget_qa_file`, `knowmem_forget_qa_icl_file`, `knowmem_retain_qa_file`, `knowmem_retain_qa_icl_file` (Optional): Specifying these file names overrides the corresponding data files. For example, setting `corpus='news'` and specifying `privleak_retain_file` would only override `privleak_retain_file`; all other files default to those associated with the `news` corpus by default.
- `temp_dir` (Optional): Same as earlier.

</details>

## ➕ Add to the Leaderboard

Submit the output CSV file generated by `eval.py` to our [HuggingFace leaderboard](https://huggingface.co/spaces/muse-bench/muse_leaderboard). You are additionally asked to specify the corpus name (either `news` or `books`) that your model(s) were evaluated on, the name of your organization, and your email.
