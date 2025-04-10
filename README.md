# 🧠 LaTeXify: Handwritten Math to LaTeX with TrOCR

> An end-to-end machine learning project that trains a model to read and translate **handwritten math** into **LaTeX code**.

This is a **fine-tuned** version of [`microsoft/trocr-base-handwritten`](https://huggingface.co/microsoft/trocr-base-handwritten), a transformer-based optical character recognition model, adapted to work with handwritten math images and structured math syntax. You can find it on Hugging Face as [`tjoab/latex_finetuned`](https://huggingface.co/tjoab/latex_finetuned).

In this repo, you'll find:

- 🧱 **Data + preprocessing pipeline** from raw InkML files to model-ready image/label pairs
- 🧠 **TrOCR fine-tuning** using custom PyTorch training loop and `DataLoaders`
- 💾 **Use of gradient accumulation + mixed precision** to train on limited hardware
- 📉 **Model logging and checkpointing** for segmented training sessions
- 🖥️ **Lightweight demo** to showace model inference
- 🚀 **(Coming soon) Docker containerization** for cloud deployment on AWS SageMaker

## 🎯 Motivation

Most OCR systems perform well on natural language, but they **struggle with mathematical notation** — especially when it's handwritten. LaTeXify aims to understand the **structure** of math.

Math expressions aren’t linear like natural text — they’re inherently **2D**. You’re not just translating symbols, you’re interpreting spatial relationships: superscripts, fractions, nested square roots, integrals with bounds, and multi-level subscripts. This makes math recognition fundamentally different from typical OCR or sequence-to-sequence tasks.

I wanted to directly output LaTeX from rasterized handwriting — no intermediate character recognition, no symbol lookup, just end-to-end learning.

## 🔩 Training Strategy: Mixed Precision & Gradient Accumulation

Training a transformer model is pretty demanding — especially on commodity hardware. So to make this proccess more accessible to more people, I used a couple tricks and was able to train this model on a NVIDIA T4 with 16GM of VRAM.

- By using **mixed precision (`torch.cuda.amp`)**
  - Reduced RAM consumption by using `float16` where possible
  - Look out for `autocast()` and `GradScaler()` calls inside of `train/train.py`
  - ```python
    with autocast():
          outputs = model(pixel_values=images, labels=labels)
          loss = outputs.loss / grad_accumulation_steps
    ```
- Small batche sizes are inherently noisey, and transformer models benefit more from larger batches
  - But increasing batch size could cause memory issues
    - Introduce **gradient accumulation**
      - Enables a larger **effective batch** by accumulating gradients over several small batches, then updating model weights
      - This improves the quality of our gradient signal without increasing **peak memory load per step**
      - Essentially trading time for memory, because compute is cheap while memory is scarce

## 📦 Dependencies

This project uses `pycairo` for rendering handwritten strokes. If you plan on using the `DataLoader` from `train/dataset.py`, you **must** install these system libraries **prior** to installing the Python dependencies:

```bash
sudo apt-get install -y libcairo2-dev libjpeg-dev libgif-dev
```

Otherwise, you can remove `pycairo` from the `requirements.txt` and run:

```bash
pip install -r requirements.txt
```

## ⚙️ Configuration via YAML

Would you like to change your training parameters, choose your own model to fine-tune, or toggle model checkpoints/logs? No need to touch any of the Python logic — everything is driven from config files. Take a look in in `train/train_config.yaml` or `evaluation/eval_config.yaml` and make your changes.

```yaml
model_name: microsoft/trocr-base-handwritten
data_dir: ./data/mathwriting-2024/train/

batch_size: 8
grad_accumulation: 8
learning_rate: 5e-5
warmup_steps: 1000

perform_logs: false
log_dir: ./train/logs/
```

## 📈 Evaluation

I decided to evaluate performance using Character Error Rate (CER) which is defined below. It basically tells you what fraction of the characters in the target output were wrong — either missing, incorrect, or extra.

- `CER = (Substitutions + Insertions + Deletions) / Total Characters in Ground Truth`

##### ✅ Why CER?

Math expressions are **structurally sensitive.** Shuffling even a single character can completely change the meaning:

- `x^2` vs. `x_2`
- `\frac{a}{b}` vs. `\frac{b}{a}`

In the past I've worked with BLEU which is a sequence level metric, however I settled on CER because it penalizes small syntax error more harshly.

Evalution of `tjoab/latex_finetuned` yeilded a CER of 14.9%.

## 🛠️ Built With

- 🤗 [**HuggingFace Transformers**](https://huggingface.co/tjoab/latex_finetuned) — for TrOCR and tokenizers
- 🔥 [**PyTorch**](https://pytorch.org/) — for training loops, data loading, and AMP
- 🖼️ [**Streamlit**](https://latexify.streamlit.app/) — model demo (👈 click the link)
