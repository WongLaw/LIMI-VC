# LIMI-VC

Here is the official implementation of the paper, [LIMI-VC: A LIGHT WEIGHT VOICE CONVERSION MODEL WITH MUTUAL
INFORMATION DISENTANGLEMENT], which is modified from [FragmentVC](https://arxiv.org/abs/2010.14150).

## Demo

[demo](https://www.notion.so/LIMI-VC-9f3e5ca1da984a41a4f00b2e52c4c662)

## Usage

You can download the pretrained model as well as the vocoder following the link under **Releases** section on the sidebar.

The whole project was developed using Python 3.8, torch 1.6, and the pretrained model as well as the vocoder were turned to [TorchScript](https://pytorch.org/docs/stable/jit.html), so it's not guaranteed to be backward compatible.
You can install the dependencies with

```bash
pip install -r requirements.txt
```

If you encounter any problems while installing *fairseq*, please refer to [pytorch/fairseq](https://github.com/pytorch/fairseq) for the installation instruction.

### Wav2Vec

Please refer to FragmentVC

### Vocoder and Pre-trained Model

The WaveRNN-based neural vocoder is from [yistLin/universal-vocoder](https://github.com/yistLin/universal-vocoder) which is based on the paper, [Towards achieving robust universal neural vocoding](https://arxiv.org/abs/1811.06292).
The pre-trained model and vocoder can be accessed [here](https://drive.google.com/drive/folders/1pw0Ceu8VhG52YN6hfmUd8ckbs91UabW7?usp=share_link).


### Training

```bash
python train.py features --save_dir ./ckpts
```

### More training details can be refered to FragmentVC