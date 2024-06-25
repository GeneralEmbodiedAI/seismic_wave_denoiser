# Seismic Wave Denoiser

## Abstract

Denoising of scismic waveform signal is crucial for seismic monitoring and seismological research. To this end, we propose an
end-to-end deep learning method for denoising seismic waveforms. The method
combines the deep convolutional network with multi-head self-attention
mechanism. We employ residual encoder-decoder structure, which is particularly
well-suited for processing signals with complex backgrounds and rich details,
while the multi-head self-attention mechanism can capture long-range
dependencies. By jointly constraining the model with a consistent correlation
loss and a frequency-domain mean squared error loss, outstanding denoising
performance is achieved in both the time and frequency domains. Additionally,
this method requires only one training process to perform jointly denoise
three-component seismic waveform signals, capturing the inter-component
relationships. Evaluation on the publicly available dataset STEAD shows that
our method outperforms traditional and existing deep learning methods in two
key metrics: peak signal-to-noise ratio (PSNR) and signal correlation
coefficient (CC), achieving a Pearson correlation of 0.918 and a PSNR of 36.79,
reaching the state-of-the-art performance. Furthermore, our method provides a
powerful solution for denoising seismic waveform signals, which is of great
significance for seismic monitoring and seismological research.

## Usage

### Preparation

Download dataset from [zenodo](https://zenodo.org/records/11094536]).

Then, put four files of `train_noise.hdf5`, `train.hdf5`, `test_noise.hdf5` and `test.hdf5` into `./datas` folder.

### Weights
Download weights from [Google Drive](https://drive.google.com/file/d/14BMXpXG57MKqMny6Qyf3vJxnp4GT05D3/view?usp=drive_link) or [KuaKe NetDisk](https://pan.quark.cn/s/c726b97f6734).

Then, put the file to './saved_model' folder.


### Training

To train the model, please run:

```python
python train.py 
```

### Evaluation

To evaluate the model, please run:

```python
python inference.py
```
