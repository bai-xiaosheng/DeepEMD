# A Method for Few-Shot Automatic Modulation Recognition Based on DeepEMD

## Introduction

This method achieves automatic modulation classification by calculating the optimal matching distance between known samples and the query sample. Initially, a feature extraction module splits the signal into I and Q paths to extract features separately. Subsequently, a channel attention module is employed to obtain signal features with channel weights. Finally, the features of known signals and the query signal are sent to the EMD (Empirical Mode Decomposition) module to determine the corresponding category. A cross-reference mechanism is utilized to derive the weights for each feature. Additionally, for multiple known samples of each category, a class prototype extraction module is used to obtain the characteristic features. Simulation experiment results on the RadioML2016.10A dataset demonstrate the effectiveness of the proposed method.

![RadioML2016.10a Dataset Example](https://github.com/bai-xiaosheng/DeepEMD/assets/68796611/e27fd884-0d7c-456b-bae3-9cddfafce24d)

### Main Contributions

1. The paper transforms the problem of signal automatic modulation classification into an optimal matching problem, using EMD as the distance measure between signal features.
2. The paper proposes a channel attention module to assign weights to the extracted channel features, in combination with the IQCNN network, effectively increasing the importance of relevant features.

### Citation

```plaintext
@manual{CN116204777A,
  author = {Zhou, Feng and Wang, Li and Bai, Dongsheng and Tan, Haoyue and Yang, Xinyao and Shi, Xiaoran},
  title = {A method for few-shot automatic modulation recognition based on DeepEMD},
  edition = {CN116204777A},
  year = {2023},
  pages = {16},
  address = {710071 Xi'an, Shaanxi, China}
}
```
### Dataset Introduction

The project utilizes the RadioML2016.10a dataset. Below is an example of the dataset samples:

![Dataset Sample](https://github.com/bai-xiaosheng/DeepEMD/assets/68796611/617f7a5b-c938-42a2-99a2-bbf910b75d4a)

| Dataset Attribute       | Description                                                                                   |
|------------------------|-----------------------------------------------------------------------------------------------|
| Data Source            | Digital modulation signals, Shakespeare's Gutenberg texts, analog modulation signals, TV series "Serial Episode" |
| Generation Method      | Gnuradio + Python                                                                           |
| Data Format            | IQ (In-phase and Quadrature)                                                               |
| Total Samples          | 220,000                                                                                    |
| Sampling Rate Offset   | Standard Deviation: 0.01Hz; Maximum Offset: 50Hz                                      |
| Carrier Frequency Offset| Standard Deviation: 0.01Hz; Maximum Offset: 500Hz                                   |
| Sine Wave Count        | 8                                                                                           |
| Modulation Schemes     | 8 digital modulation schemes: 8PSK, BPSK, CPFSK, PAM4, 16QAM, 64QAM, QPSK; 3 analog modulation schemes: AM-DSB, AM-SSB, WBFM |
| SNR Range              | Specific range not provided, interval required                                           |
| Sampling Rate          | 200KHz                                                                                       |
| Delay Settings         | Delay values: [0.0, 0.9, 1.7]; Amplitudes: [1, 0.8, 0.3]                                |
| Noise                  | Additive White Gaussian Noise (AWGN)                                                       |
| Channel Environment    | AWGN, Selective Fading (Rician + Rayleigh), Center Frequency Offset (CFO), Sample Rate Offset (SRO) |

Please note that you will need to download the RadioML2016.10a dataset from the web and place it in the `data` folder. The project provides some common datasets in the `dataset` folder.

### Authors

## Bai, Dongsheng
- **Role**: Lead Researcher / Author
- **Email**: [22021211621@stu.xidian.edu.cn](mailto:22021211621@stu.xidian.edu.cn)
- **Institution**: Xidian University, Xi'an, Shaanxi, China
