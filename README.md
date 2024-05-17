

# A method for few-shot automatic modulation recognition based on DeepEMD.


A method for automatic modulation classification with small samples.A patent has been published.

### 方法介绍
The method classifies the corresponding category by calculating the optimal matching distance between known samples and the query sample. Initially, a feature extraction module is employed to split the signal into I and Q paths and extract features separately. Subsequently, a channel attention module is utilized to obtain signal features with channel weights. Finally, the features of the known signals and the query signal are sent to the EMD (Empirical Mode Decomposition) module to determine the corresponding category. In this process, a cross-reference mechanism is used to derive the weights for each feature. Moreover, for multiple known samples of each category, a class prototype extraction module is used to obtain the characteristic features of the category. Simulation experiment results on the RadioML2016.10A dataset demonstrate the effectiveness of the proposed method.

![image](https://github.com/bai-xiaosheng/DeepEMD/assets/68796611/e27fd884-0d7c-456b-bae3-9cddfafce24d)



### 主要贡献

1.	本文将信号自动调制分类问题转化为最优匹配问题，将EMD作为信号特征间的距离度量。
2.	本文提出使用通道注意力模块对提取到的通道特征赋予权重，与IQCNN网络相结合，有效的增加了相关特征的重要性。

### 引用

@manual{CN116204777A,
author = {  周峰 and     王力 and     **白东升** and     谭浩月 and     杨鑫瑶 and 石晓然},
 title = {一种基于DeepEMD的少镜头自动调制方式识别方法},
edition = {CN116204777A},
year = {2023},
pages = {16},
address = {710071 陕西省西安市太白南路2号}
}    


# 数据集介绍 

本项目使用的RadioML2016.10a数据集，数据集样本示例：
![image](https://github.com/bai-xiaosheng/DeepEMD/assets/68796611/617f7a5b-c938-42a2-99a2-bbf910b75d4a)


| 数据集属性       | 描述                                                         |
|----------------|--------------------------------------------------------------|
| 数据源          | 数字调制信号、莎士比亚著作的古腾堡文本、模拟调制信号、连续剧《Serial Episode》 |
| 生成方式        | Gnuradio + Python                                          |
| 数据格式        | IQ (In-phase and Quadrature)                              |
| 样本总量        | 220,000                                                    |
| 采样率偏移      | 标准偏差：0.01Hz；最大偏移：50Hz                        |
| 载波频率偏移    | 标准偏差：0.01Hz；最大偏移：500Hz                       |
| 正弦波数        | 8                                                           |
| 调制方式        | 8个数字调制方式：8PSK, BPSK, CPFSK, PAM4, 16QAM, 64QAM, QPSK；3个模拟调制方式：AM-DSB, AM-SSB, WBFM |
| 信噪比范围      | 具体范围未提供，需间隔                                     |
| 采样率          | 200KHz                                                     |
| 延迟设置        | 延迟值：[0.0, 0.9, 1.7]；幅度：[1, 0.8, 0.3]            |
| 噪声            | 加性高斯白噪声 (Additive White Gaussian Noise, AWGN)     |
| 信道环境        | 加性高斯白噪声、选择性衰落（莱斯 (Rician) + 瑞利 (Rayleigh)）、中心频率偏移 (Center Frequency Offset, CFO)、采样率偏移 (Sample Rate Offset, SRO) |

注意：需要自己从网上下载RML2016.10a数据集，放在data文件夹中，本项目提供了一些常用数据集放在dataset文件夹中






### 作者

@白东升

邮箱:22021211621@stu.xidian.edu.cn




