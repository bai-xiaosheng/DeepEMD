

# A method for few-shot automatic modulation recognition based on DeepEMD.


A method for automatic modulation classification with small samples.The method classifies the corresponding category by calculating the optimal matching distance between known samples and the query sample. Initially, a feature extraction module is employed to split the signal into I and Q paths and extract features separately. Subsequently, a channel attention module is utilized to obtain signal features with channel weights. Finally, the features of the known signals and the query signal are sent to the EMD (Empirical Mode Decomposition) module to determine the corresponding category. In this process, a cross-reference mechanism is used to derive the weights for each feature. Moreover, for multiple known samples of each category, a class prototype extraction module is used to obtain the characteristic features of the category. Simulation experiment results on the RadioML2016.10A dataset demonstrate the effectiveness of the proposed method.

### 上手指南

注意：本项目使用的RadioML2016.10a数据集,需要下载将数据集放在data文件夹中
数据集样本示例：
![image](https://github.com/bai-xiaosheng/DeepEMD/assets/68796611/617f7a5b-c938-42a2-99a2-bbf910b75d4a)


### 网络介绍

![image](https://github.com/bai-xiaosheng/DeepEMD/assets/68796611/d0d73012-39e5-4bc1-a3a2-3e1a789f01f3)




### 作者

白东升

邮箱:969900860@qq.com




