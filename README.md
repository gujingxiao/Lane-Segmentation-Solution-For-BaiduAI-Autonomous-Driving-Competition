# Lane-Segmentation-Solution-For-BaiduAI-Autonomous-Driving-Competition
Lane Segmentation 1st Place Solution for Baidu AI PaddlePaddle Autonomous Driving Competition
[无人车车道线检测挑战赛](http://aistudio.baidu.com/aistudio/#/competition/detail/5),为了推广并进一步优化PaddlePaddle，本次比赛的训练框架也被确定为使用PaddlePaddle。在为期四个月的比赛中，初步了解了PaddlePaddle，并且能够学以致用，收获颇多。最终以0.63547分数取得第一名，ID为Seigato。

### 总体描述
本次比赛图像分辨率非常的大，3384x1710，我的设备是两块1070Ti，可用显存15.6G，无奈只好忍痛缩小图像。下面说下主要思路：

【1】经过对label的分析，首先裁掉了图片最上部的3384x690的图像，因为这部分都是天空和树木，没有正样本的存在，裁掉后还可以减小图像的压缩比例，一举两得。

【2】裁掉后，为了尽可能保持图像比例，在比赛不同阶段采用了3种分辨率（768x256,1024x384,1536x512）进行训练，并且大分辨率的模型是基于前一个小分辨率的预训练模型而来（这样做的原因是大分辨率由于感受野的问题，很难训练，所以先从小分辨率入手，不仅可以快速验证模型的有效性，还能够提供较好的特征与分布；事实证明，基于小分辨率的预训练，确实可以帮助大分辨率更好地进行收敛）。

【3】本次比赛精简了paddle提供的deeplabv3p的模型，并且根据以往在kaggle的比赛经验，重新设计了两个基于resnet残差模块的unet网络，最终成绩了采用了三个网络的平均结果。

【4】在训练策略上，采用了修改版的Cycle LR策略（原本写了个全自动的，但是由于训练偶尔出现loss为NaN的情况，耽误了很多时间，所以只好改为手动了）。采用Adam，前3个epochs采用默认参数训练（lr=0.001），在之后3个epochs的训练中，每个epoch平均分配出6个改变lr的地方，改变方式为:0.001-0.0006-0.0003-0.0001-0.0004-0.0008-0.001。最后两个epoch一般采用0.0004-0.0001之间学习率训练的策略。由于测试集与训练集在图像质量和视觉感知上差距不小，太小的学习率很容易导致过拟合，所以最小的学习率采用0.0001。在8-10个epochs后，训练基本上就结束了。

【5】在loss function的使用上，最先的三个epochs采用sigmoid bce，后面的训练中采用bce + dice的方式，这种方式会比单一的bce提升0.01-0.015。

【6】数据清洗上，最一开始采用了全部数据训练，发现loss经常出现不规则的跳动，经过排查，发现road 3存在几乎一半以上图像过曝的问题，并且road 3大多在强光下拍摄，不符合测试集的分布，所以很果断的舍弃了road 3，分数也提升了0.01左右（好神奇。。。）。

【7】数据增强上，由于第五类和第八类训练数据较少（但是测试集中占比不少），所以针对这两类，采用了iaa的图像处理，从亮度、饱和度、噪点、对比度、crop、scale等方面做了共计12000张图片的增强。最后排查了一遍road 2和road 4，把一些错误和过曝的图片都删掉了，最终保留了56000张图片（包括数据增强的图片）进行训练。

【8】训练时采用上述分辨率，但是在实际提交结果时，添加一层bilinear，直接将结果缩放到3384x1020，实测结果会有0.005左右的提升

【9】最终的融合先使用每个模型将每个test image的结果保存为1536x512x8的npy文件，然后加载test image的三个模型的npy文件进行求平均值，然后创建了一个bilinearNet将结果缩放成3384x1020，再将之前裁掉的3384x690的背景与结果拼接，得到最终的预测结果。

【10】【最终夺冠关键】当完成了1-9步骤后，得到的分数是0.61234分，位列第三，但是local CV的结果却不应该对应这个分数，于是我将label叠加到了原图上，发现了一个巨大的问题，由于我训练的分辨率为1536x512，原始图像为3384x1020，我首先使用了最邻近插值来缩放label，但是后来在生成结果时，却使用了bilinear，再加上本身分辨率并不是能够被原始分辨率整除，导致了我所有的prediction都向图像右侧偏移了4-5个像素。于是在results_correction.py中，我将label进行了位置修正，当使用4个像素修正时，得到了当前的0.63547分。（Note：如果使用原始图像训练、或者使用恰好被二整除关系的分辨率训练则不会出现这个问题，主要还是我设计的unet有点小问题，所以无法这样训练）

### 模型记录

|Models|Loss Function|Base LR|Batch Size|Resolution|Miou|
|:---|:---|:---|:---|:---|:---|
|Unet-base|bce + dice|0.001|8|768 x 256|0.52231|
|Unet-base|bce + dice|0.001|4|1024 x 384|0.55136|
|Unet-base|bce + dice|0.001|2|1536 x 512|0.60577|
|Unet-Simple|bce + dice|0.001|2|1536 x 512|0.60223|
|Deeplabv3p|bce + dice|0.001|2|1536 x 512|0.59909|
|Ensemble|-|-|-|1536 x 512|0.61234|
|Correction|-|-|-|1536 x 512|0.63547|

### 依赖说明
    Python 3.6
    opencv-python 3.4.3.18
    paddlepaddle-gpu 1.3.0.post97
    imgaug 0.2.7

### 代码结构
    |Projects - |data_list - train.csv  训练集数据路径
                           - val.csv  验证集数据路径
                       
                |model_weights - paddle_deeplabv3p  模型权重文件存放
                               - paddle_unet_base
                               - paddle_unet_simple
                               
                |models - deeplabv3p.py  模型deeplabv3p.py结构
                        - unet_base.py   模型unet_base.py结构
                        - unet_simple.py  模型unet_simple.py结构
                        
                |utils  - data_feeder.py  数据读取、生成
                        - image_process.py  数据预处理
                        - make_lists.py   生成数据列表
                        - process_labels.py  label的编解码
                        
                |ensemble.py   模型融合代码
                
                |train.py   训练脚本
                
                |val_inference.py  验证及生成提交数据脚本
                
                |results_correction.py  修正预测值位置
                
### 使用说明
    不论是训练还是验证，首先都要使用utils/make_lists.py脚本，将路径配好，生成存储数据路径的csv文件
#### 训练
    【1】 首先在train.py中配好data_dir、save_model_path、model_path的路径，保证数据读取和存储都不会存在问题

    【2】 配置IMG_SIZE，默认(1536,512)；配置base_lr，默认0.001;并修改需要使用的network（可选项为deeplabv3p，unet_base，unet_simple）

    【3】 确定crop_offset，就是解决方案中说的裁掉天空和树木，这里默认是690；确定log_iters打印时间和save_model_iters的模型存储时间

    【4】 配置好后就可以运行train.py训练了（需说明的是我采用的是双卡训练，采用的是fluid.ParallelExecutor，batch_size需要为偶数，如果是单卡训练，需要修改部分代码）

#### 验证
    【1】 与训练类似，配置上述参数；并将program_choice = 1，开启validation功能

    【2】 确定model_path和network，并可以选择是否显示prediction与ground truth的对比

    【3】 一切配置好后，运行val_inference.py即可

#### 测试结果生成
    【1】 与验证类似，配置上述参数；并将program_choice = 2，开启Test功能

    【2】 脚本中有一个save_test_logits参数，当False时，仅保存单模型预测结果的png；如果是True，将会保存单模型预测结果的npy文件，用于最终ensemble使用

    【3】 test_dir需要指定验证集图片保存的文件夹，脚本会自动获取路径

    【4】 一切配置好后，运行val_inference.py即可；等运行完毕，将生成的预测png文件夹打包压缩，按照官方说明，提交成绩即可

#### 模型融合
    【1】 与上述类似，配置测试集路径，分辨率等参数

    【2】 在model_lists中输入三个模型生成的npy文件路径，配置好后根据路径数量调整求均值的响应策略

    【3】 配置好后，运行ensemble.py即可；运行完毕后，将生成的预测png文件夹打包压缩，按照官方说明，提交成绩即可
    
#### 标签修正
    【1】 将测试集路径、原始预测标签路径、生成修正标签路径配好
    
    【2】 调整offset值，单位为像素；默认值为4
    
    【3】 配置好后，运行results_correction.py即可；运行完毕后，将生成的新修正预测png打包压缩，按照官方说明，提交成绩即可
