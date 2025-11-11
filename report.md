<h1 id="wNA6S">Introduction</h1>
<h2 id="vuouW">1.1 研究背景</h2>

**Metric3D**旨在通过单张图像实现zero-shot（零样本）度量3D重建。传统的3D重建方法依赖于多视图几何和相机校准，这些方法无法从单一视角进行准确的3D重建。近年来，一些基于深度学习的单目深度估计方法尝试解决这一问题，但它们通常依赖于相同相机模型的训练，无法进行跨相机和跨数据集的泛化。

<h2 id="m6x3D">1.2 Related Work</h2>
<h3 id="6023596e">1. 从单幅图像进行三维重建（Single-Image 3D Reconstruction）</h3>

早期研究主要针对**特定类别的物体**（如汽车、飞机、桌子、人体等）进行重建。这类方法能够生成高质量的 3D 模型，但依赖于**类别特定的3D监督**，因此**无法推广到复杂场景**（如整幅室内或室外图像）。

一些工作尝试从单张图像**重建整个场景**，如：

+ Saxena 等人假设场景由多个小平面组成，通过这些平面的方向与位置重建三维结构；
+ LeReS使用强大的单目深度估计模型进行场景重建，但只能恢复到**一个未知尺度**；
+ Zhang 等人提出零样本几何保持深度估计模型，能在未知尺度下预测深度。

然而，这些方法都只能恢复相对或仿射不变的结构，而metric3d可以**恢复度量（metric）级的3D结构**。

<h3 id="f0df65e4">2. 有监督的单目深度估计（Supervised Monocular Depth Estimation）</h3>
自从NYUv2 数据集、KITTI 数据集等几个基准确立以来，基于神经网络的方法占据主导地位，方法可分为：

+ **连续深度回归**（直接预测深度值）。
+ **离散深度分类**（将深度离散化视为分类问题），往往效果更好。

面向三维度量恢复的深度模型泛化问题涉及两个问题，泛化问题主要体现在：

+ **场景泛化**（不同场景的深度分布差异大）。
+ **相机泛化**（不同内参导致尺度误差）。

近年研究通过构建**相对深度数据集**（如 DIW、OASIS）来学习相对关系，但这会丢失几何信息。为了改善几何质量，提出了**仿射不变深度学习**（如 MiDaS、LeReS、HDN），通过融合大规模多样数据提升泛化性。但这些方法**无法恢复绝对度量信息**。因此，“**如何同时获得强泛化性和准确的度量深度**”成为关键研究问题。

<h3 id="7f106902">3. 大规模数据训练（Large-scale Data Training）</h3>

近年来，大规模训练在视觉与语言任务中表现出显著优势（如 CLIP、MSEG等）。

在深度预测中，也有类似趋势：

+ Ranft 等人使用超 200 万数据；
+ LeReS收集约 30 万数据；
+ Eftekhar 等人融合数百万数据构建强大的深度预测模型。

本文（Metric3D）基于这一趋势，使用**超过800万张图像、1万多种相机模型**进行混合训练，从而实现**零样本度量深度估计**。

<h2 id="PyJGS">1.3 创新点</h2>

这些方法普遍存在一个共同挑战：**尺度不一致**<sup>**[1]**</sup>。该团队认为：**解决零样本单视角深度度量问题的关键在于大规模的数据训练和解决来自各种相机的度量歧义。**

为了解决这个问题，Metric3D 提出了: 

1. 一种**标准相机变换**（**Canonical Camera Transformation，CSTM**）方法，将训练数据转换到标准相机空间，从而消除由相机内参不同带来的度量歧义。
2. 一个**随机提议归一化损失**（**Random Proposal Normalization Loss，RPNL**）来有效地提高深度精度。

通过在大量多样化的数据上训练，Metric3D 实现了对未见相机和未见场景的有效泛化。该方法在多个零样本评估基准上达到了与最先进方法相媲美的性能，尤其在度量3D重建和稠密SLAM建图任务中表现突出。

> [1] **尺度不一致（Scale Ambiguity）** 是在深度估计（尤其是单目深度估计）中常见的问题，指的是由于相机的内参（如焦距、主点）不同，导致在不同的相机设置下拍摄的同一场景可能具有不同的深度尺度。换句话说，虽然不同相机可能拍摄的是同一场景，但由于相机的标定参数（例如焦距）不同，最终的深度估计结果会存在比例上的差异。  
>



<h1 id="KmdXR">Methodolog</h1>
<h2 id="VX93e">2.1 相关背景</h2>

![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1760500185990-ad7eccf8-792b-48a5-9bff-d8ed1e4cf3e5.png)



基于上图的透视原理，可以得出如下公式：

![image](https://cdn.nlark.com/yuque/__latex/9f6c82cfa58fa882e8c676f4ad435b23.svg)						

![image](https://cdn.nlark.com/yuque/__latex/f8ffbb71f4836bc7742a45cbc4e7e4d9.svg)

![image](https://cdn.nlark.com/yuque/__latex/8b49349408c2cf216432c8c836a962ae.svg) 



<h2 id="CKIpU">2.2 前置结论</h2>
<h3 id="ebpDM">2.2.1 传感器尺寸和像素尺寸不影响度量深度估计：</h3>

（1）传感器尺寸：

传感器尺寸只会影响相机视场角（FOV），并不会影响公式（1）的![image](https://cdn.nlark.com/yuque/__latex/18d25ca4f77a9bbed9812e2bb0b350a5.svg),因此不影响深度估计。



（2）像素尺寸：

![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1760500185990-ad7eccf8-792b-48a5-9bff-d8ed1e4cf3e5.png)



假设使用两台不同像素尺寸（![image](https://cdn.nlark.com/yuque/__latex/25b3429808b03c4e0680325a7a7f1ea9.svg)），但是相同焦距![image](https://cdn.nlark.com/yuque/__latex/d4b6d95070e14ff628f267674dd81e90.svg)的相机，在相同距离（深度）捕捉同一物体的统同一个位置。

由于![image](https://cdn.nlark.com/yuque/__latex/25b3429808b03c4e0680325a7a7f1ea9.svg)，故像素焦距![image](https://cdn.nlark.com/yuque/__latex/c24d5eef5fc7c805f733cefd97da611e.svg),相应的图像分辨率![image](https://cdn.nlark.com/yuque/__latex/8ea87e026c90c640be6e00f8ad7f8e3b.svg)。

由透视原理公式：![image](https://cdn.nlark.com/yuque/__latex/9f6c82cfa58fa882e8c676f4ad435b23.svg)，![image](https://cdn.nlark.com/yuque/__latex/c2bdf1dfba589c90a9feae613128f3b5.svg)

**总结**：像素尺寸不改变物体的成像尺寸，因此不会对深度估计造成影响。



<h3 id="MWoPT">2.2.1 焦距对于度量深度估计至关重要：</h3>

![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1760498585470-01671c46-d677-450a-b785-da2bfb4f9854.png)



仅从图像的外观来看，人们可能会认为最后两张照片是由同一相机在相似的位置拍摄的。实际上，由于焦距不同，这些在不同位置被捕获。因此，**相机内参数对于单幅图像的度量估计至关重要**。



![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1760513326818-ab2fd559-3c0c-4cb0-b266-4592f98b6cd4.png)



当拍摄条件为（![image](https://cdn.nlark.com/yuque/__latex/7a4f67ad26dc2ce87d079d42cf07a53e.svg)）,两个相机拍摄的成像是相同的。由此可知，在不同物体距离下，由于焦距的设置不同，物体成像大小却可能相同。这对于网络训练是非常不友好的：

当训练数据来自很多不同相机（不同焦距）时，同样的图像外观可能对应不同的真实深度标签（见论文 Fig5示例：![image](https://cdn.nlark.com/yuque/__latex/bdf44ddb007770f9194b4f1f3e652d59.svg)），导致监督信号冲突，模型无法收敛或泛化差。  



<h2 id="ecXDB">2.3 标准相机变换空间（CSTM）</h2>
为了解决单目深度估计的度量歧义问题，论文提出了“标准相机变换（CSTM）”



**目标**：把来自不同相机/不同内参的数据“统一”到一个**canonical（标准）相机空间**，消除或显著减少因相机内参（主要是焦距）差异引起的尺度和外观歧义，从而可以在大规模、多相机混合的数据上训练一个能输出**真正度量（metric）深度**的模型。  



**核心思想**：设定一个固定的标准相机内参（![image](https://cdn.nlark.com/yuque/__latex/489913b0dae14c414dea3b4d70fb172e.svg)）。训练时把每条训练样本（图像或深度标签）通过变换映射到这个标准相机空间。网络在这个统一空间中学习。从该空间预测出的深度是“canonical depth”。推理阶段，再把预测的canonical depth反变换回原始相机尺度以得到最终metric深度。



**两种实现方法：**

论文中提出两种可选的CSTM实现方法：

+ **Method1**：Transforming depth labels（CSTM label）—— 对深度标签进行变换
+ **Method2**：Transforming input images（CSTM image）—— 对输入图像进行变换



**设原始相机模型为**![image](https://cdn.nlark.com/yuque/__latex/8adddb276f93c62a4eb2eca20972d81d.svg)

**方法1：CSTM label（变换深度标签）**

**图3的模糊度量是针对深度而言的。因此，我们的第一种方法直接转换真实深度标签来解决这个问题**。保持输入图像不变，直接缩放深度值标签，使其与canonical焦距![image](https://cdn.nlark.com/yuque/__latex/31faf2a2674ba5305a8ef8d2fde5dce1.svg)对齐。**原始相机模型变换为**![image](https://cdn.nlark.com/yuque/__latex/c7aee68545f2e006a784bdf5c7e8e3c0.svg)。

**方法流程：**

+ **深度缩放比例为：**

![image](https://cdn.nlark.com/yuque/__latex/6a4811c4cb5e92f5ff328e4ac017e63a.svg)

+ **训练时将深度标签**![image](https://cdn.nlark.com/yuque/__latex/6144d255bfc233fd699d6eb68512ae6b.svg)**缩放为：**

![image](https://cdn.nlark.com/yuque/__latex/cb2a03cbc0dcf62aac8cc97543326ba9.svg)

+ **不对输入图像做处理，保持输入图像不变：**

![image](https://cdn.nlark.com/yuque/__latex/5625defa6f43d9844cc3541d5dff87f7.svg)

+ **网络预测出canonical深度**![image](https://cdn.nlark.com/yuque/__latex/d21688f6faab1c0dce065d87d5e4eee1.svg)**,推理时反变换恢复原尺度： **

![image](https://cdn.nlark.com/yuque/__latex/9d8930a98188f19272740810f9500c03.svg)

**好处**：把“真实深度数值”按比例放大或缩小，使它们都看起来是由同一焦距fc产生的深度。这样网络在训练时看到的是尺度一致的深度标签，监督不再因焦距差异而冲突。  



**方法 2：CSTM image（变换图像）**

**从另一个角度来看，这种歧义是由相似的图像外观造成的。因此该方法是对输入图像进行变换，以模拟标准摄像机的成像效果**。通过缩放输入图像的像素表示来模拟标准相机的成像效果，从而把图像外观变为canonical相机下的外观；同时要对深度图做相应地缩放保证像素对应关系，但不对深度值本身做数值比例缩放。

**方法流程：**

+ **图像缩放比例：**

![image](https://cdn.nlark.com/yuque/__latex/2766d43a12e8b635bf21821ee7a05f16.svg)

+ **训练时对输入图像做缩放（resize）：**

![image](https://cdn.nlark.com/yuque/__latex/a0c7b7ab5ebb331e9d018037faddd0f7.svg)   (T表示图像缩放操作，相机光心u0,v0也要相应缩放![image](https://cdn.nlark.com/yuque/__latex/fb84fbf402017014aa4e558887c49101.svg)）

+ **原始相机模型变换为**![image](https://cdn.nlark.com/yuque/__latex/33042ee5a4a7e3e0d2266421ddd56d32.svg)
+ **深度图相应地缩放（空间上缩放，但深度数值不乘比例），保证像素对应关系：**

![image](https://cdn.nlark.com/yuque/__latex/148181ff33a222ce9c2391bebbc7a529.svg)

+ **推理时把预测的标准深度通过de-canonical恢复回原图尺寸（不改变数值标度）：**

![image](https://cdn.nlark.com/yuque/__latex/d2e1a98c7ee181894c18e1a6a46f4e65.svg)

**好处**：更直观地把外观统一，适用于那些图像视角/FOV不同带来主要差异的情况



<h2 id="Fay4Y">2.4 训练</h2>
<h3 id="WuaVu">2.4.1 训练流程</h3>

![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1760629842824-7e521db2-3da2-4b5e-85fe-989836120170.png)



给定一幅输入图像I，我们首先利用CSTM将其变换到标准空间。将变换后的图像Ic输入到一个标准的深度估计模型中，以产生标准空间中的预测度量深度![image](https://cdn.nlark.com/yuque/__latex/d21688f6faab1c0dce065d87d5e4eee1.svg)。在训练过程中，![image](data:image/svg+xml;utf8,%3Csvg%20xmlns%3Axlink%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxlink%22%20width%3D%222.868ex%22%20height%3D%222.509ex%22%20style%3D%22vertical-align%3A%20-0.671ex%3B%22%20viewBox%3D%220%20-791.3%201235%201080.4%22%20role%3D%22img%22%20focusable%3D%22false%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20aria-labelledby%3D%22MathJax-SVG-1-Title%22%3E%0A%3Ctitle%20id%3D%22MathJax-SVG-1-Title%22%3EEquation%3C%2Ftitle%3E%0A%3Cdefs%20aria-hidden%3D%22true%22%3E%0A%3Cpath%20stroke-width%3D%221%22%20id%3D%22E1-MJMATHI-44%22%20d%3D%22M287%20628Q287%20635%20230%20637Q207%20637%20200%20638T193%20647Q193%20655%20197%20667T204%20682Q206%20683%20403%20683Q570%20682%20590%20682T630%20676Q702%20659%20752%20597T803%20431Q803%20275%20696%20151T444%203L430%201L236%200H125H72Q48%200%2041%202T33%2011Q33%2013%2036%2025Q40%2041%2044%2043T67%2046Q94%2046%20127%2049Q141%2052%20146%2061Q149%2065%20218%20339T287%20628ZM703%20469Q703%20507%20692%20537T666%20584T629%20613T590%20629T555%20636Q553%20636%20541%20636T512%20636T479%20637H436Q392%20637%20386%20627Q384%20623%20313%20339T242%2052Q242%2048%20253%2048T330%2047Q335%2047%20349%2047T373%2046Q499%2046%20581%20128Q617%20164%20640%20212T683%20339T703%20469Z%22%3E%3C%2Fpath%3E%0A%3Cpath%20stroke-width%3D%221%22%20id%3D%22E1-MJMATHI-63%22%20d%3D%22M34%20159Q34%20268%20120%20355T306%20442Q362%20442%20394%20418T427%20355Q427%20326%20408%20306T360%20285Q341%20285%20330%20295T319%20325T330%20359T352%20380T366%20386H367Q367%20388%20361%20392T340%20400T306%20404Q276%20404%20249%20390Q228%20381%20206%20359Q162%20315%20142%20235T121%20119Q121%2073%20147%2050Q169%2026%20205%2026H209Q321%2026%20394%20111Q403%20121%20406%20121Q410%20121%20419%20112T429%2098T420%2083T391%2055T346%2025T282%200T202%20-11Q127%20-11%2081%2037T34%20159Z%22%3E%3C%2Fpath%3E%0A%3C%2Fdefs%3E%0A%3Cg%20stroke%3D%22currentColor%22%20fill%3D%22currentColor%22%20stroke-width%3D%220%22%20transform%3D%22matrix(1%200%200%20-1%200%200)%22%20aria-hidden%3D%22true%22%3E%0A%20%3Cuse%20xlink%3Ahref%3D%22%23E1-MJMATHI-44%22%20x%3D%220%22%20y%3D%220%22%3E%3C%2Fuse%3E%0A%20%3Cuse%20transform%3D%22scale(0.707)%22%20xlink%3Ahref%3D%22%23E1-MJMATHI-63%22%20x%3D%221171%22%20y%3D%22-213%22%3E%3C%2Fuse%3E%0A%3C%2Fg%3E%0A%3C%2Fsvg%3E)由一个GT深度![image](https://cdn.nlark.com/yuque/__latex/7a602cef87d217c377c859d363efdeb6.svg)监督，![image](https://cdn.nlark.com/yuque/__latex/7a602cef87d217c377c859d363efdeb6.svg)也被转换到正则空间。在推理中，在标准空间中产生度量深度![image](https://cdn.nlark.com/yuque/__latex/d21688f6faab1c0dce065d87d5e4eee1.svg)后，我们执行去标准变换将其转换回原始输入I的空间。标准空间变换和去标准变换使用相机内参执行。

**训练目标：**![image](https://cdn.nlark.com/yuque/__latex/aa2327c1a29daaa1739a7b287ba52a0d.svg)

最小化标准变换空间下模型预测深度与GT深度的差值

在训练阶段，网络不是直接学习原始深度![image](https://cdn.nlark.com/yuque/__latex/6144d255bfc233fd699d6eb68512ae6b.svg)，而是学习变换后的**canonical深度**![image](https://cdn.nlark.com/yuque/__latex/2b34d343c346a5716cc83259414e7a44.svg)：

![image](https://cdn.nlark.com/yuque/__latex/a1026a3cab90c0e3dd1bdb7cd5062354.svg)

这样网络学习到的深度处于一个固定的 canonical 相机体系中。因此，在训练阶段，深度尺度是一致的、可比较的。 Metric3D 使用了 **11 个数据集**、超过 **10K 不同相机** 的数据进行联合训练(**所有训练数据都包含相机内参，并被用于规范化相机变换模块**（**CSTM**))。每个样本都被转化到同一 canonical 相机空间，这种“统一视角”的训练让模型学会了跨相机的深度尺度。  



<h3 id="NGC9a">2.4.2 相机内参获取</h3>

**对于训练阶段：**

Metric3D 训练时共使用 **11 个公开 RGB-D 数据集**（约 800 万张图像）。论文明确指出：

“**All collected training data have included paired camera intrinsic parameters, which are used in our canonical transformation module.**”  
—— 所有训练数据都包含相机内参，并被用于规范化相机变换模块（CSTM）。

**对于测试阶段：**

在推理（测试）时，Metric3D 也需要相机内参做去标准化（de-canonical transformation）。论文说明：

“**We employ their provided camera intrinsic parameters to perform our proposed canonical space transformation**.”  
—— 在测试时使用数据集中提供的相机内参执行标准化变换。

此外，对于wild images，论文在“**Metrology in the wild**”部分说明了内参获取方式：

“**To show the robustness and accuracy of our recovered metric 3D, we download Flickr photos captured by various cameras and collect coarse camera intrinsic parameters from their metadata.**”

—— 即从照片元数据中提取粗略的相机内参

论文在"Reconstruction of in-the-wild scenes."同样提及：

**“We collect several photos from Flickr. From their associated camera metadata, we can obtain the focal length fˆ and the pixel size δ. According to fˆ/δ, we can obtain the pixel-represented focal length for 3D reconstruction and achieve the metric information.”**  
—— 即从照片元数据（metadata）中提取焦距和像素尺寸，然后计算出有效焦距。



<h3 id="ybkRW">2.4.3 损失函数</h3>

**尺度平移不变损失**被广泛用于仿射不变性深度估计。在计算损失时，会用整张图的均值或方差进行归一化，使得深度的局部差异（特别是近距离的细节）被“平均化”或“压扁”掉。

Metric3D基于这个问题提出了随机提案归一化损失（Random Proposal Normalization Loss，RPNL）。该损失不再在整幅图上归一化，而是随机**裁剪多个局部patch（论文中 M=32 个），每个patch的边长尺寸在原图尺寸的12.5%到50%随机选取**。在每个 patch 内独立进行局部归一化（使用 median absolute deviation, MAD），计算局部归一化误差最后取平均。

**具体形式：**

![image](https://cdn.nlark.com/yuque/__latex/9904e03416991a74c7ebddd81dbc791b.svg)

![image](https://cdn.nlark.com/yuque/__latex/2d80d3be2c8444c378f2556a9d86692e.svg)



**总Loss:**

![image](https://cdn.nlark.com/yuque/__latex/fe0b6c6a10cbfe7468f3f4d8aabda5c4.svg)

| **损失** | **主要约束** | **空间层级** | **功能** |
| --- | --- | --- | --- |
| 配对法线损失![image](https://cdn.nlark.com/yuque/__latex/d6a2e789eca31799ca94974cadca08b6.svg) | 局部法线方向 | 局部邻域 | 保持局部几何形态 |
| 虚拟法线损失![image](https://cdn.nlark.com/yuque/__latex/9746bd07d2150ed73a30d66895c20b54.svg) | 跨区域几何一致性 | 大范围 | 保持全局几何形态 |
| 尺度不变对数损失![image](https://cdn.nlark.com/yuque/__latex/36e3c1839b874e62f517279f7b0e32df.svg) | 全局相对深度尺度 | 全图 | 保持全局相对深度趋势  |
| 随机提案归一化损失![image](https://cdn.nlark.com/yuque/__latex/94a64cd20bc670a9ec7b7a8cd09ac2d3.svg) | 局部深度细节 | 局部 patch | 保留细节与近景差异 |


四者联合，使得模型从“图像外观”中学到**真实、稳定、结构完整**的**度量深度。**  

<h1 id="btCZJ">Experiments</h1>

**实验设置如下表所示：**

| **评估对象** | **数据集** | **评估指标** |
| --- | --- | --- |
| 零样本深度估计的鲁棒性 | 8个零样本基准数据集：NYUv2、KITTI、NuScenes、7-scenes、iBIMS-1、DIODE、ETH3D | AbsRel（绝对相对误差）<br/>δi < 1.25i（准确度）<br/>RMSE（均方根误差）<br/>RMS log（对数空间误差）<br/>log10 error（对数误差） |
| 泛化性能  | 5个零样本数据集：NYUv2、DIODE、ETH3D、ScanNet、KITTI | <br/>对比大规模数据训练模型  |
| 三维重建质量评估   | 随机选择 9个未见过的NYUv2场景，使用 COLMAP 获得相机位姿进行多帧重建 | Chamfer L1 距离 <br/>F-score |
| 密集SLAM建图 | ****KITTI odometry benchmark**  ** | 平均平移 RMS 漂移 <br/>旋转 RMS 漂移 |


<h2 id="dQrWT">3.1 零样本学习泛化性检验</h2>

为了评估预测度量深度的准确性，我们首先在NYUv2 ，KITTI上与最先进的度量深度预测方法( SOTA )进行了比较。我们使用相同的模型来做所有的评估。结果如下表。在没有任何微调或度量调整的情况下，我们可以实现与SOTA方法相媲美的性能。

![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1762851401940-a72ecb9b-ce51-4f32-a2d1-6210bdc033ac.png)



<h2 id="dRIBE">3.2 基于metric3d的应用效果</h2>
<h3 id="d272e">3.2.1 三维场景重建</h3>

为了证明我们的工作可以在野外恢复3D度量形状，我们首先在训练过程中没有看到的9个NYUv2场景上进行了定量比较。我们预测每帧的度量深度，然后将它们与提供的相机姿态融合在一起。结果报告于表2 .

![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1762852585195-b3e5c0da-d6ab-4960-8021-6940ca199ef7.png)



我们与视频一致性深度预测方法( RCVD )，无监督视频深度估计方法( SC- DepthV2 )，3D场景形状恢复方法( LeReS )，仿射不变深度估计方法( DPT )，以及多视图立体重建方法( DPSNet )进行了比较。除了DPSNet和我们的方法外，其他方法必须对齐每一帧的尺度和真实深度。虽然我们的方法不针对视频或多视图重建问题，但我们的方法可以实现帧之间的良好一致性，并且在这些零镜头场景上重建出比其他方法更精确的三维场景。从图6中的定性比较来看，我们的重建具有更少的噪声和异常值。

![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1762852742778-0e60e5a3-8398-4c21-bcbd-18f9db13d791.png)

<h3 id="KlHIN">3.2.2 Dense-slam 建图</h3>
由于有限的光度和几何约束，现有方法在大场景下面临严重的尺度漂移问题，无法恢复度量信息。我们的鲁棒度量深度估计方法是SLAM系统的强深度先验。为了展示这种好处，我们天将我们的度量深度输入到SOTA SLAM系统DroidSLAM中，并在KITTI上评估轨迹。

![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1762853029045-b748fac0-5587-4d68-9583-2afe656a09dc.png)

在我们的深度下，Droid-SLAM可以执行更稠密和更准确的三维建图：

![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1762853262680-de1152d5-669f-4ed9-9cc8-104b906472b6.png)



我们还在ETH3D SLAM基准测试集上进行了测试。结果报告于表6。具有我们深度的Droid具有更好的SLAM性能。由于ETH3D场景均为小尺度室内场景，性能提升不如KITTI。

![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1762853387704-19e5c8dc-ed78-41a5-b3c0-7ac29352ccbf.png)



<h3 id="QI3WD">3.2.4 野外深度度量</h3>
为了展示我们恢复的度量3D的鲁棒性和准确性，我们下载了不同相机捕获的Flickr照片，并从它们的元数据中收集了粗略的相机内在参数。我们使用我们的CSTM图像模型来重建它们的度量形状和测量结构的尺寸(图7以红色标示)，而真实尺寸是蓝色的。这表明我们的测量尺寸与真实尺寸非常接近。

![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1762853700745-4ec103ff-1988-4736-b9b7-8a0c9cf1c952.png)

<h2 id="V7qb5">3.3 消融实验</h2>
<h3 id="eeVfO">3.3.1 Ablation on canonical transformation</h3>
我们研究了我们提出的针对输入图像( " CSTM输入")的正则变换和针对真实标签( " CSTM输出")的正则变换的效果。结果报告于表7 .我们在采样的混合数据( 55K张图片)上训练模型，并在6个数据集上进行测试。一个朴素的基准是去除CSTM模块，并执行与我们相同的监督。没有CSTM，模型在混合度量数据集上训练时无法收敛，在零样本数据集上无法达到度量预测能力。这就是为什么最近的混合数据训练方法妥协于学习仿射不变深度以避免度量问题。

![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1762854243616-aed2c37d-9715-4959-88fd-5b26fa7c2e09.png)

<h3 id="atvgT">3.3.2 Ablation on canonical space</h3>
我们在这里研究标准相机的效应，即标准焦距。我们在小样本数据集上训练模型，并在训练数据和测试数据的验证集上进行测试。计算了平均AbsRel误差。我们实验了三种不同的焦距，即500，1000，1500。实验表明，foccal = 1000的性能略好于其他值，详见图8。因此，在我们的实验中，我们将标准焦距设置为1000。

![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1762854541247-3afab989-928e-4a7e-816f-47a218b106ed.png)

<h3 id="XOBzU">3.3.3 Effectiveness of the random proposal normalization loss</h3>
为了说明我们提出的随机提议归一化损失( RPNL )的有效性，我们在采样的小数据集上进行了实验。结果见表8 。“基线"使用了除我们的RPNL以外的所有损失。我们将其与"基线+ RPNL "和"基线+ SSIL"进行比较。我们可以观察到，我们提出的随机建议归一化损失可以进一步提高性能。

![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1762854743195-f4d7580a-1beb-4227-9927-4896986204a1.png)



<h1 id="nMqgT">项目代码</h1>

**源码链接**：[**https://github.com/YvanYin/Metric3D**](https://github.com/YvanYin/Metric3D)

<h2 id="d6Q4T">3.1 创建环境 && 安装依赖</h2>

```bash
conda create -n metric3d python = 3.8
conda activate metric3d
```

<font style="color:rgb(31, 35, 40);">对于back-bone为ConvNeXt-L的模型，请运行：</font>

```bash
pip install -r requirements_v1.txt
```

对于back-bone为VIT的模型，请运行：

```bash
pip install -r requirements_v2.txt
```

**pip会自动下载兼容版本的相关环境依赖**

****

**代码运行环境：**

**ConvNeX:**

```python
torch
torchvision
opencv-python
numpy
Pillow
DateTime
matplotlib
plyfile
HTML4Vision
timm
tensorboardX
imgaug
iopath
imagecorruptions
mmcv
```

**VIT:**

```plain
torch == 2.0.1
torchvision == 0.15.2
opencv-python
numpy == 1.23.1
xformers == 0.0.21
Pillow
DateTime
matplotlib
plyfile
HTML4Vision
timm
tensorboardX
imgaug
iopath
imagecorruptions
mmcv
```

****

**准备预训练模型文件并放置在指定文件夹下，作者提供了五种不同骨干网络、不同规模的预训练模型：**

**metric3d_v1:**

[**https://huggingface.co/JUGGHM/Metric3D/resolve/main/convtiny_hourglass_v1.pth**](https://huggingface.co/JUGGHM/Metric3D/resolve/main/convtiny_hourglass_v1.pth)

[**https://huggingface.co/JUGGHM/Metric3D/resolve/main/convlarge_hourglass_0.3_150_step750k_v1.1.pth**](https://huggingface.co/JUGGHM/Metric3D/resolve/main/convlarge_hourglass_0.3_150_step750k_v1.1.pth)

**metric3d_v2:**

[**https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_small_800k.pth**](https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_small_800k.pth)

[**https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth**](https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth)

[**https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_giant2_800k.pth**](https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_giant2_800k.pth)



如果是手动下载模型：

1. 手动修改hubconf.py:14 的对应模型的ckpt_file：将下载url更改为本地模型存放地址

```bash
MODEL_TYPE = {
  'ConvNeXt-Tiny': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/convtiny.0.3_150.py',
    'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/convtiny_hourglass_v1.pth',
  },
  'ConvNeXt-Large': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/convlarge.0.3_150.py',
    'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/convlarge_hourglass_0.3_150_step750k_v1.1.pth',
  },
  'ViT-Small': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.small.py',
    'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_small_800k.pth',
  },
  'ViT-Large': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.large.py',
    'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth',
  },
  'ViT-giant2': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.giant2.py',
    'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_giant2_800k.pth',
  },
}
```



2. 修改对应的模型参数加载方法（hubconf.py:39-142）:

```bash
# torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict']
# ->
torch.loa(ckpt_file)['model_state_dict']
```



**输入如下指令即可运行代码:**

```python
python hubconf.py
```



**准备测试数据集：**

**以 DIODE数据集为例，手动下载：**

**在官网**[**https://diode-dataset.org/**](https://diode-dataset.org/)**选择所需数据集进行下载，并放置在指定路径下（如Metric3d-main/data）**

****

<h2 id="SYSht">3.2 测试结果</h2>
从测试数据集中选取三张图像，输出对应的度量深度图：

![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1760851301549-13ea85c2-f12e-490e-a978-046ae9f16eaa.png)

![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1760851350964-ba8be6e5-4758-4ef0-87b7-07e1463c4431.png)

![](https://cdn.nlark.com/yuque/0/2025/png/58377837/1760851371276-ca803685-9d77-470e-befb-8e195cc93509.png)

<h2 id="McFNX">3.3 论文公式对应代码</h2>
**公式对照表：**

| **论文公式** | **对应文件位置** |
| --- | --- |
| 标准相机变换空间参数 | metric3d-main/momo/config |
| 标准变换空间 | metric3d-main/training/mono/utils/transform.py：123 -148 |
| 推理阶段，标准空间下深度还原为真实度量深度 | metric3d-main/hubconf.py: 199-201 |
| 尺度不变对数损失![image](https://cdn.nlark.com/yuque/__latex/36e3c1839b874e62f517279f7b0e32df.svg) | metric3d-main/training/mono/model/losses/SiLog.py: 17-26 |
| 随机提案归一化损失![image](https://cdn.nlark.com/yuque/__latex/94a64cd20bc670a9ec7b7a8cd09ac2d3.svg) | training/mono/model/losses/HDSNL_random.py |
| 虚拟法线损失![image](https://cdn.nlark.com/yuque/__latex/9746bd07d2150ed73a30d66895c20b54.svg) | training/mono/model/losses/VNL.py |
| 配对法线损失![image](https://cdn.nlark.com/yuque/__latex/d6a2e789eca31799ca94974cadca08b6.svg) | training/mono/model/losses/PWN_planes.py |


**标准相机变换空间参数（metric3d-main/momo/config）：**

```python
data_basic=dict(
    canonical_space = dict(
        img_size=(512, 960),    # 模型输入图像大小
        focal_length=1000.0,    # 标准相机焦距
    ),
    depth_range=(0, 1),         
    depth_normalize=(0.3, 150), 
    crop_size = (544, 1216),    # 数据裁剪尺寸
)
```

****

**标准变换空间（metric3d-main/training/mono/utils/transform.py：123 -148）:**

```python
def random_on_canonical_transform(self, image, label, intrinsic, cam_model, to_random_ratio):
        ori_h, ori_w, _ = image.shape
        ori_focal = (intrinsic[0] + intrinsic[1]) / 2.0 # 原始相机焦距

        to_canonical_ratio = self.canonical_focal / ori_focal # 缩放因子
        to_scale_ratio = to_random_ratio
        resize_ratio = to_canonical_ratio * to_random_ratio # 乘以随机缩放因子
        # 获取缩放后尺寸
        reshape_h = int(ori_h * resize_ratio + 0.5)
        reshape_w = int(ori_w * resize_ratio + 0.5)
        # 对原图缩放
        image = cv2.resize(image, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
        # 若有内参，同步缩放光心、内参不变
        if intrinsic is not None:
            intrinsic = [self.canonical_focal, self.canonical_focal, intrinsic[2]*resize_ratio, intrinsic[3]*resize_ratio]
        if label is not None:
            # number of other labels may be less than that of image
            label = cv2.resize(label, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_NEAREST)
            # scale the label and camera intrinsics
            label = label / to_scale_ratio
        
        if cam_model is not None:
            # Should not directly resize the cam_model.
            # Camera model should be resized in 'to canonical' stage, while it holds in 'random resizing' stage. 
            # cam_model = cv2.resize(cam_model, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
            cam_model = build_camera_model(reshape_h, reshape_w, intrinsic)
        
        return image, label, intrinsic, cam_model, to_scale_ratio
```



**推理阶段，标准空间下深度还原为真实度量深度（metric3d-main/hubconf.py: 199-201）:**

```python
  #### de-canonical变换
  canonical_to_real_scale = intrinsic[0] / 1000.0 # 1000.0 为标准相机焦距
  pred_depth = pred_depth * canonical_to_real_scale # 现在深度为真实度量值
  pred_depth = torch.clamp(pred_depth, 0, 300) # 限制深度范围
```



**尺度不变对数损失**![image](https://cdn.nlark.com/yuque/__latex/36e3c1839b874e62f517279f7b0e32df.svg)**(metric3d-main/training/mono/model/losses/SiLog.py: 17-26)：**

```python
def silog_loss(self, prediction, target, mask):
        # di = log(d_i) - log(d_i_gt)
        d = torch.log(prediction[mask]) - torch.log(target[mask])
        # 平均平方误差 sum(d**2)/N
        d_square_mean = torch.sum(d ** 2) / (d.numel() + self.eps)
        # 平均误差 sum(d)/N
        d_mean = torch.sum(d) / (d.numel() + self.eps)
        # 最后loss
        loss = d_square_mean - self.variance_focus * (d_mean ** 2) # variance_focus = 0.5
        return loss
```

**根据代码还原的公式： **![image](https://cdn.nlark.com/yuque/__latex/d106cad76e5e40d03cbfecbb34ca4ce8.svg)

****

**随机提案归一化损失**![image](https://cdn.nlark.com/yuque/__latex/94a64cd20bc670a9ec7b7a8cd09ac2d3.svg)**(training/mono/model/losses/HDSNL_random.py)：**

****![image](https://cdn.nlark.com/yuque/__latex/9904e03416991a74c7ebddd81dbc791b.svg)

```python
class HDSNRandomLoss(nn.Module):
    """
    Hieratical depth spatial normalization loss.
    Replace the original grid masks with the random created masks.
    loss = MAE((d-median(d)/s - (d'-median(d'))/s'), s = mean(d- median(d))
    """
    """	关键成员变量
    loss_weight=1.0       # 最终损失权重
    random_num=32         # 每张图随机裁剪窗口数量
    data_type=[...]       # 哪些数据来源可能使用
    disable_dataset=[...] # 在这些数据集上跳过损失计算（cfg）
    sky_id=142            # 语义标签中被视为天空的 id（会被排除）
    batch_limit=8         # 每次并行处理多少随机 mask
    eps=1e-6              # 防止除零
    """
    def __init__(self, 
                 loss_weight=1.0, 
                 random_num=32, 
                 data_type=['sfm', 
                            'stereo', 
                            'lidar', 
                            'denselidar',
                            'denselidar_nometric',
                            'denselidar_syn'], 
                 disable_dataset=['MapillaryPSD'], 
                 sky_id=142, 
                 batch_limit=8, **kwargs):
        super(HDSNRandomLoss, self).__init__()
        self.loss_weight = loss_weight
        self.random_num = random_num
        self.data_type = data_type
        self.sky_id = sky_id
        self.batch_limit = batch_limit
        self.eps = 1e-6
        self.disable_dataset = disable_dataset
        
    # 生成self.random_num个随机矩形mask（不重叠检测）用于后续局部归一化。
    def get_random_masks_for_batch(self, image_size: list)-> torch.Tensor:
        height, width = image_size
        # 确定裁剪高宽范围（基于图像大小）
        crop_h_min = int(0.125 * height)
        crop_h_max = int(0.5 * height)
        crop_w_min = int(0.125 * width)
        crop_w_max = int(0.5 * width)
        # 确定裁剪的起点范围
        h_max = height - crop_h_min
        w_max = width - crop_w_min
        # 随机选取裁剪范围
        crop_height = np.random.choice(np.arange(crop_h_min, crop_h_max), self.random_num, replace=False)
        crop_width = np.random.choice(np.arange(crop_w_min, crop_w_max), self.random_num, replace=False)
        # 随机选取裁剪起点（左下）
        crop_y = np.random.choice(h_max, self.random_num, replace=False)
        crop_x = np.random.choice(w_max, self.random_num, replace=False)
        # 确定裁剪钟点（右上）
        crop_y_end = crop_height + crop_y
        crop_y_end[crop_y_end>=height] = height # 边界检测
        crop_x_end = crop_width + crop_x
        crop_x_end[crop_x_end>=width] = width # 边界检测
        # 对每个裁剪像素标记True
        mask_new = torch.zeros((self.random_num,  height, width), dtype=torch.bool, device="cuda") #.cuda() #[N, H, W]
        for i in range(self.random_num):
           mask_new[i, crop_y[i]:crop_y_end[i], crop_x[i]:crop_x_end[i]] = True
            
        return mask_new
        #return crop_y, crop_y_end, crop_x, crop_x_end

    # 筛选有意义的语义区域
    def reorder_sem_masks(self, sem_label): 
        # sem_label：语义分割标签，形状 [B, H, W]（每个像素是一个语义类别ID）
        assert sem_label.ndim == 3
        # 从sem_label提取有效语义类别ID（0：背景类；sky_id:天空类）
        semantic_ids = torch.unique(sem_label[(sem_label>0) & (sem_label != self.sky_id)])
        # 生成语义标签对应的二值掩码（有效标签：True；无效标签：False）
        sem_masks = [sem_label == id for id in semantic_ids]
        if len(sem_masks) == 0:
            # no valid semantic labels
            out = sem_label > 0
            return out
        
        sem_masks = torch.cat(sem_masks, dim=0)	# (num_classes,H,W)
        # 过滤对应语义数量太少的区域（噪声、小目标）
        mask_batch = torch.sum(sem_masks.reshape(sem_masks.shape[0], -1), dim=1) > 500
        sem_masks = sem_masks[mask_batch]
        # 语义类别过多，随机选取self.random_num个
        if sem_masks.shape[0] > self.random_num:
            balance_samples = np.random.choice(sem_masks.shape[0], self.random_num, replace=False)
            sem_masks = sem_masks[balance_samples, ...]
        
        if sem_masks.shape[0] == 0:
            # no valid semantic labels
            out = sem_label > 0
            return out

        if sem_masks.ndim == 2:
            sem_masks = sem_masks[None, :, :]
        return sem_masks

    # 核心计算部分
    def ssi_mae(self, prediction, target, mask_valid):
        B, C, H, W = target.shape
        prediction_nan = prediction.clone().detach()
        target_nan = target.clone()
        # 忽略裁剪patch外的区域
        prediction_nan[~mask_valid] = float('nan')
        target_nan[~mask_valid] = float('nan')
        # 统计有效像素数量
        valid_pixs = mask_valid.reshape((B, C,-1)).sum(dim=2, keepdims=True) + 1e-10
        valid_pixs = valid_pixs[:, :, :, None]
        # 计算GT深度图每个patch的中位数
        gt_median = target_nan.reshape((B, C,-1)).nanmedian(2, keepdims=True)[0].unsqueeze(-1) # 最后一维为像素维，沿该维选取中位数
        gt_median[torch.isnan(gt_median)] = 0
        # 计算GT深度平均绝对偏差
        gt_diff = (torch.abs(target - gt_median) ).reshape((B, C, -1))
        gt_s = gt_diff.sum(dim=2)[:, :, None, None] / valid_pixs
        gt_trans = (target - gt_median) / (gt_s + self.eps)
        # 计算预测深度图每个patch的中位数
        pred_median = prediction_nan.reshape((B, C,-1)).nanmedian(2, keepdims=True)[0].unsqueeze(-1) # [b,c,h,w]
        pred_median[torch.isnan(pred_median)] = 0
        # 计算预测深度平均绝对偏差
        pred_diff = (torch.abs(prediction - pred_median)).reshape((B, C, -1))
        pred_s = pred_diff.sum(dim=2)[:, :, None, None] / valid_pixs
        pred_trans = (prediction - pred_median) / (pred_s + self.eps)
        # 总loss
        loss_sum = torch.sum(torch.abs(gt_trans - pred_trans)*mask_valid)
        return loss_sum
    

    def forward(self, prediction, target, mask=None, sem_mask=None, **kwargs):
        """
        Calculate loss.
        """
        B, C, H, W = target.shape
        
        loss = 0.0
        valid_pix = 0.0

        device = target.device
        # 按dataset名称决定该batch item是否有效
        batches_dataset = kwargs['dataset']
        self.batch_valid = torch.tensor([1 if batch_dataset not in self.disable_dataset else 0 \
            for batch_dataset in batches_dataset], device=device)[:,None,None,None]

        batch_limit = self.batch_limit
        # 获取裁剪区域掩码
        random_sample_masks = self.get_random_masks_for_batch((H, W)) # [N, H, W]
        # 遍历batch中每张图像
        for i in range(B):
            # each batch
            mask_i = mask[i, ...] #[1, H, W]
            if self.batch_valid[i, ...] < 0.5:
                loss += 0 * torch.sum(prediction[i, ...])
                valid_pix += 0 * torch.sum(mask_i)
                continue

            pred_i = prediction[i, ...].unsqueeze(0).repeat(batch_limit, 1, 1, 1)
            target_i = target[i, ...].unsqueeze(0).repeat(batch_limit, 1, 1, 1)

            # 获取语义掩码
            sem_label_i = sem_mask[i, ...] if sem_mask is not None else None
            if sem_label_i is not None:
                sem_masks = self.reorder_sem_masks(sem_label_i) # [N, H, W]
                random_sem_masks = torch.cat([random_sample_masks, sem_masks], dim=0)
            else:
                random_sem_masks = random_sample_masks
            #random_sem_masks = random_sample_masks


            sampled_masks_num = random_sem_masks.shape[0]
            loops = int(np.ceil(sampled_masks_num / batch_limit))
            conditional_rank_ids = np.random.choice(sampled_masks_num, sampled_masks_num, replace=False)

            for j in range(loops):
                mask_random_sem_loopi = random_sem_masks[j*batch_limit:(j+1)*batch_limit, ...]
                mask_sample = (mask_i & mask_random_sem_loopi).unsqueeze(1) # [N, 1, H, W]
                loss += self.ssi_mae(
                    prediction=pred_i[:mask_sample.shape[0], ...], 
                    target=target_i[:mask_sample.shape[0], ...], 
                    mask_valid=mask_sample)
                valid_pix += torch.sum(mask_sample)

        mask = mask * self.batch_valid.bool()
        loss += self.ssi_mae(
                    prediction=prediction, 
                    target=target, 
                    mask_valid=mask)
        valid_pix += torch.sum(mask)
        # 除以全部有效像素点数量（i * j）
        loss = loss / (valid_pix + self.eps)
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(prediction)
            print(f'HDSNL NAN error, {loss}, valid pix: {valid_pix}')
        return loss * self.loss_weight    
```

****

**虚拟法线损失**![image](https://cdn.nlark.com/yuque/__latex/9746bd07d2150ed73a30d66895c20b54.svg)**(training/mono/model/losses/VNL.py)**

```python
class VNLoss(nn.Module):
    """
    虚拟法线损失（Virtual Normal Loss, VNL）

    基本思想：
    从深度图中随机采样像素三元组（p1, p2, p3），利用相机内参将像素坐标与深度回投到
    相机坐标系，得到三点的 3D 坐标。
    分别用 GT 深度与预测深度构造三角面片的法线（通过叉乘获得），比较两者的差异（L1）
    ，以此约束几何一致性。
    """

    """
    成员变量:
    delta_cos: 过滤近线性/共线三点组的余弦阈值（越大越严格）
    delta_diff_x/y/z: 三点在 x/y/z 维度上的最小差异阈值（过近则丢弃）
    delta_z: z 方向（深度）过小视为无效
    sample_ratio: 从有效像素中采样的比例（每张图采 H*W*ratio 个三元组索引）
    loss_weight: 损失权重
    data_type: 数据类型标识（当前类内部未用作分支，仅保留兼容）
    """
    def __init__(
        self,
        delta_cos: float = 0.867,
        delta_diff_x: float = 0.01,
        delta_diff_y: float = 0.01,
        delta_diff_z: float = 0.01,
        delta_z: float = 1e-5,
        sample_ratio: float = 0.15,
        loss_weight: float = 1.0,
        data_type=['sfm', 'stereo', 'lidar', 
                   'denselidar', 'denselidar_nometric',
                   'denselidar_syn'],**kwargs) -> None:
        super(VNLoss, self).__init__()
        self.delta_cos = delta_cos
        self.delta_diff_x = delta_diff_x
        self.delta_diff_y = delta_diff_y
        self.delta_diff_z = delta_diff_z
        self.delta_z = delta_z
        self.sample_ratio = sample_ratio
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6  # 数值稳定项
        
    # 依据相机内参与图像尺寸，初始化像素坐标到主点的偏移矩阵 (u-u0, v-v0)，并缓存为 buffer。
    def init_image_coor(self, intrinsic: torch.Tensor, height: int, width: int) -> None:
        u0 = intrinsic[:, 0, 2][:, None, None, None]
        v0 = intrinsic[:, 1, 2][:, None, None, None]
        # 生成像素网格（行=Y, 列=X）
        y, x = torch.meshgrid(
            [
                torch.arange(0, height, dtype=torch.float32, device="cuda"),
                torch.arange(0, width, dtype=torch.float32, device="cuda"),
            ],
            indexing='ij'
        )
        u_m_u0 = x[None, None, :, :] - u0  # [B,1,H,W]
        v_m_v0 = y[None, None, :, :] - v0  # [B,1,H,W]
        self.register_buffer('v_m_v0', v_m_v0, persistent=False)
        self.register_buffer('u_m_u0', u_m_u0, persistent=False)
    
    # 将深度图回投到相机坐标系。
    def transfer_xyz(
        self,
        depth: torch.Tensor,
        focal_length: torch.Tensor,
        u_m_u0: torch.Tensor,
        v_m_v0: torch.Tensor
    ) -> torch.Tensor:
        x = u_m_u0 * depth / focal_length
        y = v_m_v0 * depth / focal_length
        z = depth
        pw = torch.cat([x, y, z], 1).permute(0, 2, 3, 1).contiguous()  # [B,H,W,3]
        return pw

    # 每一张图片选取 N=H×W×sample_ratio 个三点组
    def select_index(self, B: int, H: int, W: int, mask: torch.Tensor):
        """
        基于有效像素掩码，随机为每张图采样三组索引 p1/p2/p3。

        输入:
        - B,H,W: batch 与尺寸
        - mask: [B,1,H,W]，有效像素（>0）为 True

        输出:
        - p123: 字典，包含 p1_x/y, p2_x/y, p3_x/y（每个 [B, intend_sample_num]）
        """
        p1 = []
        p2 = []
        p3 = []
        pix_idx_mat = torch.arange(H * W, device="cuda").reshape((H, W))
        for i in range(B):
            inputs_index = torch.masked_select(pix_idx_mat, mask[i, ...].gt(self.eps))
            num_effect_pixels = len(inputs_index)

            intend_sample_num = int(H * W * self.sample_ratio)
            sample_num = intend_sample_num if num_effect_pixels >= intend_sample_num else num_effect_pixels

            # 三次独立采样，增强多样性
            shuffle_effect_pixels = torch.randperm(num_effect_pixels, device="cuda")
            p1i = inputs_index[shuffle_effect_pixels[:sample_num]]
            shuffle_effect_pixels = torch.randperm(num_effect_pixels, device="cuda")
            p2i = inputs_index[shuffle_effect_pixels[:sample_num]]
            shuffle_effect_pixels = torch.randperm(num_effect_pixels, device="cuda")
            p3i = inputs_index[shuffle_effect_pixels[:sample_num]]

            # 若不足，补零索引（后续会被 mask 过滤）
            cat_null = torch.tensor(([0,] * (intend_sample_num - sample_num)), dtype=torch.long, device="cuda")
            p1i = torch.cat([p1i, cat_null])
            p2i = torch.cat([p2i, cat_null])
            p3i = torch.cat([p3i, cat_null])

            p1.append(p1i)
            p2.append(p2i)
            p3.append(p3i)

        p1 = torch.stack(p1, dim=0)
        p2 = torch.stack(p2, dim=0)
        p3 = torch.stack(p3, dim=0)

        # 将一维索引还原为 (x,y)
        p1_x = p1 % W
        p1_y = torch.div(p1, W, rounding_mode='trunc').long()
        p2_x = p2 % W
        p2_y = torch.div(p2, W, rounding_mode='trunc').long()
        p3_x = p3 % W
        p3_y = torch.div(p3, W, rounding_mode='trunc').long()
        p123 = {'p1_x': p1_x, 'p1_y': p1_y, 'p2_x': p2_x, 'p2_y': p2_y, 'p3_x': p3_x, 'p3_y': p3_y}
        return p123

    def form_pw_groups(self, p123, pw: torch.Tensor) -> torch.Tensor:
        """
        按 p123 索引从 3D 点云中取点，组成 [B,N,3(xyz),3(p1,p2,p3)] 的三点组。

        输入:
        - p123: select_index 返回的索引字典
        - pw: [B,H,W,3] 3D 点云

        输出:
        - pw_groups: [B,N,3,3]
        """
        B, _, _, _ = pw.shape
        p1_x = p123['p1_x']; p1_y = p123['p1_y']
        p2_x = p123['p2_x']; p2_y = p123['p2_y']
        p3_x = p123['p3_x']; p3_y = p123['p3_y']

        pw_groups = []
        for i in range(B):
            pw1 = pw[i, p1_y[i], p1_x[i], :]
            pw2 = pw[i, p2_y[i], p2_x[i], :]
            pw3 = pw[i, p3_y[i], p3_x[i], :]
            pw_bi = torch.stack([pw1, pw2, pw3], dim=2)
            pw_groups.append(pw_bi)
        pw_groups = torch.stack(pw_groups, dim=0)  # [B,N,3,3]
        return pw_groups

    def filter_mask(
        self,
        p123,
        gt_xyz: torch.Tensor,
        delta_cos: float = 0.867,
        delta_diff_x: float = 0.005,
        delta_diff_y: float = 0.005,
        delta_diff_z: float = 0.005
    ):
        """
        根据 GT 3D 点筛除无效/不稳定的三点组：
        - 丢弃近线性/共线组合（通过三条边向量两两余弦相似度判定）
        - 丢弃三点在 xyz 方向差异过小的组合
        - 丢弃深度无效的组合

        返回:
        - mask: [B,N] 有效组标记
        - pw: [B,N,3,3] 与输入 gt_xyz 对应的采样三点组
        """
        pw = self.form_pw_groups(p123, gt_xyz)
        pw12 = pw[:, :, :, 1] - pw[:, :, :, 0]
        pw13 = pw[:, :, :, 2] - pw[:, :, :, 0]
        pw23 = pw[:, :, :, 2] - pw[:, :, :, 1]

        # 将三条边拼成 [B,N,3(xyz),3(p123)]，用于计算两两余弦
        pw_diff = torch.cat(
            [pw12[:, :, :, np.newaxis], pw13[:, :, :, np.newaxis], pw23[:, :, :, np.newaxis]],
            3
        )
        m_batchsize, groups, coords, index = pw_diff.shape
        proj_query = pw_diff.view(m_batchsize * groups, -1, index).permute(0, 2, 1).contiguous()  # [bn,3,3]
        proj_key = pw_diff.contiguous().view(m_batchsize * groups, -1, index)  # [bn,3,3]
        q_norm = proj_query.norm(2, dim=2)
        nm = torch.bmm(q_norm.contiguous().view(m_batchsize * groups, index, 1),
                       q_norm.view(m_batchsize * groups, 1, index))
        energy = torch.bmm(proj_query, proj_key)  # [bn,3,3]
        norm_energy = energy / (nm + self.eps)
        norm_energy = norm_energy.contiguous().view(m_batchsize * groups, -1)
        # 若 3x3 的余弦矩阵中大于阈值(或小于 -阈值)的数量>3，视作线性/共线，忽略
        mask_cos = torch.sum((norm_energy > delta_cos) + (norm_energy < -delta_cos), 1) > 3
        mask_cos = mask_cos.contiguous().view(m_batchsize, groups)

        # 深度有效: 三个点的z都要大于delta_z
        mask_pad = torch.sum(pw[:, :, 2, :] > self.delta_z, 2) == 3

        # 过近过滤（xyz任一轴均“太近”则忽略）
        mask_x = torch.sum(torch.abs(pw_diff[:, :, 0, :]) < delta_diff_x, 2) > 0
        mask_y = torch.sum(torch.abs(pw_diff[:, :, 1, :]) < delta_diff_y, 2) > 0
        mask_z = torch.sum(torch.abs(pw_diff[:, :, 2, :]) < delta_diff_z, 2) > 0

        mask_ignore = (mask_x & mask_y & mask_z) | mask_cos
        mask_near = ~mask_ignore
        mask = mask_pad & mask_near
        return mask, pw

    def select_points_groups(
        self,
        gt_depth: torch.Tensor,
        pred_depth: torch.Tensor,
        intrinsic: torch.Tensor,
        mask: torch.Tensor
    ):
        """
        从 GT 与预测深度分别构造 3D 三点组，并筛选有效组。

        返回:
        - pw_groups_gt_not_ignore: [1,n,3,3]
        - pw_groups_pred_not_ignore: [1,n,3,3]
        """
        B, C, H, W = gt_depth.shape
        focal_length = intrinsic[:, 0, 0][:, None, None, None]
        u_m_u0, v_m_v0 = self.u_m_u0, self.v_m_v0

        pw_gt = self.transfer_xyz(gt_depth, focal_length, u_m_u0, v_m_v0)
        pw_pred = self.transfer_xyz(pred_depth, focal_length, u_m_u0, v_m_v0)

        p123 = self.select_index(B, H, W, mask)
        # 基于 GT 的几何稳定性做筛选
        mask_valid, pw_groups_gt = self.filter_mask(
            p123, pw_gt,
            delta_cos=0.867, delta_diff_x=0.005, delta_diff_y=0.005, delta_diff_z=0.005
        )

        # 预测点的三点组
        pw_groups_pred = self.form_pw_groups(p123, pw_pred)
        # 防止 z==0 导致范数为 0
        pw_groups_pred[pw_groups_pred[:, :, 2, :] == 0] = 0.0001

        # 将 [B,N,3,3] 根据 mask 压缩为 [1,n,3,3]（n为有效3点组数量）
        mask_broadcast = mask_valid.repeat(1, 9).reshape(B, 3, 3, -1).permute(0, 3, 1, 2).contiguous()
        pw_groups_pred_not_ignore = pw_groups_pred[mask_broadcast].reshape(1, -1, 3, 3)
        pw_groups_gt_not_ignore = pw_groups_gt[mask_broadcast].reshape(1, -1, 3, 3)

        return pw_groups_gt_not_ignore, pw_groups_pred_not_ignore

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, intrinsic: torch.Tensor, select: bool = True, **kwargs):
        """
        计算 VNL 损失。

        输入:
        - prediction: [B,1,H,W] 预测深度
        - target: [B,1,H,W] GT 深度
        - mask: [B,1,H,W] 有效像素掩码
        - intrinsic: [B,3,3] 相机内参
        - select: 是否启用 top-75% 聚焦难例（丢弃前 25% 最小项）

        输出:
        - 标量损失（加权）
        """
        loss = self.get_loss(prediction, target, mask, intrinsic, select, **kwargs)
        return loss
        
    # 核心计算公式
    def get_loss(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, intrinsic: torch.Tensor, select: bool = True, **kwargs) -> torch.Tensor:
        B, _, H, W = target.shape
        # 确保u_m_u0/V_m_v0已根据当前batch大小/分辨率初始化
        if 'u_m_u0' not in self._buffers or 'v_m_v0' not in self._buffers
            or self.u_m_u0.shape != torch.Size([B, 1, H, W]) or self.v_m_v0.shape != torch.Size([B, 1, H, W]):
            self.init_image_coor(intrinsic, H, W)
        # gt/pred_points[1,N,3,3](1，样本数，坐标维，三个点)
        gt_points, pred_points = self.select_points_groups(target, prediction, intrinsic, mask)

        # 构造边向量
        gt_p12 = gt_points[:, :, :, 1] - gt_points[:, :, :, 0] # [1,N,3]
        gt_p13 = gt_points[:, :, :, 2] - gt_points[:, :, :, 0]
        pred_p12 = pred_points[:, :, :, 1] - pred_points[:, :, :, 0]
        pred_p13 = pred_points[:, :, :, 2] - pred_points[:, :, :, 0]
        # 法线
        gt_normal = torch.cross(gt_p12, gt_p13, dim=2) # [1,N,3]
        pred_normal = torch.cross(pred_p12, pred_p13, dim=2)
        # 归一化并做数值保护
        pred_norm = torch.norm(pred_normal, 2, dim=2, keepdim=True) #[1,N,1](每个虚法线的L2范数)
        gt_norm = torch.norm(gt_normal, 2, dim=2, keepdim=True)
        pred_mask = (pred_norm == 0.0).to(torch.float32) * self.eps
        gt_mask = (gt_norm == 0.0).to(torch.float32) * self.eps
        gt_norm = gt_norm + gt_mask
        pred_norm = pred_norm + pred_mask
        gt_normal = gt_normal / gt_norm	# 归一化，[1,N,3]
        pred_normal = pred_normal / pred_norm 

        # L1 差异并聚合
        loss = torch.abs(gt_normal - pred_normal)
        loss = torch.sum(torch.sum(loss, dim=2), dim=0)  # 对xyz与组内求和.[1,N]->[N]

        if select:
            # 丢弃前 25% 最小损失项（聚焦难例）
            loss, _ = torch.sort(loss, dim=0, descending=False)
            loss = loss[int(loss.size(0) * 0.25):]

        loss = torch.sum(loss) / (loss.numel() + self.eps) # 平均

        # 数值检查
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(prediction)
            print(f'VNL NAN error, {loss}')
        return loss * self.loss_weight
```

**根据代码还原的公式：**

**向量：**

![image](https://cdn.nlark.com/yuque/__latex/c8d53638caa1fa9ec29bc878cbecd63f.svg)

![image](https://cdn.nlark.com/yuque/__latex/34f54f1cca5c5d74adc63b01ff57b19d.svg)

![image](https://cdn.nlark.com/yuque/__latex/895c42cea2c4d117e7cc62fdc5093c41.svg)

![image](https://cdn.nlark.com/yuque/__latex/ebc35a8cbd2484cc6a76ec6168ed475b.svg)

**法向量：**

![image](https://cdn.nlark.com/yuque/__latex/9992919c7bfba580dfa7990239b1a746.svg)

![image](https://cdn.nlark.com/yuque/__latex/f3d9f884e9aa22f81444cd0b62d78179.svg)

**法向量单位化：**

![image](https://cdn.nlark.com/yuque/__latex/e0b8be0be8eb796d7fb5414ce269d5bc.svg)

![image](https://cdn.nlark.com/yuque/__latex/c4c4757fdbe7693e0c3d9063d69d9e4c.svg)

**Loss:**

![image](https://cdn.nlark.com/yuque/__latex/111db28ff5e3b90d297bb74993243962.svg)

![image](https://cdn.nlark.com/yuque/__latex/eb41e70bc3275892759fc17042b7c013.svg)



**配对法线损失**![image](https://cdn.nlark.com/yuque/__latex/d6a2e789eca31799ca94974cadca08b6.svg)**(training/mono/model/losses/PWN_planes.py)**

```python
import torch
import torch.nn as nn
import numpy as np


class PWNPlanesLoss(nn.Module):
    """
    平面一致性损失（PWN Planes）

    基本思想：
    - 对每个实例平面区域，在像素平面随机采样多个三点组，用预测深度回投为 3D 点，计算每组三点构成的“虚拟法线”。
    - 将同一平面内所有虚拟法线聚拢到平均法线方向（用 1-cos 作为差异度量），从而鼓励该区域内的点共面。

    与 VNL 的区别：
    - VNL 比较“GT 法线 vs 预测法线”的差异；
    - 本损失仅用“预测深度”构造法线，并在平面内做“相互一致性”约束，不直接依赖 GT 深度。
    """

    def __init__(
        self,
        delta_cos: float = 0.867,
        delta_diff_x: float = 0.007,
        delta_diff_y: float = 0.007,
        sample_groups: int = 5000,
        loss_weight: float = 1.0,
        data_type=['lidar', 'denselidar'],
        **kwargs
    ) -> None:
        """
        参数:
        - delta_cos: 过滤近线性/共线三点组的余弦阈值（越大越严格）
        - delta_diff_x/y: 在像素平面上 x/y 方向差异过小的三点组将被丢弃
        - sample_groups: 每个平面采样的三点组数（会抽取 3 * sample_groups 个像素索引并拆为三组）
        - loss_weight: 损失权重
        - data_type: 数据类型标识（当前类内部未用）
        """
        super(PWNPlanesLoss, self).__init__()
        self.delta_cos = delta_cos
        self.delta_diff_x = delta_diff_x
        self.delta_diff_y = delta_diff_y
        self.sample_groups = sample_groups
        self.loss_weight = loss_weight
        self.data_type = data_type

    def init_image_coor(self, B: int, H: int, W: int) -> None:
        """
        构造像素齐次坐标 (u, v, 1)，其中:
        - u 为行索引 [0..H-1]，v 为列索引 [0..W-1]
        - 结果形状 [B,3,H,W]，保存在 self.uv

        说明:
        - 本实现未注册为 buffer；每次 get_loss 调用都会重建。
        """
        u = torch.arange(0, H, dtype=torch.float32, device="cuda").contiguous().view(1, H, 1).expand(1, H, W)  # [1,H,W]
        v = torch.arange(0, W, dtype=torch.float32, device="cuda").contiguous().view(1, 1, W).expand(1, H, W)  # [1,H,W]
        ones = torch.ones((1, H, W), dtype=torch.float32, device="cuda")
        pixel_coords = torch.stack((u, v, ones), dim=1).expand(B, 3, H, W)  # [B,3,H,W]
        self.uv = pixel_coords

    def upproj_pcd(self, depth: torch.Tensor, intrinsics_inv: torch.Tensor) -> torch.Tensor:
        """
        将像素齐次坐标回投到相机坐标系，再乘深度得到 3D 点云。

        输入:
        - depth: [B,1,H,W] 预测深度
        - intrinsics_inv: [B,3,3] 相机内参逆矩阵

        输出:
        - [B,3,H,W] 相机坐标系下的点云
        """
        b, _, h, w = depth.size()
        assert self.uv.shape[0] == b, "self.uv 与 batch 大小不匹配，请先调用 init_image_coor"
        current_pixel_coords = self.uv.reshape(b, 3, -1)  # [B,3,H*W]
        cam_coords = (intrinsics_inv @ current_pixel_coords)  # [B,3,H*W]
        cam_coords = cam_coords.reshape(b, 3, h, w)
        out = depth * cam_coords
        return out

    def select_index(self, mask_kp: torch.Tensor):
        """
        为每个平面掩码（x 个平面）随机采样三点组索引。

        输入:
        - mask_kp: [x,1,H,W]，x 为当前图像中的平面数量。True/1 表示该平面内的像素。

        策略:
        - 对每个平面，抽取 select_size = 3*sample_groups 个像素索引；
          再切成 3 份作为 p1/p2/p3。
        - 若平面内有效像素数不足：
            * 若不足 0.6*select_size：改从“非该平面”的像素中采样，并将 valid_batch 置 False（后续会过滤）。
            * 若介于 0.6~1 倍之间：随机重复补足到 select_size。

        输出:
        - p123: 字典，包含各组的 x/y 索引及 valid_batch 标志。
        """
        x, _, h, w = mask_kp.shape
        select_size = int(3 * self.sample_groups)
        p1_x = []; p1_y = []; p2_x = []; p2_y = []; p3_x = []; p3_y = []
        valid_batch = torch.ones((x, 1), dtype=torch.bool, device="cuda")
        for i in range(x):
            mask_kp_i = mask_kp[i, 0, :, :]
            valid_points = torch.nonzero(mask_kp_i)

            if valid_points.shape[0] < select_size * 0.6:
                # 平面像素太少：改从非该平面区域采样，并记为无效批（用于后续过滤）
                valid_points = torch.nonzero(~mask_kp_i.to(torch.uint8))
                valid_batch[i, :] = False
            elif valid_points.shape[0] < select_size:
                # 介于 0.6~1 倍：重复补足
                repeat_idx = torch.randperm(valid_points.shape[0], device="cuda")[:select_size - valid_points.shape[0]]
                valid_repeat = valid_points[repeat_idx]
                valid_points = torch.cat((valid_points, valid_repeat), 0)

            select_indx = torch.randperm(valid_points.size(0), device="cuda")
            p1 = valid_points[select_indx[0:select_size:3]]
            p2 = valid_points[select_indx[1:select_size:3]]
            p3 = valid_points[select_indx[2:select_size:3]]

            p1_x.append(p1[:, 1]); p1_y.append(p1[:, 0])
            p2_x.append(p2[:, 1]); p2_y.append(p2[:, 0])
            p3_x.append(p3[:, 1]); p3_y.append(p3[:, 0])

        p123 = {
            'p1_x': torch.stack(p1_x), 'p1_y': torch.stack(p1_y),
            'p2_x': torch.stack(p2_x), 'p2_y': torch.stack(p2_y),
            'p3_x': torch.stack(p3_x), 'p3_y': torch.stack(p3_y),
            'valid_batch': valid_batch
        }
        return p123

    def form_pw_groups(self, p123, pw: torch.Tensor) -> torch.Tensor:
        """
        将 3D 点按索引组织成三点组。

        输入:
        - p123: select_index 输出
        - pw: [1,H,W,3]，单张图像的点云（已在调用前对 batch 维处理）

        输出:
        - [x,N,3,3]，x 为平面数量，N 为每平面三点组数，后三维为 (x,y,z) 与 (p1,p2,p3)
        """
        p1_x = p123['p1_x']; p1_y = p123['p1_y']
        p2_x = p123['p2_x']; p2_y = p123['p2_y']
        p3_x = p123['p3_x']; p3_y = p123['p3_y']
        batch_list = torch.arange(0, p1_x.shape[0], device="cuda")[:, None]
        pw = pw.repeat((p1_x.shape[0], 1, 1, 1))  # 将单张点云复制到每个平面
        pw1 = pw[batch_list, p1_y, p1_x, :]
        pw2 = pw[batch_list, p2_y, p2_x, :]
        pw3 = pw[batch_list, p3_y, p3_x, :]
        pw_groups = torch.cat([pw1[:, :, :, None], pw2[:, :, :, None], pw3[:, :, :, None]], 3)
        return pw_groups  # [x,N,3,3]

    def filter_mask(self, pw_pred: torch.Tensor) -> torch.Tensor:
        """
        在像素平面（仅使用 xy 分量）过滤不稳定三点组：
        - 计算三边在 xy 平面的向量，两两余弦相似度若大，则认为近线性，剔除；
        - 若在 x 或 y 方向差异过小（过近），也剔除。

        输入:
        - pw_pred: [x,N,3,3]，每组三点的 (x,y,z)

        输出:
        - mask_valid_pts: [x,N] 有效三点组的布尔掩码
        """
        xy12 = pw_pred[:, :, 0:2, 1] - pw_pred[:, :, 0:2, 0]
        xy13 = pw_pred[:, :, 0:2, 2] - pw_pred[:, :, 0:2, 0]
        xy23 = pw_pred[:, :, 0:2, 2] - pw_pred[:, :, 0:2, 1]

        xy_diff = torch.cat(
            [xy12[:, :, :, np.newaxis], xy13[:, :, :, np.newaxis], xy23[:, :, :, np.newaxis]],
            3
        )  # [x,N,2,3]
        m_batchsize, groups, coords, index = xy_diff.shape
        proj_query = xy_diff.contiguous().view(m_batchsize * groups, -1, index).permute(0, 2, 1).contiguous()  # [bn,3,2]
        proj_key = xy_diff.contiguous().view(m_batchsize * groups, -1, index)  # [bn,2,3]
        q_norm = proj_query.norm(2, dim=2)  # [bn,3]
        nm = torch.bmm(
            q_norm.contiguous().view(m_batchsize * groups, index, 1),
            q_norm.contiguous().view(m_batchsize * groups, 1, index)
        )
        energy = torch.bmm(proj_query, proj_key)  # [bn,3,3]
        norm_energy = energy / (nm + 1e-8)
        norm_energy = norm_energy.contiguous().view(m_batchsize * groups, -1)  # [bn,9]
        mask_cos = torch.sum((norm_energy > self.delta_cos) + (norm_energy < -self.delta_cos), 1) > 3
        mask_cos = mask_cos.contiguous().view(m_batchsize, groups)  # [x,N]

        # 过近过滤（在 x 或 y 方向出现“太近”的边）
        mask_x = torch.sum(torch.abs(xy_diff[:, :, 0, :]) < self.delta_diff_x, 2) > 0
        mask_y = torch.sum(torch.abs(xy_diff[:, :, 1, :]) < self.delta_diff_y, 2) > 0
        mask_near = mask_x & mask_y
        mask_valid_pts = ~(mask_cos | mask_near)
        return mask_valid_pts

    def select_points_groups(self, pcd_bi: torch.Tensor, mask_kp: torch.Tensor):
        """
        将单张图像的点云与多平面掩码结合，得到三点组与有效性掩码。

        输入:
        - pcd_bi: [1,3,H,W] 单张图像的 3D 点云
        - mask_kp: [x,1,H,W] x 个平面的二值掩码

        输出:
        - groups_pred: [x,N,3,3]
        - mask_valid: [x,N] 有效三点组掩码
        """
        p123 = self.select_index(mask_kp)  # p1_x: [x,N]
        pcd_bi = pcd_bi.permute((0, 2, 3, 1)).contiguous()  # [1,H,W,3]
        groups_pred = self.form_pw_groups(p123, pcd_bi)  # [x,N,3,3]

        mask_valid_pts = self.filter_mask(groups_pred).to(torch.bool)  # [x,N]
        mask_valid_batch = p123['valid_batch'].repeat(1, mask_valid_pts.shape[1])  # [x,N]
        mask_valid = mask_valid_pts & mask_valid_batch
        return groups_pred, mask_valid

    def constrain_a_plane_loss(self, pw_groups_pre_i: torch.Tensor, mask_valid_i: torch.Tensor):
        """
        针对单个平面，计算其三点组法线的“一致性损失”。

        输入:
        - pw_groups_pre_i: [N,3,3]，N 组三点
        - mask_valid_i: [N]，有效三点组掩码

        输出:
        - (loss_sum, valid_num): 标量损失与有效法线数量
        """
        if torch.sum(mask_valid_i) < 2:
            # 有效三点组太少时返回 0（保持梯度图形兼容）
            return 0.0 * torch.sum(pw_groups_pre_i), 0

        pw_groups_pred_i = pw_groups_pre_i[mask_valid_i]  # [n,3,3]
        p12 = pw_groups_pred_i[:, :, 1] - pw_groups_pred_i[:, :, 0]
        p13 = pw_groups_pred_i[:, :, 2] - pw_groups_pred_i[:, :, 0]
        virtual_normal = torch.cross(p12, p13, dim=1)  # [n,3]

        # 归一化 + 稳定
        norm = torch.norm(virtual_normal, 2, dim=1, keepdim=True)
        virtual_normal = virtual_normal / (norm + 1e-8)

        # 统一法线朝向：用与点位置的点乘符号来翻转不一致的法线
        orient_mask = torch.sum(torch.squeeze(virtual_normal) * torch.squeeze(pw_groups_pred_i[:, :, 0]), dim=1) > 0
        virtual_normal[orient_mask] *= -1

        # 平面内平均法线（单位化）
        aver_normal = torch.sum(virtual_normal, dim=0)
        aver_norm = torch.norm(aver_normal, 2, dim=0, keepdim=True)
        aver_normal = aver_normal / (aver_norm + 1e-5)  # [3]

        # 与平均法线的 1 - cos 差异
        cos_diff = 1.0 - torch.sum(virtual_normal * aver_normal, dim=1)
        loss_sum = torch.sum(cos_diff, dim=0)
        valid_num = cos_diff.numel()
        return loss_sum, valid_num

    def get_loss(self, pred_depth: torch.Tensor, gt_depth: torch.Tensor, ins_planes_mask: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor:
        """
        计算所有图像、所有实例平面的平均一致性损失。

        输入:
        - pred_depth: [B,1,H,W] 预测深度（只用它）
        - gt_depth: 兼容参数（未实际使用）
        - ins_planes_mask: [B,1,H,W] 实例平面掩码 (0 为背景，非 0 为平面 ID)
        - intrinsic: [B,3,3] 相机内参（将使用 inverse()）

        输出:
        - 标量损失
        """
        if pred_depth.ndim == 3:
            pred_depth = pred_depth[None, ...]
        if gt_depth.ndim == 3:
            gt_depth = gt_depth[None, ...]
        if ins_planes_mask.ndim == 3:
            ins_planes_mask = ins_planes_mask[None, ...]

        B, _, H, W = pred_depth.shape
        loss_sum = torch.tensor(0.0, device="cuda")
        valid_planes_num = 0

        # 每次根据当前 B/H/W 重建像素齐次坐标
        self.init_image_coor(B, H, W)
        pcd = self.upproj_pcd(pred_depth, intrinsic.inverse())  # [B,3,H,W]

        for i in range(B):
            mask_i = ins_planes_mask[i, :][None, :, :]  # [1,1,H,W]
            unique_planes = torch.unique(mask_i)
            # 跳过背景 0
            planes = [mask_i == m for m in unique_planes if m != 0]  # list of [1,1,H,W]
            if len(planes) == 0:
                continue
            mask_planes = torch.cat(planes, dim=0)  # [x,1,H,W]

            pcd_groups_pred, mask_valid = self.select_points_groups(pcd[i, ...][None, :, :, :], mask_planes)  # [x,N,3,3], [x,N]
            for j in range(unique_planes.numel() - 1):
                mask_valid_j = mask_valid[j, :]
                pcd_groups_pred_j = pcd_groups_pred[j, :]
                loss_tmp, valid_angles = self.constrain_a_plane_loss(pcd_groups_pred_j, mask_valid_j)
                valid_planes_num += valid_angles
                loss_sum += loss_tmp

        loss = loss_sum / (valid_planes_num + 1e-6) * self.loss_weight
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = torch.sum(pred_depth) * 0
            print(f'PWNPlane NAN error, {loss}')
        return loss

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, intrinsic: torch.Tensor, **kwargs):
        """
        前向接口（与其他损失保持统一签名）。

        约定:
        - 仅当 kwargs['dataset'] 中存在 'Taskonomy' 的样本时才计算此损失；
          否则返回 0（保持计算图）。
        - 实例平面掩码需从 kwargs['sem_mask'] 提供，形状 [B,1,H,W]。

        输入:
        - prediction: [B,1,H,W] 预测深度
        - target: 兼容参数（未使用）
        - mask: 兼容参数（未使用）
        - intrinsic: [B,3,3] 相机内参

        返回:
        - 标量损失
        """
        dataset = kwargs['dataset']
        batch_mask = np.array(dataset) == 'Taskonomy'
        if np.sum(batch_mask) == 0:
            return torch.sum(prediction) * 0.0

        ins_planes_mask = kwargs['sem_mask']
        assert ins_planes_mask.ndim == 4
        loss = self.get_loss(
            prediction[batch_mask],
            target[batch_mask],
            ins_planes_mask[batch_mask],
            intrinsic[batch_mask],
        )
        return loss
```

![image](https://cdn.nlark.com/yuque/__latex/5c41185875bf9d725d8cd93a3a277a99.svg)

![image](https://cdn.nlark.com/yuque/__latex/7179979adff603000d90eb4f56fc5f17.svg) ![image](https://cdn.nlark.com/yuque/__latex/3cad5dff74dada0a6c3d599e880fd9d2.svg)

![image](https://cdn.nlark.com/yuque/__latex/06ef8ec4d6492b1ea86de7cde4a353b4.svg)

![image](https://cdn.nlark.com/yuque/__latex/087b484a0c6dcd1d78436912d47a35a5.svg)

![image](https://cdn.nlark.com/yuque/__latex/1782d0c145bcee078f50feb4d5f861b9.svg)

