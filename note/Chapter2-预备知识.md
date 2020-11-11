# 第2章 预备知识

## 2.1 环境配置

### 2.1.1 Anaconda

**Q**：Anaconda是什么？

**A**：Anaconda是Python的一个开源发行版本，主要面向科学计算。我们可以简单理解为，Anaconda是一个预装了很多我们用的到或用不到的第三方库的Python。而且相比于大家熟悉的pip install命令，Anaconda中增加了conda install命令。当你熟悉了Anaconda以后会发现，conda install会比pip install更方便一些。 强烈建议先去看看[最省心的Python版本和第三方库管理——初探Anaconda](https://zhuanlan.zhihu.com/p/25198543)（[笔者笔记](相关笔记/最省心的Python版本和第三方库管理——初探Anaconda.md)）和[初学 Python 者自学 Anaconda 的正确姿势-猴子的回答](https://www.zhihu.com/question/58033789/answer/254673663)（笔者这部分内容都会了）。

Q：由于我使用了Zsh，所以在安装完Anaconda之后使用`$conda`进行测试，无法识别，如何解决这个问题?
A：
```shell
echo 'export PATH="/home/ttz/anaconda3/bin:$PATH"' >> ~/.zshrc
```

### 2.1.2 Jupyter

**Q**：Jupyter好在哪里？

**A**：边写代码边写注释、所见即所得。

### 2.1.3 PyTorch

**Q**：如何安装PyTorch？

**A**：直接去[PyTorch官网](https://pytorch.org/)找到自己的软硬件对应的安装命令即可。

​	**说明**：在没有翻墙的情况下使用`conda`命令装PyTorch可能会出现问题，因而需要换成清华的源进行下载，具体见[1]。

**Q**：安装好PyTorch之后，如何通过命令查看安装的PyTorch及版本号？

**A**：

```shell
conda list | grep torch
```

## 2.2 数据操作

**小结**：本节介绍了如何对内存中的数据进行操作。

**Q**：如何理解PyTorch中的`torch.Tensor`？

**A**：

1）在PyTorch中，`torch.Tensor`是存储和变换数据的主要工具；

2）Tensor`和NumPy的多维数组非常类似。然而，`Tensor`提供GPU计算和自动求梯度等更多功能，这些使`Tensor`**更加适合深度学习**。

> "tensor"这个单词一般可译作“张量”，张量可以看作是一个多维数组。标量可以看作是0维张量，向量可以看作1维张量，矩阵可以看作是二维张量。

### 2.2.1 创建`Tensor`

**小结**：本小节介绍了`Tensor`的最基本功能,即`Tensor`的创建、获取`Tensor`的形状、到官网查阅更多关于`Tensor`的API。

**Q**：如何创建`Tensor`?

**A**：

1）未初始化的`Tensor`；

2）随机初始化的`Tensor`；

3）0填充的`Tensor`；

4）直接根据数据创建的`Tensor`；

5）构建与`Tensor`A相同大小但是类型或元素值不同的`Tensor`B等；

**重点**：获取`Tensor`的**形状**是写代码、调试代码中使用频率非常高的功能，可以通过`shape`或者`size()`来获取`Tensor`的形状：

```python
print(x.size())
print(x.shape)
```

​	输出：

```c++
torch.Size([5, 3])
torch.Size([5, 3])
```

```
注意：返回的torch.Size其实就是一个tuple, 支持所有tuple的操作。

```

**Q**:在哪里查PyTorch `Tensor`的API？  
**A**:[PyTorch官方文档](https://pytorch.org/docs/stable/torch.html)。

### 2.2.2 操作

**小结**：本小节介绍`Tensor`的各种操作，包括算术操作、索引、改变形状、线性代数等。

**算术操作**

> 在PyTorch中，同一种操作可能有很多种形式，本教程使用了加法作为例子。

**索引**

> 索引出来的结果与原数据共享内存，也即修改一个，另一个会跟着修改。

**改变形状**

**小结**：本小节介绍了改变形状的2种方法：用`view()`和`clone()`，二者的区别是`view()`共享数据，而`clone`不共享数据。

>**注意`view()`返回的新`Tensor`与源`Tensor`虽然可能有不同的`size`，但是是共享`data`的，也即更改其中的一个，另外一个也会跟着改变。(顾名思义，view仅仅是改变了对这个张量的观察角度，内部数据并未改变)**
>
>`clone`会创造一个副本然后再使用`view`，使用`clone`还有一个好处是会被记录在计算图中，即梯度回传到副本时也会传到源`Tensor`。

**线性代数**

> PyTorch中的`Tensor`支持超过一百种操作，包括转置、索引、切片、数学运算、线性代数、随机数等等，可参考[官方文档](https://pytorch.org/docs/stable/tensors.html)。

### 2.2.3 广播机制

**小结**：本小节通过一个例子说明了什么是**广播机制**。

### 2.2.4 运算的内存开销

**小结**：本小节通过例子说明像`y = x + y`这种运算会新开内存的，然后将`y`指向新内存。

### 2.2.5 `Tensor`和Numpy相互转换

**小结**：本小节介绍了`Tensor`和NumPy**相互转换的方法**、相互转换也分为**数据共享**和**数据不共享**这2种方式。

**`Tensor`转NumPy**

**小结**：本小节通过一个例子说明如何使用`numpy()`将`Tensor`转换成NumPy数组。

**NumPy数组转`Tensor`**

### 2.2.6 `Tensor`on GPU

**说明**：关于该小结的学习，还是等使用了GPU再说吧。

## 2.3 自动求梯度

在深度学习中，我们经常需要对函数求**梯度（gradient）**。PyTorch提供的[autograd](https://pytorch.org/docs/stable/autograd.html)包能够根据**输入和前向传播过程自动构建计算图**，**并执行反向传播**。本节将介绍如何使用autograd包来进行**自动求梯度**的有关操作。

### 2.3.1 概念

### 2.3.2 `Tensor`

**小节**：本小节介绍了：

1）设置`requires_grad=True;`的方法；

2）直接创建的变量时没有`grad_fn`的，而通过运算创建的是有一个为`<AddBackward>`的`grad_fn`的；

3）直接创建的变量被称为**叶子节点**，叶子节点对应的`grad_fn`是`None`；

4）可以通过`.requires_grad_()`来用in-place的方式改变`requires_grad`属性。

### 2.3.3 梯度

**重点**：要求梯度，需要确定其分子和分母，对于一个神经网络，分子为输出变量的微分，分母为各种计算节点的微分：

```python
out.backward()		// 导数的分子
print(x.grad)		// 导数的分母
```

**原则**：**不允许张量对张量求导，只允许标量对张量求导，求导结果是和自变量同形的张量**。

## 参考资料

[1] [pytorch安装问题](https://blog.csdn.net/jiyangsb/article/details/82430794)









