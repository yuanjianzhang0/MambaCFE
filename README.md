## README [[中文](https://github.com/userpandawin/MambaCFE/blob/main/README_CN.md)][[English](https://github.com/userpandawin/MambaCFE/blob/main/README_EN.md)]

# MTS: An Efficient Stock Prediction Method Based on the Improved Mamba Model

### 项目介绍

MTS（Multiscale Time Series）是一种基于改进Mamba模型的高效股票预测方法。该模型融合了卷积、注意力机制和多尺度卷积，并引入了一个新型局部特征提取模块（CFE），用于替代传统的卷积操作。MTS模型在多个行业的股票预测任务中表现出色。

### 作者
张元鉴，北京邮电大学13组营员
邮箱：yuanjianzhang2003@163.com

### 文件结构

```
MTS/
│
├── mamba_test.ipynb        # 主测试脚本
├── requirements.txt        # 环境依赖文件      # 测试数据
│
└── README.md               # 项目说明文件
```

### 环境依赖

请在运行代码之前安装必要的依赖库。可以使用以下命令安装`requirements.txt`中的所有依赖：

```bash
pip install -r requirements.txt
```

### 使用说明

1. **克隆仓库**

   ```bash
   git clone https://github.com/userpandawin/MambaCFE.git
   cd MambaCFE
   ```

2. **安装依赖**

   确保您已经安装了Python 3.x，并使用以下命令安装项目依赖：

   ```bash
   pip install -r requirements.txt
   ```

3. **准备数据**

   请确保您的数据文件位于`data/`目录下。如果没有数据，请根据需要下载或准备相应的股票数据。

4. **运行MTS模型**

   打开并运行`mamba_test.ipynb` Jupyter Notebook文件。该Notebook文件包含了模型训练和测试的完整代码。

### 示例代码

在`mamba_test.ipynb`中，您将看到以下示例代码：

```python
# 设置参数
class Args:
    use_cuda = True
    seed = 1
    epochs = 90
    lr = 0.01
    wd = 1e-5
    hidden = 16
    layer = 2
    n_test = 46
    ts_code = '301314' ##选择股票代码
    cfe = 'True' ## 是否使用CFE
    
args = Args()
args.cuda = args.use_cuda and torch.cuda.is_available()
```

### 评价指标

模型的性能将通过以下指标进行评价：
- **MSE（均方误差）**
- **RMSE（均方根误差）**
- **MAE（平均绝对误差）**
- **R²（决定系数）**

### 项目贡献

欢迎任何形式的贡献，包括但不限于：
- 提交Bug报告或功能请求
- 创建Pull Request进行代码改进
- 提出优化建议

### 许可证

该项目采用MIT许可证。详细信息请参见LICENSE文件。

### 联系方式

如有任何问题，请通过以下方式与我们联系：
- 邮箱：yuanjianzhang2003@163.com

---

感谢您对MTS项目的关注和支持！

---

### 致谢

特别感谢所有为本项目做出贡献和支持的人们。

---

### 环境依赖文件 (requirements.txt)

```shell
numpy
pandas
scikit-learn
tensorflow
matplotlib
jupyter
```

---

请按照上述说明进行操作，确保所有步骤都能顺利完成。如果遇到任何问题，请随时联系！
