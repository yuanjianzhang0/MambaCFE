## README[[中文](https://github.com/userpandawin/MambaCFE/blob/main/README_CN.md)][[English](https://github.com/userpandawin/MambaCFE/blob/main/README_EN.md)]

# MTS: An Efficient Stock Prediction Method Based on the Improved Mamba Model

### Project Introduction

MTS (Multiscale Time Series) is an efficient stock prediction method based on the improved Mamba model. This model integrates convolution, attention mechanisms, and multiscale convolution, and introduces a novel local feature extraction module (CFE) to replace traditional convolution operations. The MTS model performs exceptionally well in stock prediction tasks across multiple industries.

### Author

Yuanjian Zhang, Group 13, Beijing University of Posts and Telecommunications
Email: yuanjianzhang2003@163.com

### File Structure

```
MTS/
│
├── mamba_test.ipynb        # Main test script
├── requirements.txt        # Environment dependencies file
└── README.md               # Project description file
```

### Environment Dependencies

Please install the necessary dependencies before running the code. You can install all dependencies listed in `requirements.txt` using the following command:

```bash
pip install -r requirements.txt
```

### Usage Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/userpandawin/MambaCFE.git
   cd MambaCFE
   ```

2. **Install Dependencies**

   Ensure you have Python 3.x installed, then install the project dependencies using:

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data**

   Ensure your data files are located in the `data/` directory. If you do not have data, please download or prepare the appropriate stock data.

4. **Run the MTS Model**

   Open and run the `mamba_test.ipynb` Jupyter Notebook file. This Notebook contains the complete code for model training and testing.

### Example Code

In `mamba_test.ipynb`, you will find the following example code:

```python
# Set parameters
class Args:
    use_cuda = True
    seed = 1
    epochs = 90
    lr = 0.01
    wd = 1e-5
    hidden = 16
    layer = 2
    n_test = 46
    ts_code = '301314'  # Select stock code
    cfe = 'True'  # Whether to use CFE
    
args = Args()
args.cuda = args.use_cuda and torch.cuda.is_available()
```

### Evaluation Metrics

The model's performance will be evaluated using the following metrics:

- **MSE (Mean Squared Error)**
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **R² (R-squared)**

### Project Contribution

We welcome any form of contribution, including but not limited to:

- Submitting bug reports or feature requests
- Creating Pull Requests for code improvements
- Providing optimization suggestions

### License

This project is licensed under the MIT License. For more details, please refer to the LICENSE file.

### Contact

If you have any questions, please contact us at:

- Email: yuanjianzhang2003@163.com

---

Thank you for your attention and support for the MTS project!

---

### Acknowledgments

Special thanks to everyone who contributed and supported this project.

---

### Environment Dependencies File (requirements.txt)

```shell
akshare==1.14.29
matplotlib==3.9.1
numpy==2.0.0
pandas==2.2.2
scikit_learn==1.5.1
statsmodels==0.14.2
tensorflow_gpu==2.10.1
torch==2.0.1+cu118
xgboost==2.1.0
```

---

Please follow the instructions above to ensure all steps are completed successfully. If you encounter any issues, feel free to contact us!
