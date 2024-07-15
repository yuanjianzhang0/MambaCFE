import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Layer
import tensorflow as tf
import akshare as ak
from tensorflow.keras.optimizers import Adam
from timeseries_predictor import TimeSeriesPredictor as tsf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.dates as mdates

matplotlib.use('Agg')

st_code = ['600519', '000001', '601318', '600276', '600036', '000002', '000858', '002415']
time_step = 5
model_name = 'Transformer'
units = 1200
dropout_rate = 0.2
patience = 500
vp = 0.2
epochs = 50


def create_dataset(dataset, time_step=1):  ## 定义一个函数，用来构建数据集，输入一个原始csv，和时间步长
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


for i in st_code:
    stock_code = i
    df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date="20230701", end_date="20240701", adjust="qfq")

    stock_info = ak.stock_zh_a_spot_em()[ak.stock_zh_a_spot_em()['代码'] == stock_code]
    stock_name = stock_info['名称'].values[0] if not stock_info.empty else "Unknown"

    data = df[['日期', '收盘']]
    data.set_index('日期', inplace=True)  ## 作为索引值

    # data_diff = data.diff().dropna()
    scaler = MinMaxScaler(feature_range=(0, 1))  ### 定义一个数据处理器
    scaled_data = scaler.fit_transform(data)


    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    ## 划分训练数据和测试数据

    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    # print(len(X_test))
    # print(X_train,X_test)
    # 重塑输入数据
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    # y_train = y_train.reshape(-1, 1)

    model = tsf(model_type=model_name, input_shape=(time_step, X_train.shape[2]), units=units,
                dropout_rate=dropout_rate, patience=patience)

    model.train(X_train, y_train, validation_split=vp, epochs=epochs, batch_size=32)

    # 保存模型配置到txt文件
    config = {
        'model_name': model_name,
        'time_step': time_step,
        'units': units,
        'dropout_rate': dropout_rate,
        'patience': patience,
    }

    config_file = f'{model_name}/best_{model_name}_config.txt'
    with open(config_file, 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    print(f"Model configuration saved to {config_file}")

    # 进行预测
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # 逆归一化数据
    # train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    # y_test_actual = scaler.inverse_transform(y_test)
    # train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
    # test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    ### 滑动窗口问题，需要截一下
    test_predict = test_predict[1:]
    y_test_actual = y_test_actual[:len(y_test_actual) - 1]

    # print(f"y_test_actual shape: {y_test_actual.shape}")
    # print(f"test_predict shape: {test_predict.shape}")
    #
    # print(test_predict)
    # print(y_test_actual)

    # 计算评价指标
    wmape = np.sum(np.abs(y_test_actual - test_predict)) / np.sum(np.abs(y_test_actual)) * 100
    mse = mean_squared_error(y_test_actual, test_predict)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_actual, test_predict)
    r2 = r2_score(y_test_actual, test_predict)
    gflops = model.calculate_flops()
    params = model.calculate_params()
    print(f"=========================={stock_name}预测========================")
    # 打印评价指标
    print(f"WMAPE: {wmape:.2f}%")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R²: {r2}")
    if gflops is not None and params is not None:
        print(f"GFLOPs: {gflops / 10 ** 9:.2f} GFLOPs, Params: {params / 10 ** 6:.2f} M")
    # 将评价指标保存到文本文件
    with open(f'{model_name}/{model_name}_{stock_name}_evaluation.txt', 'w') as f:
        f.write(f"WMAPE: {wmape:.2f}%\n")
        f.write(f"MSE: {mse}\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"MAE: {mae}\n")
        f.write(f"R2: {r2}\n")
        if gflops is not None and params is not None:
            f.write(f"GFLOPs:{gflops / 10 ** 9},Params:{params / 10 ** 9} M\n")
    # 获取测试集对应的日期
    # test_dates = data.index[len(train_data) + time_step + 1:len(data) - 1]
    #### 滑动窗口，有问题，需要截一下
    test_dates = data.index[len(train_data) + time_step:len(train_data) + time_step + len(test_predict)]

    print(len(test_dates))
    print(len(test_predict))

    # 绘制测试集预测值
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(16, 8))
    plt.plot(data.index[:train_size], scaler.inverse_transform(train_data), label='训练集')
    plt.plot([data.index[train_size - 1], data.index[train_size]],
             [scaler.inverse_transform(train_data[-1].reshape(1, -1))[0][0],
              scaler.inverse_transform(test_data[0].reshape(1, -1))[0][0]],
             color='black', linestyle='--')
    # 绘制测试集
    plt.plot(data.index[train_size:], scaler.inverse_transform(test_data), label='测试集')
    plt.title(f'{stock_name}股票收盘价', fontsize=24)
    plt.xlabel('时间')
    plt.ylabel('收盘价/元')
    plt.legend()

    plt.savefig(f'{model_name}/数据集划分——{stock_name}.png', dpi=300, bbox_inches='tight')
    # plt.show()

    plt.figure(figsize=(24, 16))
    plt.plot(test_dates, y_test_actual, label='真实值')
    plt.plot(test_dates, test_predict, label='预测值')

    # 添加每个点的注释
    for i, txt in enumerate(y_test_actual):
        plt.annotate(f'{txt[0]:.2f}', (test_dates[i], y_test_actual[i]), textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=8)

    for i, txt in enumerate(test_predict):
        plt.annotate(f'{txt[0]:.2f}', (test_dates[i], test_predict[i]), textcoords="offset points", xytext=(0, -15),
                     ha='center', fontsize=8, color='red')

    plt.title(f'{model_name}{stock_name}股票预测值', fontsize=36)
    plt.xlabel('时间')
    plt.ylabel('收盘价/元')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))  # 设置日期间隔
    plt.gcf().autofmt_xdate()  # 自动旋转日期标签
    plt.savefig(f'{model_name}/{model_name}{stock_name}.png', dpi=300, bbox_inches='tight')
    # plt.show()
