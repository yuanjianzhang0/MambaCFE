import torch
import torch.nn as nn
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.dates as mdates

matplotlib.use('Agg')  # 使用无GUI的后端
import akshare as ak
stock_code = "600015"
stock_info = ak.stock_zh_a_spot_em()[ak.stock_zh_a_spot_em()['代码'] == stock_code]
# stock_name = stock_info['名称'].values[0] if not stock_info.empty else "Unknown"
stock_name = '招商银行'
torch.manual_seed(0)
np.random.seed(0)
lr = 0.005
epochs = 50 #45
input_window = 8
output_window = 1
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date='20230701', end_date='20240701', adjust="qfq")

# 选择“收盘”列
df = df[['日期', '收盘']]  # 假设你需要保留日期列

# df = pd.read_csv(f'dataset/{stock_name}.csv', usecols=['收盘'])

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):
    def __init__(self, feature_size=250, num_layers=1, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + output_window:i + tw + output_window]
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(inout_seq)


def get_data():
    df = pd.read_csv(f'dataset/{stock_name}.csv', usecols=['日期', '收盘'])
    series = df['收盘'].values
    scaler = MinMaxScaler(feature_range=(-1, 1))
    series = scaler.fit_transform(series.reshape(-1, 1)).reshape(-1)

    train_samples = int(0.8 * len(series))
    train_data = series[:train_samples]
    test_data = series[train_samples:]

    train_sequence = create_inout_sequences(train_data, input_window)
    train_sequence = train_sequence[:-output_window]

    test_data = create_inout_sequences(test_data, input_window)
    test_data = test_data[:-output_window]

    test_dates = df['日期'][train_samples + input_window:].reset_index(drop=True)

    return train_sequence.to(device), test_data.to(device), scaler, test_dates

def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    return input, target


def evaluation_metric(y_test, y_hat):
    MSE = mean_squared_error(y_test, y_hat)
    RMSE = MSE ** 0.5
    MAE = mean_absolute_error(y_test, y_hat)
    R2 = r2_score(y_test, y_hat)
    print('%.4f %.4f %.4f %.4f' % (MSE, RMSE, MAE, R2))
    with open(f'Transformer/MTS_{stock_name}_evaluation.txt', 'w') as f:
        f.write(f"MSE: {MSE}\n")
        f.write(f"RMSE: {RMSE}\n")
        f.write(f"MAE: {MAE}\n")
        f.write(f"R2: {R2}\n")
    print('%.4f %.4f %.4f %.4f' % (MSE,RMSE,MAE,R2))


def train(train_data):
    model.train()

    for batch_index, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        start_time = time.time()
        total_loss = 0
        data, targets = get_batch(train_data, i, batch_size)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.item()
        log_interval = max(1, int(len(train_data) / batch_size / 5))  # 确保 log_interval 至少为 1
        if batch_index % log_interval == 0 and batch_index > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | {:5.2f} ms | loss {:5.5f} | ppl {:8.2f}'
                  .format(epoch, batch_index, len(train_data) // batch_size, scheduler.get_lr()[0],
                          elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0


def evaluate(eval_model, data_source):
    eval_model.eval()
    total_loss = 0
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            output = eval_model(data)
            total_loss += len(data[0]) * criterion(output, targets).cpu().item()
    return total_loss / len(data_source)


def plot_and_loss(eval_model, data_source, scaler, test_dates):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1)
            output = eval_model(data)
            total_loss += criterion(output, target).item()
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)

    test_result = scaler.inverse_transform(test_result.reshape(-1, 1)).reshape(-1)
    truth = scaler.inverse_transform(truth.reshape(-1, 1)).reshape(-1)
    test_dates = test_dates[2:]
    test_dates = test_dates.reset_index(drop=True)  # 重置索引，确保索引是连续的
    plt.rcParams['font.sans-serif'] = ['SimHei']

    truth = truth[:len(truth) - 1]
    test_result = test_result[1:]
    test_dates = test_dates[1:]

    test_dates = test_dates.reset_index(drop=True)

    plt.figure(figsize=(24, 16))
    plt.plot(test_dates, truth, label='真实值')
    plt.plot(test_dates, test_result, label='预测值')

    for i in range(len(truth)):
        plt.annotate(f'{truth[i]:.2f}', (test_dates[i], truth[i]), textcoords="offset points", xytext=(0, 10), ha='center',
                     fontsize=12)
        plt.annotate(f'{test_result[i]:.2f}', (test_dates[i], test_result[i]),
                     textcoords="offset points", xytext=(0, -15), ha='center', fontsize=12, color='red')

    plt.title(f'Transformer{stock_name}预测', fontsize=36)
    plt.xlabel('时间', fontsize=12, verticalalignment='top')
    plt.ylabel('收盘价', fontsize=14, horizontalalignment='center')
    plt.legend()
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))  # 设置日期间隔
    plt.gcf().autofmt_xdate()  # 自动旋转日期标签
    plt.savefig(f'Transformer/{stock_name}预测.png', dpi=300, bbox_inches='tight')
    evaluation_metric(test_result, truth)
    return total_loss / i


train_data, val_data, scale, test_dates = get_data()
model = TransAm().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)


for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data)

    if epoch == epochs:
        val_loss = plot_and_loss(model, val_data, scale, test_dates)
    else:
        val_loss = evaluate(model, val_data)

    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (
            time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
    print('-' * 89)
    scheduler.step()
