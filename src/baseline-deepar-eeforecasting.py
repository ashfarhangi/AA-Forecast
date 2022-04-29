import util
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from progressbar import *
from datetime import date
import argparse
from time import time
from torch.optim import Adam
import random
import os
from joblib import load, dump
import pickle
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import sklearn
from matplotlib import rc
from pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('/data/electricity.csv', parse_dates=['Date'])
df = df[df.region == 'MT_200']
df.reset_index(drop=True, inplace=True)
feature = ['observed', 'weekday', 'month', 'year']
feature = ['observed']

target = ['observed']
df_og = df
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(df_og[feature])
df[feature] = scaler.transform(df_og[feature])
df


def create_seq(df, feature, target, seq_window, hor_window):
    Xs = []
    ys = []
    for j in range(len(df)-seq_window-1):
        X = df[feature][j:seq_window+j]
        y = df[target][seq_window+j:seq_window+j+hor_window]
        Xs.append(X)
        ys.append(y)
    return np.array(Xs), np.array(ys)


seq_window = 8
hor_window = 1
X, y = create_seq(df, feature, target, seq_window, hor_window)
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()
print('X shape:', X.shape)
print('y shape:', y.shape)


def MAE(ytrue, ypred):
    ytrue = np.array(ytrue).ravel()
    ypred = np.array(ypred).ravel()
    return np.mean(np.abs((ytrue - ypred)))


def MSE(ytrue, ypred):
    ytrue = np.array(ytrue).ravel()
    ypred = np.array(ypred).ravel()
    return np.mean(np.square((ytrue - ypred)))


def RMSE(ypred, ytrue):
    rsme = np.sqrt(np.mean(np.square(ypred - ytrue)))
    return rsme


def get_data_path():
    folder = os.path.dirname(__file__)
    return os.path.join(folder, "data")


def RSE(ypred, ytrue):
    rse = np.sqrt(np.square(ypred - ytrue).sum()) / \
        np.sqrt(np.square(ytrue - ytrue.mean()).sum())
    return rse


def quantile_loss(ytrue, ypred, qs):
    L = np.zeros_like(ytrue)
    for i, q in enumerate(qs):
        yq = ypred[:, :, i]
        diff = yq - ytrue
        L += np.max(q * diff, (q - 1) * diff)
    return L.mean()


def SMAPE(ytrue, ypred):
    ytrue = np.array(ytrue).ravel()
    ypred = np.array(ypred).ravel() + 1e-4
    mean_y = (ytrue + ypred) / 2.
    return np.mean(np.abs((ytrue - ypred)
                          / mean_y))


def MAPE(ytrue, ypred):
    ytrue = np.array(ytrue).ravel() + 1e-4
    ypred = np.array(ypred).ravel()
    return np.mean(np.abs((ytrue - ypred)
                          / ytrue))


def train_test_split(X, y, train_ratio=0.7):
    num_ts, num_periods, num_features = X.shape
    train_periods = int(num_periods * train_ratio)
    random.seed(2)
    Xtr = X[:, :train_periods, :]
    ytr = y[:, :train_periods]
    Xte = X[:, train_periods:, :]
    yte = y[:, train_periods:]
    return Xtr, ytr, Xte, yte


class StandardScalerManual:

    def fit_transform(self, y):
        self.mean = np.mean(y)
        self.std = np.std(y) + 1e-4
        return (y - self.mean) / self.std

    def inverse_transform(self, y):
        return y * self.std + self.mean

    def transform(self, y):
        return (y - self.mean) / self.std


class MaxScaler:

    def fit_transform(self, y):
        self.max = np.max(y)
        return y / self.max

    def inverse_transform(self, y):
        return y * self.max

    def transform(self, y):
        return y / self.max


class MeanScaler:

    def fit_transform(self, y):
        self.mean = np.mean(y)
        return y / self.mean

    def inverse_transform(self, y):
        return y * self.mean

    def transform(self, y):
        return y / self.mean


class LogScaler:

    def fit_transform(self, y):
        return np.log1p(y)

    def inverse_transform(self, y):
        return np.expm1(y)

    def transform(self, y):
        return np.log1p(y)


def gaussian_likelihood_loss(z, mu, sigma):
    negative_likelihood = torch.log(
        sigma + 1) + (z - mu) ** 2 / (2 * sigma ** 2) + 6
    return negative_likelihood.mean()


def negative_binomial_loss(ytrue, mu, alpha):
    batch_size, seq_len = ytrue.size()
    likelihood = torch.lgamma(ytrue + 1. / alpha) - torch.lgamma(ytrue + 1) - torch.lgamma(1. / alpha) \
        - 1. / alpha * torch.log(1 + alpha * mu) \
        + ytrue * torch.log(alpha * mu / (1 + alpha * mu))
    return - likelihood.mean()


def batch_generator(X, y, num_obs_to_train, seq_len, batch_size):
    num_ts, num_periods, _ = X.shape
    if num_ts < batch_size:
        batch_size = num_ts
    t = random.choice(range(num_obs_to_train, num_periods-seq_len))
    batch = random.sample(range(num_ts), batch_size)
    X_train_batch = X[batch, t-num_obs_to_train:t, :]
    y_train_batch = y[batch, t-num_obs_to_train:t]
    Xf = X[batch, t:t+seq_len]
    yf = y[batch, t:t+seq_len]
    return X_train_batch, y_train_batch, Xf, yf


class AutoEncoder(nn.Module):

    def __init__(self, input_size, encoder_hidden_units):
        super(AutoEncoder, self).__init__()
        self.layers = []
        self.dropout = nn.Dropout()
        last_ehu = None
        for idx, ehu in enumerate(encoder_hidden_units):
            if idx == 0:
                layer = nn.LSTM(input_size, ehu, 1,
                                bias=True, batch_first=True)
            else:
                layer = nn.LSTM(last_ehu, ehu, 1, bias=True, batch_first=True)
            last_ehu = ehu
            self.layers.append(layer)

    def forward(self, x):
        batch_size, seq_len, input_size = x.size()
        for layer in self.layers:
            hs = []
            for s in range(seq_len):
                _, (h, c) = layer(x)
                h = h.permute(1, 0, 2)
                h = F.relu(h)
                h = self.dropout(h)
                hs.append(h)
            x = torch.cat(hs, dim=1)
        return x


class Forecaster(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers):
        super(Forecaster, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            n_layers, bias=True, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mu):
        '''
        Args:
        x (tensor): 
        mu (tensor): model uncertainty
        '''
        batch_size, seq_len, hidden_size = x.size()
        out = []
        for s in range(seq_len):
            xt = x[:, s, :].unsqueeze(1)
            xt = torch.cat([xt, mu], dim=1)
            _, (h, c) = self.lstm(xt)
            ht = h[-1, :, :].unsqueeze(0)
            h = ht.permute(1, 0, 2)
            h = F.relu(h)
            h = self.dropout(h)
            out.append(h)
        out = torch.cat(out, dim=1)
        out = self.fc(out)
        return out

class ExtremeModel(nn.Module):

    def __init__(
        self,
        input_size,
        encoder_hidden_units=[512, 128, 64],
        hidden_size_forecaster=512,
        n_layers_forecaster=3
    ):
        super(ExtremeModel, self).__init__()
        self.embed = nn.Linear(input_size, encoder_hidden_units[-1])
        self.auto_encoder = AutoEncoder(
            encoder_hidden_units[-1], encoder_hidden_units)
        self.forecaster = Forecaster(encoder_hidden_units[-1],
                                     hidden_size_forecaster, n_layers_forecaster)

    def forward(self, xpast, xnew):
        if isinstance(xpast, type(np.empty(1))):
            xpast = torch.from_numpy(xpast).float()
        if isinstance(xnew, type(np.empty(1))):
            xnew = torch.from_numpy(xnew).float()
        xpast = self.embed(xpast)
        xnew = self.embed(xnew)
        # auto-encoder
        ae_out = self.auto_encoder(xpast)
        ae_out = torch.mean(ae_out, dim=1).unsqueeze(1)
        # concatenate x
        # x = torch.cat([xnew, ae_out], dim=1)
        x = self.forecaster(xnew, ae_out)
        return x


def batch_generator(X, y, num_obs_to_train, seq_len, batch_size):
    num_ts, num_periods, _ = X.shape
    if num_ts < batch_size:
        batch_size = num_ts
    t = random.choice(range(num_obs_to_train, num_periods-seq_len))
    batch = random.sample(range(num_ts), batch_size)
    X_train_batch = X[batch, t-num_obs_to_train:t, :]
    y_train_batch = y[batch, t-num_obs_to_train:t]
    Xf = X[batch, t:t+seq_len]
    yf = y[batch, t:t+seq_len]
    return X_train_batch, y_train_batch, Xf, yf


def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat-y)**2))


def MAELoss(yhat, y):
    loss = torch.nn.L1Loss()
    output = loss(input, target)
    return output.backward()


"""### Train"""


def train(
    X,
    y,
    seq_len,
    num_obs_to_train,
    lr,
    num_epoches,
    step_per_epoch,
    batch_size
):

    num_ts, num_periods, num_features = X.shape
    Xtr, ytr, Xte, yte = train_test_split(X, y)
    yscaler = None
    # if args.standard_scaler:
    yscaler = StandardScalerManual()
    # elif args.log_scaler:
    #     yscaler = LogScaler()
    # elif args.mean_scaler:
    #     yscaler = util.MeanScaler()
    if yscaler is not None:
        ytr = yscaler.fit_transform(ytr)

    progress = ProgressBar()
    seq_len = seq_len
    num_obs_to_train = num_obs_to_train

    model = ExtremeModel(num_features)
    optimizer = Adam(model.parameters(), lr=lr)
    losses = []
    MAE_losses = []
    mape_list = []
    mse_list = []
    rmse_list = []
    mae_list = []

    cnt = 0
    for epoch in progress(range(num_epoches)):
        # print("Epoch {} starts...".format(epoch))
        for step in range(step_per_epoch):
            Xtrain, ytrain, Xf, yf = batch_generator(Xtr, ytr, num_obs_to_train,
                                                     seq_len, batch_size)
            Xtrain_tensor = torch.from_numpy(Xtrain).float()
            ytrain_tensor = torch.from_numpy(ytrain).float()
            Xf = torch.from_numpy(Xf).float()
            yf = torch.from_numpy(yf).float()
            ypred = model(Xtrain_tensor, Xf)
            # loss = F.mse_loss(ypred, yf)
            loss = RMSELoss(ypred, yf)
            loss_mae = F.l1_loss(ypred, yf)
            MAE_losses.append(np.float(loss_mae))
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt += 1

    # select skus with most top K
    X_test = Xte[:, -seq_len-num_obs_to_train:-
                 seq_len, :].reshape((num_ts, -1, num_features))
    Xf_test = Xte[:, -seq_len:, :].reshape((num_ts, -1, num_features))
    y_test = yte[:, -seq_len-num_obs_to_train:-seq_len].reshape((num_ts, -1))
    yf_test = yte[:, -seq_len:].reshape((num_ts, -1))
    if yscaler is not None:
        y_test = yscaler.transform(y_test)
    ypred = model(X_test, Xf_test)
    ypred = ypred.data.numpy()
    if yscaler is not None:
        ypred = yscaler.inverse_transform(ypred)

    mape = MAPE(yf_test, ypred)
    mae = MAE(yf_test, ypred)
    mse = MSE(yf_test, ypred)
    rmse = RMSE(yf_test, ypred)
    # print("MAE: {}".format(mae))
    # print("RMSE: {}".format(rmse))
    # print("MSE: {}".format(mse))
    # print("MAPE: {}".format(mape))
    mape_list.append(mape)
    mse_list.append(mse)
    mae_list.append(mae)
    rmse_list.append(rmse)

    plt.figure(1)
    plt.plot([k + seq_len + num_obs_to_train - seq_len
              for k in range(seq_len)], ypred[-1], "r-")
    plt.title('EE-Forecasting')
    yplot = yte[-1, -seq_len-num_obs_to_train:]
    plt.plot(range(len(yplot)), yplot, "k-")
    plt.legend(["forecast", "true"], loc="upper left")
    plt.xlabel("Periods")
    plt.ylabel("Y")
    plt.show()

    return yf_test, ypred, losses, MAE_losses, mape_list, mse_list, mae_list, rmse_list


df

X = np.c_[np.asarray(hours), np.asarray(dows)]
num_features = X.shape[1]
num_periods = len(df)
X = np.asarray(X).reshape((-1, num_periods, num_features))
y = np.asarray(df["observed"]).reshape((-1, num_periods))
y_test, y_pred, losses, MAE_losses, mape_list, mse_list, mae_list, rmse_list = train(X, y, seq_len=8,
                                                                                     num_obs_to_train=4,
                                                                                     lr=1e-3,
                                                                                     num_epoches=50,
                                                                                     step_per_epoch=2,
                                                                                     batch_size=32
                                                                                     )
plt.plot(range(len(losses)), losses, "k-")
plt.xlabel("Period")
plt.ylabel("RMSE")
plt.title('RMSE: '+str(np.average(losses))+'MAE:'+str(np.average(MAE_losses)))
plt.show()
plt.savefig('training_EE.png')

"""## DeepAR"""


# import util


class Gaussian(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(Gaussian, self).__init__()
        self.mu_layer = nn.Linear(hidden_size, output_size)
        self.sigma_layer = nn.Linear(hidden_size, output_size)

    def forward(self, h):
        _, hidden_size = h.size()
        sigma_t = torch.log(1 + torch.exp(self.sigma_layer(h))) + 1e-6
        sigma_t = sigma_t.squeeze(0)
        mu_t = self.mu_layer(h).squeeze(0)
        return mu_t, sigma_t


class NegativeBinomial(nn.Module):

    def __init__(self, input_size, output_size):
        '''
        Negative Binomial Supports Positive Count Data
        Args:
        input_size (int): hidden h_{i,t} column size
        output_size (int): embedding size
        '''
        super(NegativeBinomial, self).__init__()
        self.mu_layer = nn.Linear(input_size, output_size)
        self.sigma_layer = nn.Linear(input_size, output_size)

    def forward(self, h):
        _, hidden_size = h.size()
        alpha_t = torch.log(1 + torch.exp(self.sigma_layer(h))) + 1e-6
        mu_t = torch.log(1 + torch.exp(self.mu_layer(h)))
        return mu_t, alpha_t


def gaussian_sample(mu, sigma):
    gaussian = torch.distributions.normal.Normal(mu, sigma)
    ypred = gaussian.sample(mu.size())
    return ypred


def negative_binomial_sample(mu, alpha):
    var = mu + mu * mu * alpha
    ypred = mu + torch.randn(mu.size()) * torch.sqrt(var)
    return ypred


class DeepAR(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, num_layers, lr=1e-3, likelihood="g"):
        super(DeepAR, self).__init__()

        # network
        self.input_embed = nn.Linear(1, embedding_size)
        self.encoder = nn.LSTM(embedding_size+input_size, hidden_size,
                               num_layers, bias=True, batch_first=True)
        if likelihood == "g":
            self.likelihood_layer = Gaussian(hidden_size, 1)
        elif likelihood == "nb":
            self.likelihood_layer = NegativeBinomial(hidden_size, 1)
        self.likelihood = likelihood

    def forward(self, X, y, Xf):
        if isinstance(X, type(np.empty(2))):
            X = torch.from_numpy(X).float()
            y = torch.from_numpy(y).float()
            Xf = torch.from_numpy(Xf).float()
        num_ts, seq_len, _ = X.size()
        _, output_horizon, num_features = Xf.size()
        ynext = None
        ypred = []
        mus = []
        sigmas = []
        h, c = None, None
        for s in range(seq_len + output_horizon):
            if s < seq_len:
                ynext = y[:, s].view(-1, 1)
                yembed = self.input_embed(ynext).view(num_ts, -1)
                x = X[:, s, :].view(num_ts, -1)
            else:
                yembed = self.input_embed(ynext).view(num_ts, -1)
                x = Xf[:, s-seq_len, :].view(num_ts, -1)
            # num_ts, num_features + embedding
            x = torch.cat([x, yembed], dim=1)
            inp = x.unsqueeze(1)
            if h is None and c is None:
                # h size (num_layers, num_ts, hidden_size)
                out, (h, c) = self.encoder(inp)
            else:
                out, (h, c) = self.encoder(inp, (h, c))
            hs = h[-1, :, :]
            hs = F.relu(hs)
            mu, sigma = self.likelihood_layer(hs)
            mus.append(mu.view(-1, 1))
            sigmas.append(sigma.view(-1, 1))
            if self.likelihood == "g":
                ynext = gaussian_sample(mu, sigma)
            elif self.likelihood == "nb":
                alpha_t = sigma
                mu_t = mu
                ynext = negative_binomial_sample(mu_t, alpha_t)
            # if without true value, use prediction
            if s >= seq_len - 1 and s < output_horizon + seq_len - 1:
                ypred.append(ynext)
        ypred = torch.cat(ypred, dim=1).view(num_ts, -1)
        mu = torch.cat(mus, dim=1).view(num_ts, -1)
        sigma = torch.cat(sigmas, dim=1).view(num_ts, -1)
        return ypred, mu, sigma


def batch_generator(X, y, num_obs_to_train, seq_len, batch_size):
    num_ts, num_periods, _ = X.shape
    if num_ts < batch_size:
        batch_size = num_ts
    t = random.choice(range(num_obs_to_train, num_periods-seq_len))
    batch = random.sample(range(num_ts), batch_size)
    X_train_batch = X[batch, t-num_obs_to_train:t, :]
    y_train_batch = y[batch, t-num_obs_to_train:t]
    Xf = X[batch, t:t+seq_len]
    yf = y[batch, t:t+seq_len]
    return X_train_batch, y_train_batch, Xf, yf


def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat-y)**2))


def train(
    X,
    y,
    seq_len,
    num_obs_to_train,
    lr,
    num_epoches,
    step_per_epoch,
    batch_size,
    likelihood,
    embedding_size,
    n_layers,
    sample_size,
    hidden_size
):
    num_ts, num_periods, num_features = X.shape
    model = DeepAR(num_features, embedding_size,
                   hidden_size, n_layers, lr, likelihood)
    optimizer = Adam(model.parameters(), lr=lr)
    random.seed(2)
    # select sku with most top n quantities
    Xtr, ytr, Xte, yte = util.train_test_split(X, y)
    losses = []
    cnt = 0

    yscaler = None
    # if args.standard_scaler:
    yscaler = util.StandardScaler()
    # elif args.log_scaler:
    #     yscaler = util.LogScaler()
    # elif args.mean_scaler:
    #     yscaler = util.MeanScaler()
    if yscaler is not None:
        ytr = yscaler.fit_transform(ytr)
    rmse_losses = []
    mae_losses = []
    # training
    seq_len = seq_len
    num_obs_to_train = num_obs_to_train
    progress = ProgressBar()
    for epoch in progress(range(num_epoches)):
        # print("Epoch {} starts...".format(epoch))
        for step in range(step_per_epoch):
            Xtrain, ytrain, Xf, yf = batch_generator(
                Xtr, ytr, num_obs_to_train, seq_len, batch_size)
            Xtrain_tensor = torch.from_numpy(Xtrain).float()
            ytrain_tensor = torch.from_numpy(ytrain).float()
            Xf = torch.from_numpy(Xf).float()
            yf = torch.from_numpy(yf).float()
            ypred, mu, sigma = model(Xtrain_tensor, ytrain_tensor, Xf)
            # ypred_rho = ypred
            # e = ypred_rho - yf
            # loss = torch.max(rho * e, (rho - 1) * e).mean()
            # gaussian loss
            loss_rmse_inter = RMSELoss(ypred, yf)
            mae_losses_inter = mean_absolute_error(ypred, yf)
            mae_losses.append(mae_losses_inter)
            rmse_losses.append(loss_rmse_inter)
            ytrain_tensor = torch.cat([ytrain_tensor, yf], dim=1)
            if likelihood == "g":
                loss = util.gaussian_likelihood_loss(ytrain_tensor, mu, sigma)
            elif likelihood == "nb":
                loss = util.negative_binomial_loss(ytrain_tensor, mu, sigma)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt += 1

    # test
    mape_list = []
    # select skus with most top K
    X_test = Xte[:, -seq_len-num_obs_to_train:-
                 seq_len, :].reshape((num_ts, -1, num_features))
    Xf_test = Xte[:, -seq_len:, :].reshape((num_ts, -1, num_features))
    y_test = yte[:, -seq_len-num_obs_to_train:-seq_len].reshape((num_ts, -1))
    yf_test = yte[:, -seq_len:].reshape((num_ts, -1))
    if yscaler is not None:
        y_test = yscaler.transform(y_test)
    result = []
    n_samples = sample_size
    for _ in tqdm(range(n_samples)):
        y_pred, _, _ = model(X_test, y_test, Xf_test)
        y_pred = y_pred.data.numpy()
        if yscaler is not None:
            y_pred = yscaler.inverse_transform(y_pred)
        result.append(y_pred.reshape((-1, 1)))

    result = np.concatenate(result, axis=1)
    p50 = np.quantile(result, 0.5, axis=1)
    p90 = np.quantile(result, 0.9, axis=1)
    p10 = np.quantile(result, 0.1, axis=1)

    mape = util.MAPE(yf_test, p50)
    print("P50 MAPE: {}".format(mape))
    mape_list.append(mape)

    # if args.show_plot:
    plt.figure(1, figsize=(20, 5))
    plt.plot([k + seq_len + num_obs_to_train - seq_len
              for k in range(seq_len)], p50, "r-")
    plt.fill_between(x=[k + seq_len + num_obs_to_train - seq_len for k in range(seq_len)],
                     y1=p10, y2=p90, alpha=0.5)
    plt.title('Prediction uncertainty')
    yplot = yte[-1, -seq_len-num_obs_to_train:]
    plt.plot(range(len(yplot)), yplot, "k-")
    plt.legend(["P50 forecast", "true", "P10-P90 quantile"], loc="upper left")
    ymin, ymax = plt.ylim()
    plt.vlines(seq_len + num_obs_to_train - seq_len, ymin, ymax,
               color="blue", linestyles="dashed", linewidth=2)
    plt.ylim(ymin, ymax)
    plt.xlabel("Periods")
    plt.ylabel("Y")
    plt.show()
    return yf_test, ypred, losses, rmse_losses, mae_losses, mape_list, mse_list, mae_list, rmse_list


df["year"] = df["Date"].apply(lambda x: x.year)
df["day_of_week"] = df["Date"].apply(lambda x: x.dayofweek)
df["hour"] = df["Date"].apply(lambda x: x.hour)

features = ["hour", "day_of_week"]
hours = df["hour"]
dows = df["day_of_week"]
X = np.c_[np.asarray(hours), np.asarray(dows)]
num_features = X.shape[1]
num_periods = len(df)
X = np.asarray(X).reshape((-1, num_periods, num_features))
y = np.asarray(df["observed"]).reshape((-1, num_periods))
y_test, y_pred, losses, rmse_losses, mae_losses, mape_list, mse_list, mae_list, rmse_list = train(X, y, seq_len=7,
                                                                                                  num_obs_to_train=1,
                                                                                                  lr=1e-3,
                                                                                                  num_epoches=1000,
                                                                                                  step_per_epoch=2,
                                                                                                  batch_size=32,
                                                                                                  sample_size=100,
                                                                                                  n_layers=3,
                                                                                                  hidden_size=64,
                                                                                                  embedding_size=64,
                                                                                                  likelihood="g"
                                                                                                  )
plt.plot(range(len(rmse_losses)), rmse_losses, "k-")
plt.xlabel("Period")
plt.ylabel("RMSE")
plt.title('RMSE: '+str(np.average(rmse_losses)) +
          'MAE:' + str(np.average(mae_losses)))
plt.show()
plt.savefig('training_DeepAR.png')

rmse_losses

plt.title('RMSE average: '+str(np.average(rmse_losses)) +
          'MAE average: ' + str(np.average(mae_losses)))

plt.plot(range(len(rmse_losses)), rmse_losses, "k-")
plt.xlabel("Period")
plt.ylabel("RMSE")
plt.title('RMSE average: '+str(np.average(rmse_losses)) +
          'MAE average: ' + str(np.average(mae_losses)))
plt.show()
plt.savefig('training_EE.png')
