class Metrics:    
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
    

