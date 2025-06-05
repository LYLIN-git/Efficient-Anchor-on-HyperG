import pandas as pd
import numpy as np

# 1. 指定正確路徑
df = pd.read_csv(
    'data/Paper_datasets/mushroom/agaricus-lepiota.data', header=None)

# 2. encoding 轉數字
X = df.iloc[:, 1:].apply(lambda x: pd.factorize(x)[0]).values  # features
y = pd.factorize(df.iloc[:, 0])[0]                             # label

# 3. 儲存
np.save('data/Paper_datasets/mushroom.npy', X)
np.save('data/Paper_datasets/mushroom.label', y)


df = pd.read_csv('data/Paper_datasets/zoo/zoo.data', header=None)
X = df.iloc[:, 1:-1].values        # 去掉第0欄(名字)和最後一欄(label)
y = pd.factorize(df.iloc[:, -1])[0]  # 直接 factorize label

np.save('data/Paper_datasets/zoo.npy', X)
np.save('data/Paper_datasets/zoo.label', y)


df = pd.read_csv('data/Paper_datasets/covertype/covtype.data', header=None)
X = df.iloc[:, :-1].values           # 去掉最後一欄（label）
y = pd.factorize(df.iloc[:, -1])[0]  # label

np.save('data/Paper_datasets/covertype.npy', X)
np.save('data/Paper_datasets/covertype.label', y)
