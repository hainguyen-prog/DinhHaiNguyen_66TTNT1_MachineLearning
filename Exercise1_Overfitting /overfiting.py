import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score

df = pd.read_csv('du_lieu_gia_nha.csv') 
X = df.iloc[:, :-1].values  #Input: chứa thông tin diện tích, số phòng ngủ, khoảng cách 
y = df.iloc[:, -1].values   #Output: giá nhà

sort_idx = np.argsort(y)
X, y = X[sort_idx], y[sort_idx]
truc_x = np.arange(len(y))

# Lấy ngẫu nhiên các vị trí (index) để chia 80% Train - 20% Test
train_idx, test_idx = train_test_split(truc_x, test_size=0.2, random_state=42)

#Sắp xếp lại index Train/Test từ trái sang phải
train_idx = np.sort(train_idx)
test_idx = np.sort(test_idx)

# Lấy dữ liệu X, y tương ứng với các vị trí đã sắp xếp
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# PHẦN 1: DECISIONTREE => OVERFITTING

print("--- ĐANG CHẠY PHẦN 1: MÔ HÌNH BỊ OVERFITTING ---")
tree_overfit = DecisionTreeRegressor(random_state=42)
tree_overfit.fit(X_train, y_train)

# Lấy kết quả dự đoán riêng biệt cho Train và Test
y_train_pred_tree = tree_overfit.predict(X_train)
y_test_pred_tree = tree_overfit.predict(X_test)

plt.figure(figsize=(10, 5))
plt.plot(truc_x, y, label='1. Thực tế (Chuẩn)', color='green', marker='o', alpha=0.5)
plt.plot(train_idx, y_train_pred_tree, label='2. Train (Khớp 100%)', color='blue', linestyle='--', marker='^')
plt.plot(test_idx, y_test_pred_tree, label='3. Test (Văng tung tóe)', color='red', linestyle='-.', marker='x')
plt.title('PHẦN 1: BỊ OVERFITTING (Cây quyết định)')
plt.xlabel('Thứ tự căn nhà (Từ rẻ nhất -> đắt nhất)')
plt.ylabel('Giá nhà')
plt.legend()
plt.grid(True, linestyle=':')
plt.show()


# PHẦN 2: K-FOLD + HỒI QUY TUYẾN TÍNH

print("\n--- ĐANG CHẠY PHẦN 2: K-FOLD + HỒI QUY TUYẾN TÍNH ---")
lr_model = LinearRegression()

# Dùng K-Fold kiểm tra độ ổn định của Hồi quy trên tập Train
kf = KFold(n_splits=5, shuffle=True, random_state=42)
sai_so = cross_val_score(lr_model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
print(f"=> Lỗi K-Fold (MSE): {-sai_so.mean():.2f}. Mô hình rất ổn định, tiến hành huấn luyện!")

lr_model.fit(X_train, y_train)

# Lấy kết quả dự đoán riêng biệt cho Train và Test
y_train_pred_lr = lr_model.predict(X_train)
y_test_pred_lr = lr_model.predict(X_test)

plt.figure(figsize=(10, 5))
plt.plot(truc_x, y, label='1. Thực tế (Chuẩn)', color='green', marker='o', alpha=0.5)
plt.plot(train_idx, y_train_pred_lr, label='2. Train (Bám xu hướng)', color='blue', linestyle='--', marker='^')
plt.plot(test_idx, y_test_pred_lr, label='3. Test (Dự đoán tốt)', color='red', linestyle='-.', marker='x')
plt.title('PHẦN 2: ĐÃ KHẮC PHỤC (Linear Regression + K-Fold)')
plt.xlabel('Thứ tự căn nhà (Từ rẻ nhất -> đắt nhất)')
plt.ylabel('Giá nhà')
plt.legend()
plt.grid(True, linestyle=':')
plt.show()
