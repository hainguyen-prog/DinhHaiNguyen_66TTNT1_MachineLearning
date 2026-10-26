import numpy as np
import pandas as pd

data = {
    "Độ tuổi": [25, 40, 35, 27, 31, 36, 48, 26, 33, 29, 38, 44, 42, 28, 30],
    "Hôn nhân": ["Độc thân", "Đã kết hôn", "Từng ly hôn", "Đã kết hôn", "Độc thân", "Đã kết hôn", "Độc thân", "Đã kết hôn", "Từng ly hôn", "Độc thân", "Đã kết hôn", "Độc thân", "Đã kết hôn", "Độc thân", "Đã kết hôn"],
    "Sở hữu BĐS": ["Ở cùng bố mẹ", "Nhà sở hữu", "Nhà thuê", "Ở cùng bố mẹ", "Nhà thuê", "Nhà sở hữu", "Nhà thuê", "Nhà thuê", "Ở cùng bố mẹ", "Nhà thuê", "Nhà sở hữu", "Nhà sở hữu", "Nhà sở hữu", "Nhà thuê", "Ở cùng bố mẹ"],
    "Thu nhập": [7000000, 18000000, 12000000, 9000000, 6000000, 8000000, 7000000, 8000000, 5000000, 10000000, 15000000, 14000000, 10000000, 7000000, 6000000],
    "Rủi ro tín dụng": [0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
}
df = pd.DataFrame(data)

df["Độ tuổi"] = pd.cut (
    df["Độ tuổi"],
    bins=[0, 30, 40, 100],
    labels=["<=30", "30-40", ">40"]
)
df["Thu nhập"] = pd.cut (
    df["Thu nhập"],
    bins=[0, 8000000, 12000000, np.inf],
    labels=["<=8tr", "8tr-12tr", ">12tr"]
)

def entropy(target_col):
    values, counts = np.unique(target_col, return_counts=True)
    total = len(target_col)
    ent = 0.0
    for count in counts:
        p = count / total
        if p > 0:
            ent -= p * np.log2(p)
    return ent

def info_gain(df, col_name, target_name="Rủi ro tín dụng"):
    total_entropy = entropy(df[target_name])
    vals, counts = np.unique(df[col_name], return_counts=True)

    new_entropy = 0.0
    for i in range (len(vals)):
        new_df = df[df[col_name] == vals[i]]
        new_entropy += (counts[i]/len(df)) * entropy(new_df[target_name])
    return total_entropy - new_entropy

def id3(df, original_df, features, target_attribute_name="Rủi ro tín dụng", parent_node_class=None):
    # Điều kiện dừng 1: Nhóm này đã thuần nhất (100% cùng 1 nhãn)
    if len(np.unique(df[target_attribute_name])) <= 1:
        return np.unique(df[target_attribute_name])[0]
    
    # Điều kiện dừng 2: Nhóm này không có người nào (tập rỗng)
    elif len(df) == 0:
        values, counts = np.unique(original_df[target_attribute_name], return_counts=True)
        return values[np.argmax(counts)]
    
    # Điều kiện dừng 3: Đã dùng hết câu hỏi / thuộc tính
    elif len(features) == 0:
        return parent_node_class
    
    # Tiếp tục chia nhánh:
    else:
        # Lưu lại nhãn chiếm đa số của nút hiện tại (đề phòng nhánh con bị rỗng)
        values, counts = np.unique(df[target_attribute_name], return_counts=True)
        parent_node_class = values[np.argmax(counts)]
        
        # 1. Tính điểm Gain cho tất cả các cột còn lại
        gains = [info_gain(df, feature, target_attribute_name) for feature in features]
        
        # 2. Chọn cột có Gain cao nhất làm nút phân nhánh
        best_feature_index = np.argmax(gains)
        best_feature = features[best_feature_index]
        
        # 3. Tạo gốc cây với thuộc tính tốt nhất vừa tìm được
        tree = {best_feature: {}}
        
        # 4. Loại bỏ cột này khỏi danh sách câu hỏi vì đã dùng rồi
        remaining_features = [f for f in features if f != best_feature]
        
        # 5. Rẽ nhánh theo từng giá trị của cột đó và lặp lại quá trình
        for value in np.unique(df[best_feature]):
            sub_df = df[df[best_feature] == value]
            subtree = id3(sub_df, original_df, remaining_features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree
            
        return tree

def print_tree(tree, indent=""):
    # Trường hợp 1: Nếu nút này không phải là từ điển, tức nó là kết quả cuối cùng (nút lá: 0 hoặc 1)
    if not isinstance(tree, dict):
        print(f" -> Dự đoán (Rủi ro): {tree}")
        return  # return này PHẢI nằm THỤT VÀO trong if

    # Trường hợp 2: Duyệt qua các câu hỏi và các nhánh rẽ
    for attribute, branches in tree.items():
        for branch_val, subtree in branches.items():
            print(f"{indent}[{attribute} = {branch_val}]", end="")
            
            # Nếu nhánh con vẫn là một từ điển con -> xuống dòng và đệ quy in tiếp
            if isinstance(subtree, dict):
                print()
                print_tree(subtree, indent + "    ")
            # Nếu nhánh con là kết luận cuối cùng (0 hoặc 1)
            else:
                print(f" -> Dự đoán (Rủi ro): {subtree}")

# ==========================================
# PHẦN 8: CHẠY THUẬT TOÁN VÀ IN KẾT QUẢ
# ==========================================

# Lấy danh sách tên các thuộc tính dự đoán (bỏ cột cuối 'Rủi ro tín dụng')
features = list(df.columns[:-1])

# Chạy thuật toán ID3 để dựng cây
cay_quyet_dinh = id3(df, df, features)

# In cây ra màn hình
print("\n--- CÂY QUYẾT ĐỊNH XÂY DỰNG TỪ ID3 ---")
print_tree(cay_quyet_dinh)
