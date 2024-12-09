import os
import numpy as np
from glob import glob
from tqdm import tqdm

# Định nghĩa tên các lớp
class_names = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry']

# Sử dụng đường dẫn tuyệt đối
data_dir = r"C:\PyCharm\Ktra_05_11_24\Fruits Classification"  # Thay đổi thành đường dẫn tuyệt đối trên máy bạn
train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "valid")
test_dir = os.path.join(data_dir, "test")

# Kiểm tra sự tồn tại của các thư mục
for directory in [train_dir, valid_dir, test_dir]:
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Không tìm thấy thư mục: {directory}")

# Phần còn lại của mã không thay đổi
# Thu thập tất cả các đường dẫn hình ảnh cho mỗi lớp
all_class_paths = [glob(os.path.join(train_dir, name, '*')) +
                   glob(os.path.join(valid_dir, name, '*')) +
                   glob(os.path.join(test_dir, name, '*'))
                   for name in class_names]

# Tính tổng số lượng hình ảnh
total_size = sum([len(paths) for paths in all_class_paths])

# In thông tin về kích thước dữ liệu
print("Tổng số hình ảnh:", total_size)
print("Kích thước của mỗi lớp:", [len(paths) for paths in all_class_paths])

# Xáo trộn các đường dẫn hình ảnh
for paths in all_class_paths:
    np.random.shuffle(paths)

# Thông báo rằng dữ liệu đã được xáo trộn và sẵn sàng
print("Dữ liệu đã được xáo trộn và sẵn sàng để huấn luyện!")

# Ghi chú: Không di chuyển hoặc xóa bất kỳ tệp nào, vì dữ liệu đã được phân chia theo cấu trúc mong muốn.
print("ALL DONE!!")
