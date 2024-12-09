import torch
from torch import nn
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import Label


# Định nghĩa lại kiến trúc của mô hình (SimpleCNN)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 5)  # Số lớp là 5 (Apple, Banana, Grape, Mango, Strawberry)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Tạo lại mô hình
model = SimpleCNN()
# Tải trạng thái của mô hình đã lưu với weights_only=True
model.load_state_dict(torch.load("fruit_classifier.pth", weights_only=True))
model.eval()  # Chuyển mô hình sang chế độ dự đoán



# Chuẩn bị các chuyển đổi dữ liệu (Data Transforms)
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Kích thước cần khớp với khi huấn luyện
    transforms.ToTensor()
])


# Hàm để dự đoán nhãn của hình ảnh
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    class_names = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry']
    return class_names[predicted.item()]


# Hàm mở và hiển thị hình ảnh, sau đó dự đoán
def open_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    try:
        # Dự đoán nhãn
        label = predict_image(file_path)

        # Hiển thị hình ảnh trong giao diện
        image = Image.open(file_path)
        image = image.resize((200, 200))  # Thay đổi kích thước hình ảnh để phù hợp với giao diện
        photo = ImageTk.PhotoImage(image)

        # Cập nhật hình ảnh trong giao diện
        image_label.config(image=photo)
        image_label.image = photo

        # Hiển thị kết quả dự đoán
        result_label.config(text=f"Nhãn dự đoán: {label}")
    except Exception as e:
        messagebox.showerror("Lỗi", str(e))


# Tạo giao diện người dùng với Tkinter
root = tk.Tk()
root.geometry("200x300")
root.title("Fruit Classifier")

# Nút để tải hình ảnh
upload_button = tk.Button(root, text="Tải hình ảnh", command=open_image)
upload_button.pack(pady=10)

# Nhãn để hiển thị hình ảnh
image_label = Label(root)
image_label.pack(pady=10)

# Nhãn để hiển thị kết quả dự đoán
result_label = Label(root, text="Nhãn dự đoán sẽ hiển thị ở đây")
result_label.pack(pady=10)

# Chạy giao diện Tkinter
root.mainloop()
