import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('D:\LEARNING AI\BTL PYTHON\model.h5') 

# Class names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Function to open a file dialog and get the path of the selected file
def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        process_image(file_path)

# Function to preprocess the input image and make a prediction
def process_image(file_path):
    # Load original image
    original_img = Image.open(file_path)
    
    # Convert to grayscale
    img = original_img.convert('L')
    img = img.resize((28, 28))  # Thay đổi kích thước ảnh thành (28, 28)

    # Chuyển đổi sang numpy array
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Chuẩn hóa giá trị pixel về khoảng [0, 1]

    # Dự đoán lớp của ảnh
    predictions = model.predict(img_array)

    # Lấy lớp có xác suất cao nhất
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]

    # Hiển thị ảnh gốc
    original_img.thumbnail((500, 500))
    original_img_tk = ImageTk.PhotoImage(original_img)
    original_img_label.config(image=original_img_tk)
    original_img_label.image = original_img_tk

    # Hiển thị kết quả
    result_text = f"Predicted class: {predicted_class_index} - {predicted_class_name}"
    result_label.config(text=result_text, font=('Helvetica', 16))  # Điều chỉnh kích thước chữ

    # Hiển thị thông báo phù hợp với kết quả
    message_label.config(text=f"Bức ảnh của bạn có {predicted_class_name}", font=('Helvetica', 18))  # Điều chỉnh kích thước chữ

# Create the main window
window = tk.Tk()
window.title("Image Classification Demo")

# Set window size to full screen
window.geometry("{0}x{1}+0+0".format(window.winfo_screenwidth(), window.winfo_screenheight()))

# Create buttons and labels
open_button = tk.Button(window, text="Hãy tải ảnh của bạn vào đây", command=open_file, font=('Helvetica', 18))  # Điều chỉnh kích thước chữ
open_button.pack(pady=20)

original_img_label = tk.Label(window)
original_img_label.pack(pady=20)

result_label = tk.Label(window, text="", font=('Helvetica', 16))  # Điều chỉnh kích thước chữ
result_label.pack(pady=20)

message_label = tk.Label(window, text="", font=('Helvetica', 18))  # Điều chỉnh kích thước chữ
message_label.pack(pady=20)

# Run the main loop
window.mainloop()
