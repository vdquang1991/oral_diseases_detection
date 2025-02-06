import os
from flask import Flask, redirect, render_template, request
import cv2
from PIL import Image
import uuid
from ultralytics import YOLO
# import numpy as np
# from model import build_model, resnet
import pandas as pd
import google.generativeai as genai
import markdown

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Choose GPU for training

disease_info = pd.read_csv('disease_info.csv')
supplement_info = pd.read_csv('supplement_info.csv')

# model = YOLO("best_v6.pt")
model = YOLO("best_v7.pt")
print('Load model weights completed!')

# genai.configure(api_key="AIzaSyB96-XbJiia-A_R9pRAn22oxjhWKlb_Bok")
genai.configure(api_key="AIzaSyAiN7eeOODib-VdCi1L2feJ-gFPlljHH_Y")

gemini_model = genai.GenerativeModel("gemini-1.5-flash")

conf = 0.2

def prediction(image_path, conf=0.5):
    results = model(image_path, conf=conf)
    img = None
    file_name = str(uuid.uuid4()) + '.jpg'
    file_path = os.path.join('static/uploads', file_name)
    if os.path.exists(file_path):
       os.remove(file_path)

    count_0 = 0
    count_1 = 0
    count_2 = 0
    for result in results:
        result.save(filename=file_path)
        boxes = result.boxes  # Boxes object for bounding box outputs
        cls = boxes.cls.detach().cpu().numpy()
        for b in cls:
            if b==0:
                count_0+=1
            elif b==1:
                count_1+=1
            else:
                count_2 +=1

        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print("count Calculus: ", count_0)
    print("count Caries: ", count_1)
    print("count Healthy: ", count_2)
    return img, results, count_0, count_1, count_2, file_path


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        # filename = image.filename
        filename = str(uuid.uuid4()) + '.jpg'
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)

        img, results, count_0, count_1, count_2, outpath = prediction(file_path, conf=conf)
        str_result = "Kết quả dự đoán"
        if count_0 == 0 and count_1 == 0:
            pred = 2
        else:
            if count_0 > count_1:
                pred = 0
            else:
                pred = 1

        title = str_result
        # description =disease_info['description'][pred]
        # prevent = disease_info['Possible Steps'][pred]

        img = Image.open(file_path)
        prompt = "Mô tả hình ảnh răng miệng mà người dùng tải lên bằng tiếng việt. Lưu ý: Nếu hình ảnh không liên quan đến răng miệng, hãy lịch sự đề nghị người dùng tải ảnh chính xác và tuyệt đối không mô tả hay đưa ra kết luận gì cả."
        description = gemini_model.generate_content([img, prompt]).text
        description = markdown.markdown(description)

        if (count_0==0 and count_1==0 and count_2==0):
            prevent = "Có vẻ hình ảnh bạn vừa tải lên không phải hình ảnh về răng miệng hoặc nó có chất lượng ảnh quá thấp khiến chúng tôi gặp khó khăn trong việc nhận dạng bệnh. \n Xin vui lòng tải ảnh khác có chất lượng tốt hơn. \n Xin cảm ơn!!!"
        else:
            if count_0>0 and count_1>0:
                s = "Hãy tạo một câu chuyện sinh động bằng tiếng Việt về tình trạng răng miệng trong hình ảnh (gồm có" + str(count_0) + " vị trí cao răng và " + str(count_1) + " vị trí sâu răng.\n"
            else:
                s = "Hãy tạo một câu chuyện sinh động bằng tiếng Việt về tình trạng răng miệng trong hình ảnh"
                s += """
        I. Cốt Truyện và Nhân Vật:
1. Mở đầu thân thiện:
   - Tạo không khí thoải mái như trò chuyện với người thân
   - Giới thiệu bản thân là một nha sĩ đồng hành
   - Tạo sự gắn kết với bệnh nhân

2. Phát triển câu chuyện:
   - Ví von răng miệng như một thành phố nhỏ
   - Mỗi răng là một ngôi nhà
   - Nướu như những khu vườn xanh
   - Vi khuẩn như "những vị khách không mời"

II. Các Vấn Đề Cần Phát Hiện (qua câu chuyện):
1. "Những ngôi nhà" cần quan tâm:
   - Sâu răng ("những lỗ hổng đen tối")
   - Mục răng ("ngôi nhà đang mục nát")
   - Nứt, vỡ răng ("tường nhà bị nứt, đổ")
   - Mòn men răng ("lớp sơn bảo vệ bị mòn")
   - Đổi màu răng ("ngôi nhà đổi màu")
   - Cao răng ("rêu phong bám lâu ngày")
   - Mảng bám ("bụi bẩn đóng cặn")
   Lưu ý: Nếu có nhiều phát hiện trùng lặp hãy đặt tên theo số thứ tự (ví dụ: "Sâu răng 1", "Sâu răng 2")

2. "Khu vườn và không gian xung quanh" cần chăm sóc:
   - Viêm nướu ("khu vườn đỏ ửng, sưng tấy")
   - Tụt nướu ("đất bị xói mòn")
   - Chảy máu nướu ("những giọt sương hồng")
   - Túi nha chu ("những hố sâu bí ẩn")
   - Viêm quanh implant ("cây ghép đang gặp trở ngại")

3. "Quy hoạch khu phố" cần điều chỉnh:
   - Lệch khớp cắn ("tầng trệt và tầng trên không khớp")
   - Răng mọc lệch lạc ("nhà xây không đúng hàng")
   - Khoảng hở răng ("những khoảng trống cô đơn")
   - Khớp cắn sâu/hở ("mái nhà quá cao hoặc quá thấp")
   - Mất răng ("ngôi nhà đã chuyển đi")

III. Phân Tích Qua Câu Chuyện:
1. Mức độ nghiêm trọng:
   - Nhẹ ("chỉ cần trang trí lại")
   - Vừa ("cần sửa chữa sớm")
   - Nặng ("cần xây dựng lại")

2. Mối liên hệ:
   - Giữa các vấn đề ("câu chuyện liên hoàn")
   - Nguyên nhân và hậu quả
   - Dự đoán tương lai

IV. Lời Khuyên và Kế Hoạch:
1. Giải pháp điều trị:
   - Phục hồi răng như "tân trang ngôi nhà"
   - Điều trị nha chu như "phục hồi khu vườn"
   - Chỉnh nha như "quy hoạch lại khu phố"
   - Implant như "xây nhà mới"

2. Hướng dẫn phòng ngừa:
   - Chải răng như "quét dọn nhà cửa"
   - Xỉa răng như "làm vườn hàng ngày"
   - Súc miệng như "tưới cây buổi sáng"
   - Khám định kỳ như "kiểm tra an toàn công trình"

3. Kết thúc hy vọng:
- Viễn cảnh tương lai tươi sáng
   - Lời động viên thân thiện
   - Hẹn gặp lại

V. Yêu Cầu Kỹ Thuật:
1. Ngôn ngữ:
   - Dễ hiểu, gần gũi
   - Giải thích thuật ngữ qua hình ảnh đời thường
   - Sử dụng ví dụ cụ thể

2. Độ chính xác:
   - Đảm bảo thông tin chuyên môn
   - Đánh dấu vị trí chính xác
   - Không bỏ sót chi tiết quan trọng
   - Phân tích mối liên hệ giữa các vấn đề

Chủ đề: "Hành trình của những nụ cười" - Câu chuyện về thành phố răng miệng của bạn.
Hãy làm đậm những từ liên quan đến bệnh, tổn thương hoặc phát hiện bất thường trong câu chuyện của bạn.
Lưu ý: Nếu người dùng tải ảnh không liên quan đến răng, miệng. Hãy lịch sư đề nghị người dùng tải ảnh chính xác và không đưa ra câu chuyện.
        """

            prevent = gemini_model.generate_content([img, s]).text

        prevent = markdown.markdown(prevent)

        print(prevent)

        image_url = outpath
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]

        # print(supplement_name)
        # print(supplement_image_url)
        # print(supplement_buy_link)

        return render_template('submit.html' , title = title , desc = description , prevent = prevent ,
                               input_image_url = file_path,
                               image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

