from flask import Flask, render_template,request, Response
import pickle
import os, string, re
import glob
from pyvi import ViTokenizer, ViPosTagger
import pandas as pd
import numpy as np

app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = "uploads"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Load các mô hình và các hàm tiền xử lý đặc trưng
svm_model = "pickle/svm_classifier.pkl"  
with open(svm_model, 'rb') as f:
    classifier = pickle.load(f)

tuoi_scalered_path = "pickle/scaler_tuoi.pkl"  
with open(tuoi_scalered_path, 'rb') as f:
    scaler_tuoi = pickle.load(f)
    
kinh_nghiem_scalered_path = "pickle/scaler_kinhnghiem.pkl"  
with open(kinh_nghiem_scalered_path, 'rb') as f:
    scaler_kinhnghiem = pickle.load(f)
      
vectorizer_bangcap_path = "pickle/vectorizer_bangcap.pkl"  
with open(vectorizer_bangcap_path, 'rb') as f:
    vectorizer_bangcap = pickle.load(f)

vectorizer_chuyennganh_path = "pickle/vectorizer_chuyennganh.pkl"  
with open(vectorizer_chuyennganh_path, 'rb') as f:
    vectorizer_chuyennganh = pickle.load(f)
    
vectorizer_ngoaingu_path = "pickle/vectorizer_ngoaingu.pkl"  
with open(vectorizer_ngoaingu_path, 'rb') as f:
    vectorizer_ngoaingu = pickle.load(f)

vectorizer_tinhoc_path = "pickle/vectorizer_tinhoc.pkl"  
with open(vectorizer_tinhoc_path, 'rb') as f:
    vectorizer_tinhoc = pickle.load(f)
    
# Hàm tiền xử lý dữ liệu
def tien_xu_ly(text):
    text = text.replace(','," ").strip()
    text = re.sub('\s+', ' ', text)
    tokens = ViTokenizer.tokenize(text).lower()
    return tokens

# Trang home
@app.route("/")
def home():
    return render_template("homepage.html")

# Trang VIDEO DEMO
@app.route("/videodemo/")
def videodemo():
    return render_template("videodemo.html")

# Trang nhập Kết quả nghiên cứu
@app.route("/ketquanghiencuu/")
def ketquanghiencuu():
    return render_template("ketquanghiencuu.html")


# Trang nhập PHAN TICH DU LIEU
@app.route("/ungdung/")
def ungdung():
    return render_template("ungdung.html")

# Trang nhập PHAN TICH DU LIEU
@app.route("/phantichdulieu/")
def phantichdulieu():
    return render_template("phantichdulieu.html")

# Trang nhập file dự đoán
@app.route("/nhapfiledudoan/")
def nhapfiledudoan():
    return render_template("uploadfile.html")

classes = ['Bảo Vệ', 'Cơ Khí', 'Công Nhân', 'Công Nghệ Thông Tin', 'Giáo Viên', 'Kế Toán', 'Lái Xe', 'Nhân Viên Nhân Sự', 'Nhân Viên Kinh Doanh', 'Nhân Viên Kỹ Thuật', 'Nhân Viên Phiên Dịch', 'Nhân Viên Văn Phòng', 'Nhân Viên Y Tế', 'Nấu Ăn', 'Quản Lý', 'Xây Dựng', 'Nhân Viên Xuất Nhập Khẩu']


# Trang dự đoán trực tiếp
@app.route("/dudoan/", methods=['POST'])
def dudoan():
    try:
        tuoi = request.form['tuoi']
        tuoi = int(tuoi)
    except:
        notification1 = "Vui lòng nhập đúng thông tin định dạng tuổi!"
        return render_template("Error_input.html", data = [{"notification":notification1}])
    
    if tuoi == 0:
        tuoi = 18
    kinhnghiem = request.form['kinhnghiem']
    gioitinh = request.form['gioitinh']
    bangcap = request.form['bangcap']
    chuyennganh = request.form['chuyennganh']
    ngoaingu = request.form['ngoaingu']
    tinhoc = request.form['tinhoc']
    print(tuoi,kinhnghiem,gioitinh,bangcap,chuyennganh,ngoaingu,tinhoc)

    tuoi_vec  = scaler_tuoi.transform([[int(tuoi)]])
    kinh_nghiem_vec = scaler_kinhnghiem.transform([[int(kinhnghiem)]])
    gioitinh_vec = [[float(gioitinh)/2]]
    bangcap_vec = vectorizer_bangcap.transform([tien_xu_ly(bangcap)])
    ngoaingu_vec = vectorizer_ngoaingu.transform([tien_xu_ly(ngoaingu)])
    tinhoc_vec = vectorizer_tinhoc.transform([tien_xu_ly(tinhoc)])
    chuyenganh_vec = vectorizer_chuyennganh.transform([tien_xu_ly(chuyennganh)])

    df = pd.DataFrame(list(zip(tuoi_vec, gioitinh_vec,kinh_nghiem_vec,bangcap_vec.toarray(),chuyenganh_vec.toarray(),\
                           ngoaingu_vec.toarray(),tinhoc_vec.toarray())), \
                  columns =['Tuoi', 'GioiTinh','KinhNghiem','BangCap','ChuyenNganh','NgoaiNgu','TinHoc']) 
    a = np.concatenate((df['Tuoi'].tolist(), df['GioiTinh'].tolist(), df['KinhNghiem'].tolist(),\
                    df['BangCap'].tolist(), df['ChuyenNganh'].tolist(), df['NgoaiNgu'].tolist(), df['TinHoc'].tolist()), axis=1)

    X_predict = np.asarray(a)
    predicted = classifier.predict(X_predict)
    prob = classifier.predict_proba(X_predict)[0]

    label1 = classes[predicted[0]]
    prob[predicted[0]] = 0.0
    maxIndex  = np.where(prob == np.amax(prob))
    label2 = classes[maxIndex[0][0]]

    prob[maxIndex[0][0]] = 0.0
    maxIndex  = np.where(prob == np.amax(prob))
    label3 = classes[maxIndex[0][0]]

    prob[maxIndex[0][0]] = 0.0
    maxIndex  = np.where(prob == np.amax(prob))
    label4 = classes[maxIndex[0][0]]

    prob[maxIndex[0][0]] = 0.0
    maxIndex  = np.where(prob == np.amax(prob))
    label5 = classes[maxIndex[0][0]]

    return render_template("ketquadudoan.html", data =[{"label1":label1,"label2":label2,"label3":label3,"label4":label4,"label5":label5}])
    
   

# Hàm dự đoán kết quả file .csv
def predict_file_csv(data):
    try:
        data = data.fillna(value="Không có")
        tuoi = data['Tuổi'].tolist()
        tuoi  = [[x] for x in tuoi]
        tuoi_scalered = scaler_tuoi.transform(tuoi)

        gioi_tinh = data['Giới Tính'].tolist()
        gioi_tinh  = [[float(x)/2] for x in gioi_tinh]

        kinh_nghiem = data['Năm Kinh Nghiệm'].tolist()
        kinh_nghiem  = [[int(x)] for x in kinh_nghiem]
        kinh_nghiem_scalered = scaler_kinhnghiem.transform(kinh_nghiem)
        print(kinh_nghiem_scalered)

        bang_cap = data['Trình Độ'].tolist()
        bang_cap = [tien_xu_ly(s) for s in bang_cap]
        X_bangcap = vectorizer_bangcap.transform(bang_cap)

        chuyen_nganh = data['Ngành'].tolist()
        chuyen_nganh = [tien_xu_ly(s) for s in chuyen_nganh]
        X_chuyen_nganh = vectorizer_chuyennganh.transform(chuyen_nganh)

        ngoai_ngu = data['Ngoại Ngữ'].tolist()
        ngoai_ngu = [tien_xu_ly(s) for s in ngoai_ngu]
        X_ngoai_ngu = vectorizer_ngoaingu.transform(ngoai_ngu)

        tin_hoc = data['Tin Học'].tolist()
        tin_hoc = [tien_xu_ly(s) for s in tin_hoc]
        X_tin_hoc = vectorizer_tinhoc.transform(tin_hoc)

        df = pd.DataFrame(list(zip(tuoi_scalered, gioi_tinh,kinh_nghiem_scalered,X_bangcap.toarray(),X_chuyen_nganh.toarray(),\
                               X_ngoai_ngu.toarray(),X_tin_hoc.toarray())), \
                      columns =['Tuoi', 'GioiTinh','KinhNghiem','BangCap','ChuyenNganh','NgoaiNgu','TinHoc']) 
        a = np.concatenate((df['Tuoi'].tolist(), df['GioiTinh'].tolist(), df['KinhNghiem'].tolist(),\
                        df['BangCap'].tolist(), df['ChuyenNganh'].tolist(), df['NgoaiNgu'].tolist(), df['TinHoc'].tolist()), axis=1)
        X_predict = np.asarray(a)

        predicted = classifier.predict(X_predict)
        print("Predict: ",predicted)
        predict_label = [classes[index] for index in predicted]
        return predict_label
    except:
        return "Error"

# Hàm nhập file dự đoán
@app.route("/uploadFile", methods=['POST'])
def uploadFile():
    try:
        if request.method == 'POST':
            fileupload = request.files['file']
            try:
                data = pd.read_csv(fileupload) 
                predict_list = predict_file_csv(data)
                if predict_list != "Error":
                    data = data.fillna(value=" ")
                    data = data.drop(['Ngày Sinh', 'Nơi Làm Việc','Loại Công Việc'], 1)  
                    data["Dự đoán"]=predict_list
                    max_value = data.shape[0] + 1
                    stt = list(range(1, max_value))
                    print("STT: ", stt)
                    stt_df = pd.Series(stt,name="STT").astype(int)
                    data = pd.concat([pd.Series(stt_df).astype(int), data], axis=1)
                    data.style.set_properties(**{'text-align': 'center'})
                    data.to_csv('uploads/result.csv', index = False, header=True)
                    data['Giới Tính'] = data['Giới Tính'].map({2: 'Nam', 1: 'Nữ'})
                    return render_template('ketquadudoanfile.html',  tables=[data.to_html(classes='data',index=False)], titles=data.columns.values)
                else:
                    return render_template("Error.html")
            except:
                notification1 = "Vui lòng upload File đúng định dạng!"
                return render_template("Error.html", data = [{"notification":notification1}])
    except:
        notification1 = "Vui lòng lựa chọn File trước khi nộp lên!"
        return render_template("Error.html", data = [{"notification":notification1}])
# Hàm xuất file
@app.route("/getPlotCSV")
def getPlotCSV():
    with open("uploads/result.csv",'r',encoding='utf-8') as fp:
         csv = fp.read()
    return Response( csv, mimetype="text/csv", headers={"Content-disposition": "attachment; filename=result.csv"})    

if __name__ == "__main__":
    app.run(debug=True)
