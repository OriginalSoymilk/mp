from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import pickle

app = Flask(__name__)

# 使用 Google Drive API 进行身份验证
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # 使用本地 Web 服务器进行身份验证
drive = GoogleDrive(gauth)

# 下载模型文件
file_id = '16QIrPY3FNjri4tZZeH18cM0YIt1kMusz'  # 文件 ID
model_file = drive.CreateFile({'id': file_id})
model_file.GetContentFile('pushup.pkl')  # 将模型文件下载到本地

# 加载模型
with open('pushup.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/classify', methods=['POST'])
def classify_pose():
    # 接收 POST 请求中的 JSON 数据
    data = request.get_json()
    poses = data['poses']
    
    # 将坐标转换为特征向量
    rows = []
    for pose in poses:
        row = []
        for landmark in pose:
            row.append(landmark['x'])
            row.append(landmark['y'])
            row.append(landmark['z'])
            row.append(landmark['visibility'])
        rows.append(row)
    X = pd.DataFrame(rows, columns=['x', 'y', 'z', 'visibility'])

    # 使用模型进行预测
    body_language_class = model.predict(X)[0]
    body_language_prob = model.predict_proba(X)[0]

    # 返回预测结果
    return jsonify({'class': body_language_class, 'prob': body_language_prob.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
