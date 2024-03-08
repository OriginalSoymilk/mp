from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import requests
import pickle

app = Flask(__name__)

# 下载模型文件
github_url = 'https://raw.githubusercontent.com/OriginalSoymilk/mp/main/pushup.pkl'
response = requests.get(github_url)
with open('pushup.pkl', 'wb') as f:
    f.write(response.content)

# 加载模型
with open('pushup.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/classify', methods=['POST'])
def classify_pose():
    # 接收 POST 请求中的 JSON 数据
    data = request.get_json()
    poses = data['jsonPoses']
    
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
