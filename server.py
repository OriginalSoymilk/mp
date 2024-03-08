# app/server.py

from flask import Flask, jsonify, request
import pickle
import re
import json
import numpy as np
import pandas as pd

app = Flask(__name__)

# 定义姿势的关键点
landmarks=['class']
for val in range(1, 33+1):
    landmarks+=['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

# 从 Google Drive 下载模型
# 请确保你已经按照之前的说明将 export_file_url 替换为正确的链接
export_file_url = 'https://www.googleapis.com/drive/v3/files/16QIrPY3FNjri4tZZeH18cM0YIt1kMusz?alt=media&key=YOUR_API_KEY'
# 使用你的下载模型的代码

# 加载模型
with open('pushup.pkl','rb') as f:
    model= pickle.load(f)

class PoseLandmarkType:
    nose = "nose"
    leftEyeInner = "leftEyeInner"
    leftEye = "leftEye"
    leftEyeOuter = "leftEyeOuter"
    rightEyeInner = "rightEyeInner"
    rightEye = "rightEye"
    rightEyeOuter = "rightEyeOuter"
    leftEar = "leftEar"
    rightEar = "rightEar"
    leftMouth = "leftMouth"
    rightMouth = "rightMouth"
    leftShoulder = "leftShoulder"
    rightShoulder = "rightShoulder"
    leftElbow = "leftElbow"
    rightElbow = "rightElbow"
    leftWrist = "leftWrist"
    rightWrist = "rightWrist"
    leftPinky = "leftPinky"
    rightPinky = "rightPinky"
    leftIndex = "leftIndex"
    rightIndex = "rightIndex"
    leftThumb = "leftThumb"
    rightThumb = "rightThumb"
    leftHip = "leftHip"
    rightHip = "rightHip"
    leftKnee = "leftKnee"
    rightKnee = "rightKnee"
    leftAnkle = "leftAnkle"
    rightAnkle = "rightAnkle"
    leftHeel = "leftHeel"
    rightHeel = "rightHeel"
    leftFootIndex = "leftFootIndex"
    rightFootIndex = "rightFootIndex"
    pass  # 这里省略了关键点类型的定义，请根据你的实际需要进行定义

# 处理 JSON 数据并预测结果
@app.route('/predict', methods=['POST'])
def predict():
    # 从 POST 请求中获取 JSON 数据
    data = request.json
    poses = data['jsonPoses']
    raw_string = json.dumps(poses)

    # 使用正则表达式进行匹配和替换
    pattern = r'(?<=[ {,])(x|y|z|v): (?=-?\d)'
    formatted_string = re.sub(pattern, r'"\1": ', raw_string)

    # 将字典的键名修正为字符串形式
    formatted_string = re.sub(r'([A-Za-z_][A-Za-z_0-9.]*)(?=:)', r'"\1"', formatted_string)

    # 将字符串解析为字典列表
    poses = json.loads(formatted_string)

    # 将 poses 转换为标准格式
    formatted_poses = []

    for pose in poses:
        formatted_pose = {}
        for landmark, data in pose.items():
            formatted_pose[str(landmark)] = {
                "x": data['x'],
                "y": data['y'],
                "z": data['z'],
                "visibility": data['v']
            }
        formatted_poses.append(formatted_pose)

    # 将 formatted_poses 转换为 JSON 格式
    json_data = json.dumps(formatted_poses, indent=4)

    # 提取 x、y、z、visibility 的值
    rows = []
    for pose in formatted_poses:
        for landmark, values in pose.items():
            x = values["x"]
            y = values["y"]
            z = values["z"]
            visibility = values["visibility"]
            rows.append([x, y, z, visibility])

    # 将数据转换为 numpy 数组，并展开为一维列表
    row = np.array(rows).flatten().tolist()
    X=pd.DataFrame([row], columns=landmarks[1:])
    body_language_class=model.predict(X)[0]
    body_language_prob=model.predict_proba(X)[0]
    
    # 返回预测结果
    return jsonify({'body_language_class': body_language_class, 'body_language_prob': body_language_prob})

if __name__ == '__main__':
    app.run(debug=True)
