from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['POST'])
def add_numbers():
    data = request.get_json()
    num1 = data['num1']
    num2 = data['num2']
    result = num1 + num2
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
