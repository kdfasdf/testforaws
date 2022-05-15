from flask import Flask, request, jsonify, redirect

app = Flask(__name__)
@app.route("/")
def hello():
    return "Hello server!"

@app.route("/test")
def test():
    return "test"

@app.route("/Person",methods=['GET','POST'])
def Person():
    req = request.get_json()
    
    Person_Number = req["action"]["detailParams"]["Person_Number"]["value"]	# json파일 읽기

    answer = Person_Number
    
    # 답변 텍스트 설정
    res = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": answer
                    }
                }
            ]
        }
    }

    # 답변 전송
    return jsonify(res)

if __name__=="__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True,debug=True)