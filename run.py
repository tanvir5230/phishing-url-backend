from audioop import add
import time
from flask import Flask, jsonify, render_template, request
import pickle
from functionsForFeatures import *

app = Flask(__name__);

# @app.route("/", methods=['GET','POST'])
@app.route("/url", methods = ["GET"])   
def sendStatusToFrontEnd():
    address = request.args.get('name');
    have_IP_Addr = is_ip(address);
    url_length = url_length_feature(address);
    having_At_Symbol_feat = having_At_Symbol(address);
    prefix_Suffix_feat = prefix_Suffix(address);
    model = pickle.load(open('catboost.pkl','rb'))
    arr_of_features = [have_IP_Addr,	url_length,	1,	having_At_Symbol_feat,	prefix_Suffix_feat,	-1,	-1,	-1,	-1,	1,	-1,	0,	-1,	1,	0,	1,	-1,	-1,	-1,	-1,	1,	1,	1]
    prediction = model.predict([arr_of_features])[0]
    predictedText = ""
    if(prediction==0):
        predictedText = "The url is malicious."
    else:
        predictedText = "The url is not malicious."
    time.sleep(5)
    return jsonify({'prediction':predictedText})
if __name__ == "__main__":
    app.run(host='192.168.0.102', port = 3000, debug=True)

# address = "http://federmacedoadv.com.br/3f/aze/ab51e2e319e51502f416dbe46b773a5e/?cmd=_home&dispatch=11004d58f5b74f8dc1e7c2e8dd4105e811004d58f5b74f8dc1e7c2e8dd4105e8@phishing.website.html";
# print(prefix_Suffix(address))