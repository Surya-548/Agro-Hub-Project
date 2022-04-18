from flask import Flask, g, session, redirect, request, render_template, url_for
import pymongo
import os
from bson.objectid import ObjectId
import string
import random

app = Flask(__name__)

app.secret_key = os.urandom(24)

client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['hackathon1']
users = db['users']
answers = db['answers']
questions = db['questions']
contributions = db['contributions']
admin2 = db['admin']


################ globals

username_taken_msg = ""
invalid_user = ""
contribution_successful = None

#######################


@app.before_request
def before_request():
    g.user = None 
    if 'user' in session:
        g.user = session['user']
    
    g.admin = None
    if 'admin' in session:
        g.admin = session['admin']

@app.route('/',methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        if g.user:
            # print("Current User: ",session['user'])
            return render_template('index.html',username=session['user'])
        return redirect(url_for('login'))

@app.route('/login',methods=['GET', 'POST'])
def login():
    global invalid_user
    if request.method == 'POST':
        session.pop('user',None)
        user_list = users.find_one({"username":request.form['username']})
        if user_list:
            if request.form['password'] == user_list['password']:
                print("if password...",user_list)
                session['user'] = request.form['username']
                print("logged in")
                return render_template('index.html',username=session['user'])
            return render_template('login.html',invalid_user="Invalid Username or Password")
        return render_template('login.html',invalid_user="Invalid Username or Password")
    return render_template('login.html')
        

@app.route('/logout')
def logout():
    session.pop('user', None)
    return render_template('login.html')

@app.route('/signup',methods=['GET', 'POST'])
def signup():
    global username_taken_msg
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        user_list = users.find_one({"username":username})
        if user_list:
            username_taken_msg = "Username already taken, try another one"
            return render_template('signup.html',username_taken_msg=username_taken_msg)
        users.insert_one({"username":username,"password":password})
        return render_template('index.html',username=username)
    return render_template('signup.html')


@app.route('/models')
def models():
    if g.user:
        return render_template('models.html',username =session['user'])
    return redirect(url_for('login'))
@app.route('/forum')
def forum():
    if g.user:
        return render_template('forum.html',username =session['user'])
    return redirect(url_for('login'))


####################### crop crop_recommendation #########################

import numpy as np
from keras.models import load_model 

cr_model = load_model('models/Crop_Recommendation.h5')

Crops = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
       'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
       'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
       'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']

sc_mean = np.array([ 50.39261364,  52.88238636,  47.23579545,  25.58610532,
        71.20236431,   6.46069391, 104.99553826])
sc_std = np.array([36.6511325 , 32.50093228, 49.19601994,  5.08611542, 22.50626072,
        0.78173883, 55.66283749])
        


def scale(x):
    sc = (x-sc_mean)/sc_std
    sc = np.array(sc)
    return sc


@app.route('/cr_page')
def cr_page():
    if g.user:
        return render_template('models/crop_recommendation/crop_recommend_index.html',username =session['user'])
    return redirect(url_for('login'))



@app.route('/cr_predict', methods=['POST'])
def cr_predict():
    d1 = dict()
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    T = float(request.form['T'])
    H = float(request.form['H'])
    PH = float(request.form['PH'])
    R = float(request.form['R']) 
	
    arr = np.array([[N,P,K,T,H,PH,R]])
    arr = scale(arr)
    ind = cr_model.predict(arr)
    for i,j in zip(Crops,ind[0]):
        d1.update({i:j})
    k = sorted(d1.items(), key = lambda kv:(kv[1], kv[0]))
    k1 = k[-3:]
    pred = k1[::-1]
    llist = []
    for i in pred:
        llist.append(i[0])
     
    return render_template('models/crop_recommendation/crop_recommend_result.html', result=llist,username =session['user'])


####################### crop crop_recommendation ends #########################

# -----------------------------------------------------------------------------

###################### cotton leaf disease detection ##############################
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array

cotton_leaf_model = load_model('models/cotton_leaf_disease_detection_model.h5')

def cotton_predict(filepath):
  img = plt.imread(filepath)
  temp_img = img
  img = cv2.resize(img,(150,150))
  img = img_to_array(img)/255
  img = np.expand_dims(img,axis=0)
  prediction = cotton_leaf_model.predict(img) >= 0.5
  if prediction==1:
    prediction = "Fresh Cotton Leaf"
    print("Prediction: "+prediction)
    return 1
  else:
    prediction = "Diseased Cotton Leaf"
    print("Prediction: "+prediction)
    return 0


@app.route('/ct_page')
def ct_page():
    if g.user:
        return render_template('models/cotton_leaf_disease_detection/cotton_leaf_disease_detection.html',username =session['user'])
    return redirect(url_for('login'))



@app.route('/ct_predict', methods=['POST'])
def ct_predict():
    img = request.files['image']
    img_name = img.filename
    
    path = os.path.join('static/cotton_images/',img_name)
    img.save(path)    
    
    prediction = cotton_predict(path)
    
    if prediction == 0:
        pred = "Prediction : Diseased Cotton Leaf"
    else :
        pred = "Prediction : Fresh Cotton Leaf"
    
    return render_template("models/cotton_leaf_disease_detection/cotton_leaf_disease_detection.html", img_src = path, result = pred,username =session['user'])
  


###################### cotton leaf disease detection ends #########################

#-------------------------------------------------------------

##################### ask query ##############################################

@app.route('/ask_query',methods=['GET', 'POST'])
def ask_query():
    if g.user:
        if request.method == 'POST':
            
            username = session['user']
            msg = request.form['msg']
            questions.insert_one({"username":username, "message":msg})

            
        return render_template("askQuery.html",username=session['user'])
    return render_template("login.html")

##################### ask query ends ##############################################

#---------------------------------------------------------------------------------

######################### Answer query ###########################################

@app.route('/answer_query')
def answer_query():
    if g.user:
        questions_list = questions.find()
        query_list=[]
        print(questions_list)
        for item in questions_list:
            # print(item)
            query_list.append([item['_id'],item['username'],item['message']])
        return render_template("answerQuery.html",queries=query_list,username =session['user'])
    return redirect(url_for('login'))

######################### Answer query ends ######################################


######################### View query ############################################

@app.route('/view_query/<qid>',methods=['GET','POST'])
def view_query(qid):
    if g.user:
        if request.method == 'GET':
            question = questions.find_one({"_id":ObjectId(qid)})
            answer_list=[]
            for item in answers.find({"question_id":qid}):
                answer_list.append([item["username"],item["answer"]])
            print(answer_list)
            asked_by = questions.find_one({"_id":ObjectId(qid)})
            # return redirect(url_for('view_query',question=question['message'],qid=qid,answer_list=answer_list,asked_by=asked_by["username"]))
            return render_template('view_query.html',question=question['message'],qid=qid,answer_list=answer_list,asked_by=asked_by["username"],username =session['user'])
        else:
            print("in post-----------------------------")
            username = session['user']
            answer = request.form['answer']
            question_id = qid  
            user_id = users.find_one({"username":username})
            print(answer,username,question_id,user_id)
            answers.insert_one({"answered_by":user_id,"answer":answer,"question_id":question_id,"username":username})
            return redirect(url_for('view_query',qid=qid))    
    return redirect(url_for('login'))

######################### View query ends ############################################


######################### Contribute ###############################

def get_name():
    S = 6  # number of characters in the string.  
    ran = ''.join(random.choices(string.ascii_lowercase + string.digits, k = S))    
    return str(ran)+".docx"

@app.route('/contribute', methods=['GET','POST'])
def contribute():
    if g.user:
        global contribution_successful
        if request.method == 'GET':
            return render_template('contribute.html',username =session['user'])
        else:
            name = request.form['name']
            mail = request.form['mail']
            code_link = request.form['code_link']
            problem_statement = request.form['problem_statement']
            contributions.insert_one({"username":session['user'],"name":name,"mail":mail,"code_link":code_link,"problem_statement":problem_statement})
            return render_template('contribute.html',contribution_successful = "Successfully Submitted, Thanks you.",username =session['user'])
    return redirect(url_for('login'))

######################### Contribute ends ###############################3

@app.route('/admin_login',methods=['GET', 'POST'])
def admin_login():
    if request.method == 'GET':
        return render_template('admin_login.html')
    else:
        admin_id = request.form['admin_id']
        password = request.form['password']
        admin_list = admin2.find_one({"admin_id": admin_id})
        if admin_list:
            if password == admin_list["password"]:
                session['admin'] = admin_id
                return redirect(url_for('admin'))
        else:
            return render_template('admin_login.html',admin_msg="Invalid Admin Id or Password")
        
    
@app.route('/admin')
def admin():
    if g.admin:
        contributions_list = []
        for i in contributions.find():
            contributions_list.append([i["username"],i["name"],i["mail"],i["code_link"],i["problem_statement"]])
        return render_template('admin.html',contributions_list=contributions_list)
    return redirect(url_for('admin_login'))
    

if __name__ == '__main__':
    app.run(host='0.0.0.0')