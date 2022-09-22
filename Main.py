from tkinter import *
from tkinter import ttk
import os
import time
import sqlite3
import random
import cv2
import pyautogui as pyt
import datetime
from sklearn import svm
from sklearn.metrics import accuracy_score
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
import tkinter
from tkinter import messagebox
a=Tk()
a.geometry("1366x768+0+0")
a.title("Attendance Management System using OpenCv and Machine Learning")
a.iconbitmap("abc.ico")
image1 = Image.open('bg1.jpg').resize((1366,768))
test = ImageTk.PhotoImage(image1)
label1 = tkinter.Label(image=test,border="0")
label1.image = test
label1.place(x=0, y=0)
lb=Label(a,text="Attendance Management System using OpenCv and Machine Learning",fg="#3794DF",bg="black",font=('Arial',30,'bold','underline')).place(x=23,y=5)
class Marquee(tkinter.Canvas):
    def __init__(self, parent, text, margin=0, borderwidth=5, relief='flat', fps=100):
        tkinter.Canvas.__init__(self, parent, borderwidth=borderwidth, relief=relief,bg="#3794DF")
        self.fps = fps 
        text = self.create_text(0, -1000, text=text, anchor="w", tags=("text",))
        (x0, y0, x1, y1) = self.bbox("text")
        width = (x1 - x0) + (2*margin) + (2*borderwidth)
        height = (y1 - y0) + (2*margin) + (2*borderwidth)
        self.configure(width=width, height=height)
        
        # start the animation
        self.animate()
    def animate(self):
        (x0, y0, x1, y1) = self.bbox("text")
        if x1 < 0 or y0 < 0:
            # everything is off the screen; reset the X
            # to be just past the right margin
            x0 = self.winfo_width()
            y0 = int(self.winfo_height()/2)
            self.coords("text", x0, y0)
        else:
            self.move("text", -1, 0)

        # do again in a few milliseconds
        self.after_id = self.after(int(1000/self.fps), self.animate)
path=Path("E:\Chrome\Dataset")
directory=path.glob("*")
labels={"Abhay":0 ,"Atul":1, "Deepak":2, "Ketan":3, "Prateek":4, "Pritam":5, "Sanjeev":6, "Swapan":7 }
img_data=[]
labelList=[]
for folder in directory:
    label=os.path.basename(folder)
    for images in folder.glob("*.jpg"):
        img=load_img(images,target_size=(128,128))
        im_array=img_to_array(img)
        img_data.append(im_array)
        labelList.append(labels[label])
img_data=np.array(img_data)
print(img_data)
labelList=np.array(labelList)
np.transpose(labelList)
for val in img_data:
    val/=255.0
print(img_data)
trainX,testX,trainY,testY=train_test_split(img_data,labelList,test_size=0.2,random_state=2)
datagen=ImageDataGenerator()
model=svm.SVC(kernel='poly',degree=2,C=1,gamma=0.01,probability=True)
epochs=10
train_validation_x=[]
train_validation_y=[]
for val in range(epochs):
    print("epoch: ", val)
    b=0
    for batchX,batchY  in datagen.flow(trainX,trainY,batch_size=104,seed=7):
        train_validation_x=[x for x in batchX]
        train_validation_y=[y for y in batchY]
        model.fit(batchX.reshape(len(batchX),-1),batchY)
        b+=1
        if b>=len(trainX)/104:     
                  break
    prediction=model.predict(np.array(train_validation_x).reshape(77,49152))
    ac=accuracy_score(train_validation_y,prediction)*100
    print("\t\tTraining Accuracy: ",ac,"%")
    train_validation_x.clear()
    train_validation_y.clear()
prediction=model.predict(testX.reshape(46,49152))
ac=accuracy_score(testY,prediction)*100
print("Testing Accuracy : ",ac,"%")
def predict():
    cap = cv2.VideoCapture(0)
    while True:
        val,frame = cap.read()
        cv2.imshow('webcam', frame)
        frame = cv2.resize(frame, (128,128), interpolation = cv2.INTER_AREA)/255.0
        # press escape to exit
        if (cv2.waitKey(30) == 27):
           break
    cap.release()
    cv2.destroyAllWindows()
    prediction=model.predict(frame.reshape(1,49152))
    a=['1','2','3','4','5','6','7','8','9','0']
    id_=int(random.choice(a)+random.choice(a)+random.choice(a)+random.choice(a))
    conn=sqlite3.connect("student.db")
    x=datetime.datetime.now()
    date=x.strftime("%D")
    time=x.strftime("%H")+":"+x.strftime("%M")
    #attendance(stud_id int,stud_name text,date text,time text,atten_status text)
    #{"Abhay":0 ,"Atul":1, "Deepak":2, "Ketan":3, "Prateek":4, "Pritam":5, "Sanjeev":6, "Swapan":7 }
    if prediction[0]==0:
        conn.execute("insert into attendance values({},'Abhey','{}','{}','P')".format(id_,date,time))
        conn.commit()
    elif prediction[0]==1:
        conn.execute("insert into attendance values({},'Atul','{}','{}','P')".format(id_,date,time))
        conn.commit()
    elif prediction[0]==2:
        conn.execute("insert into attendance values({},'Deepak','{}','{}','P')".format(id_,date,time))
        conn.commit()
    elif prediction[0]==3:
        conn.execute("insert into attendance values({},'Ketan','{}','{}','P')".format(id_,date,time))
        conn.commit()
    elif prediction[0]==4:
        conn.execute("insert into attendance values({},'Prateek','{}','{}','P')".format(id_,date,time))
        conn.commit()
    elif prediction[0]==5:
        conn.execute("insert into attendance values({},'Pritam','{}','{}','P')".format(id_,date,time))
        conn.commit()
    elif prediction[0]==6:
        conn.execute("insert into attendance values({},'Sanjeev','{}','{}','P')".format(id_,date,time))
        conn.commit()
    elif prediction[0]==7:
        conn.execute("insert into attendance values({},'Swapan','{}','{}','P')".format(id_,date,time))
        conn.commit()
    else:
        messagebox.showinfo("Attendance Status","Try Again")
    conn.close()
    messagebox.showinfo("Attendance Status","Your Attendance Marked Successfully")
    print(prediction)
def train():
    pass
def exit():
    os._exit(1)
"""
def att():
    os.system('python attendance.py')
    name=int(round(time.time()*1000))
    a.deiconify()
"""
def delay1():
    import attendance
    attendance.main()
def student_list():
    messagebox.showinfo("Students List"," 1.) Abey Singh\n 2.) Ketan Tiwari\n 3.) Deepak Bora\n 4.) Pritam Kumar\n 5.) Swapan Chetri\n 6.) Atual kumar Jha\n 7.) Prateek Kumar Verma\n 8.) Sanjeev Kumar Prajapati")
marquee = Marquee(a, text="Attendance Management System using OpenCv and Machine Learning\t\t\t Attendance Management System using OpenCv and Machine Learning\t\t\t Attendance Management System using OpenCv and Machine Learning \t\t\t Attendance Management System using OpenCv and Machine Learning \t\t\t Attendance Management System using OpenCv and Machine Learning",borderwidth=1,relief="sunken")
marquee.pack(side="top", fill="x", pady=60)
image1 = Image.open('tra1.jpg').resize((200,260))
test = ImageTk.PhotoImage(image1)
label1 = tkinter.Label(image=test,border="10")
label1.image = test
label1.place(x=25, y=100)
tra_b1 = Button(a,text="Train Your Model",height=3,width=20,bg="#3794DF",bd=7,fg="white",font=('Arial',10,'bold'),command=train).place(x=47,y=345)
image1 = Image.open('det1.jpg').resize((200,260))
test = ImageTk.PhotoImage(image1)
label1 = tkinter.Label(image=test,border="10")
label1.image = test
label1.place(x=310, y=250)
det1_b1 = Button(a,text="Mark Your Attendance",height=3,width=20,bg="#3794DF",bd=7,fg="white",font=('Arial',10,'bold'),command=predict).place(x=330,y=495)
image1 = Image.open('exit.jpg').resize((200,260))
test = ImageTk.PhotoImage(image1)
label1 = tkinter.Label(image=test,border="10",fg="blue")
label1.image = test
label1.place(x=600, y=350)
tra_b1 = Button(a,text="Exit",height=3,width=15,bg="#3794DF",bd=7,fg="white",font=('Arial',10,'bold'),command=exit).place(x=638,y=595)
image1 = Image.open('att.jpg').resize((200,260))
test = ImageTk.PhotoImage(image1)
label1 = tkinter.Label(image=test,border="10")
label1.image = test
label1.place(x=870, y=250)
det1_b1 = Button(a,text="Show Your Attendance",height=3,width=20,bg="#3794DF",bd=7,fg="white",font=('Arial',10,'bold'),command=delay1).place(x=890,y=495)
image1 = Image.open('std1.jpg').resize((200,260))
test = ImageTk.PhotoImage(image1)
label1 = tkinter.Label(image=test,border="10")
label1.image = test
label1.place(x=1140, y=100)
tra_b1 = Button(a,text="Students List",height=3,width=20,bg="#3794DF",bd=7,fg="white",font=('Arial',10,'bold'),command=student_list).place(x=1160,y=345)
a.mainloop()

