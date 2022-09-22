from tkinter import *
import tkinter
import sqlite3
import random
import datetime
import os
from tkinter import messagebox
import pyautogui as pyt
from PIL import ImageTk, Image
def main():
    a=Tk()
    a.configure(bg="white")
    """
    a.maxsize(1366,768)
    a.minsize(1366,768)
    """
    a.geometry("1366x768+0+0")
    a.title("Attendance Panel")
    a.iconbitmap("abc.ico")
    conn=sqlite3.connect("student.db")
    b=conn.execute("select * from attendance")
    id_=[]
    name=[]
    date=[]
    time=[]
    status=[]
    for i in b:
        id_.append(i[0])
        name.append(i[1])
        date.append(i[2])
        time.append(i[3])
        status.append(i[4])
    #print(id_,name,date,time,status)
    def delete_data():
        a=pyt.prompt("Enter Your Student Id ")
        if int(a) in id_:
            con=sqlite3.connect("student.db")
            con.execute("delete from attendance where stud_id={}".format(int(a)))
            con.commit()
            con.close()
            messagebox.showwarning("Successfull","Your Student ID {} successfully delete from database !!".format(int(a)))
        else:
            messagebox.showwarning("Warning","No Data Found!!")
    def update_data():
        ab=pyt.prompt("Enter Your Student ID ")
        if(int(ab) in id_):
            aa=pyt.prompt("What you want to update ['Name','Date','Time','Status']")
            b=aa.lower()
            if b=="name":
                bb=pyt.prompt("Enter Student Name")
                con=sqlite3.connect("student.db")
                con.execute("update attendance set stud_name='{}' where stud_id={}".format(bb,int(ab)))
                con.commit()
                con.close()
                messagebox.showwarning("Successfull","Name Updated Successfully!!")
            elif b=="date":
                bb=pyt.prompt("Enter Attendance Date")
                con=sqlite3.connect("student.db")
                con.execute("update attendance set date='{}' where stud_id={}".format(bb,int(ab)))
                con.commit()
                con.close()
                messagebox.showwarning("Successfull","Date Updated Successfully!!")
            elif b=="time":
                bb=pyt.prompt("Enter Attendace Time")
                con=sqlite3.connect("student.db")
                con.execute("update attendance set time='{}' where stud_id={}".format(bb,int(ab)))
                con.commit()
                con.close()
                messagebox.showwarning("Successfull","Time Updated Successfully!!")
            
            elif b=="status":
                bb=pyt.prompt("Enter Your Attendance Status")
                conn=sqlite3.connect("student.db")
                conn.execute("update attendance set  atten_status='{}' where stud_id={}".format(bb,int(ab)))
                conn.commit()
                conn.close()
                messagebox.showwarning("Successfull","Status Updated Successfully!!")
            else:
                messagebox.showwarning("Warning","Sorry something went wrong")
        else:
            messagebox.showwarning("Warning","Student Id is invalid!!")
    def add_new():
        con=sqlite3.connect("student.db")
        a=['1','2','3','4','5','6','7','8','9','0']
        idd=int(random.choice(a)+random.choice(a)+random.choice(a)+random.choice(a))
        x=datetime.datetime.now()
        b=x.strftime("%D")
        c=x.strftime("%H")+":"+x.strftime("%M")
        n=pyt.prompt("Enter Student Name")
        st=pyt.prompt("Enter Student Status(P/A)")
        con.execute("insert into attendance values({},'{}','{}','{}','{}')".format(idd,n,b,c,st))
        con.commit()
        con.close()
        messagebox.showwarning("Successfull","Data Added Successfully!!")
    serial_no=1
    y_value=110
    lb1=Label(a,text="Attendance Panel",font=("Arial",30,"bold","underline"),fg="black",bg="white").pack()
    lb2=Label(a,text="Student ID",font=("Arial",13,"bold"),fg="black",bg="white").place(x=100,y=70)
    lb3=Label(a,text="Student Name",font=("Arial",13,"bold"),fg="black",bg="white").place(x=300,y=70)
    lb4=Label(a,text="Date",font=("Arial",13,"bold"),fg="black",bg="white").place(x=560,y=70)
    lb5=Label(a,text="Time",font=("Arial",13,"bold"),fg="black",bg="white").place(x=750,y=70)
    lb6=Label(a,text="Attendance Status",font=("Arial",13,"bold"),fg="black",bg="white").place(x=900,y=70)
    for i in range(len(name)):
        lb2=Label(a,text="{}.)\t         {}".format(serial_no,id_[i]),font=("Arial",10),fg="black",bg="white").place(x=30,y=y_value)
        lb3=Label(a,text="         {}".format(name[i]),font=("Arial",10),fg="black",bg="white").place(x=300,y=y_value)
        lb4=Label(a,text="{}".format(date[i]),font=("Arial",10),fg="black",bg="white").place(x=550,y=y_value)
        lb5=Label(a,text=" {}".format(time[i]),font=("Arial",10),fg="black",bg="white").place(x=750,y=y_value)
        lb6=Label(a,text="  {}".format(status[i]),font=("Arial",10),fg="black",bg="white").place(x=950,y=y_value)
        btn3=Button(a,text="Delete",bd=5,width=7,bg="red",fg="white",font=("Arial",8,"bold"),command=delete_data).place(x=1100,y=y_value-5)
        btn3=Button(a,text="Add",bd=5,width=7,bg="green",fg="white",font=("Arial",8,"bold"),command=add_new).place(x=1200,y=y_value-5)
        btn3=Button(a,text="Update",bd=5,width=7,bg="#FFA012",fg="white",font=("Arial",8,"bold"),command=update_data).place(x=1300,y=y_value-5)
        serial_no+=1
        y_value+=50
    conn.close()

