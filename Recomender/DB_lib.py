import mysql.connector
import MySQLdb
import pandas as pd
import pandas.io.sql
import csv
import numpy as np
import sys
import Rec_main


class dataBASE():

    def __init__(self) -> None:
        
        self.connection = MySQLdb.connect(host='localhost',
                                                database='dormitory',
                                                user='root',
                                                password='security2KL')
        # if self.connection.is_connected():
        #      self.cursor = self.connection.cursor()
        #      self.cursor.execute("select database();")
        #      self.record = self.cursor.fetchone()
        #      print("You're connected to database: ", self.record)

    def insert_Data(self, csv_path = 'C:\\Users\\quach\\Desktop\\Personal\\FPT University\\SEMESTER 9\\Dataset\\SORT_dATA.csv'):
        with open(csv_path, encoding= "utf-8") as file_obj:
            cursor = self.connection.cursor()
            query_dl = "DELETE FROM dormitory.student"
            cursor.execute(query_dl)
            # Skips the heading
            # Using next() method
            heading = next(file_obj)
            
            # Create reader object by passing the file 
            # object to reader method
            reader_obj = csv.reader(file_obj)
            
            # Iterate over each row in the csv file 
            # using reader object
            query_id = "SELECT coalesce(max(Student.id), 0) from Student"
            i = 1
            id_student = 0
            for row in reader_obj:
                cursor.execute(query_id)
                result = cursor.fetchall()
                for k in result:
                    id_student = k[0]
                id_student +=1
                gender = 0
                if row[2].lower() == "nam":
                    gender = 1
                row = (row[0], row[1], gender, row[3], row[4], row[5],row[6],row[7],row[8],row[9],row[10])

                sql = "INSERT INTO Student (id, name, gender, hometown, Bio_personality, food_drink, hob_inter, smoking, refer_roommate, Cleanliess, Privacy) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                cursor.execute(sql, row)
                self.connection.commit()
                print(i)
                i = i+1
            last_id = str(cursor.lastrowid)
            cursor.close()
    def get_students(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM Student")
        myresult = cursor.fetchall()
        np_result = np.array(myresult)
        col = ['','Name','Sex','Hometown','Bio_personality','food_drink','hob_inter','smoking','refer_roommate','Cleanliess','Privacy']
        res = pd.DataFrame(columns= col,data= np_result)
        return res
            
    def user_room(self,keep_data = False, start_room = 1,split_gender = False, room_size = 3):
        cursor = self.connection.cursor()
        if keep_data is False:
            cursor.execute("TRUNCATE dormitory.student_room;")
        # cursor = self.connection.cursor()
        # query_dl = "DELETE FROM dormitory.student"
        # cursor.execute(query_dl)
        df_student = self.get_students()
        AR_room = Rec_main.RS(df_student)
        res = AR_room.arrange_ROOM(room_size = room_size, start_room = start_room, split_gender = split_gender)
        # Insert Dataframe into SQL Server:
        for index, row in res.iterrows():
            cursor.execute("INSERT INTO student_room (id,room) values(%s, %s)", (row.id, row.room))
        self.connection.commit()
        cursor.close()

if __name__ == '__main__':
    DBer = dataBASE()
    DBer.user_room(keep_data = False,start_room = 6)
    print("FINISH")