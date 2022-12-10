import mysql.connector
import MySQLdb
import pandas as pd
import pandas.io.sql
import csv
import numpy as np
import sys
sys.path.append('c:\\Users\\quach\\Desktop\\Personal\\FPT University\\SEMESTER 9\\Arranging-rooms-based-on-student-information')
from Recomender.Rec_main import RS



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
    def get_students(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM Student")
        myresult = cursor.fetchall()
        np_result = np.array(myresult)
        return np.delete(np_result,1,1)
            
    def user_room(self):
        pass
if __name__ == '__main__':
    DBer = dataBASE()
    DBer.get_students()
