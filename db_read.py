import sqlite3
import pandas as pd

con = sqlite3.connect("OpenSSL_dataset.db")
sql_query = """SELECT name FROM sqlite_master  
  WHERE type='table';"""
cursor = con.cursor()
cursor.execute(sql_query)
list_of_tables = cursor.fetchall()
print(list_of_tables)