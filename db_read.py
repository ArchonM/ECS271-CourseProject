import sqlite3
import pandas as pd

db_path = "/Users/asmitaa/binarySimiliarityProject/OpenSSL_dataset.db"
try : 
    con = sqlite3.connect(db_path)
    sql_query = """SELECT name FROM sqlite_master  
    WHERE type='table';"""
    cursor = con.cursor()
    cursor.execute(sql_query)
    list_of_tables = cursor.fetchall()
    print(list_of_tables)
    df = pd.read_sql_query("SELECT * from acfg", con)

    # Verify that result of SQL query is stored in the dataframe
    print(df.head())


except sqlite3.Error as error:
    print("Failed to execute above queries", error)

finally:
    if con:
      con.close()
      print("the sqlite connection is closed")