import sqlite3
from sqlite3 import Error
    
import psycopg2
from psycopg2 import OperationalError

class Sqlite:
    
    def __init__(self):
        pass
    
  
    def create_connection(self, path):
        connection = None
        try:
            connection = sqlite3.connect(path)
            print("Connection to SQLite DB successful")
        except Error as e:
            print(f"The error '{e}' occurred")
            
        return connection
    
    
    def execute_query(self, connection, query):
        cursor = connection.cursor()
        try:
            cursor.execute(query)
            connection.commit()
            print("Query executed successfully")
        except Error as e:
            print(f"The error '{e}' occurred")

            
    def execute_read_query(self, connection, query):
        cursor = connection.cursor()
        result = None
        try:
            cursor.execute(query)
            result = cursor.fetchall()
            return result
        except Error as e:
            print(f"The error '{e}' occurred")
        
        

# # This is an example

# select_users = "SELECT * from users"
# users = execute_read_query(connection, select_users)

# for user in users:
#     print(user)

# column_names = [description[0] for description in cursor.description]
# print(column_names)

class MongoDB:

    def __init__(self, user, password, host, db_name ,port='27017', authSource='admin'):
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.db_name = db_name
        self.authSource = authSource
        self.uri = f'mongodb://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}?authSource={self.authSource}'
        try:
            self.client = MongoClient(self.uri)
            self.db = self.client[self.db_name]
            print('MongoDB Connection Successful. CHEERS!!!')
        except Exception as e:
            print('Connection Unsuccessful!! ERROR!!')
            print(e)


    def insert_into_db(self, data, collection):
        if isinstance(data, pd.DataFrame):
            try:
                self.db[collection].insert_many(data.to_dict('records'))
                print('Data Inserted Successfully')
            except Exception as e:
                print('OOPS!! Some ERROR Occurred')
                print(e)
            finally:
                self.client.close()
                print('Connection Closed!!!')
        else:
            try:
                self.db[collection].insert_one(data)
                print('Data Inserted Successfully')
            except Exception as e:
                print('OOPS!! Some ERROR Occurred')
                print(e)
            finally:
                self.client.close()
                print('Connection Closed!!!')

class Postgres:

    def create_connection(db_name, db_user, db_password, db_host, db_port):
        connection = None
        try:
            connection = psycopg2.connect(
                database=db_name,
                user=db_user,
                password=db_password,
                host=db_host,
                port=db_port,
            )
            print("Connection to PostgreSQL DB successful")
        except OperationalError as e:
            print(f"The error '{e}' occurred")
        return connection
    
    
    def execute_query(connection, query):
        connection.autocommit = True
        cursor = connection.cursor()
        try:
            cursor.execute(query)
            print("Query executed successfully")
        except OperationalError as e:
            print(f"The error '{e}' occurred")
        
        
#     create_users_table = """
#     CREATE TABLE IF NOT EXISTS users (
#       id SERIAL PRIMARY KEY,
#       name TEXT NOT NULL, 
#       age INTEGER,
#       gender TEXT,
#       nationality TEXT
#     )
#     """
    
    def execute_read_query(connection, query):
        cursor = connection.cursor()
        result = None
        try:
            cursor.execute(query)
            result = cursor.fetchall()
            return result
        except OperationalError as e:
            print(f"The error '{e}' occurred")

        select_users = "SELECT * FROM users"
        users = execute_read_query(connection, select_users)

#     for user in users:
#         print(user)
    

class Mysql:
    pass






