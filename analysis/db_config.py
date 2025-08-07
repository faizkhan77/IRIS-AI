# app/db_config.py
from sqlmodel import create_engine, Session

DATABASE_URL = "mysql+pymysql://root:root@localhost:3306/accord_base_live"
# DATABASE_URL = "mysql+pymysql://root:root@0.tcp.in.ngrok.io:14987/accord_base_live"


# Replace with your actual MySQL connection details:
# USER: your_mysql_user
# PASSWORD: your_mysql_password
# HOST: your_mysql_host (e.g., localhost)
# PORT: your_mysql_port (e.g., 3306)
# DATABASE_NAME: your_database_name

engine = create_engine(DATABASE_URL, echo=False) # Set echo=True for debugging SQL queries

def get_db_session():
    with Session(engine) as session:
        yield session