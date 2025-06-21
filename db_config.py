from sqlalchemy import create_engine

# Replace these with your actual credentials
DATABASE_URL = "mysql+pymysql://root:root@localhost:3306/accord_base_live"

engine = create_engine(DATABASE_URL)
