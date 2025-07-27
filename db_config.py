from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import create_engine

ASYNC_DB_USER = "root"
ASYNC_DB_PASSWORD = "R22.Mane"
ASYNC_DB_HOST = "localhost"
ASYNC_DB_NAME = "accord_base_live"

ASYNC_DATABASE_URL = f"mysql+aiomysql://{ASYNC_DB_USER}:{ASYNC_DB_PASSWORD}@{ASYNC_DB_HOST}/{ASYNC_DB_NAME}"
async_engine = create_async_engine(ASYNC_DATABASE_URL)


SYNC_DB_USER = "root"
SYNC_DB_PASSWORD = "R22.Mane"
SYNC_DB_HOST = "localhost"
SYNC_DB_NAME = "accord_base_live"

SYNC_DATABASE_URL = f"mysql+pymysql://{SYNC_DB_USER}:{SYNC_DB_PASSWORD}@{SYNC_DB_HOST}/{SYNC_DB_NAME}"
sync_engine = create_engine(SYNC_DATABASE_URL)

# # Replace these with your actual credentials
# DATABASE_URL = "mysql+pymysql://root:root@localhost:3306/accord_base_live"

# engine = create_engine(DATABASE_URL)
