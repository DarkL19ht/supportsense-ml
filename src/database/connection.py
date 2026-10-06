import os

from dotenv import load_dotenv
from sqlalchemy import create_engine


load_dotenv()


def get_engine():
    database_url = os.getenv("DATABASE_URL")

    print(database_url)

    if not database_url:
        raise ValueError(
            "DATABASE_URL environment variable is not set."
        )

    return create_engine(database_url)