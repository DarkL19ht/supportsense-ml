from connection import get_engine


engine = get_engine()

try:
    with engine.connect() as connection:
        print("SUCCESS: Connected to PostgreSQL!")
except Exception as error:
    print("FAILED to connect.")
    print(error)