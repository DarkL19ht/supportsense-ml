from sqlalchemy import text

from connection import get_engine


engine = get_engine()


with engine.connect() as connection:

    result = connection.execute(
        text(
            """
            SELECT COUNT(*)
            FROM support_queries
            """
        )
    )

    total_rows = result.scalar()

    print(
        f"Rows in support_queries: {total_rows}"
    )