import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv("../stocks/db/.env")

async def check_tables():
    url = os.getenv("DATABASE_URL")
    print(f"Connecting to: {url}")
    conn = await asyncpg.connect(url)
    rows = await conn.fetch("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
    """)
    print("Tables found:", [row['table_name'] for row in rows])
    await conn.close()

asyncio.run(check_tables())
