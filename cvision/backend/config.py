
from pymongo import MongoClient
import os

# קבלת URI ממשתנה סביבה, או שימוש בברירת מחדל (localhost)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "cv_matching_db"

# יצירת חיבור למסד הנתונים
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# שימוש באוספים
users_collection = db["users"]
resumes_collection = db["resumes"]
explanations_collection = db["explanations"]

