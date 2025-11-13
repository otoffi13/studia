import pymongo
from pymongo import MongoClient
from pymongo.errors import OperationFailure, ConfigurationError
import datetime

MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client["restaurants_db"]
coll = db["restaurants"]

def count_all():
    return coll.count_documents({})

def insert_demo():
    before = count_all()
    doc = {
        "name": f"Demo_Restaurant_{datetime.datetime.now().isoformat()}",
        "borough": "Bronx",
        "cuisine": "Italian",
        "grades": [{"grade": "A", "score": 10}],
        "address": {"building": "100", "street": "Demo St", "zipcode": "10451"},
        "temp_demo": True,
        "created_at": datetime.datetime.now()
    }
    res = coll.insert_one(doc)
    after = count_all()
    print(f"INSERT: inserted_id={res.inserted_id}. Count before={before}, after={after}")
    print(coll.find_one({"_id": res.inserted_id}))

def find_demo():
    filter_q = {"cuisine": "Italian"}
    print(f"FIND: {filter_q}")
    for i, doc in enumerate(coll.find(filter_q).limit(5), start=1):
        print(f"{i}. _id={doc.get('_id')} name={doc.get('name')} borough={doc.get('borough')}")
    total = coll.count_documents(filter_q)
    print(f"Łącznie znaleziono: {total}")

def update_demo():
    filter_q = {"cuisine": "Italian"}
    before = coll.count_documents(filter_q)
    res = coll.update_many(filter_q, {"$set": {"changed_by_lab": True}})
    after = coll.count_documents(filter_q)
    print(f"UPDATE: {filter_q} \nmatched={res.matched_count}, modified={res.modified_count}.")
    print(coll.find_one(filter_q))
    
def delete_demo():
    filter_q = {"temp_demo": True}
    before = coll.count_documents({})
    res = coll.delete_many(filter_q)
    after = coll.count_documents({})
    print(f"DELETE: {filter_q} \ndeleted_count={res.deleted_count}. Count before={before}, after={after}")



print("Połączenie z serwerem MongoDB. Dostępne bazy:")
for name in client.list_database_names():
    print(" -", name)


print("------------------")

insert_demo()
print("")
find_demo()
print("")
update_demo()
print("")
delete_demo()