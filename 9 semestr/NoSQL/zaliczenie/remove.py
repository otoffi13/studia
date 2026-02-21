import redis

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

poll_ids = r.smembers("polls:all")

for poll_id in poll_ids:
    r.delete(f"poll:{poll_id}:question")
    r.delete(f"poll:{poll_id}:max_choices")
    r.delete(f"poll:{poll_id}:end_time")
    r.delete(f"poll:{poll_id}:voters")
    r.delete(f"poll:{poll_id}:options")

r.delete("polls:all")

print("Baza ankiet została wyczyszczona!")
