from flask import Flask, request, jsonify, render_template
import redis
import time
from datetime import datetime

app = Flask(__name__)
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

@app.route("/")
def home():
    return render_template("index.html")

#tworzenie ankiety
@app.route("/poll", methods=["POST"])
def create_poll():
    data = request.json

    #id
    all_ids = r.smembers("polls:all")
    if all_ids:
        max_id = max(int(pid) for pid in all_ids)
        poll_id = str(max_id + 1)
    else:
        poll_id = "1"

    question = data["question"]
    options = data["options"]
    max_choices = int(data["max_choices"])
    end_time = data["end_time"]

    r.set(f"poll:{poll_id}:question", question)
    r.set(f"poll:{poll_id}:max_choices", max_choices)
    r.set(f"poll:{poll_id}:end_time", end_time)

    for option in options:
        r.hset(f"poll:{poll_id}:options", option, 0)

    r.sadd("polls:all", poll_id)

    return jsonify({"status": "Poll created", "id": poll_id}), 201

#lista ankiet
@app.route("/polls", methods=["GET"])
def list_polls():
    all_ids = r.smembers("polls:all")
    polls = []

    now = time.time()

    for poll_id in all_ids:
        question = r.get(f"poll:{poll_id}:question")
        end_time = float(r.get(f"poll:{poll_id}:end_time"))
        max_choices = int(r.get(f"poll:{poll_id}:max_choices"))

        status = "active" if end_time > now else "ended"
        end_datetime = datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")

        polls.append({
            "id": poll_id,
            "question": question,
            "max_choices": max_choices,
            "end_time": end_datetime,
            "status": status
        })

    polls.sort(key=lambda x: x["end_time"])
    return jsonify(polls)

#wyniki ankiety
@app.route("/results/<poll_id>", methods=["GET"])
def results(poll_id):
    question = r.get(f"poll:{poll_id}:question")
    max_choices = r.get(f"poll:{poll_id}:max_choices")
    end_time = r.get(f"poll:{poll_id}:end_time")
    options = r.hgetall(f"poll:{poll_id}:options")

    return jsonify({
        "question": question,
        "max_choices": max_choices,
        "end_time": end_time,
        "results": options
    })

#głosowanie
@app.route("/vote", methods=["POST"])
def vote():
    data = request.json
    poll_id = data["poll_id"]
    selected_options = data["options"]
    voter_id = request.remote_addr

    end_time = r.get(f"poll:{poll_id}:end_time")
    if end_time is None or time.time() > float(end_time):
        return jsonify({"error": "Poll has ended"}), 403

    voters_key = f"poll:{poll_id}:voters"
    if r.sismember(voters_key, voter_id):
        return jsonify({"error": "You have already voted"}), 403

    max_choices = int(r.get(f"poll:{poll_id}:max_choices"))

    if len(selected_options) == 0 or len(selected_options) > max_choices:
        return jsonify({"error": f"You can choose from 1 to {max_choices} options"}), 400

    for option in selected_options:
        r.hincrby(f"poll:{poll_id}:options", option, 1)

    r.sadd(voters_key, voter_id)
    return jsonify({"status": "Vote counted"}), 200

if __name__ == "__main__":
    app.run(debug=True)
