from flask import Flask, request, jsonify, render_template
import base64, os

app = Flask(__name__)
SAVE_DIR = "server_images"
os.makedirs(SAVE_DIR, exist_ok=True)

alerts = []

@app.route("/")
def dashboard():
    return render_template("index.html")

@app.route("/alert", methods=["POST"])
def alert():
    try:
        data = request.get_json()
        event = data.get("event", "unknown")
        timestamp = data.get("timestamp", "unknown")
        location = data.get("location", "unknown")
        img_b64 = data.get("image", None)

        # Save image locally
        if img_b64:
            img_bytes = base64.b64decode(img_b64)
            path = os.path.join(SAVE_DIR, f"{event}_{timestamp.replace(':','-')}.jpg")
            with open(path, "wb") as f:
                f.write(img_bytes)

        alerts.append(data)  # Save in memory for SSE

        print(f"Alert received: {event} at {timestamp} ({location})")
        return jsonify({"status":"ok"}), 200
    except Exception as e:
        print("Server error:", e)
        return jsonify({"status":"error", "msg": str(e)}), 500

@app.route("/alert_stream")
def alert_stream():
    def event_stream():
        last_index = 0
        while True:
            if len(alerts) > last_index:
                for alert_item in alerts[last_index:]:
                    yield f"data: {alert_item}\n\n"
                last_index = len(alerts)
    return app.response_class(event_stream(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
