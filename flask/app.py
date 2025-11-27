from flask import Flask, jsonify

app = Flask(__name__)

# Sample data
items = [
    {"id": 1, "name": "item1", "description": "This is item one"},
    {"id": 2, "name": "item2", "description": "This is item two"}
]

@app.route("/")
def home():
    return "Welcome to the Flask To-Do App!"

@app.route("/items", methods=["GET"])
def get_items():
    return jsonify(items)

@app.route("/items/<int:item_id>", methods=["GET"])
def get_item(item_id):
    item = next((item for item in items if item["id"] == item_id), None)
    if item is None:
        return jsonify({"error": "Item not found"}), 404
    return jsonify(item)

if __name__ == "__main__":
    app.run(debug=True)
