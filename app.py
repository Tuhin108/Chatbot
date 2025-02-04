import os
from dotenv import load_dotenv
import base64
from flask import Flask, render_template, request, jsonify
from mistralai import Mistral

load_dotenv()

app = Flask(__name__)

api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY environment variable not set in .env")

client = Mistral(api_key=api_key)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])  # Changed route to /api/chat
def chat():
    try:
        message = request.json.get("message")
        if not message:
            return jsonify({"error": "Message is required"}), 400

        model = "mistral-large-latest"

        stream_response = client.chat.stream(
            model=model,
            messages=[
                {"role": "user", "content": message},
            ]
        )

        generated_text = ""
        for chunk in stream_response:
            if chunk.data.choices[0].delta.content is not None:
                generated_text += chunk.data.choices[0].delta.content

        return jsonify({"response": generated_text})

    except Exception as e:
        print(f"Error in chat generation: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/classify_image", methods=["POST"]) # Changed route to /api/classify_image
def classify_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No selected image file"}), 400

        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        model = "pixtral-12b-2409"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                ]
            }
        ]

        chat_response = client.chat.complete(model=model, messages=messages)
        classified_text = chat_response.choices[0].message.content

        return jsonify({"response": classified_text})

    except Exception as e:
        print(f"Error in image classification: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)