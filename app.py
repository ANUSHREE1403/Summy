from flask import Flask, request, jsonify, render_template
import os
import whisper
from pydub import AudioSegment
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Whisper model for speech-to-text
model = whisper.load_model("base")

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def summarize_with_gemini(transcript):
    try:
        # Initialize the Gemini model
        model = genai.GenerativeModel('gemini-pro')

        # Create a prompt for summarization
        prompt = f"Summarize the following text in 2-3 concise sentences:\n\n{transcript}"

        # Generate the summary
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "Failed to generate summary."

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Extract audio if it's a video file
    if file.filename.endswith(('.mp4', '.avi', '.mov')):
        audio_path = os.path.join(UPLOAD_FOLDER, "extracted_audio.wav")
        file_path = extract_audio(file_path, audio_path)

    # Generate transcript
    try:
        transcript = model.transcribe(file_path)["text"]
    except Exception as e:
        print(f"Error during transcription: {e}")
        return jsonify({"error": "Failed to transcribe the file"}), 500

    # Summarize transcript using Gemini
    summary = summarize_with_gemini(transcript)

    return jsonify({
        "transcript": transcript,
        "summary": summary
    })

def extract_audio(file_path, output_path):
    try:
        # Load the video file and extract audio
        audio = AudioSegment.from_file(file_path)
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)