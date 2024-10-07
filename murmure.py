import argparse
from openai import OpenAI
from pydub import AudioSegment
import os
import tempfile
from dotenv import load_dotenv
import yaml

# Load the configuration from settings.yaml
with open("settings.yaml", "r") as file:
    config = yaml.safe_load(file)

# Accessing the configuration
input_source_path = config["input"]["source_path"]
output_transcript_path = config["output"]["transcript_path"]
transcription_enabled = config["transcription"]["enabled"]

# Load secret .env file
load_dotenv()

# Initialize the OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "API key not found. Please set the OPENAI_API_KEY in the .env file."
    )

client = OpenAI(api_key=api_key)

# Debugging output to confirm the API key is loaded
print("Loaded environment variables:")
print(f"OPENAI_API_KEY: {api_key}")

# Argument parsing
parser = argparse.ArgumentParser(description="Process audio file for transcription.")
parser.add_argument(
    "--gpt_post_process",
    action="store_true",
    help="Enable GPT-4 post-processing",
)

args = parser.parse_args()

# Ensure output_transcript_path is treated as a directory
if not os.path.exists(output_transcript_path):
    os.makedirs(output_transcript_path)


# Function to convert m4a to mp3 and save it in the same 'audios' folder
def convert_m4a_to_mp3(m4a_file_path):
    try:
        # Construct the output .mp3 file path in the same folder as the input .m4a file
        mp3_file_path = os.path.join(
            os.path.dirname(m4a_file_path),  # Same directory as the input file
            os.path.splitext(os.path.basename(m4a_file_path))[0]
            + ".mp3",  # Change extension to .mp3
        )

        # Check if mp3 file already exists
        if os.path.exists(mp3_file_path):
            print(f"MP3 already exists for {m4a_file_path}. Skipping conversion.")
            return mp3_file_path

        # Load the .m4a file
        audio = AudioSegment.from_file(m4a_file_path, format="m4a")
        # Export the file as .mp3
        audio.export(mp3_file_path, format="mp3")
        print(f"Conversion successful! File saved as: {mp3_file_path}")
        return mp3_file_path

    except Exception as e:
        print(f"An error occurred during conversion: {e}")
        return None


# Function to split the audio file
def split_audio(file_path, chunk_length_ms):
    audio = AudioSegment.from_mp3(file_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunks.append(audio[i : i + chunk_length_ms])
    return chunks


# Transcription function using OpenAI Whisper
def transcribe(audio_chunk):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        audio_chunk.export(temp_file.name, format="mp3")
        with open(temp_file.name, "rb") as audio_file_path:
            response = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file_path, response_format="text"
            )
        # Check the format of the response and extract text accordingly
        if isinstance(response, str):
            return response
        else:
            return response.choices[0].text


# Post-process function using GPT-4
def post_process_transcript(transcript, system_prompt):
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
    )
    return response.choices[0].message.content


# Function to process all files in the input directory
def process_all_files(chunk_length_ms, system_prompt, use_gpt_post_process):
    for file_name in os.listdir(input_source_path):
        file_path = os.path.join(input_source_path, file_name)

        # Only process .m4a or .mp3 files
        if file_name.endswith((".m4a", ".mp3")):
            print(f"Processing file: {file_name}")
            file_extension = os.path.splitext(file_name)[1].lower()

            if file_extension == ".m4a":
                print("Input file is in .m4a format. Converting to .mp3...")
                mp3_file = convert_m4a_to_mp3(file_path)
            elif file_extension == ".mp3":
                print("Input file is already in .mp3 format. No conversion needed.")
                mp3_file = file_path
            else:
                print(f"Skipping unsupported file format: {file_extension}")
                continue

            # Proceed with transcription if enabled
            if transcription_enabled and mp3_file:
                full_transcript = process_audio(
                    mp3_file, chunk_length_ms, system_prompt, use_gpt_post_process
                )

                # Save the transcript to the specified folder
                transcript_file_name = os.path.splitext(file_name)[0] + ".txt"
                transcript_file_path = os.path.join(
                    output_transcript_path, transcript_file_name
                )

                with open(transcript_file_path, "w") as file:
                    file.write(full_transcript)
                print(f"Transcript saved at: {transcript_file_path}")
            else:
                if transcription_enabled:
                    print("Conversion to mp3 failed.")
                else:
                    print("Transcription is disabled.")


# Main function to split and transcribe
def process_audio(file_path, chunk_length_ms, system_prompt, use_gpt_post_process):
    chunks = split_audio(file_path, chunk_length_ms)
    full_transcript = ""
    for chunk in chunks:
        transcript = transcribe(chunk)
        if use_gpt_post_process:
            transcript = post_process_transcript(transcript, system_prompt)
        full_transcript += transcript + "\n"
    return full_transcript


if __name__ == "__main__":
    chunk_length_ms = 10 * 60 * 1000  # 10 minutes in milliseconds
    system_prompt = "Add your system prompt here"  # Customize this prompt as needed
    process_all_files(chunk_length_ms, system_prompt, args.gpt_post_process)
