from TTS.api import TTS

# Initialize the TTS with a pre-trained model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

# Convert text to speech and save it to a file
tts.tts_to_file(text="Hello, this is Coqui TTS running locally.", file_path="output.wav")
