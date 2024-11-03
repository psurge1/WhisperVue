import pyaudio
import wave
import whisper
import numpy as np

model = whisper.load_model("base")

def record_audio(filename="output.wav", silence_duration=2.0, threshold=500):
    chunk = 1024  # Record in chunks of 1024 samples
    format = pyaudio.paInt16  # 16 bits per sample
    channels = 1  # Single channel for microphone
    rate = 44100  # Sample rate

    p = pyaudio.PyAudio()

    stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)

    frames = []
    silent_chunks = 0  # Counter for silent chunks

    while True:
        data = stream.read(chunk)
        frames.append(data)

        # Convert the data to an array of integers
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Check the amplitude (volume)
        if np.abs(audio_data).mean() < threshold:
            silent_chunks += 1
        else:
            silent_chunks = 0  # Reset counter if sound is detected

        # Stop recording if silence exceeds the defined duration (2 seconds)
        if silent_chunks * (chunk / rate) >= silence_duration:
            break

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded data as a WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
    print(f"Audio recorded and saved to {filename}")
    result = model.transcribe("output.wav")
    return result["text"]