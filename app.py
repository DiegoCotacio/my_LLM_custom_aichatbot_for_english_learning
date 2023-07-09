import gradio as gr
import openai, config
import requests
import soundfile as sf
import pandas as pd
import numpy as np
import os

openai.api_key = config.OPENAI_API_KEY

messages = [{"role": "system", "content": 'You are machine learning engineer from new york city. Respond to all input in 50 words or less. Speak in the first person, be funny. Do not use quotation marks. Do not say you are an AI language model.'}]

def transcribe(audio):
    global messages

     # Convertir el audio a int de 16 bits
    audio_data, sample_rate = sf.read(audio)
    audio_data = (audio_data * 32767).astype("int16")

    # Guardar el audio convertido en un archivo temporal
    converted_audio = "converted_audio.wav"
    sf.write(converted_audio, audio_data, sample_rate)

    audio_file = open(converted_audio, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    messages.append({"role": "user", "content": transcript["text"]})

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

    system_message = response["choices"][0]["message"]["content"]

    messages.append({"role": "system", "content": system_message})

    #Text to speech request with eleven labs
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{config.ADVISOR_VOICE_ID}/stream"
    data = {
        "text" : system_message, #["content"].replace('"',''),
        "voice_settings":{
            "stability" : 0.1,
            "similarity_boost": 0.8
        }
    }
    headers = {'xi-api-key': config.ELEVEN_LABS_API_KEY}
    response = requests.post(url, headers=headers, json=data)

    output_filename = "replay.mp3"
    with open (output_filename, "wb") as ouput:
        ouput.write(response.content)
    
    chat_transcript = ""
    for message in messages:
        #if message['role'] != 'system':
        chat_transcript += message['role'] + ": " + message['content'] + "\n\n"

    return chat_transcript, output_filename #converted_output

# set a custom theme
theme = gr.themes.Default().set(
    body_background_fill="#000000",
)

with gr.Blocks(theme=theme) as ui:
    # advisor image input and microphone input
    #advisor = gr.Image(value=config.ADVISOR_IMAGE).style(width=config.ADVISOR_IMAGE_WIDTH, height=config.ADVISOR_IMAGE_HEIGHT)
    audio_input = gr.Audio(source="microphone", type="filepath")

    # text transcript output and audio 
    text_output = gr.Textbox(label="Conversation Transcript")
    audio_output = gr.Audio()

    btn = gr.Button("Run")
    btn.click(fn=transcribe, inputs=audio_input, outputs=[text_output, audio_output])

ui.launch(debug=True, share=True)


#ui = gr.Interface(fn=transcribe, inputs=gr.Audio(source="microphone", type="filepath"), outputs=["text", gr.Audio()])

