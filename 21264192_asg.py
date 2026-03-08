from transformers import pipeline
import streamlit as st
from PIL import Image

#introduce models

pipe_image2txt = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
pipe_txt2story = pipeline("text-generation",model="Qwen/Qwen2.5-0.5B-Instruct")
pipe_txt2audio = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")

#function 1 : image to text'
def image2txt(image):
    image = Image.open(image)
    text = pipe_image2txt(image)[0]['generated_text']
    return text
    
#function 2 : text to a story'
def txt2story(text):

    prompt = (
        f"You are a storyteller for children aged 3 to 10. "
        f"Write a simple story of no more than 100 words based on this description: {text}"
    )

    output = pipe_txt2story(
        prompt,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.7
    )[0]["generated_text"]

    if output.startswith(prompt):
        output = output[len(prompt):].strip()

    return output  

#function 3 : text to audio'
def txt2audio(story_txt):
    audio_data = pipe_txt2audio(story_txt)
    return audio_data


def main():
  st.set_page_config(page_title="Your Image to Audio Story", page_icon="😊")
  st.header("Turn Your Image to Audio Story")
  uploaded_file = st.file_uploader("Select an Image...")

  if uploaded_file is not None:
    print(uploaded_file)
    bytes_data = uploaded_file.getvalue()
    with open(uploaded_file.name, "wb") as file:
        file.write(bytes_data)
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)


    #Stage 1: image to text
    st.text('Processing image2txt...')
    scenario = image2txt(uploaded_file.name)
    st.write(scenario)

    #Stage 2: text to story
    st.text('Generating a story...')
    story = txt2story(scenario)
    st.write(story)

    #Stage 3: story to audio
    st.text('Generating audio data...')
    audio_data =txt2audio(story)

    # Play button
    if st.button("Play Audio"):
      # Get the audio array and sample rate
      audio_array = audio_data["audio"]
      sample_rate = audio_data["sampling_rate"]

      # Play audio directly using Streamlit
      st.audio(audio_array,
        sample_rate=sample_rate)

if __name__ == "__main__":
    main()
