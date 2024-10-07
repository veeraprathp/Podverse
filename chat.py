import streamlit as st
import pypdf
from io import BytesIO
import os
from groq import Groq # Importing Groq for ChatGroq API
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO
from openai import OpenAI

# Initialize ChatGroq with your API key
llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = pypdf.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
        
    if not text.strip():  # Check if the extracted text is empty
        raise ValueError("The PDF is empty or contains no extractable text.")
    
    return text

# Function to convert text to speech and return as audio
def text_to_speech(text):
    # Step 1: Condense paper to insights and create a podcast transcript
    summary = condense_paper_to_insights_in_chunks(text)
    podcast_transcript = create_podcast_transcript(summary)
    
    # Step 2: Initialize OpenAI Client
    client = OpenAI()

    # Step 3: Call OpenAI's TTS API to get the audio stream
    response = client.audio.speech.create(
        model="tts-1",  # Assuming "tts-1" is the correct model name
        voice="alloy",  # Assuming "alloy" is the correct voice name
        input=podcast_transcript  # Pass the podcast transcript to the TTS model
    )

    # Step 4: Manually stream the response content to a BytesIO object
    audio_content = response.read()

    # Step 5: Create a BytesIO object and write the binary content
    audio_file = BytesIO()
    audio_file.write(audio_content)

    # Move back to the start of the BytesIO object
    audio_file.seek(0)

    # Return the in-memory audio file
    return audio_file

def condense_paper_to_insights_in_chunks(extracted_text, chunk_size=4096, overlap=200):
    """
    Condenses a technical paper into key insights by processing the text in chunks,
    focusing on results and implications.
    """
    # Initialize text splitter with a chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    
    # Split the text into chunks
    chunks = text_splitter.split_text(extracted_text)
    
    # Store summaries of each chunk
    summaries = []
    
    for chunk in chunks:
        # Create a prompt for each chunk
        prompt = (f"Summarize the following technical paper section with a focus on retaining critical context, "
          f"key results, and their implications. Ensure that the summary accurately conveys the meaning "
          f"and avoids excessive technical jargon while preserving essential details:\n\n{chunk}")


        # Call to ChatGroq's API to get the summary for each chunk
        chat_completion = llm.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gemma2-9b-it",
        )
        
        # Extracting the generated text for the chunk
        summary = chat_completion.choices[0].message.content.strip()
        summaries.append(summary)
    
    # Combine all chunk summaries into a final summary
    combined_summary = " ".join(summaries)
    
    print("=============================================================")
    print(combined_summary)
    print("=============================================================")

    return combined_summary

def create_podcast_transcript(insights):
    """
    Creates a podcast transcript from key insights, including an introduction, questions, callbacks,
    and key takeaways for professional podcast delivery.
    """
    prompt = (f"Using the following insights:\n\n{insights}\n\n"
          f"Create a podcast transcript. The transcript should have a clear introduction, "
          f"engage the audience with relatable content, ask a question in the middle, "
          f"include a joke that relates to the topic of AI and transformers, "
          f"and end with a summary of the key takeaways and a professional outro. "
          f"Do not include any stage directions like 'upbeat music fades' or speaker labels like 'Host:'."
          f"Write it as a fluid, natural conversation, and ensure it remains engaging.")

    # Call to ChatGroq's API to create the podcast transcript
    chat_completion = llm.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gemma2-9b-it",
    )
    
    # Extracting the generated transcript
    transcript = chat_completion.choices[0].message.content.strip()
    print("=============================================================")
    print(transcript)
    print("=============================================================")
    return transcript

def main():
    st.title("PDF to Podcast Converter")

    # Upload PDF file
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if pdf_file is not None:
        # Extract text from the uploaded PDF
        text = extract_text_from_pdf(pdf_file)
        st.text_area("Extracted Text", text, height=200)
        
        if st.button("Convert to Podcast"):
            if text.strip() == "":
                st.error("No text found in the PDF.")
            else:
                # Convert extracted text to speech
                audio_file = text_to_speech(text)
                
                # Save the audio file
                with open("output.mp3", "wb") as f:
                    f.write(audio_file.read())
                
                # Play the generated audio file
                st.audio("output.mp3", format="audio/mp3")
                st.success("Podcast generated successfully!")

if __name__ == "__main__":
    main()
