Here's a README file for the provided code:

---

# PDF to Podcast Converter

This Streamlit application converts text from PDF files into a podcast-style audio file using ChatGroq and OpenAI's Text-to-Speech (TTS) API. The application processes PDF content to extract insights and generate a podcast transcript, which is then converted to an audio format.

## Features
- Upload a PDF file and extract its text.
- Summarize extracted text into key insights.
- Generate a podcast-style transcript with an engaging introduction, middle, and conclusion.
- Convert the podcast transcript into an audio file using OpenAI's TTS API.

## Requirements
- **Python 3.7+**
- **Streamlit** - Web application framework
- **pypdf** - For reading PDF files
- **Groq API** - For ChatGroq language model processing
- **OpenAI API** - For Text-to-Speech (TTS) conversion
- **LangChain** - Text splitting utility for processing large text

## Installation
1. **Clone this repository** and navigate to the project folder:
   ```bash
   git clone <repository-url>
   cd pdf-to-podcast
   ```

2. **Install dependencies**:
   ```bash
   pip install streamlit pypdf groq langchain openai
   ```

3. **Set up environment variables**:
   - Set up your **Groq API key**:
     ```bash
     export GROQ_API_KEY='your_groq_api_key'
     ```
   - Set up your **OpenAI API key**:
     ```bash
     export OPENAI_API_KEY='your_openai_api_key'
     ```

## File Structure
- **app.py**: Main application file containing all code for the PDF-to-Podcast conversion.
- **README.md**: Documentation for setting up and running the application.

## How to Run
1. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Use the application**:
   - Upload a PDF file through the application interface.
   - Click on **Convert to Podcast** to start the process.
   - The extracted text will display in a text area, and upon conversion, an audio file will be generated and played.

## Code Explanation
- **extract_text_from_pdf**: Reads the uploaded PDF file and extracts text.
- **condense_paper_to_insights_in_chunks**: Splits extracted text into manageable chunks, summarizes each chunk using ChatGroq's language model, and combines these summaries.
- **create_podcast_transcript**: Converts key insights into a podcast-style transcript, structuring it with an introduction, middle, and conclusion for engaging delivery.
- **text_to_speech**: Passes the podcast transcript to OpenAI's TTS API, retrieves the audio file, and returns it as a playable audio file in the Streamlit app.

## Error Handling
- Checks if the PDF file contains any text.
- Displays an error if no text is found in the PDF.
- Prints generated summaries and podcast transcripts in the console for debugging.

## Notes
- Make sure you have valid **Groq** and **OpenAI** API keys with access to the necessary models.
- Configure the model names in the code as per your specific API version.

## License
This project is licensed under the MIT License.

---
