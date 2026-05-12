# RAG Based AI Teaching Assistant

A local RAG (Retrieval-Augmented Generation) pipeline designed to provide semantic search and conversational guidance for a web development course. This system transcribes video audio, generates vector embeddings, and uses a local LLM to point users to specific timestamps based on their questions.

##  Key Features

* **Multilingual Transcription:** Utilizes OpenAI's **Whisper (large-v2)** to transcribe Hindi audio and translate it into English text segments [cite: create_chunks.py, speech_to_text (1).py].
* **Local Vector Embeddings:** Powered by **Ollama** using the `bge-m3` model for high-dimensional semantic representation [cite: request_chunks (1).py].
* **Smart Retrieval:** Uses **Cosine Similarity** to match user queries with the most relevant video chunks [cite: process_incoming (1).py].
* **Conversational Guidance:** Integrates **Llama 3.2** to explain where and how much of a topic is covered, directing users to exact timestamps [cite: process_incoming (1).py].

##  Project Structure

- `create_chunks.py`: Extracts metadata (video number/title) from filenames and handles the translation/transcription loop [cite: create_chunks.py].
- `speech_to_text (1).py`: Core utility for converting audio segments into JSON objects containing `start`, `end`, and `text` [cite: speech_to_text (1).py].
- `request_chunks (1).py`: Processes JSON transcripts into embeddings and saves them as a serialized dataframe (`embeddings.joblib`) [cite: request_chunks (1).py].
- `process_incoming (1).py`: The inference engine that accepts user queries and generates responses using the context of retrieved video chunks [cite: process_incoming (1).py].

##  Tech Stack

* **AI Models:** Whisper large-v2, Llama 3.2, BGE-M3 (via Ollama).
* **Language:** Python.
* **Libraries:** `pandas`, `scikit-learn`, `joblib`, `requests`.

##  Setup & Installation

1.  **Install Ollama:** Download and install [Ollama](https://ollama.com/).
2.  **Pull Models:**
    ```bash
    ollama pull llama3.2
    ollama pull bge-m3
    ```
3.  **Install Python Packages:**
    ```bash
    pip install openai-whisper pandas scikit-learn joblib requests
    ```

##  Usage

1.  **Transcribe:** Run `create_chunks.py` to generate text segments from your `/audios` folder.
2.  **Index:** Run `request_chunks (1).py` to create the `embeddings.joblib` vector store.
3.  **Query:** Execute `process_incoming (1).py` and ask questions about the course content (e.g., "Where is CSS Grid explained?").

##  License
This project is licensed under the MIT License.
