# Chatbot Conversation Analysis

## Overview

This project is designed to analyze and manage conversations between a user and an AI chatbot. It utilizes various libraries to handle natural language processing, embedding, and conversation state management. The chatbot can remember user details, respond to queries, and maintain a casual conversation.

## Features

- **Conversation Logging**: Stores user and AI messages with timestamps.
- **Dynamic Responses**: The AI can respond based on the context of the conversation.
- **Embedding Model**: Utilizes a sentence transformer for embedding text data.
- **Custom Relevance Scoring**: Implements a custom function to score the relevance of retrieved documents.
- **State Management**: Maintains the state of the conversation using a state graph.

## Requirements

- Python 3.7 or higher
- Flask
- Gradio
- Sentence Transformers
- LangChain
- LangGraph
- dotenv
- Chroma

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Naidu-Suraj-Vardhan/Chatbot-Conversation-Analysis.git
   cd chatbot-conversation-analysis
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   - Create a `.env` file in the root directory and add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

## Usage

1. To start the application, run:
   ```bash
   python3 app.py
   ```

2. You can interact with the chatbot through the Gradio interface.

3. To embed documents from a JSON file, run:
   ```bash
   python3 embed.py
   ```

## JSON Structure

The input JSON file should follow json structure


## License

This project is licensed under the MIT License. See the LICENSE file for details.
