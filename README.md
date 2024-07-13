# DocQnA Chatbot

![DocQnA Chatbot](https://via.placeholder.com/800x200.png)

## Overview

DocQnA is an advanced chatbot designed to answer questions based on document content. Leveraging the power of GroQ AI API for natural language understanding, Cohere embedding API for semantic search, and Streamlit for a user-friendly interface, DocQnA provides accurate and quick responses to user queries.

## Features

- **Natural Language Processing**: Utilizes GroQ AI API for sophisticated language understanding.
- **Semantic Search**: Employs Cohere embedding API to find the most relevant document segments.
- **User-friendly Interface**: Built with Streamlit for an interactive and easy-to-use experience.
- **Document Upload**: Allows users to upload documents in various formats.
- **Real-time Q&A**: Provides instant answers to user queries based on the uploaded documents.

## Installation

Follow these steps to set up the DocQnA chatbot on your local machine:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/DocQnA.git
    cd DocQnA
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python3 -m venv env
    source env/bin/activate
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up API keys**: 
   Create a `.env` file in the root directory and add your GroQ AI API and Cohere API keys.
    ```env
    GROQ_API_KEY=your_groq_api_key
    COHERE_API_KEY=your_cohere_api_key
    ```

## Usage

1. **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```

2. **Upload Documents**: Use the UI to upload documents you want the chatbot to analyze.
3. **Ask Questions**: Type your questions in the input box and get instant answers based on the document content.


## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature/your-feature`).
6. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- [GroQ AI](https://groq.com/) for their powerful NLP API.
- [Cohere](https://cohere.ai/) for their embedding API.
- [Streamlit](https://streamlit.io/) for their intuitive UI framework.

## Contact

For any inquiries or feedback, please contact [yourname@example.com](mailto:yourname@example.com).

---

Thank you for using DocQnA! We hope it serves your needs well.


