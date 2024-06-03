# RagWithLangChain
# Question Answering App with Streamlit

This is a Streamlit application that allows users to upload a text file and ask questions about its content. The app uses LangChain for text splitting, embeddings, and question answering.

## Features

- Upload a text file.
- Split the text into manageable chunks.
- Generate sentence embeddings.
- Retrieve relevant document chunks based on the user's query.
- Answer questions using a pre-trained language model.

## Requirements

- Python 3.7 or higher
- Streamlit
- LangChain
- LangChain Community
- Sentence Transformers

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yuvanrajjjj/RagWithLangChain.git
    cd yuvanrajjjj/RagWithLangChain
    ```

2. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:

    ```sh
    streamlit run app.py
    ```

2. Upload a text file using the provided file uploader in the app.

3. Enter a question related to the content of the uploaded text file.

4. Click "Submit" to get the answer.

## File Structure

- `Langchain_Rag.py`: The main application code.
- `requirements.txt`: List of required Python packages.
- `uploaded_files/`: Directory where uploaded files will be stored.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [LangChain](https://github.com/hwchase17/langchain)
- [Streamlit](https://streamlit.io/)
- [Sentence Transformers](https://www.sbert.net/)
