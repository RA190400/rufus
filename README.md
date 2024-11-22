Here’s an updated README for your **AI-Driven Web Scraping and Answer Retrieval System** project:

---

# AI-Driven Web Scraping and Answer Retrieval System  

An advanced **AI-powered Retrieval-Augmented Generation (RAG)** system designed to intelligently scrape the web, perform recursive link searches, and provide context-sensitive semantic retrieval using large language models (LLMs).  

Built with **Python**, **Selenium**, **FAISS**, and **Ollama Mistral**, this system integrates robust web scraping, intelligent search, and fast answer generation with a user-friendly **Streamlit** interface.  

---

## Features  

- **Intelligent Web Scraping**: Automates data extraction with Selenium and recursive link exploration.  
- **Semantic Answer Retrieval**: Uses FAISS for similarity search and the Ollama Mistral LLM for delivering accurate, context-sensitive answers.  
- **Streamlit Frontend**: Offers an interactive interface for query customization, URL input, and downloading results.  
- **Fast & Scalable**: Optimized for efficient scraping and retrieval across multiple URLs.  

---

## Installation  

1. Clone the repository:  
   ```bash  
   git clone https://github.com/RA190400/rufus.git  
   cd rufus  
   ```  

2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. Set up the necessary APIs or configurations:  
   - Configure **Ollama Mistral LLM** API access.  
   - Ensure **FAISS** and **Selenium** are correctly installed.  

---

## Usage  

### Running the Application  

1. Start the Streamlit frontend:  
   ```bash  
   streamlit run main.py  
   ```  

2. Input the URL(s) for scraping and your query in the interface.  

3. Customize the query or parameters and download the results as needed.  

### CLI Usage  
```bash  
python main.py --url <url> --query "<your query>"  
```  

#### Available Arguments:  
- `--url`: Specify the URL to scrape.  
- `--query`: Enter your question or prompt for retrieval.  

---

## Project Structure  

```
AI-Driven-Web-Scraping-System/  
│  
├── scraping/           # Web scraping logic with Selenium  
├── retrieval/          # FAISS-based semantic search  
├── model/              # Integration with Ollama Mistral LLM  
├── frontend/           # Streamlit-based user interface  
├── tests/              # Unit and integration tests  
├── app.py              # Main Streamlit app entry point  
├── main.py             # CLI entry point  
├── requirements.txt    # Python dependencies  
└── README.md           # Project documentation  
```  

---

## Contributing  

Contributions are encouraged!  

1. Fork the repository.  
2. Create a feature branch (`git checkout -b feature-branch`).  
3. Commit your changes (`git commit -m "Add feature"`).  
4. Push the branch (`git push origin feature-branch`).  
5. Submit a pull request.  

---


## Contact  

For questions or feedback, please reach out to:  

**Rakshit Grover**  
Email: rakshitg@usc.edu  
GitHub: [RA190400](https://github.com/RA190400)  

--- 

Feel free to refine this based on specific project needs!
