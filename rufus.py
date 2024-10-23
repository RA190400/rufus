import time
import json
import numpy as np
from collections import deque
from transformers import pipeline, AutoModel, AutoTokenizer
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import validators
import faiss
from docx import Document
import torch

class Rufus:
    def __init__(self, max_input_length=450, similarity_threshold=0.5):
        self.text_generation_pipeline = pipeline("text-generation", model="gpt2", device=-1)
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        self.model = AutoModel.from_pretrained(self.embedding_model_name)
        
        self.embedding_dimension = 384  # Adjust based on the embedding model
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        self.texts = []
        
        self.MAX_INPUT_LENGTH = max_input_length
        self.SIMILARITY_THRESHOLD = similarity_threshold

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings

    def store_chunks_in_faiss(self, chunks):
        for chunk in chunks:
            embedding = self.get_embedding(chunk)
            self.index.add(embedding)
            self.texts.append(chunk)

    def retrieve_relevant_chunks(self, question, top_k=7):
        query_embedding = self.get_embedding(question)
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.texts[i] for i in indices[0]]  # Fix: use self.texts instead of texts

    def get_all_relevant_urls(self, start_url, max_depth=2, max_urls=10):
        visited = set()
        queue = deque([(start_url, 0)])
        all_urls = []

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)

        while queue and len(all_urls) < max_urls:
            current_url, depth = queue.popleft()
            if depth > max_depth or current_url in visited:
                continue

            visited.add(current_url)

            try:
                driver.get(current_url)
                time.sleep(2)  # Adjust sleep time as necessary
                all_urls.append(current_url)

                links = driver.find_elements(By.TAG_NAME, 'a')
                for link in links:
                    href = link.get_attribute('href')
                    if href and validators.url(href) and href not in visited:
                        queue.append((href, depth + 1))
            except Exception as e:
                print(f"An error occurred while accessing {current_url}: {e}")

        driver.quit()
        return all_urls

    def extract_data_from_url(self, url):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)

        try:
            driver.get(url)
            time.sleep(2)  # Adjust sleep time as necessary
            body = driver.find_element(By.TAG_NAME, 'body')
            full_text = body.text
            return full_text
        except Exception as e:
            print(f"An error occurred while accessing {url}: {e}")
            return None
        finally:
            driver.quit()

    def chunk_text(self, text, max_length):
        sentences = text.split('. ')
        chunks = []
        current_chunk = []

        for sentence in sentences:
            if len(' '.join(current_chunk)) + len(sentence) + 1 <= max_length:
                current_chunk.append(sentence)
            else:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]

        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')

        return chunks

    def generate_long_answer(self, relevant_text, question):
        prompt = (
            f"You are a knowledgeable assistant. Use the following context, along with your pretrained knowledge, "
            f"to answer the question accurately.\n\n"
            f"Question: {question}\n"
            f"Context: {relevant_text}\n"
            f"Answer:"
        )
        
        generated_text = self.text_generation_pipeline(prompt, max_new_tokens=150, num_return_sequences=1)
        answer_text = generated_text[0]['generated_text'].strip()
        
        if "Answer:" in answer_text:
            answer_start = answer_text.index("Answer: ") + len("Answer: ")
            return answer_text[answer_start:].strip()
        else:
            return answer_text

    def save_to_json(self, data, filename):
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Data saved to {filename}")

    def save_to_docx(self, data, filename):
        doc = Document()
        doc.add_heading('RAG System Report', level=1)

        for entry in data:
            doc.add_heading(entry['question'], level=2)
            doc.add_paragraph(f"Extracted Context: {entry['context']}")
            doc.add_paragraph(f"Generated Answer: {entry['answer']}\n")
        
        doc.save(filename)
        print(f"Data saved to {filename}")

    def run(self, target_url, question, file_format):
        main_url_text = self.extract_data_from_url(target_url)
        relevant_data_found = False
        results = []

        if main_url_text:
            chunks = self.chunk_text(main_url_text, self.MAX_INPUT_LENGTH)
            self.store_chunks_in_faiss(chunks)

            relevant_chunks = self.retrieve_relevant_chunks(question)
            if relevant_chunks:
                combined_context = ' '.join(relevant_chunks)
                long_answer = self.generate_long_answer(combined_context, question)
                print(f"Generated Long Answer from main URL ({target_url}):\n{long_answer}\n")
                results.append({
                    'question': question,
                    'context': combined_context,
                    'answer': long_answer
                })
                relevant_data_found = True

        if not relevant_data_found:
            relevant_urls = self.get_all_relevant_urls(target_url)
            if relevant_urls:
                print(f"Found {len(relevant_urls)} relevant URLs.")
                for url in relevant_urls:
                    print(f"Checking relevant URL: {url}")
                    text = self.extract_data_from_url(url)
                    if text:
                        chunks = self.chunk_text(text, self.MAX_INPUT_LENGTH)
                        self.store_chunks_in_faiss(chunks)
                        relevant_chunks = self.retrieve_relevant_chunks(question)

                        if relevant_chunks:
                            combined_context = ' '.join(relevant_chunks)
                            long_answer = self.generate_long_answer(combined_context, question)
                            print(f"Generated Long Answer from {url}:\n{long_answer}\n")
                            results.append({
                                'question': question,
                                'context': combined_context,
                                'answer': long_answer
                            })
                            relevant_data_found = True
                            break

            if not relevant_data_found:
                print("No relevant data found in the provided URLs.")
        else:
            print("Relevant data found in the main URL.")

        if results:
            if file_format == 'json':
                self.save_to_json(results, 'rag_results.json')
            elif file_format == 'docx':
                self.save_to_docx(results, 'rag_results.docx')
            else:
                print("Invalid file format. Please choose 'json' or 'docx'.")








