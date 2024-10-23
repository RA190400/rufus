from rufus import Rufus

def main():
    target_url = input("Enter the URL of the website to scrape ")
    question = input("Enter a prompt/question: ")
    file_format = input("Enter the file format to save the answers (json/docx): ").strip().lower()

    rag_system = Rufus()  # Create an instance of the RAGSystem class
    rag_system.run(target_url, question, file_format)  # Call the run method

if __name__ == "__main__":
    main()