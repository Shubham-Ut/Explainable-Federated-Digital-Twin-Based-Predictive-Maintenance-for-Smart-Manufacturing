import os


RAG_DIRS = [
    "rag/manuals",
    "rag/failure_reports",
    "rag/sop_documents",
    "rag/troubleshooting_guides"
]


def load_documents():
    documents = []

    for folder in RAG_DIRS:
        if not os.path.exists(folder):
            continue

        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                path = os.path.join(folder, filename)

                with open(path, "r", encoding="utf-8") as file:
                    content = file.read()

                documents.append({
                    "filename": filename,
                    "path": path,
                    "content": content
                })

    return documents