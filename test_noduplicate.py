"""This is not a pytest function, but simply a program to ensure noduplicate program works as proposed
Utilized simulated passage content from langchain and metadata to test whether noduplicate work successfully.
    Expected Output:
                Original passages: 3
                Remaining passages: 2      

"""
from langchain_core.documents import Document
from gpt_researcher.context.noduplicate import remove_duplicate_passages

documents = [
    Document(
        page_content="Deepfakes can cause serious harm to younger generations utilizing social media.",
        metadata={"source": "source1.com"}
    ),
    Document(
        page_content="Deepfakes can cause serious harm to younger generations utilizing social media.",
        metadata={"source": "source2.com"}
    ),
    Document(
        page_content="Many social media platforms are utilizing machine learning capabilities within their infrastructure.",
        metadata={"source": "source3.com"}
    )
]

result = remove_duplicate_passages(documents)

print("Original passages:", len(documents))
print("Remaining passages:", len(result))

for doc in result:
    print(doc.page_content)
    print("Source:", doc.metadata["source"])
