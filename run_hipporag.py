import os
import multiprocessing

# 在导入 HippoRAG 之前设置多进程启动方法
if os.name != 'nt':  # 非 Windows 系统
    multiprocessing.set_start_method('fork', force=True)

from hipporag import HippoRAG
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    
    # Prepare datasets and evaluation
    docs = [
        "Oliver Badman is a politician.",
        "George Rankin is a politician.",
        "Thomas Marwick is a politician.",
        "Cinderella attended the royal ball.",
        "The prince used the lost glass slipper to search the kingdom.",
        "When the slipper fit perfectly, Cinderella was reunited with the prince.",
        "Erik Hort's birthplace is Montebello.",
        "Marina is bom in Minsk.",
        "Montebello is a part of Rockland County."
    ]

    save_dir = './outputs'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    llm_base_url = 'http://192.168.1.22:1234/v1'
    llm_model_name = 'qwen3-4b' # Any OpenAI model name
    embedding_base_url = llm_base_url + '/embeddings'
    embedding_model_name = 'text-embedding-bge-m3'  # Embedding model name (NV-Embed, GritLM or Contriever for now)

    # Startup a HippoRAG instance
    hipporag = HippoRAG(
        save_dir=save_dir,
        llm_model_name=llm_model_name,
        llm_base_url=llm_base_url,
        embedding_model_name=embedding_model_name,
        embedding_base_url=llm_base_url
        )

    # Run indexing
    hipporag.index(docs=docs)

    # Separate Retrieval & QA
    queries = [
        "What is George Rankin's occupation?",
        "How did Cinderella reach her happy ending?",
        "What county is Erik Hort's birthplace a part of?"
    ]

    retrieval_results = hipporag.retrieve(queries=queries, num_to_retrieve=2)
    qa_results = hipporag.rag_qa(retrieval_results)

    # Combined Retrieval & QA
    rag_results_v1 = hipporag.rag_qa(queries=queries)
    print(f"=== rag_results_v1 length:\n {len(rag_results_v1)}")
    for i, rag_result in enumerate(rag_results_v1, start=1):
        print(f"{i} rag_results_v1:\n {rag_result}")
        # print(f"{i} rag_results_v1:\n {rag_result}")
        # print(f"{i} rag_results_v1:\n {rag_result}")

    # For Evaluation
    answers = [
        ["Politician"],
        ["By going to the ball."],
        ["Rockland County"]
    ]

    gold_docs = [
        ["George Rankin is a politician."],
        ["Cinderella attended the royal ball.",
        "The prince used the lost glass slipper to search the kingdom.",
        "When the slipper fit perfectly, Cinderella was reunited with the prince."],
        ["Erik Hort's birthplace is Montebello.",
        "Montebello is a part of Rockland County."]
    ]

    rag_results_v2 = hipporag.rag_qa(
        queries=queries,
        gold_docs=gold_docs,
        gold_answers=answers
    )
    print(f"=== rag_results_v2:\n {rag_results_v2}")