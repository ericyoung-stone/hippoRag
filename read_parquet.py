import os
import pandas as pd

# 定义parquet文件所在的基础目录
BASE_DIR = '/Users/ericyoung/ysx/code/github-study/RAG/hippoRag/outputs/qwen3-4b_text-embedding-bge-m3'

# 定义要读取的parquet文件路径
parquet_files = {
    'chunk': os.path.join(BASE_DIR, 'chunk_embeddings', 'vdb_chunk.parquet'),
    'entity': os.path.join(BASE_DIR, 'entity_embeddings', 'vdb_entity.parquet'),
    'fact': os.path.join(BASE_DIR, 'fact_embeddings', 'vdb_fact.parquet')
}

def read_parquet_files():
    """读取并显示所有parquet文件的基本信息和内容"""
    for name, file_path in parquet_files.items():
        print(f'\n读取{name}嵌入向量文件：{file_path}')
        try:
            # 读取parquet文件
            df = pd.read_parquet(file_path)
            
            # 显示基本信息
            print(f'\n{name}数据基本信息：')
            print(f'行数：{len(df)}')
            print(f'列数：{len(df.columns)}')
            print('\n列名：')
            print(df.columns.tolist())
            
            # 显示数据预览
            print('\n数据预览：')
            print(df.head())
            
        except Exception as e:
            print(f'读取{name}文件时发生错误：{str(e)}')

if __name__ == '__main__':
    read_parquet_files()