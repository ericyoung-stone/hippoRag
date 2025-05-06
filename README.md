# hippoRag
- [GitHub 地址](https://github.com/OSU-NLP-Group/HippoRAG)

## 环境依赖安装
### 1. 创建 conda 环境,下载依赖
```bash
conda create -n hipporag python=3.10
conda activate hipporag
pip install hipporag
```

### 2. 设置环境变量
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 非必要
export HF_HOME=<path to Huggingface home directory> # 非必要
export OPENAI_API_KEY=<your openai api key>   # 需要设置, local LLM 设置为lmstudio
```

### 3. 运行py
- 报错... EOFError
  - The "freeze_support()" line can be omitted if the program is not going to be frozen to produce an executable.
  - 解决: 在 macOS 上，多进程的启动方法需要在任何多进程相关的模块被导入之前就设置好,hipporag 包在导入时就会初始化一些多进程相关的组件,通过提前设置启动方法，可以确保所有组件都使用相同的多进程策略
- 报错 openai.OpenAIError: 
  - The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable、
  - 解决: export OPENAI_API_KEY=xxx