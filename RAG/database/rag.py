from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.document_loaders import DirectoryLoader,PyPDFLoader,TextLoader
from sentence_transformers import CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
import re
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


class SmartDocumentProcessor:
    def __init__(self):
        # 初始化嵌入模型，使用HuggingFace的BAAI/bge-small-zh-v1.5模型-这个模型专为RAG而生
        self.embed_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"batch_size": 16}
        )

    def _detect_content_type(self, text):
        if re.search(r'def |import |print\(|代码示例', text):
            return "code"
        elif re.search(r'\|.+\|', text) and '%' in text: 
            return "table"
        return "normal" 

    def process_documents(self):
        loaders = [
            DirectoryLoader("./database", glob="**/*.pdf", loader_cls=PyPDFLoader),
            DirectoryLoader("./database", glob="**/*.txt", loader_cls=TextLoader)
        ]
        documents = []
        for loader in loaders:
            documents.extend(loader.load())
            
        chunker = SemanticChunker(
            embeddings=self.embed_model,  
            breakpoint_threshold_amount=82,  
            add_start_index=True  
        )
        base_chunks = chunker.split_documents(documents) 

        # 二次动态分块
        final_chunks = []
        for chunk in base_chunks:
            content_type = self._detect_content_type(chunk.page_content)
            if content_type == "code":
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=256, chunk_overlap=64)
            elif content_type == "table":
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=384, chunk_overlap=96)
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=256, chunk_overlap=36)
            final_chunks.extend(splitter.split_documents([chunk]))
        for i, chunk in enumerate(final_chunks):
            chunk.metadata.update({
                "chunk_id": f"chunk_{i}",
                "content_type": self._detect_content_type(chunk.page_content)
            })  
        return final_chunks


class HybridRetriever:
    def __init__(self, chunks):
        self.vector_db = Chroma.from_documents(
            chunks,
            embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5"),
            persist_directory="./vector_db"
        )

        self.bm25_retriever = BM25Retriever.from_documents(
            chunks,
            k=5  # 初始检索数量多于最终需要
        )

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.vector_db.as_retriever(search_kwargs={"k": 5}),
                self.bm25_retriever
            ],
            weights=[0.6, 0.4]
        )

        self.reranker = CrossEncoder(
            "BAAI/bge-reranker-large",
            device="cuda" if torch.has_cuda else "cpu"
        )

    def retrieve(self, query, top_k=3):
        docs = self.ensemble_retriever.get_relevant_documents(query)

        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked_docs[:top_k]]


class EnhancedRAG:
    def __init__(self,model_name_or_path,cache_dir=None):
        self.model_name_or_path=model_name_or_path
        
        processor = SmartDocumentProcessor()
        chunks = processor.process_documents()  

        self.retriever = HybridRetriever(chunks)
        if cache_dir is not None:
            self.cache_dir=cache_dir
            self.model= AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            ).to('cuda')
        else:
            self.model= AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
            ).to('cuda')
        self.tokenizer=AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )

        torch.manual_seed(42)

        self.model.eval()

    def generate_prompt(self, question, contexts):
        context_str = "\n\n".join([
            f"[来源：{doc.metadata['source']}，类型：{doc.metadata['content_type']}]\n{doc.page_content}"
            for doc in contexts
        ])
        print(context_str)
        instruction=f"请基于以下法律条文内容{context_str}，用中文回答以下法律问题："
        prompt = [
            {
                "role": "system", "content": instruction},
            {
                "role": "user", "content": question}
        ]
        return prompt

    def ask(self, question):
        contexts = self.retriever.retrieve(question)

        messages = self.generate_prompt(question, contexts)

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        model_inputs = self.tokenizer([text], return_tensors="pt").to('cuda')

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
def main():
    model_name_or_path = "../sft_merged_model"
    #cache_dir="../Qwen_model_file"
    rag = EnhancedRAG(model_name_or_path)
    question1="请根据基本案情，给出适用的法条。基本案情：经审理查明，2017年6月7日20时许，在长春市绿园区西新镇开元村小东沟屯吕某某、王桂荣家东屋，因琐事产生争执后，被告人王桂荣用手将被害人户某某推倒在地，致户某某右股骨粉碎性骨折。经长春市司法鉴定中心鉴定：户某某外伤致右股骨粉碎性骨折构成轻伤一级。2017年8月9日，民警在长春市绿园区西新镇开元村小东沟屯王桂荣家将王桂荣传唤到派出所。上述事实，被告人王桂荣在开庭审理过程中无异议，并有被告人王桂荣在侦查机关的供述、被害人户某某的陈述、证人于某某的证言、受案登记表及立案决定书、到案经过、户籍证明、指认现场笔录及照片、长春市公安司法鉴定中心法医学人体损伤程度鉴定意见书等证据证实，足以认定。"
    question2="以下情况是否属于刑事犯罪？一名人员在公共场合醉酒滋事，损坏公共财物。"
    question3="小明在街头摆摊贩卖一些明显是盗版的电影光盘，属于侵犯著作权行为吗？"
    question4="某政府部门的工作人员在办理某企业的审批手续时，向该企业索要了一定财物，以此为由加快审批进度。根据我国《刑法》相关规定，这名工作人员是否构成受贿罪？"
    question5="李某是一名国家工作人员，在履行公务期间窃取了国家机密并逃往了美国，如何处罚他？"
    prompt=[question1,question2,question3,question4,question5]
    for i in range(len(prompt)):
        answer = rag.ask(prompt[i])
        print(f"prompt{i+1}：{prompt[i]}")
        print(f"response：{answer}")
if __name__ == '__main__':
    main()
