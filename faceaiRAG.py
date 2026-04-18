import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import time
import re
from collections import defaultdict
import jieba

os.environ["TRANSFORMERS_NO_TORCHDISTRIBUTED"] = "1"

class FaceAiSystem:
    def __init__(self,
                 dataPath="./knowledgeBase.json",
                 index_path="./knowledge.index",
                 embeddings_path="./knowledge_embeddings.npy",
                 inverted_index_path="./inverted_index.json"):
        """
        初始化系统，优先加载已存在的向量库，否则创建并保存。
        """
        print("--- 系统初始化开始 ---")
        start_time = time.time()  # 记录开始时间

        self.test_data = []
        self.model = None
        self.index = None
        self.embeddings = None
        self.initialized = False
        self.dataPath = dataPath
        self.index_path = index_path
        self.embeddings_path = embeddings_path
        self.inverted_index = defaultdict(set)
        self.inverted_index_path = inverted_index_path
        self.inverted_index = {}

        # --- 1. 加载文档 ---
        print("1. 正在加载文档...")
        try:
            with open(self.dataPath, "r", encoding="utf-8") as f:
                self.test_data = json.load(f)
            if not self.test_data or not isinstance(self.test_data, list):
                print(f"   错误：知识库文件 {self.dataPath} 为空或格式不正确。")
                return
            print(f"   导入成功，共 {len(self.test_data)} 条记录。")
        except FileNotFoundError:
            print(f"   错误：知识库文件 {self.dataPath} 未找到。")
            return
        except json.JSONDecodeError:
            print(f"   错误：知识库文件 {self.dataPath} 格式错误。")
            return

        # --- 2. 尝试加载现有向量库 ---
        print("2. 检查现有向量数据库...")
        load_success = False
        if os.path.exists(self.index_path) and os.path.exists(self.embeddings_path):
            print(f"   发现现有文件。正在加载 {self.index_path} 和 {self.embeddings_path} ...")
            try:
                self.index = faiss.read_index(self.index_path)
                self.embeddings = np.load(self.embeddings_path)
                # 校验数量是否匹配
                if self.index.ntotal == len(self.test_data) and self.embeddings.shape[0] == len(self.test_data):
                    print(f"   向量数据库加载成功，包含 {self.index.ntotal} 条向量。")
                    load_success = True
                else:
                    print(
                        f"   警告：向量库记录数 ({self.index.ntotal}) 与知识库 ({len(self.test_data)}) 不匹配，将重新创建。")
            except Exception as e:
                print(f"   错误：加载向量数据库失败: {e}。将重新创建。")
        else:
            print("   未发现现有向量数据库，将创建新的。")

        # --- 3. 加载 Embedding 模型 (无论如何都需要加载，用于查询) ---
        print("3. 正在加载 embedding 模型...")
        try:
            self.model = SentenceTransformer('shibing624/text2vec-base-chinese')
            print("   模型加载成功。")
        except Exception as e:
            print(f"   错误：加载 embedding 模型失败: {e}")
            return  # 如果模型加载失败，则无法继续

        # --- 4. 如果加载失败，则创建并保存 ---
        if not load_success:
            print("4. 正在创建并保存新的向量数据库...")
            if not self._build_and_save_index():
                print("   创建向量数据库失败，系统初始化中止。")
                return  # 如果创建失败，则中止

        # --- 5. 加载或构建倒排索引 ---
        print("5. 正在加载或构建倒排索引...")
        if os.path.exists(self.inverted_index_path):
            self._load_inverted_index()
            print(f"   倒排索引加载成功，包含 {len(self.inverted_index)} 个关键词。")
        else:
            self._build_and_save_inverted_index()
            print(f"   倒排索引构建完成并保存，包含 {len(self.inverted_index)} 个关键词。")

        # --- 6. 设置初始化状态 ---
        self.initialized = True
        end_time = time.time()  # 记录结束时间
        print(f"--- 系统初始化完成 (耗时: {end_time - start_time:.2f} 秒) ---")

        if self.initialized:
            self._build_inverted_index()

    def _load_inverted_index(self):
        with open(self.inverted_index_path, "r", encoding="utf-8") as f:
            self.inverted_index = json.load(f)

    def _build_and_save_inverted_index(self):
        self.inverted_index = {}
        for idx, doc in enumerate(self.test_data):
            keywords = doc.get("keywords", [])
            content = doc.get("context", "")
            tokens = set(jieba.cut(" ".join(keywords) + " " + content))
            for word in tokens:
                word = word.strip().lower()
                if len(word) < 2:
                    continue
                self.inverted_index.setdefault(word, []).append(idx)
        with open(self.inverted_index_path, "w", encoding="utf-8") as f:
            json.dump(self.inverted_index, f, ensure_ascii=False)
    def _build_inverted_index(self):
        print("5. 正在构建倒排索引...")
        for idx, doc in enumerate(self.test_data):
            content = doc.get('context', '')
            keywords = doc.get('keywords', [])
            text = ' '.join(keywords) + ' ' + content
            tokens = self._extract_keywords_from_query(text)
            for token in tokens:
                self.inverted_index.setdefault(token, []).append(idx)
        print(f"   倒排索引构建完成，共索引关键词数: {len(self.inverted_index)}")

    def _build_and_save_index(self) -> bool:
        """
        内部方法：构建嵌入和索引，并将其保存到文件。
        返回 True 表示成功，False 表示失败。
        """
        if not self.test_data or not self.model:
            print("   错误：无法创建索引，数据或模型未加载。")
            return False

        try:
            print("   4a. 正在生成文档嵌入向量...")
            texts = [f"{d.get('keywords', [])}。{d.get('context', '')}" for d in self.test_data]
            self.embeddings = self.model.encode(
                texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
            )
            print("      嵌入向量生成完成。")

            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings.astype(np.float32))
            print(f"   4b. Faiss 索引构建完成，向量数: {self.index.ntotal}")

            print(f"   4c. 正在保存向量数据库到 {self.index_path} 和 {self.embeddings_path}...")
            faiss.write_index(self.index, self.index_path)
            np.save(self.embeddings_path, self.embeddings)
            print("      保存成功。")
            return True
        except Exception as e:
            print(f"      错误：创建或保存向量数据库失败: {e}")
            # 清理可能已创建的部分文件
            if os.path.exists(self.index_path): os.remove(self.index_path)
            if os.path.exists(self.embeddings_path): os.remove(self.embeddings_path)
            self.index = None
            self.embeddings = None
            return False

    def _extract_keywords_from_query(self, query: str) -> list:
        words = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', query.lower())
        stopwords = ["的", "了", "是", "和", "有", "什么", "如何", "怎么", "作用", "特点", "方法", "原因", "表现"]
        return [word for word in words if word not in stopwords and len(word) > 1]

    def _semantic_search(self, query: str, topk: int = 10) -> list:
        if not self.initialized: return []
        q_vec = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, idxs = self.index.search(q_vec, topk)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0: continue
            doc = self.test_data[idx]
            text = f"{doc.get('keywords', [])}。{doc.get('context', '')}"
            meta = {k: v for k, v in doc.items() if k not in ('keywords', 'context')}
            results.append({"idx": int(idx), "score": float(score), "text": text, "meta": meta})
        return results

    def _keyword_search(self, query: str, topk: int = 10) -> list:
        if not self.initialized or not self.inverted_index:
            return []

        query_words = self._extract_keywords_from_query(query)
        if not query_words:
            return []

        doc_scores = {}
        for word in query_words:
            word = word.lower()
            matched_doc_ids = self.inverted_index.get(word, [])
            for idx in matched_doc_ids:
                doc_scores[idx] = doc_scores.get(idx, 0) + 1  # 可加权处理

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in sorted_docs[:topk]:
            doc = self.test_data[idx]
            text = f"{' '.join(doc.get('keywords', []))}。{doc.get('context', '')}"
            meta = {k: v for k, v in doc.items() if k not in ('keywords', 'context')}
            results.append({"idx": idx, "score": score, "text": text, "meta": meta})
        return results
    def _hybrid_search(self, query: str, topk: int = 5, k_rrf: int = 60) -> list:
        if not self.initialized:
            print("错误：系统未成功初始化，无法搜索。")
            return []

        semantic_results = self._semantic_search(query, topk=topk * 2)
        keyword_results = self._keyword_search(query, topk=topk * 2)

        semantic_rank_map = {res['idx']: rank + 1 for rank, res in enumerate(semantic_results)}
        keyword_rank_map = {res['idx']: rank + 1 for rank, res in enumerate(keyword_results)}

        all_docs = {}
        for res in semantic_results + keyword_results:
            idx = res['idx']
            if idx not in all_docs:
                all_docs[idx] = {"text": res['text'], "meta": res['meta'], "idx": idx}

        fused_results = []
        for idx, doc_data in all_docs.items():
            rank_s = semantic_rank_map.get(idx)
            rank_k = keyword_rank_map.get(idx)
            score_s = (1 / (k_rrf + rank_s)) if rank_s else 0
            score_k = (1 / (k_rrf + rank_k)) if rank_k else 0
            doc_data['score'] = score_s + score_k
            fused_results.append(doc_data)

        return sorted(fused_results, key=lambda x: x['score'], reverse=True)[:topk]

    def retrieve_top_contexts(self, query: str, topk: int = 5) -> list[str]:
        print(f"\n正在为查询 '{query}' 检索 Top {topk} Contexts...")
        if not self.initialized:
            print("系统未初始化，无法检索。")
            return []
        hybrid_results = self._hybrid_search(query, topk=topk)
        contexts = [result['text'] for result in hybrid_results]
        print(f"检索完成，找到 {len(contexts)} 条 Context。")
        return contexts


def main():
    system = FaceAiSystem()
    while True:
        q = input("❓ 用户输入提问：").strip()
        # print("1. 正在生成回答…")
        results = system.retrieve_top_contexts(q,5)
        for result in results:
            print(result)

if __name__ == "__main__":
    main()