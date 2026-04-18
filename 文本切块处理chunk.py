import json
import markdown
from bs4 import BeautifulSoup, Tag
from pathlib import Path
import re
from typing import List, Dict, Generator
import jieba
import jieba.analyse
import tempfile
import os


class FlatMarkdownProcessor:
    def __init__(self, chunk_size=500):
        self.chunk_size = chunk_size
        self._init_regex()
        self._init_jieba()
        self.current_chunk = []
        self.chunks = []

    def _init_regex(self):
        """初始化正则表达式"""
        self.keyword_pattern = re.compile(r"【(.*?)】|《(.*?)》")  # 显式关键词匹配
        self.split_pattern = re.compile(r"[\n。！？]+")  # 分句分隔符

    def _init_jieba(self):
        """初始化分词组件（修复停用词设置问题）"""
        jieba.initialize()

        # 创建临时停用词文件
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
            stopwords = {'的', '了', '和', '是', '就', '都', '而', '及', '与'}
            f.write('\n'.join(stopwords))
            temp_path = f.name

        jieba.analyse.set_stop_words(temp_path)
        os.unlink(temp_path)  # 删除临时文件

    def _extract_keywords(self, text: str) -> List[str]:
        """组合式关键词提取"""
        explicit_kws = list(set(
            [m for group in self.keyword_pattern.findall(text) for m in group if m]
        ))

        try:
            tfidf_kws = jieba.analyse.extract_tags(
                text,
                topK=10,
                allowPOS=('n', 'vn', 'eng')
            )
        except Exception:
            tfidf_kws = []

        return list(set(explicit_kws + tfidf_kws))[:10]

    def _flush_chunk(self):
        """将当前缓冲区内容生成一个知识块"""
        if self.current_chunk:
            full_text = '\n'.join(self.current_chunk)
            self.chunks.append({
                "context": full_text,
                "keywords": self._extract_keywords(full_text)
            })
            self.current_chunk = []

    def parse_markdown(self, file_path: str) -> List[Dict]:
        """解析Markdown生成平铺结构"""
        try:
            raw_text = Path(file_path).read_text(encoding='utf-8')
            html = markdown.markdown(raw_text)
            soup = BeautifulSoup(html, 'html.parser')

            for element in self._traverse_md_elements(soup):
                text_content = self._extract_element_text(element)

                # 按分句规则拆分
                sentences = self.split_pattern.split(text_content.strip())

                for sent in sentences:
                    if not sent:
                        continue

                    # 累积到当前块
                    if len('\n'.join(self.current_chunk + [sent])) > self.chunk_size:
                        self._flush_chunk()
                    self.current_chunk.append(sent)

            # 处理最后一块
            self._flush_chunk()
            return self.chunks

        except Exception as e:
            print(f"解析失败: {str(e)}")
            return []

    def _traverse_md_elements(self, soup) -> Generator[str, None, None]:
        """遍历有效Markdown元素"""
        for element in soup.descendants:
            if isinstance(element, Tag) and element.name in ['h1', 'h2', 'h3', 'h4', 'p', 'li']:
                yield element.get_text().strip()
            elif isinstance(element, Tag) and element.name == 'hr':
                yield '---'  # 用分隔符标识段落结束

    def _extract_element_text(self, element) -> str:
        """提取元素文本并添加语义标记"""
        tag_mapping = {
            'h1': '# ',
            'h2': '## ',
            'h3': '### ',
            'h4': '#### ',
            'li': '* '
        }
        if isinstance(element, str):
            return element
        prefix = tag_mapping.get(element.name, '')
        return f"{prefix}{element.get_text().strip()}"


if __name__ == "__main__":
    processor = FlatMarkdownProcessor(chunk_size=500)
    result = processor.parse_markdown("资料.txt")

    if result:
        with open("平铺知识库.json", "w", encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"生成成功，共处理 {len(result)} 个知识块")
        print("示例片段：")
        print(json.dumps(result[0], indent=2, ensure_ascii=False)[:500])
    else:
        print("未生成有效内容")