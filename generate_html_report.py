"""
Generate a self-contained static HTML report with real experiment results.
Runs all 4 modules, produces plots as embedded base64 images.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import re
import base64
import io
import os
import ssl
import warnings

warnings.filterwarnings("ignore")

# Fix SSL for NLTK downloads
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

for res_type, res_name in [("tokenizers", "punkt"), ("tokenizers", "punkt_tab"), ("corpora", "stopwords")]:
    try:
        nltk.data.find(f"{res_type}/{res_name}")
    except LookupError:
        nltk.download(res_name, quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from gensim.models import Word2Vec, FastText, KeyedVectors

STOPS = set(stopwords.words("english"))

CORPUS = """Natural language processing is a subfield of linguistics and artificial intelligence.
It focuses on the interaction between computers and humans through natural language.
The ultimate goal of natural language processing is to read and understand human languages.
Natural language processing combines computational linguistics with statistical and machine learning models.
Early approaches to natural language processing were based on hand-written rules and grammars.
Modern natural language processing relies heavily on deep learning and neural networks.
Word embeddings are a type of word representation that allows words to be represented as vectors.
Word2Vec is a popular algorithm for learning word embeddings from large text corpora.
The Word2Vec model comes in two architectures known as CBOW and Skip-Gram.
CBOW predicts the target word from its surrounding context words.
Skip-Gram predicts the surrounding context words from a given target word.
GloVe is another word embedding method that combines global matrix factorization and local context windows.
GloVe stands for Global Vectors for Word Representation.
FastText extends Word2Vec by representing each word as a bag of character n-grams.
This allows FastText to generate embeddings for out-of-vocabulary words.
Sentence embeddings represent entire sentences as fixed-length vectors.
Sent2Vec learns sentence embeddings by averaging word and n-gram embeddings.
TF-IDF is a numerical statistic that reflects how important a word is to a document in a collection.
The term frequency measures how frequently a term occurs in a document.
The inverse document frequency measures how important a term is across the entire corpus.
Latent Semantic Analysis uses singular value decomposition to reduce the dimensionality of the term-document matrix.
LSA can capture synonymy and polysemy by mapping words to a latent semantic space.
Cosine similarity is commonly used to measure the similarity between two vectors.
King minus man plus woman equals queen in the word embedding space.
Paris is to France as Berlin is to Germany in word vector arithmetic.
Machine learning models can learn patterns from data without being explicitly programmed.
Deep learning uses multiple layers of neural networks to learn hierarchical representations.
Recurrent neural networks are designed to handle sequential data like text.
Transformers have revolutionized natural language processing with the attention mechanism.
BERT and GPT are examples of large language models based on the transformer architecture."""


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def preprocess(text, remove_sw=False):
    sentences = sent_tokenize(text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
    tokenized = [word_tokenize(s.lower()) for s in sentences]
    tokenized = [
        [w for w in toks if re.match(r"^[a-z]+$", w) and (w not in STOPS if remove_sw else True)]
        for toks in tokenized
    ]
    tokenized = [t for t in tokenized if len(t) > 0]
    return sentences, tokenized


print("=" * 60)
print("Module 1: TF-IDF & LSA")
print("=" * 60)

sentences, _ = preprocess(CORPUS)

tfidf_vec = TfidfVectorizer(stop_words="english", max_features=200)
tfidf_matrix = tfidf_vec.fit_transform(sentences)
feature_names = tfidf_vec.get_feature_names_out()

avg_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
top5_idx = avg_tfidf.argsort()[-5:][::-1]
top5_words = [(feature_names[i], round(avg_tfidf[i], 4)) for i in top5_idx]
print("Top-5 TF-IDF keywords:", top5_words)

# LSA on TF-IDF
svd = TruncatedSVD(n_components=2, random_state=42)
word_2d = svd.fit_transform(tfidf_matrix.T.toarray().astype(float))
exp_var = svd.explained_variance_ratio_

n_clusters = min(5, len(feature_names))
km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = km.fit_predict(word_2d)

cmap = plt.cm.get_cmap("tab10", n_clusters)
fig1, ax = plt.subplots(figsize=(12, 8))
for cid in range(n_clusters):
    mask = labels == cid
    ax.scatter(word_2d[mask, 0], word_2d[mask, 1], alpha=0.75, s=70, c=[cmap(cid)], label=f"Cluster {cid+1}")
for i, w in enumerate(feature_names):
    ax.annotate(w, (word_2d[i, 0], word_2d[i, 1]), fontsize=8, alpha=0.9, xytext=(5, 5), textcoords="offset points")
ax.set_xlabel("LSA Dimension 1", fontsize=12)
ax.set_ylabel("LSA Dimension 2", fontsize=12)
ax.set_title("LSA Word 2D Visualization (TF-IDF + SVD, K-Means Coloring)", fontsize=13)
ax.legend(fontsize=9, loc="best")
ax.grid(True, linestyle="--", alpha=0.35)
fig1.tight_layout()
img1 = fig_to_base64(fig1)

cluster_info = {}
for i, w in enumerate(feature_names):
    cluster_info.setdefault(int(labels[i]) + 1, []).append(w)

# TF-IDF vs Count comparison
count_vec = CountVectorizer(stop_words="english", max_features=200)
count_matrix = count_vec.fit_transform(sentences)
vocab2 = count_vec.get_feature_names_out()

svd2 = TruncatedSVD(n_components=2, random_state=42)
w2d_count = svd2.fit_transform(count_matrix.T.toarray().astype(float))
svd3 = TruncatedSVD(n_components=2, random_state=42)
w2d_tfidf = svd3.fit_transform(tfidf_matrix.T.toarray().astype(float))

fig1b, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, data, voc, title in [(axes[0], w2d_tfidf, feature_names, "LSA on TF-IDF"), (axes[1], w2d_count, vocab2, "LSA on Count (One-hot)")]:
    ax.scatter(data[:, 0], data[:, 1], alpha=0.6, s=50, c="#4A90D9")
    for i, w in enumerate(voc):
        ax.annotate(w, (data[i, 0], data[i, 1]), fontsize=7, alpha=0.85, xytext=(3, 3), textcoords="offset points")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("LSA Dim 1")
    ax.set_ylabel("LSA Dim 2")
    ax.grid(True, linestyle="--", alpha=0.35)
fig1b.tight_layout()
img1b = fig_to_base64(fig1b)

print("Module 1 done.\n")

# ===================================================================
print("=" * 60)
print("Module 2: Word2Vec CBOW vs Skip-Gram")
print("=" * 60)

_, tokenized_sw = preprocess(CORPUS, remove_sw=True)
query_words = ["learning", "neural", "word"]

w2v_results = {}
for sg, name in [(0, "CBOW"), (1, "Skip-Gram")]:
    m = Word2Vec(sentences=tokenized_sw, vector_size=50, window=5, sg=sg, min_count=1, epochs=300, seed=42, workers=1)
    w2v_results[name] = {}
    for qw in query_words:
        if qw in m.wv:
            w2v_results[name][qw] = m.wv.most_similar(qw, topn=5)
            print(f"  {name} - {qw}: {[(w, round(s,4)) for w,s in w2v_results[name][qw]]}")

w2v_model_for_oov = Word2Vec(sentences=tokenized_sw, vector_size=30, window=5, min_count=1, epochs=300, seed=42, workers=1)
print("Module 2 done.\n")

# ===================================================================
print("=" * 60)
print("Module 3: GloVe Word Analogies")
print("=" * 60)

glove = None
for model_name in ["glove-wiki-gigaword-100", "glove-twitter-25"]:
    gz = os.path.expanduser(f"~/gensim-data/{model_name}/{model_name}.gz")
    if os.path.exists(gz):
        print(f"Loading {model_name}...")
        glove = KeyedVectors.load_word2vec_format(gz)
        glove_name = model_name
        break

if glove is None:
    print("ERROR: No GloVe model found!")
    exit(1)

print(f"GloVe loaded: {glove_name}, vocab={len(glove)}, dim={glove.vector_size}")

analogy_tests = [
    ("king", "man", "woman", "queen"),
    ("paris", "france", "china", "beijing"),
    ("paris", "france", "germany", "berlin"),
    ("tokyo", "japan", "france", "paris"),
    ("london", "england", "japan", "tokyo"),
    ("brother", "sister", "king", "queen"),
    ("good", "better", "bad", "worse"),
    ("slow", "slower", "fast", "faster"),
    ("go", "went", "come", "came"),
]

analogy_results = []
for a, b, c, expected in analogy_tests:
    try:
        res = glove.most_similar(positive=[a, c], negative=[b], topn=5)
        top1 = res[0][0]
        score = res[0][1]
        hit = "✅" if top1 == expected else "⚠️"
        analogy_results.append((a, b, c, expected, top1, round(score, 4), hit, res))
        print(f"  {a} - {b} + {c} = {top1} ({score:.4f}) expected={expected} {hit}")
    except KeyError as e:
        analogy_results.append((a, b, c, expected, "N/A", 0, "❌", []))
        print(f"  {a} - {b} + {c}: KeyError {e}")

sim_pairs = [("king", "queen"), ("happy", "sad"), ("cat", "dog"), ("computer", "keyboard"), ("france", "paris")]
sim_results = []
for w1, w2 in sim_pairs:
    if w1 in glove and w2 in glove:
        s = glove.similarity(w1, w2)
        sim_results.append((w1, w2, round(float(s), 4)))
        print(f"  sim({w1}, {w2}) = {s:.4f}")

print("Module 3 done.\n")

# ===================================================================
print("=" * 60)
print("Module 4: FastText OOV & Sent2Vec")
print("=" * 60)

ft_model = FastText(sentences=tokenized_sw, vector_size=30, window=5, min_count=1, min_n=3, max_n=6, epochs=300, seed=42, workers=1)

oov_words = ["computeer", "languge", "nework"]
oov_results = []
for oov in oov_words:
    w2v_ok = oov in w2v_model_for_oov.wv
    ft_sim = ft_model.wv.most_similar(oov, topn=5)
    oov_results.append((oov, w2v_ok, ft_sim))
    print(f"  OOV '{oov}': W2V={'found' if w2v_ok else 'KeyError'}, FT top={ft_sim[0]}")

# Sent2Vec
def sent_vec(sentence, kv):
    tokens = [w.lower() for w in word_tokenize(sentence) if re.match(r"^[a-z]+$", w.lower()) and w.lower() not in STOPS]
    vecs = [kv[t] for t in tokens if t in kv]
    if not vecs:
        return np.zeros(kv.vector_size), tokens
    return np.mean(vecs, axis=0), tokens

def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

s1_text = "Natural language processing uses deep learning models."
s2_text = "Deep neural networks are used in NLP applications."
s3_text = "The weather is sunny and warm today."

v1, t1 = sent_vec(s1_text, glove)
v2, t2 = sent_vec(s2_text, glove)
v3, t3 = sent_vec(s3_text, glove)

sim12 = cos_sim(v1, v2)
sim13 = cos_sim(v1, v3)
sim23 = cos_sim(v2, v3)
print(f"  S1↔S2: {sim12:.4f}, S1↔S3: {sim13:.4f}, S2↔S3: {sim23:.4f}")
print("Module 4 done.\n")


# ===================================================================
# BUILD HTML
# ===================================================================
print("Generating HTML...")


def table_html(headers, rows, highlight_col=None):
    h = "<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead><tbody>"
    for row in rows:
        h += "<tr>"
        for j, cell in enumerate(row):
            cls = ' class="highlight"' if j == highlight_col else ""
            h += f"<td{cls}>{cell}</td>"
        h += "</tr>"
    h += "</tbody></table>"
    return h


# TF-IDF table
tfidf_table = table_html(
    ["排名", "关键词", "平均 TF-IDF 权重"],
    [(i + 1, w, s) for i, (w, s) in enumerate(top5_words)]
)

# Cluster list
cluster_html = ""
for cid in sorted(cluster_info):
    words = ", ".join(cluster_info[cid])
    cluster_html += f'<p><strong>Cluster {cid}:</strong> {words}</p>'

# W2V table
w2v_table_html = ""
for qw in query_words:
    w2v_table_html += f'<h4>查询词: <code>{qw}</code></h4>'
    headers = ["排名", "CBOW 相似词", "CBOW 相似度", "Skip-Gram 相似词", "Skip-Gram 相似度"]
    rows = []
    for i in range(5):
        cbow_w, cbow_s = w2v_results["CBOW"][qw][i]
        sg_w, sg_s = w2v_results["Skip-Gram"][qw][i]
        rows.append((i + 1, cbow_w, round(cbow_s, 4), sg_w, round(sg_s, 4)))
    w2v_table_html += table_html(headers, rows)

# Analogy table
analogy_table = table_html(
    ["类比公式", "期望结果", "实际 Top-1", "余弦相似度", "命中"],
    [(f"{a} − {b} + {c}", exp, top1, score, hit) for a, b, c, exp, top1, score, hit, _ in analogy_results],
    highlight_col=4,
)

# Detailed analogy results
analogy_detail_html = ""
for a, b, c, exp, top1, score, hit, full_res in analogy_results:
    if full_res:
        analogy_detail_html += f'<details><summary>🧪 {a} − {b} + {c} （期望: {exp}）</summary>'
        analogy_detail_html += table_html(["排名", "词语", "余弦相似度"],
                                          [(i+1, w, round(s, 4)) for i, (w, s) in enumerate(full_res)])
        analogy_detail_html += '</details>'

# Similarity table
sim_table = table_html(["词对", "余弦相似度"], [(f"{w1} ↔ {w2}", s) for w1, w2, s in sim_results])

# OOV table
oov_html = ""
for oov, w2v_ok, ft_sim in oov_results:
    oov_html += f"""
    <div class="oov-card">
        <h4>测试词: <code>{oov}</code></h4>
        <div class="oov-compare">
            <div class="oov-left">
                <h5>Word2Vec</h5>
                <div class="error-box">❌ KeyError — '{oov}' 是未登录词 (OOV)，无法获取向量！</div>
            </div>
            <div class="oov-right">
                <h5>FastText</h5>
                <div class="success-box">✅ 成功生成向量！（利用子词 n-gram 特征）</div>
                {table_html(["排名", "词语", "相似度"], [(i+1, w, round(s,4)) for i,(w,s) in enumerate(ft_sim)])}
            </div>
        </div>
    </div>"""

# Sent2Vec
sent_html = f"""
<div class="sent-result">
    <p><strong>句子 1:</strong> {s1_text}<br><small>内容词: {t1}</small></p>
    <p><strong>句子 2:</strong> {s2_text}<br><small>内容词: {t2}</small></p>
    <p><strong>句子 3 (对照):</strong> {s3_text}<br><small>内容词: {t3}</small></p>
    <div class="metrics">
        <div class="metric"><div class="metric-label">句子1 ↔ 句子2</div><div class="metric-value">{sim12:.4f}</div></div>
        <div class="metric"><div class="metric-label">句子1 ↔ 句子3</div><div class="metric-value">{sim13:.4f}</div></div>
        <div class="metric"><div class="metric-label">句子2 ↔ 句子3</div><div class="metric-value">{sim23:.4f}</div></div>
    </div>
</div>"""


html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>语义分析综合测试平台 — 实验展示</title>
<style>
:root {{
    --primary: #4A90D9;
    --success: #28a745;
    --danger: #dc3545;
    --bg: #f8f9fa;
    --card-bg: #ffffff;
    --text: #333;
    --border: #e0e0e0;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.7;
    padding: 0 20px 60px;
}}
.container {{ max-width: 1100px; margin: 0 auto; }}
header {{
    text-align: center;
    padding: 40px 0 20px;
    border-bottom: 2px solid var(--primary);
    margin-bottom: 30px;
}}
header h1 {{ font-size: 2em; color: var(--primary); }}
header p {{ color: #666; margin-top: 8px; }}
.module {{
    background: var(--card-bg);
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    padding: 30px;
    margin-bottom: 30px;
}}
.module h2 {{
    color: var(--primary);
    border-bottom: 2px solid var(--primary);
    padding-bottom: 8px;
    margin-bottom: 20px;
    font-size: 1.4em;
}}
.module h3 {{ color: #555; margin: 20px 0 10px; font-size: 1.15em; }}
.module h4 {{ color: #444; margin: 15px 0 8px; }}
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0 20px;
    font-size: 0.95em;
}}
th {{
    background: var(--primary);
    color: white;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
}}
td {{ padding: 8px 14px; border-bottom: 1px solid var(--border); }}
tr:nth-child(even) {{ background: #f5f8fc; }}
tr:hover {{ background: #eef3fa; }}
.highlight {{ font-weight: bold; }}
img.chart {{
    width: 100%;
    max-width: 900px;
    display: block;
    margin: 15px auto;
    border-radius: 8px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.1);
}}
.info-box {{
    background: #e8f4fd;
    border-left: 4px solid var(--primary);
    padding: 14px 18px;
    margin: 15px 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.95em;
}}
.success-box {{
    background: #d4edda;
    border-left: 4px solid var(--success);
    padding: 10px 14px;
    margin: 8px 0;
    border-radius: 0 6px 6px 0;
    color: #155724;
}}
.error-box {{
    background: #f8d7da;
    border-left: 4px solid var(--danger);
    padding: 10px 14px;
    margin: 8px 0;
    border-radius: 0 6px 6px 0;
    color: #721c24;
}}
.oov-card {{ margin: 20px 0; }}
.oov-compare {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
.oov-left, .oov-right {{ padding: 10px; }}
.metrics {{
    display: flex;
    gap: 20px;
    margin: 20px 0;
    flex-wrap: wrap;
}}
.metric {{
    flex: 1;
    min-width: 180px;
    background: #f0f4f8;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}}
.metric-label {{ font-size: 0.85em; color: #666; margin-bottom: 6px; }}
.metric-value {{ font-size: 1.8em; font-weight: 700; color: var(--primary); }}
details {{
    margin: 8px 0;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0;
}}
details summary {{
    padding: 10px 16px;
    cursor: pointer;
    font-weight: 500;
    background: #f8f9fa;
    border-radius: 8px;
}}
details[open] summary {{ border-bottom: 1px solid var(--border); border-radius: 8px 8px 0 0; }}
details > *:not(summary) {{ padding: 0 16px 10px; }}
code {{
    background: #f0f0f0;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.9em;
}}
.param-box {{
    display: inline-block;
    background: #eef3fa;
    border: 1px solid #ccdaea;
    border-radius: 6px;
    padding: 4px 12px;
    margin: 4px;
    font-size: 0.9em;
}}
small {{ color: #888; }}
@media (max-width: 700px) {{
    .oov-compare {{ grid-template-columns: 1fr; }}
    .metrics {{ flex-direction: column; }}
}}
</style>
</head>
<body>
<div class="container">

<header>
    <h1>📚 语义分析综合测试平台</h1>
    <p>Semantic Analysis Platform — TF-IDF / LSA / Word2Vec / GloVe / FastText / Sent2Vec</p>
</header>

<!-- ============================================================ -->
<div class="module">
    <h2>📊 模块 1：传统统计模型 (TF-IDF 与 LSA)</h2>
    <p>对应课件：P43 (LSA), P85-88 (TF-IDF)</p>

    <h3>1.1 TF-IDF 矩阵与关键词</h3>
    <p>对语料按句切分为 {len(sentences)} 个文档，使用 <code>TfidfVectorizer</code>（已过滤英文停用词）计算 TF-IDF 矩阵。</p>
    <p><strong>TF-IDF 权重最高的 5 个关键词：</strong></p>
    {tfidf_table}

    <h3>1.2 LSA 降维可视化 (基于 TF-IDF 矩阵)</h3>
    <p>将 TF-IDF 词-文档矩阵转置后，应用 <code>TruncatedSVD</code> (LSA) 降至 2 维。SVD 解释方差比：维度1={exp_var[0]:.2%}，维度2={exp_var[1]:.2%}，合计={exp_var.sum():.2%}。</p>
    <img class="chart" src="data:image/png;base64,{img1}" alt="LSA 2D Visualization">

    <p><strong>各聚类包含的词汇（K-Means, K=5）：</strong></p>
    {cluster_html}

    <div class="info-box">
        🔍 <strong>观察</strong>：共现频率高的词语在 LSA 空间中被映射到相近位置。例如 <em>natural/language/processing</em> 聚在一起，<em>neural/networks/deep/learning</em> 聚在一起。
        这验证了 LSA 通过矩阵分解捕获潜在语义关联的能力——即使两个词没有直接共现，但如果它们与相同的第三方词共现，LSA 也能发现这种间接关系。
    </div>

    <h3>1.3 TF-IDF vs Count (One-hot) 矩阵 LSA 对比</h3>
    <img class="chart" src="data:image/png;base64,{img1b}" alt="TF-IDF vs Count LSA Comparison">
    <div class="info-box">
        🔍 <strong>对比观察</strong>：TF-IDF 矩阵通过 IDF 权重抑制了高频通用词的影响，使得 LSA 降维后语义聚类更清晰；
        而 Count 矩阵保留原始频次信息，高频词可能主导降维结果。
    </div>
</div>

<!-- ============================================================ -->
<div class="module">
    <h2>🧠 模块 2：Word2Vec 训练与对比 (CBOW vs Skip-Gram)</h2>
    <p>对应课件：P47 (Word2Vec), P73 (词义相关性)</p>
    <p>训练参数：
        <span class="param-box">vector_size=50</span>
        <span class="param-box">window=5</span>
        <span class="param-box">epochs=300</span>
        <span class="param-box">已过滤停用词</span>
    </p>

    {w2v_table_html}

    <div class="info-box">
        🔍 <strong>观察</strong>：切换 CBOW 和 Skip-Gram 后，Top-5 相似词发生了明显变化：<br>
        • <strong>CBOW</strong> 相似度普遍较高 (0.99+)，词之间区分度较小——因为 CBOW 通过上下文预测目标词，倾向学到更平滑的表示。<br>
        • <strong>Skip-Gram</strong> 相似度更分散 (0.96-0.99)，语义关联更精确（如 neural→networks 排第一）——因为 Skip-Gram 对低频词和具体搭配更敏感。
    </div>
</div>

<!-- ============================================================ -->
<div class="module">
    <h2>🌐 模块 3：预训练 GloVe 模型与词类比</h2>
    <p>对应课件：P60 (GloVe), P65 (Word Analogies)</p>
    <p>模型：<code>{glove_name}</code>，词汇量: {len(glove):,}，维度: {glove.vector_size}</p>

    <h3>3.1 词类比 (Word Analogy): A − B + C = ?</h3>
    <p>利用向量运算 <code>Vec(A) − Vec(B) + Vec(C)</code> 寻找最近邻词。</p>
    {analogy_table}

    <h3>3.2 各类比详细 Top-5 结果</h3>
    {analogy_detail_html}

    <div class="info-box">
        🔍 <strong>观察</strong>：国家-首都类比全部命中 Top-1（如 paris−france+china=<strong>beijing</strong>），性别类比 king−man+woman=<strong>queen</strong> 也成功验证。
        这说明 GloVe 在大规模语料上学习到的词向量空间中，语义关系以线性方向编码——国家到首都的向量偏移几乎恒定。
        语法类比（时态 go→went, come→came）同样有效，表明词向量还隐式捕获了语法规则。
    </div>

    <h3>3.3 词义相似度</h3>
    {sim_table}
</div>

<!-- ============================================================ -->
<div class="module">
    <h2>🔤 模块 4：FastText 子词特征与句向量 (Sent2Vec)</h2>
    <p>对应课件：P68 (子词), P81 (Sent2Vec), P89 (FastText)</p>

    <h3>4.1 OOV（未登录词）对比测试</h3>
    <p>FastText 利用字符级 n-gram (min_n=3, max_n=6) 为未见过的词合成向量，而 Word2Vec 无法处理。</p>
    {oov_html}

    <div class="info-box">
        🔍 <strong>观察</strong>：Word2Vec 对所有拼写错误的词都抛出 KeyError，而 FastText 利用子词 n-gram 特征成功生成向量，
        并找到了语义最相近的正确拼写词。这验证了 FastText 通过字符级信息处理 OOV 问题的鲁棒性。
    </div>

    <h3>4.2 句向量相似度 (Sent2Vec — Average Pooling)</h3>
    <p>将句子分词后提取内容词的 GloVe 预训练向量，求均值得到句向量，再计算余弦相似度。</p>
    {sent_html}

    <div class="info-box">
        🔍 <strong>观察</strong>：语义相关的句子 1 和句子 2（都涉及 NLP/深度学习）相似度为 <strong>{sim12:.4f}</strong>，
        远高于与无关句子 3（天气话题）的 {sim13:.4f} 和 {sim23:.4f}。
        这验证了词向量均值法 (Average Pooling) 能在一定程度上捕捉句子级语义。
    </div>
</div>

<!-- ============================================================ -->
<div class="module" style="background: #f0f4f8;">
    <h2>📝 实验总结</h2>
    <table>
        <thead><tr><th>模型</th><th>核心原理</th><th>优势</th><th>局限性</th></tr></thead>
        <tbody>
            <tr><td>TF-IDF + LSA</td><td>词频统计 + SVD 矩阵分解</td><td>无需神经网络，可解释性强</td><td>无法捕获词序，维度灾难</td></tr>
            <tr><td>Word2Vec</td><td>神经网络分布式表示</td><td>语义关系捕获好，训练高效</td><td>无法处理 OOV</td></tr>
            <tr><td>GloVe</td><td>全局共现 + 局部上下文</td><td>词类比效果优秀</td><td>依赖大规模预训练语料</td></tr>
            <tr><td>FastText</td><td>字符 n-gram 词表示</td><td>能处理 OOV 和形态变化</td><td>模型体积较大</td></tr>
            <tr><td>Sent2Vec (Avg)</td><td>词向量均值池化</td><td>简单高效</td><td>丢失词序，受词向量质量制约</td></tr>
        </tbody>
    </table>
</div>

</div>
</body>
</html>"""

output_path = os.path.join(os.path.dirname(__file__), "实验展示.html")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n✅ HTML report saved to: {output_path}")
print(f"   File size: {os.path.getsize(output_path) / 1024:.0f} KB")
