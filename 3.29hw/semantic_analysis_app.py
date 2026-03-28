"""
语义分析综合测试平台 - Semantic Analysis Platform
====================================================
集成 TF-IDF/LSA、Word2Vec、GloVe、FastText/Sent2Vec 四大模块的交互式 Web 系统。
使用 Streamlit 框架构建，结合 gensim、scikit-learn、nltk 等库。
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec, FastText
from scipy.spatial.distance import cosine
import warnings
import re

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# NLTK data (quiet download)
# ---------------------------------------------------------------------------
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

for res_type, res_name in [("tokenizers", "punkt"), ("tokenizers", "punkt_tab"), ("corpora", "stopwords")]:
    try:
        nltk.data.find(f"{res_type}/{res_name}")
    except LookupError:
        nltk.download(res_name, quiet=True)

# ---------------------------------------------------------------------------
# Default corpus (~700 words)
# ---------------------------------------------------------------------------
DEFAULT_CORPUS = """Natural language processing is a subfield of linguistics and artificial intelligence.
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
BERT and GPT are examples of large language models based on the transformer architecture.
"""

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="语义分析综合测试平台", page_icon="📚", layout="wide")
st.title("📚 语义分析综合测试平台")
st.caption("Semantic Analysis Platform — TF-IDF / LSA / Word2Vec / GloVe / FastText / Sent2Vec")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 模块1: TF-IDF 与 LSA",
    "🧠 模块2: Word2Vec (CBOW / Skip-Gram)",
    "🌐 模块3: GloVe 词类比",
    "🔤 模块4: FastText & Sent2Vec",
])


# ===================================================================
# Helper: preprocess corpus into sentences and tokenised sentences
# ===================================================================
def preprocess_corpus(raw_text: str, remove_stopwords: bool = False):
    from nltk.corpus import stopwords
    stops = set(stopwords.words("english")) if remove_stopwords else set()
    sentences = sent_tokenize(raw_text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
    tokenized = [word_tokenize(s.lower()) for s in sentences]
    tokenized = [
        [w for w in tokens if re.match(r"^[a-z]+$", w) and w not in stops]
        for tokens in tokenized
    ]
    tokenized = [t for t in tokenized if len(t) > 0]
    return sentences, tokenized


# ===================================================================
# MODULE 1 — TF-IDF & LSA
# ===================================================================
with tab1:
    st.header("模块 1：传统统计模型 (TF-IDF 与 LSA)")
    st.markdown("""
    - **TF-IDF**：衡量词语在文档集合中的重要程度  
    - **LSA (Latent Semantic Analysis)**：利用 SVD 矩阵分解将高维词向量降维到潜在语义空间
    """)

    corpus_input_1 = st.text_area(
        "输入英文语料（每句话将作为一个文档）",
        value=DEFAULT_CORPUS, height=250, key="corpus1"
    )

    if st.button("运行 TF-IDF & LSA 分析", key="run_lsa"):
        sentences_1, _ = preprocess_corpus(corpus_input_1)
        if len(sentences_1) < 3:
            st.error("语料太短，请输入更多句子（至少 3 句）。")
        else:
            # --- TF-IDF ---
            st.subheader("1.1 TF-IDF 矩阵与关键词")
            tfidf_vec = TfidfVectorizer(stop_words="english", max_features=200)
            tfidf_matrix = tfidf_vec.fit_transform(sentences_1)
            feature_names = tfidf_vec.get_feature_names_out()

            avg_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            top5_idx = avg_tfidf.argsort()[-5:][::-1]
            top5_words = [(feature_names[i], round(avg_tfidf[i], 4)) for i in top5_idx]

            st.markdown("**TF-IDF 权重最高的 5 个关键词：**")
            kw_df = pd.DataFrame(top5_words, columns=["关键词", "平均 TF-IDF 权重"])
            kw_df.index = kw_df.index + 1
            st.table(kw_df)

            tfidf_dense = tfidf_matrix.toarray()
            show_df = pd.DataFrame(tfidf_dense, columns=feature_names,
                                   index=[f"Doc {i+1}" for i in range(len(sentences_1))])
            with st.expander("查看完整 TF-IDF 矩阵（部分）"):
                st.dataframe(show_df.iloc[:10, :20], use_container_width=True)

            # --- LSA (TF-IDF based) ---
            st.subheader("1.2 LSA 降维可视化 (基于 TF-IDF 矩阵)")
            st.markdown("""
            **原理**：将 TF-IDF 词-文档矩阵转置后，对每个词得到一个文档维度的向量，
            再用 `TruncatedSVD`（即 LSA）将其降维到 2 维进行可视化。
            同一篇文档中频繁共现的词语，其 TF-IDF 行向量模式相似，降维后会被映射到相近位置。
            """)

            lsa_basis = st.radio(
                "选择 LSA 输入矩阵",
                ["TF-IDF 矩阵", "One-hot (CountVectorizer) 矩阵"],
                key="lsa_basis", horizontal=True,
            )

            if "TF-IDF" in lsa_basis:
                lsa_matrix = tfidf_matrix
                lsa_vocab = feature_names
            else:
                count_vec = CountVectorizer(stop_words="english", max_features=200)
                count_matrix = count_vec.fit_transform(sentences_1)
                lsa_matrix = count_matrix
                lsa_vocab = count_vec.get_feature_names_out()

            n_components = min(2, lsa_matrix.shape[1] - 1, lsa_matrix.shape[0] - 1)
            if n_components < 2:
                st.warning("词汇或文档数量过少，无法降维到 2 维。")
            else:
                svd = TruncatedSVD(n_components=2, random_state=42)
                # Transpose: rows=words, cols=documents
                word_matrix_t = lsa_matrix.T.toarray().astype(float)
                word_2d = svd.fit_transform(word_matrix_t)

                st.markdown(f"SVD 解释方差比: 维度1={svd.explained_variance_ratio_[0]:.2%}, "
                            f"维度2={svd.explained_variance_ratio_[1]:.2%}, "
                            f"合计={svd.explained_variance_ratio_.sum():.2%}")

                # K-Means clustering for color coding
                from sklearn.cluster import KMeans
                n_clusters = min(5, len(lsa_vocab))
                km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = km.fit_predict(word_2d)

                cmap = plt.cm.get_cmap("tab10", n_clusters)

                fig, ax = plt.subplots(figsize=(12, 8))
                for cluster_id in range(n_clusters):
                    mask = labels == cluster_id
                    ax.scatter(word_2d[mask, 0], word_2d[mask, 1],
                               alpha=0.75, s=70, c=[cmap(cluster_id)],
                               label=f"Cluster {cluster_id + 1}")

                for i, w in enumerate(lsa_vocab):
                    ax.annotate(w, (word_2d[i, 0], word_2d[i, 1]),
                                fontsize=8, alpha=0.9,
                                xytext=(5, 5), textcoords="offset points")

                ax.set_xlabel("LSA 维度 1", fontsize=12)
                ax.set_ylabel("LSA 维度 2", fontsize=12)
                ax.set_title(f"LSA 词汇 2D 可视化 (基于{'TF-IDF' if 'TF-IDF' in lsa_basis else 'Count'}矩阵 SVD 分解，K-Means 聚类着色)",
                             fontsize=13)
                ax.legend(fontsize=9, loc="best")
                ax.grid(True, linestyle="--", alpha=0.35)
                fig.tight_layout()
                st.pyplot(fig)

                # Show cluster members
                st.markdown("**各聚类包含的词汇：**")
                cluster_info = {}
                for i, w in enumerate(lsa_vocab):
                    cid = int(labels[i]) + 1
                    cluster_info.setdefault(cid, []).append(w)
                for cid in sorted(cluster_info):
                    st.markdown(f"- **Cluster {cid}**: {', '.join(cluster_info[cid])}")

                st.info(
                    "🔍 **观察任务**：查看同一聚类中的词汇是否具有语义关联。例如：\n"
                    "- *word / embeddings / representation / vectors* 等「词向量」主题词聚在一起\n"
                    "- *neural / networks / deep / learning* 等「深度学习」主题词聚在一起\n"
                    "- *language / processing / natural* 等「NLP」主题词聚在一起\n\n"
                    "这说明 LSA 通过对共现矩阵做 SVD 分解，能将在相同文档中高频共现的词语（如主语与相关动词/宾语）"
                    "映射到潜在语义空间的相近位置。"
                )

            # --- LSA (Count-based) for comparison ---
            st.subheader("1.3 One-hot (CountVectorizer) 矩阵的 LSA 对比")
            st.markdown("对比 TF-IDF 和 One-hot 矩阵在 LSA 降维后的差异。Count 矩阵不区分词的全局重要性。")

            count_vec2 = CountVectorizer(stop_words="english", max_features=200)
            count_matrix2 = count_vec2.fit_transform(sentences_1)
            vocab2 = count_vec2.get_feature_names_out()

            nc2 = min(2, count_matrix2.shape[1] - 1, count_matrix2.shape[0] - 1)
            if nc2 >= 2:
                svd2 = TruncatedSVD(n_components=2, random_state=42)
                w2d_count = svd2.fit_transform(count_matrix2.T.toarray().astype(float))

                svd_tfidf = TruncatedSVD(n_components=2, random_state=42)
                w2d_tfidf = svd_tfidf.fit_transform(tfidf_matrix.T.toarray().astype(float))

                fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
                for ax, data, voc, title in [
                    (axes[0], w2d_tfidf, feature_names, "LSA on TF-IDF"),
                    (axes[1], w2d_count, vocab2, "LSA on Count (One-hot)"),
                ]:
                    ax.scatter(data[:, 0], data[:, 1], alpha=0.6, s=50, c="#4A90D9")
                    for i, w in enumerate(voc):
                        ax.annotate(w, (data[i, 0], data[i, 1]),
                                    fontsize=7, alpha=0.85,
                                    xytext=(3, 3), textcoords="offset points")
                    ax.set_title(title, fontsize=13)
                    ax.set_xlabel("LSA 维度 1")
                    ax.set_ylabel("LSA 维度 2")
                    ax.grid(True, linestyle="--", alpha=0.35)
                fig2.tight_layout()
                st.pyplot(fig2)

                st.info("🔍 **对比观察**：TF-IDF 矩阵通过 IDF 权重抑制了高频通用词的影响，"
                        "使得 LSA 降维后语义聚类更清晰；而 Count 矩阵保留了原始频次信息，"
                        "高频词可能主导降维结果。")


# ===================================================================
# MODULE 2 — Word2Vec (CBOW vs Skip-Gram)
# ===================================================================
with tab2:
    st.header("模块 2：Word2Vec 训练与对比 (CBOW vs Skip-Gram)")
    st.markdown("""
    - **CBOW**：通过上下文词预测目标词  
    - **Skip-Gram**：通过目标词预测上下文词  
    - 调整窗口大小、向量维度等参数，观察训练效果变化
    """)

    corpus_input_2 = st.text_area(
        "输入英文语料", value=DEFAULT_CORPUS, height=200, key="corpus2"
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        arch = st.radio("训练架构", ["CBOW (sg=0)", "Skip-Gram (sg=1)"], key="w2v_arch")
    with col_b:
        window_size = st.slider("上下文窗口 (window)", 2, 10, 5, key="w2v_win")
    with col_c:
        vec_dim = st.slider("向量维度 (vector_size)", 20, 200, 50, step=10, key="w2v_dim",
                            help="小语料建议 30-80 维；维度过高会导致所有词向量趋同")

    sg_flag = 0 if "CBOW" in arch else 1

    query_word = st.text_input("输入查询单词（小写），查找 Top-5 相似词", value="learning", key="w2v_query")

    if st.button("训练 Word2Vec 并查询", key="run_w2v"):
        _, tokenized_2 = preprocess_corpus(corpus_input_2, remove_stopwords=True)
        if len(tokenized_2) < 3:
            st.error("语料太短，请输入更多文本。")
        else:
            with st.spinner("正在训练 Word2Vec 模型（已自动过滤停用词）..."):
                model_w2v = Word2Vec(
                    sentences=tokenized_2,
                    vector_size=vec_dim,
                    window=window_size,
                    sg=sg_flag,
                    min_count=1,
                    epochs=300,
                    seed=42,
                    workers=1,
                )

            st.success(f"模型训练完成！词汇量: {len(model_w2v.wv)}（已过滤停用词）, "
                       f"架构: {'Skip-Gram' if sg_flag else 'CBOW'}, "
                       f"window={window_size}, vector_size={vec_dim}, epochs=300")

            qw = query_word.strip().lower()
            if qw in model_w2v.wv:
                similar = model_w2v.wv.most_similar(qw, topn=5)
                st.subheader(f"与 *{qw}* 最相似的 5 个词 (余弦相似度)")
                sim_df = pd.DataFrame(similar, columns=["词语", "余弦相似度"])
                sim_df.index = sim_df.index + 1
                sim_df["余弦相似度"] = sim_df["余弦相似度"].round(4)
                st.table(sim_df)
            else:
                available = sorted(model_w2v.wv.index_to_key[:20])
                st.warning(f"词汇 **{qw}** 不在词表中（停用词已被过滤）。可用词示例: {', '.join(available)}")

            # Store in session for module 4
            st.session_state["w2v_model"] = model_w2v


# ===================================================================
# MODULE 3 — GloVe (Pre-trained) & Word Analogies
# ===================================================================
with tab3:
    st.header("模块 3：预训练 GloVe 模型与词类比")
    st.markdown("""
    - 加载预训练 **GloVe** 模型 (`glove-wiki-gigaword-100`，基于 Wikipedia + Gigaword 语料，100 维，40 万词汇)  
    - 实现词类比运算：**A - B + C ≈ ?**  
    - 计算任意两词的语义相似度
    """)

    @st.cache_resource(show_spinner="正在加载 GloVe 预训练模型 (glove-wiki-gigaword-100)...")
    def load_glove():
        from gensim.models import KeyedVectors
        import os

        # Prefer the wiki-gigaword-100 model (better quality for analogies)
        for model_name in ["glove-wiki-gigaword-100", "glove-twitter-25"]:
            model_dir = os.path.expanduser(f"~/gensim-data/{model_name}")
            gz_path = os.path.join(model_dir, f"{model_name}.gz")
            if os.path.exists(gz_path):
                return KeyedVectors.load_word2vec_format(gz_path)

        import gensim.downloader as api
        return api.load("glove-wiki-gigaword-100")

    glove_model = load_glove()
    st.success(f"GloVe 模型加载完成！词汇量: {len(glove_model)}, 维度: {glove_model.vector_size}")

    # --- Word Analogy ---
    st.subheader("3.1 词类比 (Word Analogy): A - B + C = ?")
    st.markdown("经典示例：**king** − **man** + **woman** ≈ **queen**")

    col1, col2, col3 = st.columns(3)
    with col1:
        word_a = st.text_input("Word A", value="king", key="glove_a")
    with col2:
        word_b = st.text_input("Word B", value="man", key="glove_b")
    with col3:
        word_c = st.text_input("Word C", value="woman", key="glove_c")

    if st.button("计算词类比", key="run_analogy"):
        a, b, c = word_a.strip().lower(), word_b.strip().lower(), word_c.strip().lower()
        missing = [w for w in [a, b, c] if w not in glove_model]
        if missing:
            st.error(f"以下单词不在 GloVe 词表中: {', '.join(missing)}")
        else:
            result_vec = glove_model[a] - glove_model[b] + glove_model[c]
            similar = glove_model.most_similar(positive=[a, c], negative=[b], topn=5)

            st.markdown(f"**{a}** − **{b}** + **{c}** 的最近邻词：")
            analogy_df = pd.DataFrame(similar, columns=["词语", "余弦相似度"])
            analogy_df.index = analogy_df.index + 1
            analogy_df["余弦相似度"] = analogy_df["余弦相似度"].round(4)
            st.table(analogy_df)

    # --- More analogy examples ---
    st.subheader("3.2 更多词类比示例")
    examples = [
        ("paris", "france", "china", "国家→首都: 期望 beijing"),
        ("paris", "france", "germany", "国家→首都: 期望 berlin"),
        ("tokyo", "japan", "france", "国家→首都: 期望 paris"),
        ("london", "england", "japan", "国家→首都: 期望 tokyo"),
        ("man", "king", "woman", "性别类比: 期望 queen"),
        ("brother", "sister", "king", "性别类比: 期望 queen"),
        ("good", "better", "bad", "语法类比(比较级): 期望 worse"),
        ("slow", "slower", "fast", "语法类比(比较级): 期望 faster"),
        ("go", "went", "come", "语法类比(时态): 期望 came"),
    ]
    for a_ex, b_ex, c_ex, note in examples:
        with st.expander(f"🧪 {a_ex} − {b_ex} + {c_ex}    ({note})"):
            try:
                res = glove_model.most_similar(positive=[a_ex, c_ex], negative=[b_ex], topn=5)
                ex_df = pd.DataFrame(res, columns=["词语", "余弦相似度"])
                ex_df.index = ex_df.index + 1
                ex_df["余弦相似度"] = ex_df["余弦相似度"].round(4)
                st.table(ex_df)
            except KeyError as e:
                st.warning(f"词汇不在词表中: {e}")

    # --- Similarity ---
    st.subheader("3.3 词义相似度计算")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        sim_w1 = st.text_input("单词 1", value="happy", key="sim1")
    with col_s2:
        sim_w2 = st.text_input("单词 2", value="sad", key="sim2")

    if st.button("计算相似度", key="run_sim"):
        w1, w2 = sim_w1.strip().lower(), sim_w2.strip().lower()
        if w1 not in glove_model or w2 not in glove_model:
            st.error("输入的单词不在 GloVe 词表中。")
        else:
            score = glove_model.similarity(w1, w2)
            st.metric(label=f"'{w1}' 与 '{w2}' 的余弦相似度", value=f"{score:.4f}")


# ===================================================================
# MODULE 4 — FastText & Sent2Vec
# ===================================================================
with tab4:
    st.header("模块 4：FastText 子词特征与句向量 (Sent2Vec)")
    st.markdown("""
    - **FastText**：基于字符 n-gram，能处理未登录词 (OOV)  
    - **Sent2Vec (简单实现)**：对句子中所有词向量取平均 (Average Pooling)，计算句子相似度
    """)

    corpus_input_4 = st.text_area(
        "输入英文语料（用于训练 FastText）", value=DEFAULT_CORPUS, height=200, key="corpus4"
    )

    ft_vec_dim = st.slider("FastText 向量维度", 10, 100, 30, step=5, key="ft_dim",
                           help="小语料建议使用较低维度 (20-50)，否则向量区分度不足")

    if st.button("训练 FastText 模型", key="run_ft"):
        _, tokenized_4 = preprocess_corpus(corpus_input_4, remove_stopwords=True)
        if len(tokenized_4) < 3:
            st.error("语料太短。")
        else:
            with st.spinner("正在训练 FastText 模型（已自动过滤停用词）..."):
                ft_model = FastText(
                    sentences=tokenized_4,
                    vector_size=ft_vec_dim,
                    window=5,
                    min_count=1,
                    min_n=3,
                    max_n=6,
                    epochs=300,
                    seed=42,
                    workers=1,
                )
            st.success(f"FastText 模型训练完成！词汇量: {len(ft_model.wv)}（已过滤停用词）, 维度: {ft_vec_dim}")
            st.session_state["ft_model"] = ft_model
            st.session_state["ft_tokenized"] = tokenized_4

            if "w2v_model" not in st.session_state:
                w2v_cmp = Word2Vec(
                    sentences=tokenized_4, vector_size=ft_vec_dim, window=5,
                    min_count=1, epochs=300, seed=42, workers=1
                )
                st.session_state["w2v_model"] = w2v_cmp

    # --- OOV Test ---
    st.subheader("4.1 OOV（未登录词）对比测试")
    st.markdown("FastText 利用字符级 n-gram 为未见过的词也能生成向量，而 Word2Vec 无法处理。")

    oov_word = st.text_input("输入一个拼写错误的词（如 computeer、languge、nework）",
                             value="computeer", key="oov_input")

    if st.button("测试 OOV 处理", key="run_oov"):
        oov = oov_word.strip().lower()

        if "ft_model" not in st.session_state:
            st.warning("请先在上方训练 FastText 模型。")
        else:
            ft_m = st.session_state["ft_model"]
            w2v_m = st.session_state.get("w2v_model")

            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("**Word2Vec 结果：**")
                try:
                    if w2v_m and oov in w2v_m.wv:
                        sim_w2v = w2v_m.wv.most_similar(oov, topn=5)
                        st.write(pd.DataFrame(sim_w2v, columns=["词语", "相似度"]))
                    else:
                        raise KeyError(oov)
                except KeyError:
                    st.error(f"❌ Word2Vec: KeyError — '{oov}' 是未登录词 (OOV)，无法获取向量！")

            with col_right:
                st.markdown("**FastText 结果：**")
                try:
                    sim_ft = ft_m.wv.most_similar(oov, topn=5)
                    st.success(f"✅ FastText 成功为 '{oov}' 生成向量！（利用子词 n-gram 特征）")
                    ft_df = pd.DataFrame(sim_ft, columns=["词语", "相似度"])
                    ft_df.index = ft_df.index + 1
                    ft_df["相似度"] = ft_df["相似度"].round(4)
                    st.table(ft_df)
                except Exception as e:
                    st.error(f"FastText 也无法处理: {e}")

    # --- Sent2Vec ---
    st.subheader("4.2 句向量相似度 (Sent2Vec — Average Pooling)")
    st.markdown("""
    将句子中所有词的向量取均值 (Average Pooling)，模拟课件 P81 的 Sent2Vec 基本思想，计算句子间余弦相似度。
    
    > **注意**：提供两种词向量来源供对比——  
    > - **FastText (小语料训练)**：由上方训练的模型提供，因语料量有限区分度可能较低  
    > - **GloVe (预训练)**：基于海量 Twitter 语料训练的 25 维向量，语义区分度更好
    """)

    from nltk.corpus import stopwords
    _stops = set(stopwords.words("english"))

    vec_source = st.radio(
        "选择词向量来源",
        ["GloVe 预训练向量 (推荐)", "FastText 小语料训练向量"],
        key="sent_vec_source",
    )

    sent1 = st.text_input("句子 1",
                          value="Natural language processing uses deep learning models.",
                          key="sent1")
    sent2 = st.text_input("句子 2",
                          value="Deep neural networks are used in NLP applications.",
                          key="sent2")
    sent3_ref = st.text_input("句子 3（对照句，可选）",
                              value="The weather is sunny and warm today.",
                              key="sent3")

    if st.button("计算句子相似度", key="run_sent"):
        use_glove = "GloVe" in vec_source

        if use_glove:
            from gensim.models import KeyedVectors
            import os

            @st.cache_resource(show_spinner="正在加载 GloVe 向量用于句向量计算...")
            def _load_glove_for_sent():
                for name in ["glove-wiki-gigaword-100", "glove-twitter-25"]:
                    gz = os.path.expanduser(f"~/gensim-data/{name}/{name}.gz")
                    if os.path.exists(gz):
                        return KeyedVectors.load_word2vec_format(gz)
                import gensim.downloader as api
                return api.load("glove-wiki-gigaword-100")

            _kv = _load_glove_for_sent()
        else:
            if "ft_model" not in st.session_state:
                st.warning("请先在上方训练 FastText 模型。")
                st.stop()
            _kv = st.session_state["ft_model"].wv

        def sentence_vector(sentence: str, kv):
            tokens = [w.lower() for w in word_tokenize(sentence)
                      if re.match(r"^[a-z]+$", w.lower()) and w.lower() not in _stops]
            if not tokens:
                return np.zeros(kv.vector_size), tokens
            vecs = [kv[t] for t in tokens if t in kv]
            if not vecs:
                return np.zeros(kv.vector_size), tokens
            return np.mean(vecs, axis=0), tokens

        def safe_cosine_sim(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na == 0 or nb == 0:
                return 0.0
            return float(np.dot(a, b) / (na * nb))

        v1, t1 = sentence_vector(sent1, _kv)
        v2, t2 = sentence_vector(sent2, _kv)

        cos_sim_12 = safe_cosine_sim(v1, v2)

        st.markdown("---")
        st.markdown(f"**句子 1**: {sent1}")
        st.markdown(f"  内容词: `{t1}`")
        st.markdown(f"**句子 2**: {sent2}")
        st.markdown(f"  内容词: `{t2}`")
        st.metric("句子 1 ↔ 句子 2 余弦相似度", f"{cos_sim_12:.4f}")

        if sent3_ref.strip():
            v3, t3 = sentence_vector(sent3_ref, _kv)
            cos_sim_13 = safe_cosine_sim(v1, v3)
            cos_sim_23 = safe_cosine_sim(v2, v3)

            st.markdown(f"**句子 3 (对照)**: {sent3_ref}")
            st.markdown(f"  内容词: `{t3}`")

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("句子 1 ↔ 句子 3 余弦相似度", f"{cos_sim_13:.4f}")
            with col_m2:
                st.metric("句子 2 ↔ 句子 3 余弦相似度", f"{cos_sim_23:.4f}")

            if cos_sim_12 > cos_sim_13 and cos_sim_12 > cos_sim_23:
                st.success("🔍 **观察**：语义相关的句子 1 和句子 2 相似度最高 "
                           f"({cos_sim_12:.4f})，而与无关的句子 3 相似度较低 "
                           f"({cos_sim_13:.4f}, {cos_sim_23:.4f})。"
                           "这验证了词向量均值法 (Average Pooling) 能在一定程度上捕捉句子级语义。")
            else:
                st.info("💡 **提示**：小语料训练的向量区分度有限。"
                        "建议切换到 'GloVe 预训练向量' 以获得更好的语义区分效果。")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📖 使用说明")
    st.markdown("""
    1. **模块 1** — 输入语料 → 分析 TF-IDF 关键词 → LSA 降维可视化  
    2. **模块 2** — 选择 CBOW/Skip-Gram → 训练 Word2Vec → 查询相似词  
    3. **模块 3** — 使用预训练 GloVe 模型 → 词类比与相似度计算  
    4. **模块 4** — 训练 FastText → OOV 对比测试 → 句向量相似度  
    """)
    st.markdown("---")
    st.markdown("### ⚙️ 技术栈")
    st.markdown("""
    - `Streamlit` — Web 界面  
    - `scikit-learn` — TF-IDF, LSA (TruncatedSVD)  
    - `gensim` — Word2Vec, FastText, GloVe  
    - `nltk` — 分词  
    - `matplotlib` — 可视化  
    """)
    st.markdown("---")
    st.caption("语义分析综合测试平台 v1.0")
