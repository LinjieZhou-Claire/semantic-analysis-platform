import streamlit as st
import spacy
from spacy import displacy
import nltk
from nltk import Tree
import benepar
import os
import subprocess
import sys

# 设置页面配置
st.set_page_config(page_title="自然语言处理：句法分析器可视化", layout="wide")

st.title("🌳 句法分析器可视化 (Syntax Parser Visualization)")
st.markdown("这是一个用于同时展示一句话的 **核心依存句法 (Dependency Parsing)** 和 **成分句法 (Constituent Parsing)** 的 Web 原型。")

@st.cache_resource(show_spinner="正在加载语言模型，首次运行可能需要下载，请稍候...")
def load_models():
    """
    加载并初始化所需的语言模型。如果本地没有，则自动下载。
    """
    # 1. 确保安装并加载 spaCy 的英文小模型
    model_name = "en_core_web_sm"
    try:
        nlp_spacy = spacy.load(model_name)
    except OSError:
        st.warning(f"未找到 {model_name}，正在尝试自动下载...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
        nlp_spacy = spacy.load(model_name)
    
    # 2. 确保下载 benepar 模型
    benepar_model = 'benepar_en3'
    try:
        benepar.download(benepar_model)
    except Exception as e:
        st.warning(f"下载 benepar 模型时出现提示（可能是已存在或网络波动）：{e}")
        
    # 3. 将 benepar 组件添加到 spaCy pipeline 中用于成分句法分析
    # 注意：如果加载同一个nlp对象可能会有冲突或者变得极其缓慢，我们重新加载一个专门用于benepar
    try:
         nlp_benepar = spacy.load(model_name)
    except OSError:
         nlp_benepar = spacy.load(model_name)
         
    # 检查是否已经添加了 benepar
    if "benepar" not in nlp_benepar.pipe_names:
        nlp_benepar.add_pipe("benepar", config={"model": benepar_model})
        
    return nlp_spacy, nlp_benepar

# 加载模型
nlp_spacy, nlp_benepar = load_models()

# 页面顶部的输入框，使用经典例句作为默认值
default_text = "The boy saw the man with the telescope."
text_input = st.text_input("请输入要分析的英文句子：", value=default_text)

if text_input:
    # 将页面展示方式从并排 (columns) 改为标签页 (tabs)，让每个图表都有充足的横向空间
    tab1, tab2 = st.tabs(["🔗 依存关系 (Dependency)", "🌿 成分结构 (Constituent)"])
    
    with tab1:
        st.subheader("依存句法树")
        st.markdown("使用 **spaCy** 和 **displaCy** 渲染")
        
        # 使用 spaCy 处理文本
        doc_dep = nlp_spacy(text_input)
        
        # 使用 displacy 生成 SVG 格式的 HTML
        # options 可以调整图表的样式，如距离、颜色等。为了防止拥挤，可以增大 distance，并设置紧凑/宽屏参数
        html_dep = displacy.render(doc_dep, style="dep", jupyter=False, options={'distance': 140, 'compact': False, 'word_spacing': 45})
        
        # 在 Streamlit 中渲染 HTML/SVG
        # 使用 components.html 如果内容被截断可以增大 width 或者允许自适应滚动
        html_wrapper = f"""
        <div style="overflow-x: auto; width: 100%; white-space: nowrap; padding-bottom: 20px;">
            {html_dep}
        </div>
        """
        st.components.v1.html(html_wrapper, height=450, scrolling=True)
        
    with tab2:
        st.subheader("成分句法树")
        st.markdown("使用 **benepar** (Berkeley Neural Parser) 生成")
        
        try:
            # 使用包含 benepar 的 spaCy pipeline 处理文本
            doc_const = nlp_benepar(text_input)
            
            # 提取第一句话的成分句法树（如果有多个句子，这里只展示第一个）
            sent = list(doc_const.sents)[0]
            parse_string = sent._.parse_string
            
            # 使用 nltk 读取解析字符串
            tree = Tree.fromstring(parse_string)
            
            # 尝试使用 svgling 把树画成漂亮的 SVG（如果安装了的话）
            try:
                import svgling
                # 生成 SVG 对象，直接调用不加 global_font_size 参数以兼容各个版本的 svgling
                svg_tree = svgling.draw_tree(tree)
                # 获取 SVG 的字符串表示
                svg_xml = svg_tree.get_svg().get_xml()
                if hasattr(svg_xml, "decode"): # bytes 转 string
                     svg_str = svg_xml.decode('utf-8')
                else:
                     # 对于某些 xml 库返回的 ET 对象，可能需要序列化
                     import xml.etree.ElementTree as ET
                     svg_str = ET.tostring(svg_xml, encoding='unicode')
                     
                # 包装一下 SVG 字符串使其能在 Streamlit 中良好显示，增加内外边距
                svg_html = f"""
                <div style="background-color: white; padding: 20px; border-radius: 5px; overflow-x: auto; width: 100%; display: flex; justify-content: center;">
                    {svg_str}
                </div>
                """
                st.components.v1.html(svg_html, height=450, scrolling=True)
                
            except ImportError:
                # 如果没有安装 svgling，退回到美观的文本缩进显示
                st.info("提示：您可以安装 `svgling` (pip install svgling) 以获取更美观的图形化树状图。现在采用文本缩进展示。")
                st.code(tree.pformat(), language="text")
                
        except Exception as e:
            st.error(f"成分句法分析时出现错误：{e}")
            st.info("可能是模型正在下载中或者未正确安装 tensorflow/torch 依赖。")

    # 🎉 第二轮进化：增加核心论元提取器
    st.divider() # 添加一条分割线
    st.subheader("🎯 核心论元提取器 (Core Arguments Extractor)")
    st.markdown("通过遍历 spaCy 的依存句法树，提取句子的主干信息（如主语、宾语、谓语动词等），这对于信息抽取和知识图谱构建至关重要。")
    
    # 提取核心论元
    extracted_args = []
    for token in doc_dep:
        # 寻找根节点 (Root，通常是句子的核心谓语动词)
        if token.dep_ == "ROOT":
            extracted_args.append({"词 (Token)": token.text, "关系 (Dependency)": token.dep_, "说明 (Description)": "句子根节点 (核心谓语)"})
        # 寻找名词性主语 (nsubj)
        elif token.dep_ in ["nsubj", "nsubjpass"]:
            extracted_args.append({"词 (Token)": token.text, "关系 (Dependency)": token.dep_, "说明 (Description)": "名词性主语"})
        # 寻找直接宾语 (dobj)
        elif token.dep_ == "dobj":
            extracted_args.append({"词 (Token)": token.text, "关系 (Dependency)": token.dep_, "说明 (Description)": "直接宾语"})
        # 寻找介词宾语 (pobj)
        elif token.dep_ == "pobj":
            extracted_args.append({"词 (Token)": token.text, "关系 (Dependency)": token.dep_, "说明 (Description)": "介词宾语"})
            
    # 展示提取结果
    if extracted_args:
        # 使用 Streamlit 的 dataframe 渲染漂亮的表格
        st.dataframe(extracted_args, use_container_width=True)
    else:
        st.info("未提取到标准的主谓宾结构。")
