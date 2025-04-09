import streamlit as st
from streamlit_image_select import image_select
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.row import row
import time
import random


'''
STREAMLIT DEMO APP
------------------
Note that all inference was done offline beforehand, this way no inference
if being done online causing bog downs
'''

# Little HTML hack to left align the latex in its parent container
st.markdown('''
<style>
.katex-html {
    text-align: left;
}
</style>''',
unsafe_allow_html=True
)

# Dictionary for data paths, labels and metrics
path_latex_pairs = {
    "static/diff.jpg": {"pred": "\\frac{d1}{dx}+\\frac{2}{5}y=0", 
                        "gt": "\\frac{dy}{dx}+\\frac{2}{5}y=0", 
                        "cer": 0.03571},
    "static/integral.jpg": {"pred": "\\int_{-\\infty}^{\\infty}e^{-x^{2}}dx", 
                            "gt": "\\int_{-\\infty}^{\\infty}e^{-x^{2}}dx", 
                            "cer": 0.00},
    "static/expectation.jpg": {"pred": "\\mathbb{E}[X]=\\sum_{i}x_{i}P(X=x)",
                            "gt": "\\mathbb{E}[X]=\\sum_{i}x_{i}P(X=x_{i})",  
                            "cer": 0.1081}
}

# ---------------------------------------------
# App begins
# ---------------------------------------------
st.set_page_config(page_title="latexify", page_icon="🧠", initial_sidebar_state="collapsed")
st.markdown(
    """
    # 🧠✨ LaTeXify

    Welcome to the demo! 👋 Explore how this model reads handwritten math and 
    turns it into clean LaTeX code — like magic, but with transformers 🪄
    """
)

# ---------------------------------------------
st.divider()
# ---------------------------------------------


st.markdown("""
<span style='font-size: 2rem;'>📷👇</span>
<span style='font-size: 1rem;'> Select a handwritten math sample below to see the results:</span>
""", unsafe_allow_html=True)

# Image selector, depending on image selected, correct latex and metrics are selected
img = image_select("", [im_path for im_path in path_latex_pairs.keys()])
latex_pred = path_latex_pairs[img]['pred']
latex_gt = path_latex_pairs[img]['gt']
cer_score = float(path_latex_pairs[img]['cer'])

# Pseudo inference, just to make it seem like something is happening, duration is randomly set between 1-2.5s
message_slot = st.empty()
my_bar = st.progress(0)
message_slot.markdown("""
<span style='font-size: 1.5rem;'>🤖</span>
<span style='font-size: 1rem;'> Robots are reading and translating those scribbles, one moment please.</span>
""", unsafe_allow_html=True)

duration = random.uniform(1, 2.5)
sleep_step = duration/100
for i in range(100):
    time.sleep(sleep_step)
    my_bar.progress(i+1)
time.sleep(0.75)
# Reset bar and message once loaded
my_bar.empty()
message_slot.empty()


# ---------------------------------------------
st.divider()
# ---------------------------------------------


# Creat columns to display results
latex_col, metric_col = st.columns([2, 1])  

with latex_col:
    st.markdown("### 🧾 Predicted LaTeX")
    st.latex(latex_pred)

with metric_col:
    st.markdown('### 🧮 CER Score')
    add_vertical_space(1)
    st.markdown(f"#### {cer_score:.1%}")

add_vertical_space(1)
st.markdown('### ⚖️ Ground Truth Comparison')
comparison = f'''
    % Prediction
    {latex_pred}
    % Truth
    {latex_gt}
'''
st.code(comparison, language="latex")

# Links section
add_vertical_space(2)
st.markdown("#### Explore more")
links_row = row(3, vertical_align="center")
links_row.link_button(
    "👨‍💻  My other projects",
    "https://tjoab.com",
    use_container_width=True,
)
links_row.link_button(
    "🐙  Visit the repository",
    "https://github.com/tjoab/latexify",
    use_container_width=True,
)
links_row.link_button(
    "🤗  Hugging Face",
    "https://huggingface.co/tjoab/latex_finetuned",
    use_container_width=True,
)
