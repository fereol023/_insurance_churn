from contents import *
from contents import page0, page1, page2, page3

# st.set_option('deprecation.showPyplotGlobalUse', False)

pages = {
    "Page 0 - Contexte": page0.main,
    "Page 1 - EDA": page1.main,
    "Page 2 - Modèle": page2.main,
    "Page 3 - Généralisation": page3.main
}

st.sidebar.title('Navigation')
p = st.sidebar.radio('Aller à  ', list(pages.keys()))

pages[p]()