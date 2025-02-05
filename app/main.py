from contents import *
from contents import page1, page2, page3

# st.set_option('deprecation.showPyplotGlobalUse', False)

pages = {
    "Page 1 - EDA": page1.main,
    "Page 2 - Modèle 1": page2.main,
    "Page 3 - Modèle 2": page3.main
}

st.sidebar.title('Navigation')
p = st.sidebar.radio('Aller à  ', list(pages.keys()))

pages[p]()