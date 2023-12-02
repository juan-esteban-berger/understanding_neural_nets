import streamlit as st
# Setup
import warnings
warnings.filterwarnings('ignore')

from utils.animate import animate

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.markdown("# Understanding Neural Networks")

st.write("Forward")

st.write("When writing the derivative of the loss, 2/N when subtracting y_pred - y, use -2/N when subtracting y - y_pred, bc that is the derivative.")

st.markdown("## Appendix A: Matrix Multiplication and Dot Product Review")

st.markdown("### Matrix Multiplication")
st.latex(r'''
\begin{equation}
\begin{bmatrix}
a & b & c \\
d & e & f
\end{bmatrix}
\times
\begin{bmatrix}
g & h \\
i & j \\
k & l
\end{bmatrix}
=
\begin{bmatrix}
ag + bi + ck & ah + bj + cl \\
dg + ei + fk & dh + ej + fl
\end{bmatrix}
\end{equation}
''')

st.markdown("### Dot Product")
st.latex(r'''
\begin{equation}
\mathbf{a} \cdot \mathbf{b} =
\begin{bmatrix}
a_1 \\
a_2 \\
\vdots \\
a_n
\end{bmatrix}
\cdot
\begin{bmatrix}
b_1 \\
b_2 \\
\vdots \\
b_n
\end{bmatrix}
= a_1b_1 + a_2b_2 + \cdots + a_nb_n
\end{equation}
''')
st.markdown("### Dot Product (Multiplying by Transpose)")
st.latex(r'''
\begin{equation}
\mathbf{a}^T \mathbf{b} =
\begin{bmatrix}
a_1 & a_2 & \cdots & a_n
\end{bmatrix}
\begin{bmatrix}
b_1 \\
b_2 \\
\vdots \\
b_n
\end{bmatrix}
= a_1b_1 + a_2b_2 + \cdots + a_nb_n
\end{equation}
''')
