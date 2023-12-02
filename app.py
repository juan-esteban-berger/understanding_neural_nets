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
st.write("By Juan Esteban Berger")

st.markdown("## Ordinary Least Squares with Gradient Descent")

st.markdown("### Linear Regression Model")

st.latex(r'''
y = w_1x_1 + w_2x_2 + \beta + \epsilon
''')

st.write("Furthermore, the computation graph for a multiple linear regression model can be visualized as follows:")

col1, col2, col3 = st.columns(3)
with col2:
    st.image("diagrams/01_Linear_Regression.png")

st.write("We can generate a data from the following equation:")

st.latex(r"y = 2x_1 - 3x_2 + 4 + \epsilon")

st.markdown("<br>", unsafe_allow_html = True)

# Generate Random Data
np.random.seed(42)
x1 = np.random.uniform(-10, 10, 100)
x2 = np.random.uniform(-10, 10, 100)

# Define the Plane y = sigmoid(2x1 - 3x2 + 4 + Noise)
noise = np.random.normal(0, 3, 100)
y = 2 * x1 - 3 * x2 + 4 + noise

# Preview Data
df = pd.DataFrame({
    'x1': x1,
    'x2': x2,
    'y': y
})

col1, col2 = st.columns(2)

with col1:
    st.dataframe(df)
with col2:
    # Create 3D Scatter Plot
    scatter = go.Scatter3d(x=df['x1'], y=df['x2'], z = df['y'], mode='markers')

    fig = go.Figure(data = [scatter])
    fig.update_layout(template="plotly_dark")
    fig.update_layout(margin=dict(t=0, b=50, l=0, r=0))
    # Adjust camera settings for default zoom and angle
    fig.update_layout(
        scene_camera=dict(
            eye=dict(x=2, y=2, z=2),  # Adjust eye position for zoom and angle
            center=dict(x=0, y=0, z=0)  # Center the camera view
        )
    )

    st.plotly_chart(fig, use_container_width=True)

st.write(r'''Gradient descent is an optimization algorithm used to find the minimum value of a function iteratively. In this case, we will use gradient descent in order to find the values for parameters $w_1$, $w_2$, and $\beta$ that minimize the mean squared error. This can be achieved with the following algorithm:

1. Initialize Random Weights and Biases.
2. Make a Prediction.
3. Calculate the error by subtracting your prediction from the true y.
4. Calculate the gradients (derivative of the loss with respect to each parameter).
5. Update the parameters by subtracting the product of a pre-defined learning rate and its respective gradient values.
6. Repeat steps 2 - 6 for a desired number of steps or until the error reaches a desired value.
<br>
The Mean Squared Error is given by:
$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - (wx_i + b))^2
$$

Differentiating this with respect to $w_1$ or $w_2$ yields:

$$
\frac{\partial}{\partial w} MSE = \frac{-2}{n} \sum_{i=1}^{n} x_i(y_i - (wx_i + b))
$$

Differentiating this with respect to $\beta$ yields:
$$
\frac{\partial}{\partial b} MSE = \frac{-2}{n} \sum_{i=1}^{n} (y_i - (wx_i + b))
$$
''', unsafe_allow_html=True)

st.write("We can find the set of parameters that minimize the sum of squared error by implementing gradient descent in python as demonstrated below:")

st.code('''
# Iterate over a predefined number of training steps
for i in range(num_steps):

    # Make Prediction
    y_pred = w1 * x1 + w2 * x2 + b
    
    # Calculate Error
    error = y - y_pred
    
    # Calculate Gradients
    N = len(y)
    grad_w1 = (2/N) * np.dot(error, x1)
    grad_w2 = (2/N) * np.dot(error, x2)
    grad_b = (2/N) * np.sum(error)
    
    # Update w1, w2, b using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
    b -= learning_rate * grad_b
'''
, language="python")

st.write("The animation below visualizes the use of gradient descent to find the plane that minimizes the sum of squared residuals across 30 training steps.")

# Set random seed for reproducibility
np.random.seed(42)

# Initialize Random Weights and Biases
mean = 0
std_dev = 0.25
w1 = np.random.normal(mean, std_dev)
w2 = np.random.normal(mean, std_dev)
b = np.random.normal(mean, std_dev)
learning_rate = 0.001

# Set Colorscale
colorscale_val = 'oranges'

#  Initialize a 3D scatter plot with no data points (x, y, z are empty lists) and sets the marker style.
scatter = go.Scatter3d(x=x1, y=x2, z=y, mode="markers")

# Initialize a 3D Surface Plot using meshgrids
x1_surface = np.linspace(-10, 10, 100)
x2_surface = np.linspace(-20, 20, 100)
x1_surface, x2_surface = np.meshgrid(x1_surface, x2_surface)
surface = go.Surface(z=w1*x1_surface + w2*x2_surface + b, x=x1_surface, y=x2_surface,
                     colorscale=colorscale_val,
                     showscale=False)

# Initialize Figure
fig = go.Figure(data = [scatter, surface])

# Perform Gradient Descent and Generate Frames for the Figure
frames = []
for k in range(30): 
    # Make Prediction
    y_pred = w1 * x1 + w2 * x2 + b

    # Calculate Error
    error = y - y_pred

    # Calculate Gradients
    N = len(y)
    grad_w1 = (-2/N) * np.dot(error, x1)
    grad_w2 = (-2/N) * np.dot(error, x2)
    grad_b = (-2/N) * np.sum(error)

    # Update w1, w2, b using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
    b -= learning_rate * grad_b

    # Create a new frame with the updated plane
    new_surface = go.Surface(z=w1*x1_surface + w2*x2_surface + b,
                             x=x1_surface,
                             y=x2_surface,
                             colorscale=colorscale_val)
    frames.append(go.Frame(data=[new_surface], traces=[1], name=f'frame{k}'))
fig.update(frames=frames)

fig = animate(fig, 250, x_range=(-10, 10), y_range=(-10, 10), z_range=(-40, 40)) 

st.plotly_chart(fig, use_container_width=True)

st.markdown('## Adding an Activation Function (i.e. Logistic Regression')

st.write(r'''
The next step in understanding Neural Networks is to understand how activation functions work. The simplest example of the use of an activation function is a Logistic Classification model. Given the following sigmoid function:
$$
\sigma(x) = \frac{1}{1+ e^{-x}}
$$
we can derive a Logistic Classification model from our previosly defined Linear Regression Model as follows:
$$
y = \sigma(w_1x_1 + w_2x_2 + \beta + \epsilon)
$$
$$
y = \frac{1}{1+e^{-(w_1x_1 + w_2x_2 + \beta + \epsilon)}}
$$
Furthermore, we can continue our interactive demonstration of neural networks by generating data from the following equation:
$$
y = \frac{1}{1+e^{-(2x_1 + -3x_2 + 4 + \epsilon)}}
$$
''', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html = True)

# Generate Random Data
np.random.seed(42)
x1 = np.random.uniform(-10, 10, 100)
x2 = np.random.uniform(-10, 10, 100)

# Define the Plane y = 2x1 - 3x2 + 4 with Noise
noise = np.random.normal(0, 3, 100)
y = 1 / (1 + np.exp(-(2 * x1 - 3 * x2 + 4 + noise)))

# Preview Data
df = pd.DataFrame({
    'x1': x1,
    'x2': x2,
    'y': y
})

col1, col2 = st.columns(2)

with col1:
    st.dataframe(df)
with col2:
    # Create 3D Scatter Plot
    scatter = go.Scatter3d(x=df['x1'], y=df['x2'], z = df['y'], mode='markers')

    fig = go.Figure(data = [scatter])
    fig.update_layout(template="plotly_dark")
    fig.update_layout(margin=dict(t=0, b=50, l=0, r=0))
    # Adjust camera settings for default zoom and angle
    fig.update_layout(
        scene_camera=dict(
            eye=dict(x=2, y=2, z=2),  # Adjust eye position for zoom and angle
            center=dict(x=0, y=0, z=0)  # Center the camera view
        )
    )

    st.plotly_chart(fig, use_container_width=True)

st.write("The animation below visualizes the use of gradient descent to find the plane that minimizes the sum of squared residuals across 150 training steps (of which only every 5th training step is shown).")

# Set random seed for reproducibility
np.random.seed(42)

# Initialize Random Weights and Biases
mean = 0
std_dev = 0.25
w1 = np.random.normal(mean, std_dev)
w2 = np.random.normal(mean, std_dev)
b = np.random.normal(mean, std_dev)
learning_rate = 0.001

# Set Colorscale
colorscale_val = 'oranges'

#  Initialize a 3D scatter plot with no data points (x, y, z are empty lists) and sets the marker style.
scatter = go.Scatter3d(x=x1, y=x2, z=y, mode="markers")

# Initialize a 3D Surface Plot using meshgrids
x1_surface = np.linspace(-10, 10, 100)
x2_surface = np.linspace(-20, 20, 100)
x1_surface, x2_surface = np.meshgrid(x1_surface, x2_surface)
surface = go.Surface(z=1/ (1 + np.exp(-(w1*x1_surface + w2*x2_surface + b))), x=x1_surface, y=x2_surface,
                     colorscale=colorscale_val,
                     showscale=False)

# Initialize Figure
fig = go.Figure(data = [scatter, surface])

# Perform Gradient Descent and Generate Frames for the Figure
frames = []
for k in range(150): 
    # Make Prediction
    y_pred = 1 / (1 + np.exp(-(w1 * x1 + w2 * x2 + b)))

    # Calculate Error
    error = y - y_pred

    # Calculate Gradients
    N = len(y)
    grad_w1 = (-2/N) * np.dot(error, x1)
    grad_w2 = (-2/N) * np.dot(error, x2)
    grad_b = (-2/N) * np.sum(error)

    # Update w1, w2, b using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
    b -= learning_rate * grad_b

    # Create a new frame with the updated plane
    new_surface = go.Surface(z=1/ (1 + np.exp(-(w1*x1_surface + w2*x2_surface + b))),
                             x=x1_surface,
                             y=x2_surface,
                             colorscale=colorscale_val)
    if k % 5 == 0:
        frames.append(go.Frame(data=[new_surface], traces=[1], name=f'frame{k}'))
fig.update(frames=frames)

fig = animate(fig, 250, x_range=(-10, 10), y_range=(-10, 10), z_range=(-0.5, 1.5)) 

st.plotly_chart(fig, use_container_width=True)
st.markdown('## Adding a Hidden Layer')

st.markdown('## Adding a Second Output')

st.markdown('## Deep Neural Network (Python)')

st.markdown('## Deep Neural Network (C++)')

st.markdown('## Deep Neural Network (CUDA)')

st.markdown("## Appendix A: Neural Network Terminology")

st.markdown("## Appendix B: Matrix Multiplication and Dot Product Review")

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

st.markdown("## Appendix C: Code for Plotly Expres Charts and Animations")
