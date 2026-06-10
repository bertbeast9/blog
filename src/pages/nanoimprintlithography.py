import streamlit as st
import json
# import sys
# st.write(sys.path)
st.header ("Nano Imprint Lithography", divider=True)
st.sidebar.markdown("# Nano Imprint Lithography")
st.subheader("Problem Overview",divider=True)
st.image("./src/figures/NIL_Schematic.jpg", caption="A diagram of the nanoimprint lithography process [https://global.canon/en/technology/nil-2023.html]")
st.markdown("GIVE BREAKDOWN HERE")
st.subheader("Wafer-to-Wafer Process Control",divider=True)
st.image("./src/figures/Wafer_map.png",caption="Example of 300mm wafer with individual die shown")
st.markdown("On each wafer lie a number of die (or products), the number of which vary based on the product size. An example of a die layout on a wafer is shown above. For each die, the nanoimprint lithography tool must turn resin into a geometric pattern on each die (termed _field_) and precisely overlay each pattern onto the previous pattern. This distance between the target pattern location and the resulting pattern location is termed the _overlay error_. The sum of the overlay error across all layers of the product is termed the _stackup overlay error_. Both of these errors must be precisely controlled, else the device will not work.")
st.image("./src/figures/Overlay_Error_Notation.png",caption="Overlay error notation [Dr. Dragan Djurdjanovic (Time-Series Analysis)]")
st.markdown("There are two sets of coordinate systems: the _global coordinate system_ and the _local coordinate system_. To fully define the location of a marker on the wafer, one needs both the global coordinates $(X_i,Y_i)$ and the local coordinates $(x_i,y_i)$. The distance between the target marker and the realized marker in the $x$ and $y$ direction are termed $o_x(X_i,Y_i,x_i,y_i)$ and $o_y(X_i,Y_i,x_i,y_i)$, respectively. A nanoimprint lithography tool does not utilize the Zernike model since the physical process is very different. The controllable inputs into the tool are broken into three main categories: wafer stage inputs, template pin force inputs, and template heat inputs.")

st.subheader("Tool Inputs: Wafer Stage Inputs",divider=True)
st.markdown("*The wafer stage can be controlled in the $(x,y,\\theta)$ down to the micron-scale?* This allows for the tool to get within the ballpark of the targets before relaxing the template and immersing it into the resin. The template can also impart a downward force and tilt in both the $x$- and $y$-directions. Once immersed in the resin, the process can be modeled as a linear mapping from these three former inputs and their relationship to the resulting marker locations.")
st.latex("\\vec{u}^{(1)} \\in \\mathbb{R}^{3 \\text{x} 1} = \\left[ \\begin{matrix} \\text{template force [N]} \\newline \\text{tilt x-direction [$\\mu$rad]} \\newline  \\text{tilt y-direction [$\\mu$rad]} \\end{matrix} \\right] \\newline" \
"\\Phi^{(1)} \\in \\mathbb{R}^{2 n_m \\text{x} 3}")
st.latex("\\vec{o} = \\left[ \\begin{matrix} o_x(X_1,Y_1,x_1,y_1) \\newline o_y(X_1,Y_1,x_1,y_1) \\newline \\vdots \\newline o_y(X_{n_m},Y_{n_m},x_{n_m},y_{n_m}) \\end{matrix} \\right] = \\Phi^{(1)} \\vec{u}^{(1)} + \\vec{r}")

st.subheader("Tool Inputs: Template Pin Force Inputs",divider=True)
st.markdown("For further control over the resulting marker locations, the template can be deformed by 16 pin actuators, seen below. However, the following equations must satisfy the balance equations for static equilibrium.")
st.latex("\\vec{u}^{(2)} \\in \\mathbb{R}^{16 \\text{x} 1} = \\left[ \\begin{matrix} \\text{pin force 1 [N]} \\newline \\text{pin force 2 [N]} \\newline  \\vdots \\newline \\text{pin force 16 [N]} \\end{matrix} \\right] \\newline" \
"\\Phi^{(2)} \\in \\mathbb{R}^{2 n_m \\text{x} 16}")
st.latex("\\vec{o} = \\left[ \\begin{matrix} o_x(X_1,Y_1,x_1,y_1) \\newline o_y(X_1,Y_1,x_1,y_1) \\newline \\vdots \\newline o_y(X_{n_m},Y_{n_m},x_{n_m},y_{n_m}) \\end{matrix} \\right] = \\Phi^{(2)} \\vec{u}^{(2)} + \\vec{r}")
st.latex("\\Sigma F_x = 0 \\implies u_{13} + u_{14} + u_{15} + u_{16} - u_{5} - u_{6} - u_{7} - u_{8} = 0")
st.latex("\\Sigma F_y = 0 \\implies u_{9} + u_{10} + u_{11} + u_{12} - u_{1} - u_{2} - u_{3} - u_{4} = 0")
st.latex("\\Sigma M_0 = 0 \\implies -u_{1} l_1 - u_{2} l_2 + u_{3} l_3 + u{4} l_4 - u_{5} l_5 - u_{6} l_6 + u_{7} l_7 + u_{8} l_8 \\newline - u_9 l_9 - u_{10} l_{10} + u_{11} l_{11} + u_{12} l_{12} - u_{13} l_{13} - u_{14} l_{14} + u_{15} l_{15} + u_{16} l_{16} = 0")
st.latex("\\vec{u}^{(2)} \\leq b_u \\vec{1}")
st.image("./src/figures/Template_Pin_Forces.png", caption = "Diagram of the pin force actuators in relation to the mask, or template")


st.subheader("Tool Inputs: Template Heat Inputs",divider=True)
st.markdown("The template can further be deformed via a grid of laser inputs that will head the template. This control input is rather interesting in that each laser must pulse their pixel of the grid to achieve a desired temperature for the template. This model is likely not linear, but let's assume that we can linearize it.")
st.latex("\\vec{u}^{(3)} \\in \\mathbb{R}^{n_m \\text{x} 1} = \\left[ \\begin{matrix} \\text{grid point temp. 1 [\\degree C]} \\newline \\text{grid point temp. 2 [\\degree C]} \\newline  \\vdots \\newline \\text{grid point temp. $n_h$ [\\degree C]} \\end{matrix} \\right] \\newline" \
"\\Phi^{(3)} \\in \\mathbb{R}^{2 n_m \\text{x} n_m}")
st.latex("\\vec{o} = \\left[ \\begin{matrix} o_x(X_1,Y_1,x_1,y_1) \\newline o_y(X_1,Y_1,x_1,y_1) \\newline \\vdots \\newline o_y(X_{n_m},Y_{n_m},x_{n_m},y_{n_m}) \\end{matrix} \\right] = \\Phi^{(3)} \\vec{u}^{(3)} + \\vec{r}")

st.subheader("Questions/Requests for CNT", divider=True)
st.markdown("-Would like to know the order in which the fields were actually cured")
st.markdown("-How do you generate $\\vec{u}$ for the next field?")
st.markdown("-Do you have the true $\\vec{u}$ sent to the tool?")
st.markdown("-What are the bounds for all inputs, if any?")
st.markdown("-We would like the model for how the pin forces relate to the overlay errors?")
st.markdown("-How was outlier limit [20 nm] decided in the MatLab code? Based on physics or heuristics?")
st.markdown("-Is $\\Phi^{(1)}$ a global model for all fields on the wafer? How was this model identified?")
st.markdown("-What are the reasons for the edge markers generally having higher errors? Does weighting these outer markers' errors more make sense?")


st.markdown("Utilizing this model, we aim to model the bias terms and correct them, wafer-by-wafer. Ideally, with enough data, we would be able to utilize the full model below. This model considers the previous bias from the last field and the last wafer at the same field.")
st.latex("\\vec{o}_{i,j,k} \\sim \\text{overlay errors for wafer } i \\text{, layer } j \\text{, field } k")
st.latex("\\vec{\\chi}_{i,j,k} = A_1(\\vec{\\theta}) \\vec{\\chi}_{i,j-1,k} + A_2(\\vec{\\theta}) \\vec{\\chi}_{i,j,k-1} + \\vec{q}_{i,j,k} || \\vec{q}_{i,j,k} \\sim \\mathcal{N}(\\vec{0}, Q_{j,k}(\\vec{\\theta}))")
st.latex("\\vec{o}_{i,j,k} = \\vec{\\chi}_{i,j,k} + \\Phi_{i,j,k} \\vec{u}_{i,j,k} + \\vec{r}_{i,j,k} || \\vec{r}_{i,j,k} \\sim \\mathcal{N}(\\vec{0}, R_{j,k}(\\vec{\\theta}))")
st.markdown("However, I don't have access to that much data. So, I'll simplify the model.")
st.latex("\\vec{o}_{i} \\sim \\text{overlay errors for wafer } i")
st.latex("\\vec{\\chi}_{i} = A(\\vec{\\theta}) \\vec{\\chi}_{i-1} + \\vec{q}_{i} || \\vec{q}_{i} \\sim \\mathcal{N}(\\vec{0}, Q(\\vec{\\theta}))")
st.latex("\\vec{o}_{i} = \\vec{\\chi}_{i} + \\Phi \\vec{u}_{i} + \\vec{r}_{i} || \\vec{r}_{i} \\sim \\mathcal{N}(\\vec{0}, R(\\vec{\\theta}))")
st.markdown("We will assume that we have control over the inputs that will allow the following")
st.latex("\\Phi \\in \\mathbb{R}^{(2 \\text{x} n_m) \\text{x} 33} = \\left[ \\begin{matrix} 1 && 0 && X_1 && 0 && Y_1 && 0 && X_1^{2} && 0 && X_1 Y_1 && 0 && Y_1^{2} && 0 && X_1^{3} && 0 && X_1^{2} Y_1 && 0 && X_1 Y_1^{2} && 0 && Y_1^{3} && 0 && x_1 && 0 && y_1 && 0 && x_1^{2} && 0 && y_1^{2} && 0 && x_1^{3} && 0 && y_1^{3} && 0 && 0\\newline"
                                         "0 && 1 && 0 && X_1 && 0 && Y_1 && 0 && X_1^{2} && 0 && X_1 Y_1 && 0 && Y_1^{2} && 0 && X_1^{3} && 0 && X_1^{2} Y_1 && 0 && X_1 Y_1^{2} && 0 && Y_1^{3} && 0 && x_1 && 0 && y_1 && 0 && x_1^{2} && 0 && x_1 y_1 && 0 && y_1^{2} && 0 && x_1 y_1^{2} && y_1^{3}\\newline"
                                         " && && && && && && \\vdots \\newline"
                                         "0 && 1 && 0 && X_{n_m} && 0 && Y_{n_m} && 0 && X_{n_m}^{2} && 0 && X_{n_m} Y_{n_m} && 0 && Y_{n_m}^{2} && 0 && X_{n_m}^{3} && 0 && X_{n_m}^{2} Y_{n_m} && 0 && X_{n_m} Y_{n_m}^{2} && 0 && Y_{n_m}^{3} && 0 && x_{n_m} && 0 && y_{n_m} && 0 && x_{n_m}^{2} && 0 && x_{n_m} y_{n_m} && 0 && y_{n_m}^{2} && 0 && x_{n_m} y_{n_m}^{2} && y_{n_m}^{3}\\newline"
                                         "\\end{matrix} \\right]")
st.latex("\\vec{u}_i \\in \\mathbb{R}^{33 \\text{x} 1} = \\left[ \\begin{matrix} C_{x,X_i^{0} Y_i^{0}} \\newline  C_{y,X_i^{0} Y_i^{0}} \\newline C_{x,X_i^{1} Y_i^{0}} \\newline C_{y,X_i^{1} Y_i^{0}} \\newline C_{x,X_i^{0} Y_i^{1}} \\newline C_{y,X_i^{0} Y_i^{1}} \\newline C_{x,X_i^{2} Y_i^{0}} \\newline C_{y,X_i^{2} Y_i^{0}} \\newline C_{x,X_i^{1} Y_i^{1}} \\newline C_{y,X_i^{1} Y_i^{1}} \\newline C_{x,X_i^{0} Y_i^{2}} \\newline C_{y,X_i^{0} Y_i^{2}} \\newline C_{x,X_i^{3} Y_i^{0}} \\newline C_{y,X_i^{3} Y_i^{0}} \\newline C_{x,X_i^{2} Y_i^{1}} \\newline C_{y,X_i^{2} Y_i^{1}} \\newline C_{x,X_i^{1} Y_i^{2}} \\newline C_{y,X_i^{1} Y_i^{2}} \\newline C_{x,X_i^{0} Y_i^{3}} \\newline C_{y,X_i^{0} Y_i^{3}} \\newline C_{x,x_i^{1} y_i^{0}} \\newline C_{y,x_i^{1} y_i^{0}} \\newline C_{x,x_i^{0} y_i^{1}} \\newline C_{y,x_i^{0} y_i^{1}} \\newline C_{x,x_i^{2} y_i^{0}} \\newline C_{y,x_i^{2} y_i^{0}} \\newline C_{x,x_i^{0} y_i^{2}} \\newline C_{y,x_i^{1} y_i^{1}} \\newline C_{x,x_i^{3} y_i^{0}} \\newline C_{y,x_i^{0} y_i^{2}} \\newline C_{x,x_i^{0} y_i^{3}} \\newline C_{y,x_i^{1} y_i^{2}} \\newline C_{y,x_i^{0} y_i^{3}} \\newline \\end{matrix} \\right]")
