import streamlit as st
import json
# import sys
# st.write(sys.path)
st.header ("Photolithography", divider=True)
st.sidebar.markdown("# Photolithography")
st.subheader("Problem Overview",divider=True)
st.image("./src/figures/EUV_tool.png", caption="An ASML EUV Photolithography tool [https://www.cnbc.com/2022/03/23/inside-asml-the-company-advanced-chipmakers-use-for-euv-lithography.html]")
st.markdown("Photolithography is major process step (out of many) necessary for the construction of a semiconducting device (AKA a compute chip). In the pursuit of maintaining Moore's Law, the photolithography process has required Extreme UltraViolet (EUV) light to produce geometries at the nanometer-scale. Above, one can see inside one of ASML's EUV Photolithography tools. While these tools can achieve incredible precision, Intel and TSMC mainly use the exact same tools. What differentiates the two companies lies in their process control. Intel and TSMC still need to adjust the tool parameters to squeeze every last bit of performance out the equipment to maximize yield and throughput. Photolithography was one of the first process in semiconductor manufacturing in which wafer-to-wafer process control was applied. I will detail a fairly old version of this wafer-to-wafer process control below.")
st.subheader("Wafer-to-Wafer Process Control",divider=True)
st.image("./src/figures/Wafer_map.png",caption="Example of 300mm wafer with individual die shown")
st.markdown("On each wafer lie a number of die (or products), the number of which vary based on the product size. An example of a die layout on a wafer is shown above. For each die, the photolithography tool must expose a circuit pattern onto each die (termed _field_) and precisely overlay each pattern onto the previous circuit pattern. This distance between the target pattern location and the resulting pattern location is termed the _overlay error_. The sum of the overlay error across all layers of the product is termed the _stackup overlay error_. Both of these errors must be precisely controlled, else the device will not work.")
st.image("./src/figures/Overlay_Error_Notation.png",caption="Overlay error notation [Dr. Dragan Djurdjanovic (Time-Series Analysis)]")


