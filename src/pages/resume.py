import streamlit as st
import pandas as pd

st.header("Resume")
st.sidebar.markdown("# Resume")

st.divider()
st.subheader("Education")

with st.expander("B.S.M.E. Degree"):
    st.write("Time: Fall 2017-Spring 2021")
    st.write("GPA: 3.94")
with st.expander("M.S.M.E. & Ph.D.M.E. Degree"):
    st.write("Time: Fall 2021-Summer 2026 (Expected)")

st.subheader("Work Experience")
with st.expander("Graduate Research Assistant"):
    st.write("Company: University of Texas at Austin")
    st.write("Location: Austin, TX")
    st.write("Time: Fall 2023-Present")
    st.write("- Developed and applied a novel time-varying Kalman filter-based virtual metrology model that fuses model uncertainty, process drift, and product uncertainty for informed fab sample selection with the aim of minimizing unmeasured product quality uncertainty in Python and MatLab")
    st.write("- Demonstrated our novel sample selection policy outperformed state-of-the-art measurement selection policies by increasing adjusted $R^{2}$ by ~5%")
    st.write("- Extending novel framework for the purposes of VM-based advanced process control to correct for wafer error from target quality utilizing product recipe and bias estimation")
with st.expander("ML/AI Research Intern"):
    st.write("Company: Samsung Advanced Institute of Technology")
    st.write("Location: Su Won, South Korea")
    st.write("Time: March 2023-October 2023")
    st.write("- Applied dimensionality reduction and feature extraction of plasma-etch signals for quantitative product assessment at an industrial-scale using Python as an applied researcher on an international research grant")
    st.write("- Deployed a trace-signal feature extraction methodology on approximately 115,000 wafers utilizing plasma etch data for feature extraction and virtual metrology utilizing PySpark and several CPU clusters")
    st.write("- Improved product assessment against current state-of-the-art feature extraction by reducing RMSE by ~29% for the purposes of virtual metrology utilizing Python and SKLearn")
with st.expander("Senior Data Science Consulting Intern"):
    st.write("Company: Kalypso, A Rockwell Automation Company")
    st.write("Location: Remote")
    st.write("Time: May-August 2022")
    st.write("- Consulted for a $20 billion tire manufacturer and presented solutions for implementation to client bi-weekly")
    st.write("- Utilized PyTorch for defect detection, localization, and classification (ResNet & U-Net) on industrial product images")
    st.write("- Aided in the development of an Optical Character Recognition system (PyTesseract) for product identification")
with st.expander("Teaching Assistant"):
    st.write("University: University of Texas at Austin")
    st.write("Time: Fall 2021-Present")
    st.write("Classes: Programming and Engineering Computational Methods & Automatic System Control and Design")
    st.write("- Taught supplemental sessions/weekly labs and led recitation sections for undergrad and graduate level students for each respective class")
    st.write("- Designed and graded class material for students")
    st.write("- Mentored undergraduate students via career and coursework advice")
with st.expander("Graduate Research Assistant (CyManII)"):
    st.write("Company: Cybersecurity Manufacturing Innovation Institute")
    st.write("Location: Austin, TX")
    st.write("Time: Spring 2022-Fall 2022")
    st.write("- Aimed to develop a process with MATLAB and OpenSIM to assess and monitor worker fatigue utilizing EMG data and a system-based similarity framework")
    st.write("- Extracted joint time-frequency features from EMG signals of a cyclist for a self-organizing map-based dynamic model for fatigue assessment of the user")
    st.write("- Framework successfully captured and quantified the users fatiguing trend throughout the trial as well as identified muscle synergies fatiguing and recovering trends ")
with st.expander("UT Student-Athlete (Baseball)"):
    st.write("Time: Fall 2017-Spring 2020")
    st.write("- Managed coursework with 20+ hours of practicing/strength training and a heavy Spring travel schedule")
    st.write("- Uniquely poor Freshman year statistics")

st.subheader("Skills")
with st.expander("Python"):
    st.write("Link to Python-based projects")
with st.expander("MatLab"):
    st.write("Link to MatLab-based projects")
with st.expander("C++"):
    st.write("Link to C++-based projects")
