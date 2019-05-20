# BodyPostureClustersDetectFrustration
This project uses Kinect sensor data collected in a makerspace over the span of a semester (13 weeks) from a class of 16 students.

After data cleaning and anonymization, clustering analysis is conducted using 3 methods: DBScan, k-means, and NMF, on selected body joints of interest to the researcher, then the cluster counts per week are correlated to student self reported feelings of frustration from a weekly survey.

The results show that 1 cluster is significantly positvely correlated (P=0.011) to student reported feeling of frustration. For further details, please read the attached pdf.

The Kinect data files can be downloaded from https://drive.google.com/open?id=1AdtqPZEMdUrUtSlOpmi76EUMhX_7AJkn

The anonymized student self-reported survey results are in the repo.

To run, simply place the Kinect folder, the ipynb, correlations df, and the survey csv in one folder, then open the notebook and run

This notebook can be run as a Jupyter notebook on your desktop or on Google Colab (code is commented out)

Findings from this research can be found in the attached pdf

This project is a small step towards instrumenting a non-invasive system that can be used to track student affective and cognitive states in an open ended environment. Further analysis may be added to this notebook to expand on the findings.
