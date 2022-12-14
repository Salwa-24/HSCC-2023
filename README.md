# HSCC 2023 Submission Code Guide  

This is HSCC 2023 Repo, for the submission of the paper entitled "Response time Lag in Accessing an Application When Exposed to Dynamic and Static Security Scans:Using Data-Mining Techniques". 

The development of quicker web interfaces has enabled us a range of applications for all of our needs in communication, banking, shopping, etc. The technology and the IT infrastructure that hosts these applications is constantly evolving, there is a substantial risk of Cyber-attacks on the systems and the application.
To protect these, both static and dynamic scans are used constantly to protect online resources from harmful attacks. As a result, the infrastructure owners and application develope must be aware that it causes latency while accessing these applications.
This study is an effort to analyze the impact of response time on the application using Data Mining techniques.
The Google Cloud Platform (GCP) is used to host a sample application, which is subjected to static and dynamic scan using an industry-standard tool such as Twistlock. The user interaction response time is then recorded using the TCP/IP 3-way handshake. This data was captured to only the source and destination of the IPs involved in the response of the request while discarding re-transmission and malformed packets in the interaction. Then the collected data is compared to an Ideal scenario, where no scans are performed against captured data when static and dynamic scans are used. The data is further analyzed using industry-recommended data mining techniques and the results and observations are captured.
This study focuses on measuring lag in the response time due to constant security scanning while accessing an application. 
This is done by using TCP/IP 3-way handshake data capture, a data-mining technique such as hierarchical and k-means Clustering, and uses rationale on the anomalies observed such as outliers in the collected data.

The datasets folder contains .csv files which are passed asinput to our python code.

## Prerequisites

1. Visual Studio Code
2. Python 3

## Steps to Reproduce the code

1. Visit our URL for Github Repository.
2. Download the zip file of our code. 
3. Open Visual Studio Code and using python commands pip install the packages via the requirements.txt file, by typing the command `pip3 install -r requirements.txt` in the terminal.
4. Run the Python files using the command `python3 filename.py`, for example `python3 HSCC-2023-Hirearichal_Clustering.py`and the results are reproduced.
5. More information about the graphs and values tabulated in the tables are in brief documented in readme.txt.


