# Advanced Data Analysis and Machine Learning Projects 

This repository showcases advanced data analysis and machine learning projects applied to large-scale, statistically complex datasets. The work focuses on rare-event (signal) extraction, anomaly detection and classification, supported by a strong understanding of detector behavior and data acquisition systems.

## Core Competencies

*   **Machine Learning:** Random Forests, Boosted Decision Trees (BDT), Anomaly Detection, Multivariate Analysis (MVA), Feature Selection (mRMR), Signal/Background Classification.
*   **Statistical Data Analysis:** Advanced statistical modeling, parameter estimation and hypothesis testing on large datasets.
*   **Data Visualization:** Creation of publication-quality plots, including signal distributions, ROC curves, and performance metrics.
*   **Programming & Tools:** Python (NumPy, Pandas, Scikit-learn, Matplotlib, TensorFlow), C++, Statistics, ROOT.
*   **Sensor Characterization & Calibration:** Hands-on experience with characterizing and calibrating scintillator and silicon-based sensors.
*   **Detector Physics:** Strong understanding of the physics of particle detectors and their industrial applications.

---

## Projects

### 1. CERN: Anomaly Detection in 8.5M Events

*   **Project Goal:** To build a statistical analysis pipeline to search for anomalies in a large-scale dataset from CERN, addressing a significant signal-to-noise imbalance.
*   **Methodology & Algorithms:**
    *   Analyzed an **8.5 million instance dataset** with an extreme signal-to-noise imbalance of 1:1000.
    *   Developed a filtering pipeline based on the statistical distributions of the data to isolate regions of interest.
    *   Employed advanced visualization techniques, including **2D density maps, heatmaps, and segmentation plots**, to identify and localize anomalies.
*   **Results & Key Metrics:**
    *   Discovered a localized anomaly with a significance of **8.92σ**.
    *   Conducted a systematic uncertainty analysis, determining that equipment-related effects reduced the anomaly's significance to a still-notable **4.32σ**.
*   **Report:** [Link to Report](./experiments/cern-lhcb1/CERN_Anomaly_detection.pdf)

### 2. CERN-LHCb: Signal Classification with Multivariate Analysis (LHCB1)

*   **Project Goal:** To develop a machine learning model to effectively distinguish between signal (real particle decays) and background events in data from the LHCb experiment at CERN.
*   **Methodology & Algorithms:**
    *   Implemented a **Boosted Decision Tree (BDT)** using the TMVA toolkit.
    *   Trained the BDT on a dataset of simulated particle collisions, using kinematic variables as features.
    *   Performed hyperparameter tuning to optimize the BDT's performance.
*   **Results & Key Metrics:**
    *   The final model achieved a high signal efficiency while maintaining a low background contamination.
    *   Generated over **20 plots** to evaluate the model, including feature distributions, BDT output, and **ROC curves** to visualize the signal-background separation.
    *   The model's performance was quantified by measuring the area under the ROC curve (AUC).
*   **Report:** [Link to Report](./experiments/cern-lhcb2/Signal_classification.pdf)

### 3. IceCube: Neutrino Event Detection with Machine Learning

*   **Project Goal:** To develop a high-performance machine learning model to detect rare neutrino signal events in data from the IceCube experiment in Antarctica.
*   **Methodology & Algorithms:**
    *   Trained a **Random Forests classifier** and benchmarked its performance against Naive Bayes and K-Nearest Neighbors.
    *   Employed the **mRMR (minimum Redundancy Maximum Relevance)** algorithm to select the top 10 most informative features, ensuring a stable and consistent feature set (Jaccard index of 1).
    *   Optimized the Random Forest's decision threshold to 0.65 to maximize the F1 score and AUC.
    *   Validated the model's generalization performance using **5-fold cross-validation**.
*   **Results & Key Metrics:**
    *   The final Random Forests model achieved a **precision of 94.5%** and an **AUC of 0.981**.
    *   The model significantly outperformed Naive Bayes (AUC 0.93) and KNN (AUC 0.968).
*   **Report:** [Link to Report](./experiments/Antartica-icecube/icecube.pdf)

---

## Hardware-Focused Projects: Bridging the Gap Between Hardware and Data

The following projects demonstrate my hands-on experience in characterizing and calibrating advanced sensor technologies. This unique skill set is highly valuable in industries such as **semiconductor manufacturing**, **medical imaging**, and **aerospace**, where the quality of the data is directly tied to the performance of the underlying hardware.

### 4. Characterization of Silicon Strip Sensors

*   **Project Goal:** To measure and characterize the key performance metrics of a silicon strip sensor, a foundational technology in many advanced imaging and detection systems.
*   **Methodology & Data Analysis:**
    *   Acquired data to measure parameters such as depletion voltage, leakage current, and charge collection efficiency.
    *   Analyzed the resulting data to assess the sensor's performance and identify potential defects.
    *   Produced **10+ plots** to visualize the sensor's characteristics.
*   **Other Applications:**
    *   **Semiconductors:** My experience is directly applicable to the quality control and characterization of new sensor and processor designs.
    *   **ML Integration:** ML models can be used to automate the analysis of sensor data, predict sensor lifetime, and perform real-time calibration, all of which rely on a deep understanding of the sensor's behavior.
*   **Report:** [Link to Report](./experiments/silicon-strip-sensor-characterization/Silicon_Strip_Sensors_Report.pdf)

### 5. Characterization of Scintillator Detectors

*   **Project Goal:** To investigate the properties and performance of scintillator detectors, which are widely used in medical imaging and radiation detection.
*   **Methodology & Data Analysis:**
    *   Measured the light yield and timing resolution of the scintillator, two key performance indicators.
    *   Performed a statistical analysis of the detector's response to different radiation sources.
    *   Generated **~15 plots** to characterize the detector's performance.
*   **Other Applications:**
    *   **Medical Imaging:** This experience is highly relevant to the development and calibration of PET and CT scanners.
    *   **ML Integration:** ML algorithms are used extensively in medical imaging for image reconstruction, noise reduction, and automated diagnosis. A thorough understanding of the detector's properties, which I have demonstrated, is crucial for developing and validating these algorithms.
*   **Report:** [Link to Report](./experiments/scintillator-detector-characterization/Scintillators_Report.pdf)

---


