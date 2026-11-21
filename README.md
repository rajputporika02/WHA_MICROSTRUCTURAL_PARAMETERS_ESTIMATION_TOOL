# Automated Microstructural Characterization of Tungsten Heavy Alloys (WHA)

**Computer Vision & Deep Learning Research Engineer Intern | DRDO-funded research project | IIT (BHU), Varanasi**

This repository documents the research internship work and its associated journal publication on automated characterization of tungsten heavy alloy (WHA) microstructures from scanning electron microscopy (SEM) images. The workflow combines digital image processing, U-Net-based semantic segmentation, and geometric/image-analysis methods to estimate important microstructural parameters.

> **Reported workflow impact:** The project document describes a Python-based tool that processes a grayscale WHA SEM micrograph and calculates the required parameters in approximately **120 seconds**, compared with more than an hour for manual analysis.

---

## Contents

- [Research Overview](#research-overview)
- [Problem Statement](#problem-statement)
- [Internship Work](#internship-work)
- [Dataset Preparation](#dataset-preparation)
- [Image-Processing Pipeline](#image-processing-pipeline)
- [U-Net Segmentation](#u-net-segmentation)
- [Microstructural Parameter Estimation](#microstructural-parameter-estimation)
- [Evaluation and Validation](#evaluation-and-validation)
- [Challenges Addressed](#challenges-addressed)
- [Impact and Applications](#impact-and-applications)
- [Journal Publication](#journal-publication)
- [Repository Contents](#repository-contents)
- [Tools and Technologies](#tools-and-technologies)
- [Acknowledgement](#acknowledgement)

---

## Research Overview

Tungsten heavy alloys (WHAs) are used in defence, aerospace, and radiation-shielding applications because of their exceptional density and mechanical properties. Their properties are closely related to microstructural characteristics such as tungsten-grain contiguity, dihedral angle, tungsten grain size and size distribution, tungsten volume fraction, and tungsten precipitate size.

Traditional manual microstructural analysis is time-consuming. This research focused on automating the analysis of SEM micrographs using image processing and deep learning.

<img width="312" height="237" alt="image" src="https://github.com/user-attachments/assets/d475ca0b-159a-4af9-bc19-fbafa77b9ed8" />

*Grayscale SEM microstructure of a tungsten heavy alloy.*

## Problem Statement

Manual characterization requires identifying tungsten grains and their interfaces, measuring shared neck lengths, locating grain-boundary cusp points, and calculating grain-level statistics. Performing these measurements manually can take more than an hour for an individual image.

The objective was to build a Python-based WHA microstructural analysis workflow that:
1. Identifies tungsten grain boundaries from grayscale SEM images.
2. Predicts grain-boundary pixels using a semantic segmentation model.
3. Uses segmentation output for boundary-dependent measurements.
4. Calculates additional microstructural parameters using image-processing and geometric methods.
5. Reduces analysis time and supports more repeatable measurements.

## Internship Work

- **Role:** Computer Vision & Deep Learning Research Engineer Intern
- **Research setting:** DRDO-funded research project, IIT (BHU), Varanasi
- **Research domain:** Automated image analysis and microstructural characterization of tungsten heavy alloys

The internship work covered SEM image preparation, ground-truth annotation, dataset augmentation, U-Net model implementation and training, grain-boundary prediction, development of parameter-measurement logic, and comparison of automated measurements with manual and literature data.

## Dataset Preparation

The initial dataset consisted of **six original grayscale SEM images** of WHA.

1. **Grayscale image acquisition:** Six original SEM micrographs were used.
2. **Binarization:** Otsu's global thresholding was applied. Tungsten was represented as white and the matrix as black.
3. **Edge detection:** Canny edge detection was applied to the binarized images to extract the perimeter boundaries of individual tungsten grains.
4. **Ground-truth annotation:** Tungsten–tungsten interfaces were manually drawn on the six edge-detected images using ImageJ.
5. **Augmentation:** The annotated data was augmented to obtain 12,000 images.
6. **Dataset split:** 90% (10,800 images) for training and 10% (1,200 images) for validation.

<img width="201" height="155" alt="image" src="https://github.com/user-attachments/assets/4366ad1c-3398-4a0e-95a0-0cf61e2c2f00" />

*Binary image: tungsten grains are white and matrix regions are black.*



## Image-Processing Pipeline

### Otsu thresholding
Otsu's global thresholding algorithm was used to separate tungsten regions from the matrix in grayscale SEM images.

### Canny edge detection
Canny edge detection was applied to the binarized image to identify the perimeter boundaries of individual tungsten grains. These boundaries supported manual annotation of W–W interfaces.

### Ground-truth annotation
The W–W interfaces were manually annotated using ImageJ on the edge-detected images. These annotations were used as ground-truth masks for supervised U-Net training.

<img width="682" height="442" alt="image" src="https://github.com/user-attachments/assets/3713ee5f-346f-44bd-989d-e1bb1cc0f61c" />

*Image-analysis workflow showing grayscale input, binarization, phase segmentation, overlay, predicted boundary map, and average contiguity calculation.*

## U-Net Segmentation

A U-Net model was selected because the task requires **pixel-level segmentation and precise boundary localization**. The encoder learns contextual features while progressively reducing spatial resolution. The decoder restores spatial resolution, and skip connections transfer fine-grained spatial information from encoder layers to decoder layers.

<img width="703" height="465" alt="image" src="https://github.com/user-attachments/assets/daf98184-4c17-4220-af35-3b93f728bf78" />

*U-Net encoder–decoder architecture with skip connections.*

### Model configuration documented in the project

| Component | Configuration |
|---|---|
| Input image | 256 × 256 × 1 |
| Model | U-Net |
| Encoder filters | 16, 32, 64, 128 |
| Bottleneck filters | 256 |
| Decoder filters | 128, 64, 32, 16 |
| Convolution operations | Two convolution layers at each stage |
| Encoder downsampling | Max pooling |
| Decoder upsampling | Transposed convolution |
| Skip connections | Encoder-to-decoder concatenation |
| Output shape | 256 × 256 × 1 |
| Output activation | Sigmoid |
| Optimizer | Adam |
| Loss function | Binary cross-entropy |
| Batch size | 16 |
| Training/validation split | 90% / 10% |

The model produces a single-channel probability map. Each output pixel has a value between 0 and 1. A threshold of 0.5 is applied to convert the predicted probability map into a binary segmentation mask.

### Segmentation evaluation using IoU

Intersection over Union (IoU) compares the predicted foreground boundary pixels with the ground-truth foreground pixels:

<img width="2169" height="725" alt="image" src="https://github.com/user-attachments/assets/1a7b963b-2de7-4778-9de4-401b368db72d" />


The ground-truth mask contains binary values (0 and 1). The predicted probability map is thresholded at 0.5 to produce a Boolean mask. Logical AND gives the intersection, and logical OR gives the union. Summing the resulting Boolean arrays counts the pixels in each set.

Pixel accuracy alone can be misleading in segmentation when background pixels dominate the image, so IoU is used to assess overlap of the target regions.

## Microstructural Parameter Estimation

The trained U-Net predicts tungsten grain boundaries. The predicted boundary map supports the calculation of boundary-related parameters, while other parameters are obtained using image processing and geometric analysis.

### 1. Contiguity

Contiguity quantifies the degree of contact between tungsten grains in a tungsten heavy alloy. Two methods are documented.

**Method 1 — Line-intercept method**

<img width="697" height="168" alt="image" src="https://github.com/user-attachments/assets/ea71ec73-c123-4169-8e75-0a8c9131a0c4" />

**Method 2 — Grain-wise neck-length method**

<img width="712" height="166" alt="image" src="https://github.com/user-attachments/assets/c3292429-e0e3-45e1-9bc6-7315d1a9469d" />


<img width="695" height="307" alt="image" src="https://github.com/user-attachments/assets/ad89b818-926b-4f57-913b-f339b68ad479" />

*Illustrative stages of selecting tungsten grains and measuring shared neck lengths for grain-wise contiguity.*

### 2. Dihedral angle

The dihedral angle is measured at the cusp point between two tungsten grains. The method involves identifying the W–W interface, locating the cusp point, and determining the two tangent directions along the adjoining grain boundaries.

<img width="441" height="503" alt="image" src="https://github.com/user-attachments/assets/36d832bb-fea0-4a21-b81c-84ffc1ff9179" />


### 3. Tungsten volume fraction

The tungsten volume fraction is estimated from the ratio of tungsten pixels to the total number of image pixels

### 4. Tungsten grain size and size distribution

The equivalent circular diameter is calculated from the measured area of an individual tungsten grain:

<img width="162" height="92" alt="image" src="https://github.com/user-attachments/assets/2ffc4197-4f22-4aed-9f76-fa2ebaef0988" />


where (A) is the area of the grain. The equivalent diameters are used to obtain the grain-size distribution.

<img width="531" height="177" alt="image" src="https://github.com/user-attachments/assets/71868e5a-cfe6-4d75-93f1-687e55b93ff3" />

*Grain-size distribution plot.*

### 5. Tungsten precipitate size

The workflow also includes identifying tungsten precipitates and measuring their sizes to obtain precipitate-size statistics and distributions.

<img width="577" height="190" alt="image" src="https://github.com/user-attachments/assets/349efd7f-31c2-4c18-86fb-53fc5d928c05" />

*SEM micrograph, selected precipitates, and precipitate-size distribution.*

## Evaluation and Validation

The automated measurements were validated against manual measurements and literature data.

The supplied document includes a table comparing manual and automated measurements for parameters such as contiguity, dihedral angle, sphericity, average W grain size, and W volume fraction.

<img width="415" height="295" alt="image" src="https://github.com/user-attachments/assets/6b21b523-2d69-42f1-afb3-3853bba32d92" />

*Summary table of manual and automated measurements.*


<img width="628" height="476" alt="image" src="https://github.com/user-attachments/assets/a16e5830-b7b8-4c99-8a80-d54aeea368f3" />

*Comparison plots for contiguity, dihedral angle, grain size, and volume fraction.*

The reported conclusion is that the integrated framework can characterize important WHA microstructural features with agreement against manual and literature data. Exact model-performance metrics should be reported from the final experimental results or published paper.

## Challenges Addressed

1. **Tungsten grain-boundary identification:** identifying fine grain boundaries from grayscale SEM images.
2. **Ground-truth preparation:** manually annotating W–W interfaces on edge-detected images.
3. **Contiguity calculation:** implementing line-intercept logic on the segmented image.
4. **Neck-length measurement:** identifying W–W interfaces and measuring shared neck lengths between adjacent grains.
5. **Dihedral-angle measurement:** detecting W–W cusp points and identifying the two end points/directions needed for tangent-based angle calculation.
6. **Validation:** comparing automated measurements against manual and literature data.

## Impact and Applications

The work contributed to a Python-based WHA microstructural characterization tool as being regularly used in DMRL as part of a DRDO-funded research project.

According to the project notes, a grayscale WHA SEM micrograph (example input size: 1024 × 768) can be supplied to the tool to calculate the important microstructural parameters in approximately **120 seconds**. The notes contrast this with more than an hour for an individual performing the analysis manually.

Potential application areas include defence, aerospace, radiation shielding, and alloy processing/microstructure–property research.

## Journal Publication

The same research work was published in the **International Journal of Refractory Metals and Hard Materials**.


- **Paper title:** [Deep learning-driven automated microstructural quantification in tungsten heavy alloys using U-net segmentation and image processing]
- **Authors:** [<img width="770" height="28" alt="image" src="https://github.com/user-attachments/assets/57b41b49-df16-44f4-a5cc-1ad2b4f66518" />]
- **Journal:** International Journal of Refractory Metals and Hard Materials
- **Year / Volume / Pages:** [Volume 138, August 2026, 107719]
- **DOI:** [https://doi.org/10.1016/j.ijrmhm.2026.107719]
- **Publisher page:** [https://www.sciencedirect.com/journal/international-journal-of-refractory-metals-and-hard-materials]

### Research conclusion

The automated image-analysis framework integrating digital image processing and deep learning to characterize WHA microstructures. It reports estimation of contiguity, dihedral angle, tungsten grain size, sphericity, tungsten volume fraction, and tungsten precipitate size directly from SEM images. The workflow combines Otsu thresholding, Canny edge detection, and U-Net-based segmentation, and the reported measurements showed strong agreement with manual results.


## Tools and Technologies

- **Python** — implementation of the image-analysis workflow
- **TensorFlow / Keras** — U-Net model implementation and training workflow
- **Deep Learning** — semantic segmentation
- **OpenCV / Image Processing** — image processing operations
- **NumPy** — array-based image and mask operations
- **ImageJ** — manual ground-truth annotation
- 

## Acknowledgement

This work was carried out during a Computer Vision & Deep Learning Research Engineer internship on a DRDO-funded research project at IIT (BHU), Varanasi. The same research work was reported as published in the *International Journal of Refractory Metals and Hard Materials*.
