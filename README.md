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

![WHA grayscale SEM microstructure](assets/image1.png)

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

![Binary image after thresholding](assets/image6.png)

*Binary image: tungsten grains are white and matrix regions are black.*

![Canny edge detection](assets/image7.png)

*Grain perimeter boundaries extracted using edge detection.*

## Image-Processing Pipeline

### Otsu thresholding
Otsu's global thresholding algorithm was used to separate tungsten regions from the matrix in grayscale SEM images.

### Canny edge detection
Canny edge detection was applied to the binarized image to identify the perimeter boundaries of individual tungsten grains. These boundaries supported manual annotation of W–W interfaces.

### Ground-truth annotation
The W–W interfaces were manually annotated using ImageJ on the edge-detected images. These annotations were used as ground-truth masks for supervised U-Net training.

![Overall WHA image-analysis workflow](assets/image8.png)

*Image-analysis workflow showing grayscale input, binarization, phase segmentation, overlay, predicted boundary map, and average contiguity calculation.*

## U-Net Segmentation

A U-Net model was selected because the task requires **pixel-level segmentation and precise boundary localization**. The encoder learns contextual features while progressively reducing spatial resolution. The decoder restores spatial resolution, and skip connections transfer fine-grained spatial information from encoder layers to decoder layers.

![U-Net architecture](assets/image5.png)

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

\[
IoU = \frac{|Y_{\mathrm{true}} \cap Y_{\mathrm{pred}}|}
{|Y_{\mathrm{true}} \cup Y_{\mathrm{pred}}|}
\]

The ground-truth mask contains binary values (0 and 1). The predicted probability map is thresholded at 0.5 to produce a Boolean mask. Logical AND gives the intersection, and logical OR gives the union. Summing the resulting Boolean arrays counts the pixels in each set.

Pixel accuracy alone can be misleading in segmentation when background pixels dominate the image, so IoU is used to assess overlap of the target regions.

## Microstructural Parameter Estimation

The trained U-Net predicts tungsten grain boundaries. The predicted boundary map supports the calculation of boundary-related parameters, while other parameters are obtained using image processing and geometric analysis.

### 1. Contiguity

Contiguity quantifies the degree of contact between tungsten grains in a tungsten heavy alloy. Two methods are documented.

**Method 1 — Line-intercept method**

\[
C_{WW} = \frac{2N_{WW}}{2N_{WW}+N_{WM}}
\]

where \(N_{WW}\) is the number of tungsten–tungsten interfaces and \(N_{WM}\) is the number of tungsten–matrix interfaces. These values are obtained using the line-intercept method applied to SEM images.

**Method 2 — Grain-wise neck-length method**

\[
C_i = \frac{A_1+A_2+A_3}{P_i}
\]

where \(A_1\), \(A_2\), and \(A_3\) are the neck lengths shared with adjacent tungsten grains, and \(P_i\) is the total perimeter of the grain. This method estimates contiguity individually for each grain.

![Contiguity measurement illustration](assets/image9.png)

*Overlay, grain selection, perimeter measurement, and W–W neck-length identification.*

![Grain-wise contiguity analysis](assets/image10.png)

*Illustrative stages of selecting tungsten grains and measuring shared neck lengths for grain-wise contiguity.*

### 2. Dihedral angle

The dihedral angle is measured at the cusp point between two tungsten grains. The method involves identifying the W–W interface, locating the cusp point, and determining the two tangent directions along the adjoining grain boundaries.

### 3. Tungsten volume fraction

The tungsten volume fraction is estimated from the ratio of tungsten pixels to the total number of image pixels:

\[
V_W(\%) =
\frac{\text{Number of tungsten pixels}}
{\text{Total number of image pixels}}\times 100
\]

![Phase segmentation and volume-fraction analysis](assets/image11.png)

*Overlay and phase analysis used for tungsten volume-fraction estimation.*

### 4. Tungsten grain size and size distribution

The equivalent circular diameter is calculated from the measured area of an individual tungsten grain:

\[
D_{\mathrm{eq}} = \sqrt{\frac{4A}{\pi}}
\]

where \(A\) is the area of the grain. The equivalent diameters are used to obtain the grain-size distribution.

![Grain-size distribution](assets/image14.png)

*Grain-size distribution plots.*

### 5. Tungsten precipitate size

The workflow also includes identifying tungsten precipitates and measuring their sizes to obtain precipitate-size statistics and distributions.

![Precipitate size analysis](assets/image12.png)

*SEM micrograph, selected precipitates, and precipitate-size distribution.*

## Evaluation and Validation

The project notes state that automated measurements were validated against manual measurements and literature data.

The supplied document includes a table comparing manual and automated measurements for parameters such as contiguity, dihedral angle, sphericity, average W grain size, and W volume fraction.

![Manual and automated measurement comparison](assets/image13.png)

*Summary table of manual and automated measurements.*

The document also includes plots comparing measured parameter values with reference/literature values.

![Comparison plots for microstructural parameters](assets/image15.png)

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

The work contributed to a Python-based WHA microstructural characterization tool reported in the project document as being regularly used in DMRL as part of a DRDO-funded research project.

According to the project notes, a grayscale WHA SEM micrograph (example input size: 1024 × 768) can be supplied to the tool to calculate the important microstructural parameters in approximately **120 seconds**. The notes contrast this with more than an hour for an individual performing the analysis manually.

Potential application areas include defence, aerospace, radiation shielding, and alloy processing/microstructure–property research.

## Journal Publication

The same research work was published in the **International Journal of Refractory Metals and Hard Materials**, as stated in the supplied document.

The resume document does not provide the exact paper title, full published author list, year, volume, page range, or DOI. Insert the verified citation details here rather than guessing:

- **Paper title:** [Insert exact published title]
- **Authors:** [Insert author list in published order]
- **Journal:** International Journal of Refractory Metals and Hard Materials
- **Year / Volume / Pages:** [Insert publication details]
- **DOI:** [Insert DOI]
- **Publisher page:** [Insert article URL]

### Research conclusion

The supplied project document describes an automated image-analysis framework integrating digital image processing and deep learning to characterize WHA microstructures. It reports estimation of contiguity, dihedral angle, tungsten grain size, sphericity, tungsten volume fraction, and tungsten precipitate size directly from SEM images. The workflow combines Otsu thresholding, Canny edge detection, and U-Net-based segmentation, and the reported measurements showed strong agreement with manual results.

## Repository Contents

```text
WHA_Microstructure_Analysis/
├── README.md
└── assets/
    ├── image1.png
    ├── image2.png
    ├── image3.png
    ├── image4.png
    ├── image5.png
    ├── image6.png
    ├── image7.png
    ├── image8.png
    ├── image9.png
    ├── image10.png
    ├── image11.png
    ├── image12.png
    ├── image13.png
    ├── image14.png
    ├── image15.png
    ├── image16.png
    ├── image17.png
    ├── image18.png
    ├── image19.png
    ├── image20.png
    └── image21.png
```

**Image note:** All **21 embedded images** from the supplied Word document are preserved in `assets/`. The README embeds the figures relevant to the WHA internship and publication (images 1 and 5–15). The remaining images (2–4 and 16–21) are also included so no embedded source image is lost; they are not inserted into the WHA research narrative where they would be unrelated to this project.

## Tools and Technologies

- **Python** — implementation of the image-analysis workflow
- **TensorFlow / Keras** — U-Net model implementation and training workflow
- **Deep Learning** — semantic segmentation
- **OpenCV / Image Processing** — image processing operations
- **NumPy** — array-based image and mask operations
- **ImageJ** — manual ground-truth annotation

The supplied document does not specify exact package versions or a complete dependency list.

## Acknowledgement

This work was carried out during a Computer Vision & Deep Learning Research Engineer internship on a DRDO-funded research project at IIT (BHU), Varanasi. The same research work was reported as published in the *International Journal of Refractory Metals and Hard Materials*.
