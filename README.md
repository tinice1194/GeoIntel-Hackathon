# GeoIntel-Hackathon
AI/ML model to identify key features from drone orthophotos.

**Overview**

The project focused on developing AI/ML models to identify key features from drone orthophotos. Two segmentation architectures — UNet and DeepLab V3 — were tested. While UNet demonstrated finer spatial detail extraction, DeepLab V3 achieved higher reported accuracy (92.43% compared to UNet’s 86%). However, these metrics are ambiguous due to high background noise and insufficient feature diversity within the training data.


**Dataset Challenges**

The core issue lies with the training dataset, which was highly incomplete and inconsistently labeled.
  - A large portion of the dataset consisted of background pixels, leading to limited learning on meaningful features such as buildings, roads, and water bodies.
  - Poorly labeled roof data required script-based auto-labelling, introducing uncertainty and errors in the CNN’s understanding of roof types.
  - The dataset demanded significant preprocessing and quality refinement, but this was hindered by constrained computational resources.


**Hardware and Segmentation Limitations**

The available hardware was unable to load full raster images during training, forcing the use of segmented inputs.
  - This segmentation introduced discontinuities at image boundaries, making it harder for models to learn contextual relationships between adjoining features.
  - Memory constraints severely affected model consistency and the spatial coherence of predictions across tiles.
  - As a result, buildings and water bodies were only partially identified when they lay completely within a segment — structures falling across segment edges were                  misclassified or ignored.


**Model Performance**

Despite these challenges, both UNet and DeepLab V3 could identify major roads, certain building clusters, and water bodies when segmentation conditions were favorable.
However, due to the dominance of background data and fragmented raster input, most predictions defaulted to background classification, limiting practical utility.


**Summary**

The accuracy differences between UNet and DeepLab V3 are not definitive indicators of model superiority. Instead, the results primarily reflect the effects of hardware constraints, dataset incompleteness, and raster segmentation issues. Future model iterations should focus on acquiring high-quality labeled datasets, ensuring full raster accessibility during training, and performing robust preprocessing to minimize noise and label ambiguity.



# **Implementation Plan / Roadmap**


**1 Month:** Preparation of valid and authenticated dataset 1 for training of AI model to
classify major features with parallel testing and validation.

**2-3 Month:** Obtaining extracted buildings from GIS using AI model and preparation of
dataset 2 for training AI for classification of rooftops, along with parallel testing and
validation.

**3-6 Month:** Accessing all the obtained output from AI and further processing for minute
detailing like small features like water tanks on top of buildings using YOLO.

**6-9 Month:** Final testing of AI for a set of 50-100 village at once to see processing
capability and accuracy when applied to large data.

**9-12 Month:** Final finishes and code changes before releasing for final production. 
