
# GLCM Liteature Review

Textural analysis is a common way of analyzing remotely sensed data. Texture is a property of all surfaces and can be described by qualitative and quantitative metrics.  Qualitative descriptions of textures include:

- Smooth or Rough
- Local or Regional
- Random or repetitive

Qualitative descriptions of sidescan sonar imagery are subjective and rely on the interpreter's experience to accurately identify textures.  Additionally, visually analyzing datasets collected across large spatial extents ($10^1$ to $10^2$ $m^2$) can be prohibitively time-consuming.  Quantitative descriptions of texture offer an objective means to identify textures and can be used to develop unsupervised classification algorithms.

Stochastic approaches to developing quantitative metrics to describe textures properties have relied on first and second order statistics.  Textures exist at multiple scales and textural properties are defined by a computation window size.  First order statistics (i.e. single pixel) have been proven ineffective at describing textures in remotely sensed data because they do not account for the spatial arrangements of pixel intensities. Grey level co-occurrence matrix (GLCM) has been proven at identifying textures remotely sensed data.  GLCM describe the average spatial relationships between pixel intensities.  GLCM are hard to interpret alone and are best described by indices (i.e. properties).

Haralick et al. (1973) proposed 14 different properties of GLCM that can be used to quantify textural properties. GLCM properties describe  the contrast and orderliness of images.  Many of the GLCM properties are highly correlated and only a subset are applicable to sidescan sonar imagery.  Reed et al. (1989) were the first people to perform GLCM based textural analysis on sidescan sonar imagery and found only 5 of the properties are weakly correlated. The fire properties are:

- Angular Second Moment (ASM)
- Contrast (CON)
- Entropy (ENT)
- Angular Inverse Difference Moment (AIDM)
- Correlation (CORR)

Weakly correlated GLCM properties are desirable because the textures created by sedimentary features will either be highly disordered or organized and repetitive.  For example, rocks will display a disorderly pattern and best be characterized by entropy.  Smooth sand is highly ordered and is best described by GLCM properties related to contrast.  

Blondel et al. (1996) also conducted a GLCM based textural analysis on sidescan sonar imagery.  For his analysis, he considered 5 GLCM properties.

- Inertia
    - indicative of contrast of GLCM
    - Can be used to identify lithologic boundaries
- Uniformity
    - Also referred as Angular Second Moment
    - Identifies structures within an area
- Correlation
    - Quantifies the dependence of gray levels from one another for pixels separated by a distance
    - Low correlation means gray levels are independent of each other
    - High Correlation indicates there are repeated patterns  within a local area
- Entropy
    - Measures the lack of spatial organization inside a computation window
    - Relates to roughness
- Homogeneity
    - Also, know as inverse distance moment
    - Corresponds with textures associated with organized or poorly contrasted features
    - Used to identify differences between lava flow structures


Blondel et al. (1996) found entropy and homogeneity were the least correlated GLCM properties and could be used to reliably identify sediments, faults, and volcanic features.  Entropy reliably described rough areas (i.e. rock outcrops and lava flows), while homogeneity was a good predictor of smooth sand.  Blondel used these findings to develop the TexAn software package and authored several papers using entropy and homogeneity as reliable texture metrics to detect objects along the seafloor.  Cochrane and Lafferty (2002) also found entropy and homogeneity were suitable properties to detect rocks, sand, and thin sand.

Lucieer (2007) used a GLCM based textural analysis to automate the detection of reef, sand, and low reef in a marine environment.  Lucieer used eCognition software to perform the texture analysis and found GLCM mean, standard deviation, and mean were the least correlated.  The variables were then used to develop a fuzzy class membership applied to a larger dataset.

[TOC]

***
## 1989 

### Digital image processing techniques for enhancement and classification of SeaMARC II side scan sonar imagery

[Digital image processing techniques for enhancement and classification of SeaMARC II side scan sonar imagery](http://onlinelibrary.wiley.com/doi/10.1029/JB094iB06p07469/pdf)



***
##1993
### Textural Analysis and Structure Tracking for Geological mapping: Application to sonar images form Endeavors Segment, Juan de Fuca Ridge

[Blondel 1993 – Textural Analysis and Structure Tracking for Geological mapping: Application to sonar images form Endeavors Segment, Juan de Fuca Ridge](http://ieeexplore.ieee.org.dist.lib.usu.edu/stamp/stamp.jsp?tp=&arnumber=326187)

#### Motivation
- The collecction of sidescan sonar imagery has beecome incresingly common
- Sonar images are 2D representaions of the sediment water interface
- Textural properties of sidescan sonar imagery can be used to identify morphologic features, and roughness characteristics of riverbeds
- The spatial distribution of grey levels (i.e. texture) can be used to infer properties of sediments at the sediment water interface
- Visual analysis of sidescan sonar imagery is subjective and relies of the experiece of the user.
- GLCM offeres a quantiative approach to textural analysis
- Entropy quantifies texture (roughness)
- Homogenity measures smootheness
- Plots homogeneity vs. entropy



- Optimal parameters were determined by finding variables wiht sepearations between training zones.

#### Findings
- Entropy and homogenity were the mose useful metrics to classify sediments, faults, and volcanic features
    - Shadows identfied with 100% success
    - Talus and lava flows recogonized with accucies ranging from 60%-85%
    - Rough lavaflow identfied 100%

*********************
## 1996 

### Segmentation of the Mid-Atlantic Ridge south of the Azores, based on acoustic classification of TOBI data
[Segmentation of the Mid-Atlantic Ridge south of the Azores, based
on acoustic classification of TOBI data](http://sp.lyellcollection.org/content/118/1/17.full.pdf)

##### Motivation
- Textural analysis techniques describe the spatil organisation of grey levels within a neighborhood (texture)
- Textural Analysis is used to investigate the morphological properties and geologic processes
- Features represented in sidescan sonar imagery can be mapped on the basis of the tonal/textural processes
- Summary statistics of local areas can be used to describe the distribution of pixel intensities, but can not be used to quantify texture alone
- Textures can be qualified as:
    - Rough
    - Smooth
    - Local or Regional
    - Repetitive or random
- Stochasic techniques (i.e. GLCM) can be used to quantify textual characterics of sidescan sonar imagery
- GLCM address the average spatial relationships between pixel intensities at local areas
- GLCM is desctibed by indicies (properties)

#### Textural Properties
- Inertia
    - indictave of contrats of GLCM
    - Can be used to identify lithologic boundries
- Uniformity
    - Also refered as Angular Second Moment
    - Identifies structures within an area
- Correlation
    - Quantifies the dependec of grey levels from one another for pixels seperated by a distance
    - Low correlation means grey levels are independent of each other
    - High Correlation indicates there are repeated patterns  within a local area
- Entropy
    - Measures the lack of spatial organization inside a computation window
    - Relates to roughness
- Homogenity
    - Also know as inverse distance moment
    - Corresponends with textures associated with organized or poorly contrasted features
    - Used to identify differences between lava flow structures
- Mutational Information
- Maximum probabliity

#### Textural Analysis
- The intial step of classification of sidescan sonar imgery requres the definiton of optimal values of computation parameters and the useful textural indicies
- Tested 0°,45°, 90°, and 135° angels
- Used overlapping windows so that the transistions of textures can be observed
- Optimal seperaton of test retions was found wiht a 20 x 20 pixel window wiht interpixel displacemnt of 10 pixels
- Presents Example training zones
- Homogenity and entropy were the least correlated
- Entropy were resampled into 256 intervals, along a logritimic scale

![alt test][id]

- Backscattering mechninsms near nadir are different than the rest off the image

##### Summary
- Purpose was two fold:
    1. Demonstate the potential of textural analysis with GLCM for quanitative seafloor charactreisation
    2. Apply the mothod to a larger dataset
- Entropy and Homogeneity could accurately describe the textures

***
##1998

### TexAn: TEXTURAL ANALYSIS OF SIDESCAN SONAR IMAGERY AND GENERIC SEAFLOOR CHARACTERISATION

[TexAn: TEXTURAL ANALYSIS OF SIDESCAN SONAR IMAGERY AND GENERIC SEAFLOOR CHARACTERISATION](https://www.researchgate.net/publication/3777427_TexAn_Textural_analysis_of_sidescan_sonar_imagery_and_generic_seafloor_characterization)

- Textural analysis is a quantative means to qunitativly describe sidescan sonar imagery
- Entropy and homogeneity are the two best descriptors of sidescan sonar imagery texture
- Entropy is high when all co-occurance frequencies are wqual (i.e. low)
    - Makes it ideal to quantify rougher textures
- Homogeneity is directly proportional to the amount of local similarites
- Used Measurement-Space Guided Clustering
- 
- 
***
## 2000

### Automatic mine detection by textural analysis of COTS sidescan sonar imagery

[Automatic mine detection by textural analysis of COTS sidescan sonar imagery](http://www.tandfonline.com/doi/pdf/10.1080/01431160050144983)

### Motivation

- Automaticaly detect mine features in a marine environment
- Wanted to determine if mines could be detected using commerical grade sidescan sonar
- Textural Properties depend on the spatial arrangement of grey levels within a defined window size

### Textural Analysis

- Used entropy and homogenity to detect mines
- Homogeneity is always negative
- Qunatized homogenity has been inverted on a lograthmic scale: low quantized homogeneity values (near 0) correspond to large amplitudes, while small magnitudes correspond to high quantized homogeneity (near 255)

### Findings
![alt test][id4]

- Look-up table showing the distribution of entropy and homogeneity for the region around mine ‘A’. The two textural measurements have been quantized between 0 and 255. Purple corresponds to the lower entropies and homogeneities, red to the highest entropies and homogeneities. 




***
## 2002

### Use of acoustic classification of sidescan sonar data for mapping benthic habitat in the Northern Channel Islands, California

[Use of acoustic classification of sidescan sonar data for mapping benthic habitat in the Northern Channel Islands, California](https://scholar.google.com/scholar_url?url=http://www.academia.edu/download/40226423/Use_of_acoustic_classification_of_sidesc20151120-25765-xe5v88.pdf&hl=en&sa=T&oi=gsb-gga&ct=res&cd=0&ei=VAcIWNqfHJWhjAHKr5f4CA&scisig=AAGBfm2qvxIjS_d5QyK_AZdpySZMMF4XSA)


#### Motivation

- Wanted to classify
    - Sand
    - thin sand
    - Rock
    - Shadow

### Textural Anaysis
- Grey levels of pixels of thin sand were comparable to thos of rock outcrops
- GLCM properties are reduced to a single value for a single pixel.  
- Entropy mesures roughness of acoutic textures
- Homogeneity measure the degree of organization of the texture
- Slant range corrections indorduces an artifical texture and grey levels in the near-range pixels that mimis those of thin sand

![alt test][id2]
![alt test][id3]
***

- Disregaruded near and far range portions of the imagery

## 2007

### Object-oriented classification of sidescan sonar data for mapping benthic marine habitats

[Object‐oriented classification of sidescan sonar
data for mapping benthic marine habitats
](http://www.tandfonline.com/doi/pdf/10.1080/01431160701311309?needAccess=true)

##### Motivation
- Sidescan sonar provides one of the most cost effective methods for inital delineation of the seafloor into geological and geomorphphical regions
- Features can be identified at different scales
- Wanted to identify reef, sand, and low reef
- Habitat Mapping tool

###### GLCM variables
- Found no relationship wiht entorpy and homogeneaity
- GCLM mean, Mean, and StDev were the best
- Used nearest neighbor clasifer
- Used eCogniton software
    - Fuzzy class membership
***

## 2013

### Sidescan Sonar Imagery Segmentation with a Combination of Texture and Spectral Analysis 

[Sidescan Sonar Imagery Segmentation with a Combination of Texture and Spectral Analysis ](http://s3.amazonaws.com/academia.edu.documents/40538933/Sidescan_sonar_imagery_segmentation_with20151201-10078-1ason8w.pdf?AWSAccessKeyId=AKIAJ56TQJRTWSMTNPEA&Expires=1476926754&Signature=Dzi%2F9QJQCrOskNjU8ebeSbOgIng%3D&response-content-disposition=inline%3B%20filename%3DSidescan_sonar_imagery_segmentation_with.pdf)

#### Motivation

- Wants to detect
    - sand
    - Ripples
    - Posidonia
    - Rock
- Investiagte the ability of spectral features to discriminate betwen seabed textures
    - Directional filter bank (Fourier)
- Parametric segmentaion are based on modeling the probability distribution of signal backscattered from the seafloor
    - These models accoutn for data aquisition conditons, properties of the seabed, and the angular variationsof signal backscattering
- Nonparameteric segmentaion condiser seabed backscatter as a textured image.
- This paper uses the nonparametric method
- Two supervised classificaton algorithims:
    - Bayes naives
    - Multilayer perception
- Unsupervised classification techniques
    - K-means
    - Self organizing feature maps (SOFM)



#### Texture Analysis

- Feature extraction is based on first order statistics (single pixel), second order statistics (pixel pairs), and spectral analysis
- GLCM calculated in 4 directions (0°,45°,90°, 135°) and distances (1, 4, 8)
- Then calcuated normalized GLCM properties
- 
***

[id]: figures/homovsent.png
[id2]: figures/Cochrane_Lafferty.png
[id3]: figures/Cochrane_Lafferty2.png
[id4]: figures/blondel_2000.png
[id5]: figures/Chabane.png
