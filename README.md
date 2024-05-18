# VISION : VISION: VIdeo StabilisatION using automatic features selection

A python port for VISION: VIdeo StabilisatION using automatic features selection for image velocimetry analysis in rivers.

## Input

- `filename`: The name of the video file, e.g.: 'RawVideo.avi'
- `NameStabilisedVideo`: Name of the stabilised video.
- `algorithms`: List of algorithms to use for feature detection. You can use up to two algorithms at the same time. e.g.: ['FAST']. ['FAST', 'KAZE'].
    - The following algorithms have been implemented: 'FAST' / 'MINEIGEN' / 'KAZE' / 'BRISK' / 'ORB'
    - Default: FAST.
    - Note: The algorithms 'HARRIS' and 'SURF' are implemented in the original version but not yet available in this version.
- `PercentualStrongestFeatures`: Percentual value of the strongest features detected to be used for stabilisation analyses. Range of accepted values: ]0, 100]. 100 means all the features to be considered. An uniform filter is afterwards applied to remain with 50% of the value entered by the user of uniformly distributed features. Default: 100%.
- `TransformType`: Transformation type among 'similarity', 'affine', or 'projective'. Minimum number of matched points: similarity < affine < projective. Better accuracy of estimated transformation when greater the number of matched points is. Default: similarity.
- `ROISelection`: Two possible options: i) Binary decision, and ii) Rectangular ROI introduced a priori in a vector format. Binary decision giving the possibility to define a ROI for stabilisation analysis. 1 means ROI, while 0 means all the image in question. Rectangular ROI in a vector format: e.g., [0 0 1000 1000]. Default: all the field of view.
- `StabilisationViewer`: Binary decision giving the possibility to open a viewer to see how the stabilisation goes. 1 means viewer, while 0 means no viewer. Default: 1 (viewer)

## Outputs

    Number of Frames
    Frame Per Seconds (FPS)
    Region Of Interest (ROI)

## Dependencies

    Python 3.7+
    Numpy 1.19.5+
    opencv-contrib-python 4.9.0+

## Example

    [NFrames,FPS,ROI] = VISION('Belgrade_15frames.avi',NameStabilisedVideo =' TestVideo',
    algorithms = ['FAST'],PercentualStrongestFeatures = 30,
    TransformType = 'affine',ROISelection = [1 1 1000 1000],
    StabilisationViewer = 0)

## Authors:

Alonso Pizarro (1)
Silvano F. Dal Sasso (2)
Salvatore Manfreda (3)

#### Python porter:

Diego Fischer (4)(<difischer@uc.cl>)

- (1) Universidad Diego Portales, Chile | <alonso.pizarro@mail.udp.cl> (<https://orcid.org/0000-0002-7242-6559>)
- (2) Università Degli Studi Della Basilicata, Italy | <silvano.dalsasso@unibas.it> (<https://orcid.org/0000-0003-1376-7764>)
- (3) University of Naples Federico II, Italy | <salvatore.manfreda@unina.it> (<https://orcid.org/0000-0002-0225-144X>)
- (4) Pontificia Unviersidad Católica de Chile | <difischer@uc.cl>

# Note:

Diego Fischer is not a member of the code nor the paper's authors. He is only the Python porter of the code.

Copyright (C) 2021 Alonso Pizarro, Silvano F. Dal Sasso & Salvatore Manfreda
This program is free software (BSD 3-Clause) and distributed WITHOUT ANY
WARRANTY.