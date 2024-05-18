"""#
VISION: VIdeo StabilisatION using automatic features selection

VISION.py is a python port of VISION.m, a function aiming at stabilising videos for
image velocimetry analyses in rivers.


INPUT:                                                                
         filename  
         algorithms                                                         
[1] filename: The name of the video file, e.g.: 'RawVideo.avi'                                                                                                                                          
                                                                                                    
[2] NameStabilisedVideo: Name of the stabilised video.                  
                                                                                                    
[3] algorithms:  List of algorithms to use for feature detection. You can use up to two
                algorithms at the same time. e.g.: ['FAST']. ['FAST', 'KAZE'].
                                                                   
                The following algorithms have been implemented:                                      
                'FAST' / 'MINEIGEN' / 'KAZE' / 'BRISK' / 'ORB' /                                     
                Default: FAST.
                
                The algoritms 'HARRIS' and 'SURF' are implemented on the original
                version but not yet available in this version

[3] PercentualStrongestFeatures: Percentual value of the strongest features detected to be           
                                used for stabilisation analyses. Range of accepted values:          
                                ]0, 100]. 100 means all the features to be considered.              
                                An uniform filter is afterwards applied to remain with the          
                                50% of the value entered by the user of uniformly distributed       
                                features. Default: 100%.                                            

[4] TransformType: Transformation type among 'similarity', 'affine', or 'projective'.                
                Minimum number of matched points: similarity < affine < projective.               
                Better accuracy of estimated transformation when greater the number               
                of matched points is. Default: similarity.                                        

[5] ROISelection: Two possible options: i) Binary decision, and ii) Rectangular ROI introduced       
                                        a priori in a vector format. Binary decision giving the      
                                        possibilitity to define a ROI for stabilisation analysis.    
                                        1 means ROI, while 0 means all the image in question.        
                                        Rectangular ROI in a vector format: e.g., [0 0 1000 1000].   
                                        Default: all the field of view.                              

[6] StabilisationViewer: Binary decision giving the possibilitity to open a viewer to see how the    
                        stabilisation goes. 1 means viewer, while 0 no viewer. Default: 1 (viewer)  


OUTPUTS:

[1] Number of Frames
[2] Frame Per Seconds (FPS)
[3] Region Of Interest (ROI)

Dependencies:
    - Python 3.7+
    - Numpy 1.19.5+
    - opencv-contrib-python 4.9.0+                                                             

Example of how to call VISION:
------------------------------

[NFrames,FPS,ROI] = VISION('Belgrade_15frames.avi',NameStabilisedVideo =' TestVideo',
                            algorithms = ['FAST],PercentualStrongestFeatures = 30,
                            'TransformType','affine','ROISelection',[1 1 1000 1000],
                            'StabilisationViewer',0)

Authors:
    Alonso Pizarro (1)
    Silvano F. Dal Sasso (2)
    Salvatore Manfreda (3)
    
Python porter:
    Diego Fischer (4)(difischer@uc.cl)

Version History:
    15/02/2021 - V0.0.1
    08/04/2021 - V0.0.2
    01/03/2021 - V0.0.3
    16/04/2021 - V0.0.4 (Python port)

(1) Universidad Diego Portales, Chile | alonso.pizarro@mail.udp.cl (https://orcid.org/0000-0002-7242-6559)
(2) Università Degli Studi Della Basilicata, Italy | silvano.dalsasso@unibas.it (https://orcid.org/0000-0003-1376-7764)
(3) University of Naples Federico II, Italy | salvatore.manfreda@unina.it (https://orcid.org/0000-0002-0225-144X)
---
(4) Pontificia Unviersidad Católica de Chile -- Capta Hydro | difischer@uc.cl

Note:
Diego Fischer is not a member of the code nor the paper's authors. 
He is only the Python porter of the code.

Copyright (C) 2021 Alonso Pizarro, Silvano F. Dal Sasso & Salvatore Manfreda
This program is free software (BSD 3-Clause) and distributed WITHOUT ANY
WARRANTY.
"""

import cv2
import numpy as np
import warnings
import argparse

def VISION(
    filename,
    NameStabilisedVideo="StabilisedVideo",
    algorithms=["FAST"],
    PercentualStrongestFeatures=100,
    TransformType="similarity",
    ROISelection=1,
    StabilisationViewer=0,
):
    """
    VISION function for video stabilization using automatic features selection.

    Args:
        filename (str): The name of the video file.
        NameStabilisedVideo (str, optional): Name of the stabilised video. Defaults
            to "StabilisedVideo".
        algorithms (list, optional): The algorithm(s) used for features' detection.
            Defaults to ["FAST"].
        PercentualStrongestFeatures (int, optional): Percentual value of the strongest
            features detected to be used for stabilisation analyses. Defaults to 100.
        TransformType (str, optional): Transformation type among 'similarity',
            'affine', or 'projective'. Defaults to "similarity".
        ROISelection (int, optional): Binary decision giving the possibility to define
            a ROI for stabilisation analysis. Defaults to 1.
        StabilisationViewer (int, optional): Binary decision giving the possibility to
            open a viewer to see how the stabilisation goes. Defaults to 0.
            IMPORTANT: This parameter is not implemented in this version.

    Returns:
        tuple: A tuple containing the following outputs:
            - Numbre of Frames
            - Frame Per Seconds (FPS)
            - Region Of Interest (ROI)
    """
    # Reading Frames from a Video File
    filename = str(filename)
    print(f"abriendo video en {filename}")
    VideoFrameReader = cv2.VideoCapture(filename)
    FPS = VideoFrameReader.get(cv2.CAP_PROP_FPS)
    NFrames = int(VideoFrameReader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Processing all frames in the Video
    VideoFrameReader.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Initialising...
    ret, imgA = VideoFrameReader.read()  # First frame taken as the reference image
    imgB = imgA.copy()
    imgBp = imgA.copy()
    correctedMean = imgA.copy()

    # Creating Video file to write stabilised frames
    if len(algorithms) == 1:
        NameStabilisedVideo = NameStabilisedVideo + "_" + algorithms[0] + ".avi"
    elif len(algorithms) == 2:
        NameStabilisedVideo = (
            NameStabilisedVideo + "_" + algorithms[0] + "_" + algorithms[1] + ".avi"
        )
    else:
        raise ValueError("ERROR!!! feature detection algorithms unclear")
    print(f"intentando guardar a {NameStabilisedVideo}")
    v = cv2.VideoWriter(
        NameStabilisedVideo,
        cv2.VideoWriter_fourcc(*"XVID"),
        FPS,
        (imgA.shape[1], imgA.shape[0]),
    )  # Name of the Stabilised Video

    v.write(imgA)  # Saving reference image

    # ROI Selection
    roi_clear = True
    if type(ROISelection) == list:
        if len(ROISelection) == 4:
            ROI = ROISelection
        else:
            roi_clear = False
    elif type(ROISelection) == int:
        if ROISelection == 1:
            ROI = cv2.selectROI(imgA)
        elif ROISelection == 0:
            ROI = np.nan
        else:
            roi_clear = False
    if roi_clear == False:
        raise ValueError("ERROR!!! ROI Decision unclear")

    # Analysis
    Cont1 = 0  # Counting the frame number to filter geometric transformation
    while True:
        imgAp = imgBp
        ret, imgB = VideoFrameReader.read()

        if not ret:
            break

        # Estimating Geometric Transformations
        if type(ROISelection) == list and len(ROISelection) == 4:
            H = GeometricTransformStabilisation(
                imgA[ROI[1] : ROI[1] + ROI[3], ROI[0] : ROI[0] + ROI[2]],
                imgB[ROI[1] : ROI[1] + ROI[3], ROI[0] : ROI[0] + ROI[2]],
                algorithms,
                PercentualStrongestFeatures,
                TransformType,
            )
        elif ROISelection == 1:
            H = GeometricTransformStabilisation(
                imgA[ROI[1] : ROI[1] + ROI[3], ROI[0] : ROI[0] + ROI[2]],
                imgB[ROI[1] : ROI[1] + ROI[3], ROI[0] : ROI[0] + ROI[2]],
                algorithms,
                PercentualStrongestFeatures,
                TransformType,
            )

        # Applying Geometric Transformations
        H = np.append(H, [[0, 0, 1]], axis=0)
        H = np.float32(H)  # Convert H to float32 type
        imgBp = cv2.warpPerspective(imgB, H, (imgA.shape[1], imgA.shape[0]))

        # Saving Stabilised Frames
        v.write(imgBp)
        Cont1 += 1
        show_cont = True
        # Displaying as color composite with last corrected frame
        if StabilisationViewer == 1:
            cv2.imshow("Stabilised Frames", cv2.addWeighted(imgAp, 0.5, imgBp, 0.5, 0))
        elif StabilisationViewer == 0:
            if int(Cont1) % 10 == 0:
                print("Progress: {:.2f}%".format(100 * Cont1 / NFrames))

        else:
            raise ValueError("ERROR!!! Stabilisation Viewer Decision unclear")

        correctedMean += imgBp

    VideoFrameReader.release()
    v.release()
    return NFrames, FPS, ROI


def GeometricTransformStabilisation(
    imgA, imgB, FeaturesDetection, StrongestFeatures_Decision, TransformType
):
    """
    Perform geometric transform stabilisation on two images.

    Args:
        imgA (numpy.ndarray): First input image.
        imgB (numpy.ndarray): Second input image.
        FeaturesDetection (list): List of feature detection algorithms to be used.
        StrongestFeatures_Decision (int): Percentage of strongest and uniform features
            to be considered.
        TransformType (str): Type of geometric transform to be applied.

    Raises:
        ValueError: If the percentage of strongest and uniform features
            is out of the valid range.

    Returns:
        numpy.ndarray: Matched points in image A.
        numpy.ndarray: Matched points in image B.
    """

    MatchedPointsA = {}
    MatchedPointsB = {}
    for DetAlgorithm in FeaturesDetection:
        if 0 < StrongestFeatures_Decision <= 100:
            if DetAlgorithm == "FAST":
                pointsA = cv2.FastFeatureDetector_create().detect(imgA)
                pointsB = cv2.FastFeatureDetector_create().detect(imgB)

            elif DetAlgorithm == "MINEIGEN":
                pointsA = cv2.GFTTDetector_create().detect(imgA)
                pointsB = cv2.GFTTDetector_create().detect(imgB)

            elif DetAlgorithm == "HARRIS":
                raise ValueError(
                    "ERROR!!! Harris algorithm not yet implemented in this version"
                )
                # Rest of the code for Harris algorithm

            elif DetAlgorithm == "SURF":
                warnings.warn("SURF is an exclusively noncommercial algorithm")
                raise ValueError(
                    "ERROR!!! SURF algorithm not yet implemented in this version"
                )
                # Rest of the code for SURF algorithm

            elif DetAlgorithm == "KAZE":
                pointsA = cv2.KAZE_create().detect(imgA)
                pointsB = cv2.KAZE_create().detect(imgB)

            elif DetAlgorithm == "BRISK":
                pointsA = cv2.GFTTDetecBRISK_createtor_create().detect(imgA)
                pointsB = cv2.GFTTDetecBRISK_createtor_create().detect(imgB)

            elif DetAlgorithm == "ORB":
                pointsA = cv2.ORB_create().detect(imgA)
                pointsB = cv2.ORB_create().detect(imgB)

            else:
                raise ValueError("ERROR!!! No feature detection algorithm provided")

            strongestPointsA = sorted(pointsA, key=lambda x: x.response, reverse=True)[
                : int((StrongestFeatures_Decision / 100) * len(pointsA))
            ]
            strongestPointsB = sorted(pointsB, key=lambda x: x.response, reverse=True)[
                : int((StrongestFeatures_Decision / 100) * len(pointsB))
            ]

            pointsA = cv2.KeyPoint_convert(
                cv2.KeyPoint_convert(strongestPointsA)[
                    : int(0.5 * len(strongestPointsA))
                ]
            )
            pointsB = cv2.KeyPoint_convert(
                cv2.KeyPoint_convert(strongestPointsB)[
                    : int(0.5 * len(strongestPointsB))
                ]
            )

            # Create a BRIEF extractor object
            brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
            # Compute the descriptors with BRIEF
            pointsA, descriptorsA = brief.compute(imgA, pointsA)
            pointsB, descriptorsB = brief.compute(imgB, pointsB)

            indexPairs = cv2.BFMatcher().knnMatch(descriptorsA, descriptorsB, k=2)
            goodMatches = []
            for m, n in indexPairs:
                if m.distance < 0.75 * n.distance:
                    goodMatches.append(m)

            MatchedPointsA[DetAlgorithm] = np.float32(
                [pointsA[m.queryIdx].pt for m in goodMatches]
            ).reshape(-1, 2)
            MatchedPointsB[DetAlgorithm] = np.float32(
                [pointsB[m.trainIdx].pt for m in goodMatches]
            ).reshape(-1, 2)

        else:
            raise ValueError(
                "ERROR!!! Percentage of strongest and uniform features out of limit (give a value between ]0, 100])"
            )

    if len(FeaturesDetection) == 1:
        DetAlgorithm = FeaturesDetection[0]
        H, _ = cv2.estimateAffinePartial2D(
            MatchedPointsB[DetAlgorithm],
            MatchedPointsA[DetAlgorithm],
            method=cv2.RANSAC,
        )
    elif len(FeaturesDetection) == 2:
        matchedOriginalXY = np.concatenate(
            (
                MatchedPointsA[FeaturesDetection[0]],
                MatchedPointsA[FeaturesDetection[1]],
            ),
            axis=0,
        )
        matchedDistortedXY = np.concatenate(
            (
                MatchedPointsB[FeaturesDetection[0]],
                MatchedPointsB[FeaturesDetection[1]],
            ),
            axis=0,
        )

        H, _ = cv2.estimateAffinePartial2D(
            matchedDistortedXY, matchedOriginalXY, method=cv2.RANSAC
        )
    else:
        raise ValueError("ERROR!!! Maximum supported feature detection algorithms: 2")

    return H


def user_input():
    # Taking command line arguments from users
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        "--input",
                        help="Filename of input video",
                        type=str,
                        required=True)
    parser.add_argument(
        "-o",
        "--output",
        help="Name of Stabilised Video",
        type=str,
        required=False,
        default="StabilisedVideo"
    )
    # No se como pedir una lista así que lo dejos así no más,
    # Tal vez con un string y usando split en el espacio.
    # parser.add_argument(
    #     "-a",
    #     "--algorithms",
    #     help="Algorithms to be used",
    #     type=str,
    #     required=False,
    # )
    parser.add_argument(
        "-pr",
        "--PercentOfStrongest",
        help="Percentual of Strongest Features to keep",
        type=int,
        required=False,
        choices=range(0,100),
        default=100
    )
    parser.add_argument("-t", "--TransfomType",
                        help="Type of transform to stabilize",
                        type=str,
                        required=False,
                        choices=["similarity", "affine", "projective"],
                        default= "similarity")
    
    # parser.add_argument(
    #     "-roi",
    #     "--ROISelection",
    #     help="ROI of stable area",
    #     type=str,
    #     required=False,
    # )
    
    args = parser.parse_args()
    

    arguments = vars(args)
    return arguments


if __name__ == "__main__":
    args = user_input()
    print(args)
    [NFrames, FPS, ROI]= VISION(args["input"],
                                NameStabilisedVideo=args["output"],
                                algorithms=["KAZE", "FAST"],
                                PercentualStrongestFeatures=args["PercentOfStrongest"],
                                TransformType=args["TransfomType"])
    #Ejemplos de como correrlo a mano
    #[NFrames, FPS, ROI] = VISION("videos/olas.mp4",NameStabilisedVideo="olas", algorithms=["KAZE", "FAST"])
    #[NFrames, FPS, ROI] = VISION("dataset/videos/IMG_1737.mp4",NameStabilisedVideo="regleta_roi_agua", algorithms=["KAZE", "FAST"])
