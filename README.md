# PartialLFW

We release a dataset to evaluate face verification performance for partial faces based on [Labeled Faces in the Wild dataset](http://vis-www.cs.umass.edu/lfw/). To generate partial faces, we crop rectangular face patches of nine different areas ranging from 9% to 73.7% of the original face around four landmarks: right/left eye, nose, and mouth. Next, we either leave the cropped face patches at their initial location and fill the remaining image with zeros (*non-centered*) or move the patch to the center and zero-pad to match the input resolution (*centered*). 

![dataset](https://github.com/stefhoer/partiallfw/raw/main/resources/dataset.png)

## Download

We provide the code to generate PartialLFW given LFW. Download the unaligned images from [http://vis-www.cs.umass.edu/lfw/](http://vis-www.cs.umass.edu/lfw/).

## Alignment

To align the faces we make use of 5 landmarks extracted by MTCNN [^1].  The alignment script is provided under  `align_dataset.py`, which is based on the code from  [^2].


## Generating Partial Faces

To generate the partial faces, run `gen_partial_faces.py`.  This will output the following folder structure `PartialLFW/part(_centered)/factor_0.1-0.5/...`

```shell
PartialLFW
└───lEye
│   └───factor_0.1
│   │   └───Aaron_Eckhart
│   │   │   │   Aaron_Eckhart_0001.png
│   │   ...
│   │   └───Zydrunas_Ilgauskas
│   └───factor_0.15
│   │   ...
│   └───factor_0.5
└───lEye_centered
│   └───factor_0.1
│   │   ...
│   └───factor_0.5
└───Mouth
└───Mouth_centered
└───Nose
└───Nose_centered
└───rEye
└───rEye_centered
```

## Protocols

We follow the [standard verification protocol](http://vis-www.cs.umass.edu/lfw/pairs.txt) from LFW of 6000 face pairs separated into 10 fold with 600 pairs (300 positive and 300 negative) each.

We divide our analysis into the following categories:

- centered
  - partial - holistic (left eye - holistic, nose - holistic, mouth - holistic)
  - partial - same (left eye - left eye, nose - nose , mouth - mouth )
  - partial - cross (left eye - right eye, left eye - nose , left eye - mouth , nose - mouth)
- non-centered
  - partial - holistic (left eye - holistic, nose - holistic, mouth - holistic)
  - partial - same (left eye - left eye, nose - nose , mouth - mouth )
  - partial - cross (left eye - right eye, left eye - nose , left eye - mouth , nose - mouth)

For our final verification accuracy of e.g. *centered: partial - holistic* we average over *left eye - holistic, nose - holistic, mouth - holistic* and over all 9 non-occluded areas (factors 0.1 - 0.5) - hence in total 27 benchmarks.

Since left eye and right eye contain substantial overlap and we only treat them separately in the *partial - cross* protocol.  

For the protocols *partial - holistic* and *partial - cross* the crops are not identical, which leaves us with two options: For *left eye - holistic* we can take the holistic image from the first face and crop around the left eye from the second face, and vice versa. Therefore, we consider both cases and expand the number of face pairs to 12000 with 10 folds of 1200 pairs. Note, that we do not cycle trough the list twice but rather double every pair, with the first image always being holistic and the second always being partial (for *partial - holistic*):

```shell
Abel_Pacheco	1	4
Abel_Pacheco	4	1
Akhmed_Zakayev	1	3
Akhmed_Zakayev	3	1
```

 The following images depict the three protocols for the non-centered case:

![partial-holistic](https://github.com/stefhoer/partiallfw/raw/main/resources/partial-holistic.png)  ![partial-same](https://github.com/stefhoer/partiallfw/raw/main/resources/partial-same.png)  ![partial-cross](https://github.com/stefhoer/partiallfw/raw/main/resources/partial-cross.png)

## Results

|                         |              |      non-centered      |    non-centered    |    non-centered     |        centered        |      centered      |      centered       |
| ----------------------- | :----------: | :--------------------: | :----------------: | :-----------------: | :--------------------: | :----------------: | :-----------------: |
| **Approach**            | **holistic** | **partial - holistic** | **partial - same** | **partial - cross** | **partial - holistic** | **partial - same** | **partial - cross** |
| ResNet-41               |    99.62     |         97.71          |       97.27        |        94.53        |         97.25          |       96.80        |        93.56        |
| ResNet-50 (no finetune) |    99.58     |         94.77          |       94.93        |        88.85        |         92.05          |       92.47        |        83.92        |
| ResNet-50               |    99.60     |         97.75          |       97.36        |        94.80        |         95.48          |       94.72        |        89.60        |
| ours                    |  **99.70**   |       **98.03**        |     **97.66**      |      **94.90**      |       **97.64**        |     **97.16**      |      **93.87**      |


## Cite

If you find PartialLFW useful in your research, please cite the following papers:

~~~tex
```
tbd

@TechReport{LFWTechUpdate,
  author={Huang, Gary B and Learned-Miller, Erik},
  title={Labeled Faces in the Wild: Updates and New Reporting Procedures},
  institution={University of Massachusetts, Amherst},
  year={2014},
  number={UM-CS-2014-003},
  month= {May}}
```
~~~

## References

[^1]: K. Zhang, Z. Zhang, Z. Li, and Y. Qiao, “Joint face detection and alignment using multitask cascaded convolutional networks,” IEEE Signal Processing Letters, vol. 23, no. 10, pp. 1499–1503, 2016
[^2]: [https://github.com/davidsandberg/facenet](https://github.com/davidsandberg/facenet) and [https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection](https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection)



## Contact

[Stefan Hörmann (s.hoermann@tum.de)](mailto:s.hoermann@tum.de)

