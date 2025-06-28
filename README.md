
# Personalized Behavior-Aware Transformer for Multi-Behavior Sequential Recommendation
This is the implementation for the paper "Personalized Behavior-Aware Transformer for Multi-Behavior Sequential Recommendation", ACM Multimedia 2023.
![](PBAT.jpg)

</div>

</div>

## Quick Start
0. install pytorch and other dependencies
```bash
pip install -r requirements.txt
```
1. run the model with a `yaml` configuration file like following:
```bash
python run.py fit --config src/configs/retail.yaml
```

## Dataset
Due to file size limitations, we have not uploaded all of the data. You can download the datasets from [Google Drive](https://drive.google.com/file/d/1Ki8ibuTgkxnBITCe4gNLtrR0mG1vxevl/view?usp=sharing).

## Citation
If you find the code helpful, please cite us as:
```
@inproceedings{DBLP:conf/mm/Su0LLLZ23,
  author       = {Jiajie Su and
                  Chaochao Chen and
                  Zibin Lin and
                  Xi Li and
                  Weiming Liu and
                  Xiaolin Zheng},
  editor       = {Abdulmotaleb El{-}Saddik and
                  Tao Mei and
                  Rita Cucchiara and
                  Marco Bertini and
                  Diana Patricia Tobon Vallejo and
                  Pradeep K. Atrey and
                  M. Shamim Hossain},
  title        = {Personalized Behavior-Aware Transformer for Multi-Behavior Sequential
                  Recommendation},
  booktitle    = {Proceedings of the 31st {ACM} International Conference on Multimedia,
                  {MM} 2023, Ottawa, ON, Canada, 29 October 2023- 3 November 2023},
  pages        = {6321--6331},
  publisher    = {{ACM}},
  year         = {2023},
  url          = {https://doi.org/10.1145/3581783.3611723},
  doi          = {10.1145/3581783.3611723},
  timestamp    = {Fri, 15 Dec 2023 14:50:53 +0100},
  biburl       = {https://dblp.org/rec/conf/mm/Su0LLLZ23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
## Acknowledgements
Our code is based on the implementation of [MB-STR](https://github.com/yuanenming/mb-str).
