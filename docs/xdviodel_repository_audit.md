# Official XDVioDet Repository Audit

**Audit time (UTC):** 2026-10-02T06:21:10.936739+00:00

## Repository Identity

| Field | Value |
|---|---|
| Intended official repository | https://github.com/Roc-Ng/XDVioDet.git |
| Local path | `C:\Users\abbub\OneDrive\Desktop\Experiment\cctv-selective-retention\external\XDVioDet` |
| Remote URL | https://github.com/Roc-Ng/XDVioDet.git |
| Git commit | `f846d8cfc454943ec165bd2489b71e7dc0a1064d` |
| Clone status | CLONED |

## Research Role

This repository is inspected as the detector baseline for the XD-Violence
feature-level experiments. It is not the proposed journal contribution.

The baseline is used to generate temporally indexed video-only and audio-video
event scores. Those scores will subsequently be calibrated and used by
selective evidence-preservation policies.

## Checkpoint Files Found Locally

| File | Size in bytes | SHA-256 |
|---|---:|---|
| `ckpt\wsanodet_mix2.pkl` | 3,382,342 | `27067896258294da0a711e3f0c071190ea5848961298cbde389265627191ec2c` |

**Checkpoint decision rule:** A checkpoint can be used only if it is present
locally or is linked from an official repository/documented author source. Its
path, source URL, Git commit, SHA-256 hash, expected input features, and
inference configuration must be recorded before it is used.

## Feature References Found in Code

| File | Line | Reference |
|---|---:|---|
| `dataset.py` | 11 | `self.rgb_list_file = args.test_rgb_list` |
| `dataset.py` | 12 | `self.flow_list_file = args.test_flow_list` |
| `dataset.py` | 15 | `self.rgb_list_file = args.rgb_list` |
| `dataset.py` | 16 | `self.flow_list_file = args.flow_list` |
| `dataset.py` | 27 | `elif self.modality == 'RGB':` |
| `dataset.py` | 28 | `self.list = list(open(self.rgb_list_file))` |
| `dataset.py` | 29 | `elif self.modality == 'FLOW':` |
| `dataset.py` | 30 | `self.list = list(open(self.flow_list_file))` |
| `dataset.py` | 32 | `self.list = list(open(self.rgb_list_file))` |
| `dataset.py` | 33 | `self.flow_list = list(open(self.flow_list_file))` |
| `dataset.py` | 35 | `self.list = list(open(self.rgb_list_file))` |
| `dataset.py` | 38 | `self.list = list(open(self.flow_list_file))` |
| `dataset.py` | 41 | `self.list = list(open(self.rgb_list_file))` |
| `dataset.py` | 42 | `self.flow_list = list(open(self.flow_list_file))` |
| `dataset.py` | 54 | `features = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)` |
| `dataset.py` | 55 | `elif self.modality == 'RGB':` |
| `dataset.py` | 56 | `features = np.array(np.load(self.list[index].strip('\n')),dtype=np.float32)` |
| `dataset.py` | 57 | `elif self.modality == 'FLOW':` |
| `dataset.py` | 58 | `features = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)` |
| `dataset.py` | 60 | `features1 = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)` |
| `dataset.py` | 61 | `features2 = np.array(np.load(self.flow_list[index].strip('\n')), dtype=np.float32)` |
| `dataset.py` | 62 | `if features1.shape[0] == features2.shape[0]:` |
| `dataset.py` | 63 | `features = np.concatenate((features1, features2),axis=1)` |
| `dataset.py` | 64 | `else:# because the frames of flow is one less than that of rgb` |
| `dataset.py` | 65 | `features = np.concatenate((features1[:-1], features2), axis=1)` |
| `dataset.py` | 67 | `features1 = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)` |
| `dataset.py` | 68 | `features2 = np.array(np.load(self.audio_list[index//5].strip('\n')), dtype=np.float32)` |
| `dataset.py` | 69 | `if features1.shape[0] == features2.shape[0]:` |
| `dataset.py` | 70 | `features = np.concatenate((features1, features2),axis=1)` |
| `dataset.py` | 71 | `else:# because the frames of flow is one less than that of rgb` |
| `dataset.py` | 72 | `features = np.concatenate((features1[:-1], features2), axis=1)` |
| `dataset.py` | 74 | `features1 = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)` |
| `dataset.py` | 75 | `features2 = np.array(np.load(self.audio_list[index//5].strip('\n')), dtype=np.float32)` |
| `dataset.py` | 76 | `if features1.shape[0] == features2.shape[0]:` |
| `dataset.py` | 77 | `features = np.concatenate((features1, features2),axis=1)` |
| `dataset.py` | 78 | `else:# because the frames of flow is one less than that of rgb` |
| `dataset.py` | 79 | `features = np.concatenate((features1[:-1], features2), axis=1)` |
| `dataset.py` | 81 | `features1 = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)` |
| `dataset.py` | 82 | `features2 = np.array(np.load(self.flow_list[index].strip('\n')), dtype=np.float32)` |
| `dataset.py` | 83 | `features3 = np.array(np.load(self.audio_list[index//5].strip('\n')), dtype=np.float32)` |
| `dataset.py` | 84 | `if features1.shape[0] == features2.shape[0]:` |
| `dataset.py` | 85 | `features = np.concatenate((features1, features2, features3),axis=1)` |
| `dataset.py` | 86 | `else:# because the frames of flow is one less than that of rgb` |
| `dataset.py` | 87 | `features = np.concatenate((features1[:-1], features2, features3[:-1]), axis=1)` |
| `dataset.py` | 91 | `features = self.tranform(features)` |
| `dataset.py` | 93 | `return features` |
| `dataset.py` | 96 | `features = process_feat(features, self.max_seqlen, is_random=False)` |
| `dataset.py` | 97 | `return features, label` |
| `layers.py` | 16 | `def __init__(self, in_features, out_features, dropout, alpha, concat=True):` |
| `layers.py` | 19 | `self.in_features = in_features` |
| `layers.py` | 20 | `self.out_features = out_features` |
| `layers.py` | 24 | `self.W = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(in_features, out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)` |
| `layers.py` | 25 | `self.a = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(2*out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)` |
| `layers.py` | 33 | `a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)` |
| `layers.py` | 48 | `return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'` |
| `layers.py` | 51 | `def __init__(self, in_features, out_features):` |
| `layers.py` | 53 | `self.weight = Parameter(FloatTensor(in_features, out_features))` |
| `layers.py` | 66 | `def __init__(self, in_features, out_features, bias=False, residual=True):` |
| `layers.py` | 68 | `self.in_features = in_features` |
| `layers.py` | 69 | `self.out_features = out_features` |
| `layers.py` | 70 | `self.weight = Parameter(FloatTensor(in_features, out_features))` |
| `layers.py` | 73 | `self.bias = Parameter(FloatTensor(out_features))` |
| `layers.py` | 79 | `elif (in_features == out_features):` |
| `layers.py` | 82 | `# self.residual = linear(in_features, out_features)` |
| `layers.py` | 83 | `self.residual = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=5, padding=2)` |
| `layers.py` | 97 | `if self.in_features != self.out_features and self.residual:` |
| `layers.py` | 109 | `+ str(self.in_features) + ' -> ' \` |
| `layers.py` | 110 | `+ str(self.out_features) + ')'` |
| `layers.py` | 116 | `def __init__(self, in_features, out_features):` |
| `layers.py` | 118 | `self.in_features = in_features` |
| `layers.py` | 119 | `self.out_features = out_features` |
| `layers.py` | 121 | `self.weight0 = Parameter(FloatTensor(in_features, out_features))` |
| `layers.py` | 122 | `self.weight1 = Parameter(FloatTensor(in_features, out_features))` |
| `layers.py` | 164 | `+ str(self.in_features) + ' -> ' \` |
| `layers.py` | 165 | `+ str(self.out_features) + ')'` |
| `model.py` | 19 | `n_features = args.feature_size` |
| `model.py` | 22 | `self.conv1d1 = nn.Conv1d(in_channels=n_features, out_channels=512, kernel_size=1, padding=0)` |
| `model.py` | 33 | `self.simAdj = SimilarityAdj(n_features, 32)` |
| `option.py` | 4 | `parser.add_argument('--modality', default='MIX2', help='the type of the input, AUDIO,RGB,FLOW, MIX1, MIX2, or MIX3, MIX_ALL')` |
| `option.py` | 5 | `parser.add_argument('--rgb-list', default='list/rgb.list', help='list of rgb features ')` |
| `option.py` | 6 | `parser.add_argument('--flow-list', default='list/flow.list', help='list of flow features')` |
| `option.py` | 7 | `parser.add_argument('--audio-list', default='list/audio.list', help='list of audio features')` |
| `option.py` | 8 | `parser.add_argument('--test-rgb-list', default='list/rgb_test.list', help='list of test rgb features ')` |
| `option.py` | 9 | `parser.add_argument('--test-flow-list', default='list/flow_test.list', help='list of test flow features')` |
| `option.py` | 10 | `parser.add_argument('--test-audio-list', default='list/audio_test.list', help='list of test audio features')` |
| `option.py` | 11 | `parser.add_argument('--gt', default='list/gt.npy', help='file of ground truth ')` |
| `option.py` | 18 | `parser.add_argument('--feature-size', type=int, default=1024+128, help='size of feature (default: 2048)')` |
| `README.md` | 4 | `The project website is [XD-Violence](https://roc-ng.github.io/XD-Violence/). The features can be downloaded from our project website.` |
| `README.md` | 6 | `where we oversample each video frame with the “5-crop” augment, “5-crop” means cropping images into the center and four corners. _0.npy is the center, _1~ _4.npy is the corners.` |
| `README.md` | 9 | `* download or extract the features.` |
| `list\make_gt.py` | 9 | `feature_list = 'rgb_test.list'` |
| `list\make_gt.py` | 15 | `lists = list(open(feature_list))` |
| `list\make_gt.py` | 20 | `if '__0.npy' not in name:` |
| `list\make_gt.py` | 41 | `mod = (lens-1) % clip_len # minusing 1 is to align flow  rgb: minusing 1 when extracting features` |
| `list\make_gt.py` | 50 | `np.save('gt.npy', gt)` |
| `list\make_list.py` | 4 | `root_path = 'xx/vggish-features/test'    ## the path of features` |
| `list\make_list.py` | 5 | `files = sorted(glob.glob(os.path.join(root_path, "*.npy")))` |
| `list\make_list.py` | 8 | `with open('audio_test.list', 'w+') as f:  ## the name of feature list` |

## Label or Split References Found in Code

| File | Line | Reference |
|---|---:|---|
| `dataset.py` | 21 | `self.normal_flag = '_label_A'` |
| `dataset.py` | 49 | `label = 0.0` |
| `dataset.py` | 51 | `label = 1.0` |
| `dataset.py` | 97 | `return features, label` |
| `option.py` | 8 | `parser.add_argument('--test-rgb-list', default='list/rgb_test.list', help='list of test rgb features ')` |
| `option.py` | 9 | `parser.add_argument('--test-flow-list', default='list/flow_test.list', help='list of test flow features')` |
| `option.py` | 10 | `parser.add_argument('--test-audio-list', default='list/audio_test.list', help='list of test audio features')` |
| `train.py` | 4 | `def CLAS(logits, label, seq_len, criterion, device, is_topk=True):` |
| `train.py` | 17 | `clsloss = criterion(instance_logits, label)` |
| `train.py` | 35 | `for i, (input, label) in enumerate(dataloader):` |
| `train.py` | 38 | `input, label = input.float().to(device), label.float().to(device)` |
| `train.py` | 40 | `clsloss = CLAS(logits, label, seq_len, criterion, device, is_topk)` |
| `train.py` | 41 | `clsloss2 = CLAS(logits2, label, seq_len, criterion, device, is_topk)` |
| `list\make_gt.py` | 9 | `feature_list = 'rgb_test.list'` |
| `list\make_gt.py` | 12 | `gt_txt = 'xx/annotations.txt'     ## the path of test annotations` |
| `list\make_gt.py` | 30 | `if '_label_A' not in name:` |
| `list\make_list.py` | 8 | `with open('audio_test.list', 'w+') as f:  ## the name of feature list` |
| `list\make_list.py` | 10 | `if '_label_A' in file:` |

## URLs Found in Repository Files

- https://arxiv.org/abs/1710.10903
- https://arxiv.org/abs/1609.02907
- https://roc-ng.github.io/XD-Violence/

## Required Manual Review

1. Read the repository README.
2. Confirm whether an official pretrained checkpoint is supplied or linked.
3. Identify the exact visual feature requirement: I3D RGB, I3D Flow, or both.
4. Identify the exact audio feature requirement: VGGish or another format.
5. Identify the expected feature-directory names and file naming convention.
6. Identify train, validation, and test list files.
7. Do not download any large feature archive until these requirements are clear.
