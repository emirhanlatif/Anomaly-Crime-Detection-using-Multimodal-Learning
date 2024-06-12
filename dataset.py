import torch.utils.data as data
import numpy as np
from utils import process_feat, get_rgb_list_file
import torch
from torch.utils.data import DataLoader
torch.set_default_tensor_type('torch.cuda.FloatTensor')


class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.modality = args.modality
        self.emb_folder = args.emb_folder
        self.is_normal = is_normal
        self.dataset = args.dataset
        self.feature_size = args.feature_size
        if args.test_rgb_list is None:
            _, self.rgb_list_file = get_rgb_list_file(args.dataset, test_mode)
        else:
            self.rgb_list_file = args.test_rgb_list

        # deal with different I3D feature version
        if 'v2' in self.dataset:
            self.feat_ver = 'v2'
        elif 'v3' in self.dataset:
            self.feat_ver = 'v3'
        else:
            self.feat_ver = 'v1'

        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None


    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:  # list for training would need to be ordered from normal to abnormal
            if 'violence' in self.dataset:
                if self.is_normal:
                    self.list = self.list[1904:]
                    print('normal list for violence')
                else:
                    self.list = self.list[:1904]
                    print('abnormal list for violence')
            else:
                raise Exception("Dataset undefined!!!")

    def __getitem__(self, index):

        label = self.get_label()  # get video level label 0/1
        i3d_path = self.list[index].strip('\n')

        if self.feat_ver == 'v2':
            i3d_path = i3d_path.replace('i3d_v1', 'i3d_v2')
        elif self.feat_ver == 'v3':
            i3d_path = i3d_path.replace('i3d_v1', 'i3d_v3')

        features = np.load(i3d_path, allow_pickle=True)
        features = np.array(features, dtype=np.float32)

        if 'violence' in self.dataset:
            text_path = "save/Violence/" + self.emb_folder + "/" + i3d_path.split("\\")[-1][:-8]+"_emb.npy"
        else:
            raise Exception("Dataset undefined!!!")
        text_features = np.load(text_path, allow_pickle=True)
        text_features = np.array(text_features, dtype=np.float32)  # [snippet no., 768]
        # assert features.shape[0] == text_features.shape[0]
        if self.feature_size == 1024:
            text_features = np.tile(text_features, (5, 1, 1))  # [5,snippet no.,768]

        else:
            raise Exception("Feature size undefined!!!")
        
        if 'violence' in self.dataset:
            audio_path = 'C:/Users/Emirhan/Desktop/vggish-features/train/' + i3d_path.split("\\")[-1][:-8]+'__vggish.npy'
        
        audio_features = np.load(audio_path, allow_pickle=True)
        audio_features = np.array(audio_features,dtype=np.float32)

        if self.feature_size == 1024:
            audio_features = np.tile(audio_features, (5, 1, 1))  # [10,snippet no.,768]
        elif self.feature_size == 2048:
            audio_features = np.tile(audio_features, (10, 1, 1))  # [10,snippet no.,768]
        else:
            raise Exception("Feature size undefined!!!")


        if self.tranform is not None:
            features = self.tranform(features)

        if self.test_mode:
            text_features = text_features.transpose(1, 0, 2)  # [snippet no.,10,768]
            audio_features = audio_features.transpose(1, 0, 2)
            return features, text_features, audio_features
        else:
            # process 10-cropped snippet feature
            features = features.transpose(1, 0, 2)  # [snippet no., 10, 2048] -> [10, snippet no., 2048]
            divided_features = []
            for feature in features:  # loop 10 times
                feature = process_feat(feature, 32)  # divide a video into 32 segments/snippets/clips
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)  # [10,32,2048]

            div_feat_text = []
            for text_feat in text_features:
                text_feat = process_feat(text_feat, 32)  # [32,768]
                div_feat_text.append(text_feat)
            div_feat_text = np.array(div_feat_text, dtype=np.float32)

            div_feat_audio = []
            for audio_feat in audio_features:
                audio_feat = process_feat(audio_feat,32)
                div_feat_audio.append(audio_feat)
            div_feat_audio = np.array(div_feat_audio,dtype=np.float32)

            assert divided_features.shape[1] == div_feat_text.shape[1], str(self.test_mode) + "\t" + str(divided_features.shape[1]) + "\t" + div_feat_text.shape[1]
            assert divided_features.shape[1] == div_feat_audio.shape[1], str(self.test_mode) + "\t" + str(divided_features.shape[1]) + "\t" + div_feat_audio.shape[1]
            return divided_features, div_feat_text, div_feat_audio, label

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame
