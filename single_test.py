import matplotlib.pyplot as plt
import torch,sys
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
import numpy as np
import matplotlib.pyplot as plt
from utils import get_gt
from torch.utils.data import DataLoader
import torch.optim as optim
from model import Model
from dataset import Dataset
from test_10crop import test
import option
from utils import *
from config import *
from pathlib import Path
import shutil
import time
import ffmpeg
import cv2
from moviepy.editor import VideoFileClip
from moviepy.video.tools.drawing import color_gradient
import numpy as np
import matplotlib.pyplot as plt

from I3D_Feature_Extraction_resnet.extract_features import run
from I3D_Feature_Extraction_resnet.resnet import i3_res50
import os

from swinbert.SwinBERT.src.tasks.test_caption import *






def dataset(i3d_path,text_path, audio_path):

    i3d_features = np.load(i3d_path)
    i3d_features = np.array(i3d_features, dtype=np.float32)


    text_features = np.load(text_path, allow_pickle=True)
    text_features = np.array(text_features, dtype=np.float32)  # [snippet no., 768]
       
 
    text_features = np.tile(text_features, (5, 1, 1))  # [5,snippet no.,768]


    text_features = text_features.transpose(1, 0, 2)  # [snippet no.,5,768]

    audio_features = np.load(audio_path, allow_pickle=True)
    audio_features = np.array(audio_features, dtype=np.float32)  # [snippet no., 128]
 
    audio_features = np.tile(audio_features, (5, 1, 1))  # [5,snippet no.,128]


    audio_features = audio_features.transpose(1, 0, 2)  # [snippet no.,5,1288]
    return i3d_features, text_features, audio_features


def test(dataloader, model):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0)

        (input, text,audio) = (dataloader)
        
        input = input.to(device)
        input = input.permute(0, 2, 1, 3)
        text = text.to(device)
        text = text.permute(0, 2, 1, 3)
        audio = audio.to(device)
        audio = audio.permute(0, 2, 1, 3)
            # input.shape = (1,5,T,1024); T clips, each clip has 16frames, each frame has 10 crops

        score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
        scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(input, text, audio)  # 注意这里的score_abnormal和score_normal是一维的，是每一个video的一个分数，而logits则是一个T维的vector给每一个snippet都打了分
        logits = torch.squeeze(logits, 1)
        logits = torch.mean(logits, 0)
        sig = logits
        pred = torch.cat((pred, sig))
        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16) 

        plt.plot(pred)
        plt.xlabel('İndeksler')
        plt.ylabel('Değerler')
        plt.title('Tek Boyutlu Dizi Çizimi')
        plt.show()
      
        return pred
    




if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)
    seed_everything(args.seed)


    tek_video = 'v=w2NgAYJHnS0__#1_label_G-0-0.mp4'
    outputpath = 'C:\\Users\\Emirhan\\Desktop\\tevad\\TEVAD\\single_test\i3d_feature'
    datasetpath = 'C:\\Users\\Emirhan\\Desktop\\tevad\\TEVAD\\single_test\\ornek_video\\' + tek_video

    # pretrainedpath = 'C:\\Users\\Emirhan\\Desktop\\tevad\\TEVAD\\I3D_Feature_Extraction_resnet\\pretrained\\i3d_r50_kinetics.pth'
    # frequency = 16
    # batch_size = 20
    # sample_mode = 'oversample'
    # temppath = outputpath+ "/temp/"
    # test_video_fname = 'C:\\Users\\Emirhan\\Desktop\\tevad\\TEVAD\\single_test\\ornek_video\\A.Beautiful.Mind.2001__#00-01-45_00-02-50_label_A.mp4'
    # resume_checkpoint = 'C:\\Users\\Emirhan\\Desktop\\tevad\\TEVAD\\swinbert\\SwinBERT\\models\\table1\\vatex\\best-checkpoint\\model.bin'
    # setup the model
    #i3d = i3_res50(400, pretrainedpath)
    #i3d.cuda()
    #i3d.train(False)  # Set model to evaluate mode
    #videoname = datasetpath.split("\\")[8]qq
    #print(videoname)
    #startime = time.time()
    #print("Generating for {0}".format(datasetpath))
    #Path(temppath).mkdir(parents=True, exist_ok=True)
    #ffmpeg.input(datasetpath).output('{}%d.jpg'.format(temppath),start_number=0).global_args('-loglevel', 'quiet').run()
    #print("Preprocessing done..")
    #features = run(i3d, frequency, temppath, batch_size, sample_mode)
    #shutil.rmtree(temppath)
    #print("done in {0}.".format(time.time() - startime))
    #i3d_features = features
    #print(i3d_features.shape)
    

    torch.cuda.empty_cache()
    i3d_path = 'C:\\Users\\Emirhan\\Desktop\\tevad\\TEVAD\\save\\Violence\\Violence_five_crop_i3d_v1\\'+os.path.splitext(tek_video)[0] +'_i3d.npy'


    
    # text_features = abc(tek_video)
    #text_features = 'C:\\Users\\Emirhan\\Desktop\\tevad\\TEVAD\\save\\Violence\\sent_emb_n\\' + os.path.splitext(tek_video)[0] + '_emb.npy'
    text_features = 'C:\\Users\\Emirhan\\Desktop\\features\\text\\' + os.path.splitext(tek_video)[0] + '_emb.npy'
    #audio_features = 'C:\\Users\\Emirhan\\Desktop\\vggish-features\\train\\' + os.path.splitext(tek_video)[0] + '__vggish.npy'
    audio_features = 'C:\\Users\\Emirhan\\Desktop\\features\\audio\\' +tek_video + '.npy'

    test_loader = DataLoader(dataset(i3d_path,text_features, audio_features),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    model = Model(args)
   
    model.load_state_dict(torch.load('C:\\Users\\Emirhan\\Desktop\\tevad\\TEVAD\\sonckpt4\\violence-both-text_agg-concat-0.0001-extra_loss-410-4869-.pkl'))

    for name, value in model.named_parameters():
        print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    
    optimizer = optim.Adam(model.parameters(),
                            lr=config.lr[0], weight_decay=0.005)

    best_AUC = -1
    pred = test(test_loader, model)
    print(pred)
    
    cap=cv2.VideoCapture(datasetpath)
    i = 0

    while True:
        ret,frame = cap.read()
        if ret==0:
            break

        value = pred[i]
        if value > 0.9:

            cv2.putText(frame, 'ANOMALY !!! ', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        i = i+1
        cv2.imshow("Test Video",frame)
        if cv2.waitKey(20) & 0xFF==ord("q"):
            break
            
    cap.release()
    cv2.destroyAllWindows()
