import matplotlib.pyplot as plt
import torch,sys
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
import numpy as np
from utils import get_gt

def test(dataloader, model, args, viz, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0, device=device)

        for i, (input, text, audio) in enumerate(dataloader): 
            input = input.to(device)
            input = input.permute(0, 2, 1, 3)
            text = text.to(device)
            text = text.permute(0, 2, 1, 3)
            audio = audio.to(device)
            audio = audio.permute(0, 2, 1, 3)
            # input.shape = (1,5,T,2048); T clips, each clip has 16frames, each frame has 10 crops
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(input, text,audio)  
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            pred = torch.cat((pred, sig))
        
        gt = get_gt(args.dataset, args.gt)
        
        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)  
        fpr, tpr, threshold = roc_curve(list(gt), pred)
        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        rec_auc = auc(fpr, tpr)
        ap = average_precision_score(list(gt), pred)
        print('ap : ' + str(ap))
        print('auc : ' + str(rec_auc))
        
        viz.plot_lines('pr_auc', pr_auc)
        viz.plot_lines('auc', rec_auc)
        viz.lines('scores', pred)
        viz.lines('roc', tpr, fpr)

        if args.save_test_results:
            np.save('results/' + args.dataset + '_pred.npy', pred)
            np.save('results/' + args.dataset + '_fpr.npy', fpr)
            np.save('results/' + args.dataset + '_tpr.npy', tpr)
            np.save('results/' + args.dataset + '_precision.npy', precision)
            np.save('results/' + args.dataset + '_recall.npy', recall)
            np.save('results/' + args.dataset + '_auc.npy', rec_auc)
            np.save('results/' + args.dataset + '_ap.npy', ap)
        return rec_auc, ap

