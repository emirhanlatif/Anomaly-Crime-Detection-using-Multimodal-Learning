from torch.utils.data import DataLoader
import torch.optim as optim
from model import Model
from dataset import Dataset
from test_10crop import test
import option
from utils import *
from config import *

viz = Visualizer(env='single', use_incoming_socket=False)

if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)
    seed_everything(args.seed)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    model = Model(args)
    if args.pretrained_ckpt is not None:
        print("Loading pretrained model " + args.pretrained_ckpt)
        model.load_state_dict(torch.load(args.pretrained_ckpt))
    else:
        if "violence" in args.dataset:
            model.load_state_dict(torch.load('./ckpt/my_best/violence-both-text_agg-add-1-1-extra_loss-445-4869-BEST.pkl'))
        else:
            raise NotImplementedError

    for name, value in model.named_parameters():
        print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    optimizer = optim.Adam(model.parameters(),
                            lr=config.lr[0], weight_decay=0.005)

    best_AUC = -1
    output_path = 'sontest'   # put your own path here
    auc, ap = test(test_loader, model, args, viz, device)
