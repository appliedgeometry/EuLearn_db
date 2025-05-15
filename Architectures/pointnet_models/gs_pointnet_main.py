import torch
from dataset import get_sampled_pointclouds
from gs_pointnet import PointNet
from train_eval import train_adj_model, eval_adj_model
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--data", help="Name of files",default='train')
parser.add_argument("--test_data", help="Name of files",default='test')
parser.add_argument("--sample_size", help="Numner of data if sample=True",default=250)
parser.add_argument("--test_file", help="Name of eval files",default='results_pointnet.txt')
parser.add_argument("--epochs", help="Size of sample of points",default=10000)
parser.add_argument("--save", help="File to save model", default='gs_pointnet.model')
parser.add_argument("--load", help="File to load a model", default=None)
parser.add_argument("--dropout", help="File to load a model", default=0.3)
parser.add_argument("--d_model", help="Dimension of the model", default=512)
parser.add_argument("--load_model", help="Name of previous trained model", default='gs_pointnet.model')
parser.add_argument("--num_clases", help="Total number of expected classes", default=11)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = PointNet(in_channels=3, out_channels=int(args.num_clases), d_model=int(args.d_model), dropout=float(args.dropout), layers=0).to(device)
try:
    model.load_state_dict(torch.load(args.load_model))
except:
    print('There is not saved model or does not coincide with the actual parameters')

# Training
x,y = get_sampled_pointclouds(args.data, size=int(args.sample_size))
classes = torch.unique(y).tolist()
model.train()
train_adj_model(model, x, y, epochs=int(args.epochs), d_model=int(args.d_model), model_file=args.save)

# Evaluation
x,y = get_sampled_pointclouds(args.test_data, size=False)
model.eval()
eval_adj_model(model, x, y, test_file=args.test_file)
