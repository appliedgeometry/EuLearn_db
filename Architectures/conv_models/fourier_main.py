import torch
from dataset import Dataset
from fourier_operator import FNO
from sklearn.model_selection import train_test_split
from train_eval import train_model, eval_model
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--data", help="Name of files",default='G0,G1,G2')
parser.add_argument("--sample", help="Take only a number of files for genus",default=False)
parser.add_argument("--sample_size", help="Numner of data if sameple=True",default=500)
parser.add_argument("--test_file", help="Name of eval files",default='results.txt')
parser.add_argument("--epochs", help="Size of sample of points",default=10000)
parser.add_argument("--save", help="File to save model", default='fourier.model')
parser.add_argument("--load", help="File to load a model", default=None)
parser.add_argument("--dropout", help="File to load a model", default=0.3)
args = parser.parse_args()

data_list = args.data.split(',')
classes = len(data_list)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = Dataset(data_list, permute=bool(args.sample), size=int(args.sample_size))

x_train, x_test, y_train, y_test = train_test_split(data.x, data.y, test_size=0.3)

model = FNO(input_size=(71,71), d_model=71, output_size=int(classes), dropout=float(args.dropout)).to(device)

error = train_model(model, x_train, y_train, epochs=int(args.epochs))

eval_model(model, x_test, y_test)
