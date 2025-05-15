import torch
import torch.nn as nn
from utils import NoamOptimizer
from sklearn.metrics import classification_report
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model(model, x_train, y_train, epochs=100, d_model=71, lr=0, decay=1e-5, betas=(0.9, 0.98), model_file='fourier.model'):
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = NoamOptimizer(model.parameters(), d_model=d_model, lr=lr, decay=decay, betas=betas)

    epochs = range(epochs) # Max number of epochs
    error = [] # Error for epoch
    model.train()
    for t in tqdm(epochs):
        epoch_error = 0 # Actual epoch error
        n = len(x_train) # Number of data
        for i in torch.randperm(n):
            prediction = model(x_train[i]) # Forward

            optimizer.zero_grad() # Backward
            loss_value = criterion(prediction, y_train[i])
            loss_value.backward()
            optimizer.step()

            epoch_error += loss_value.detach().numpy() # Sum data error to epoch error
        error.append(epoch_error/n) # Average total epoch error

        torch.save(model.state_dict(), model_file) #  Filename for save the model
    return error

def train_adj_model(model, x_train, y_train, model_file, epochs=100, d_model=128, lr=0, decay=1e-5, betas=(0.9, 0.98)):
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = NoamOptimizer(model.parameters(), d_model=d_model, lr=lr, decay=decay, betas=betas)

    epochs = range(epochs) # Max number of epochs
    #error = [] # Error for epoch
    model.train()
    for t in tqdm(epochs):
        #epoch_error = 0 # Actual epoch error
        n = len(x_train) # Number of data
        for i in torch.randperm(n):
            prediction = model(x_train[i][0], x_train[i][1]) # Forward

            optimizer.zero_grad() # Backward
            loss_value = criterion(prediction, y_train[i])
            loss_value.backward()
            optimizer.step()

            #epoch_error += loss_value.detach().numpy() # Sum data error to epoch error
        #error.append(epoch_error/n) # Average total epoch error

        torch.save(model.state_dict(), model_file) #  Filename for save the model
    #return error

def eval_model(model, x_test, y_test, test_file='results.text'):
    model.eval()
    pred = [model(xi).argmax().detach().numpy() for xi in x_test]
    report = classification_report(y_test.numpy(), pred)
    print(report)
    
    f = open(test_file, "a")
    f.write(report)
    f.close()

def eval_adj_model(model, x_test, y_test, test_file='results.text'):
    model.eval()
    pred = [model(xi[0],xi[1]).argmax().detach().numpy() for xi in x_test]
    report = classification_report(y_test.numpy(), pred)
    print(report)
    
    f = open(test_file, "a")
    f.write(report)
    f.close()
