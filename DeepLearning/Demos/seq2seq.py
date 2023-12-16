import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# S: Start sequence symbol.
# E: End sequence symbol.
# P: Padding while the length of the word is shorter than the n_step (use the length of the longest word as n_step)

def make_batch(seq_data, letter2idx):
    '''
    Make data with padding, to make sure the encoder input meets the decoder input, and decoder input meets the decoder output.
    
    input: 
        seq_data: sequence data to input.
        letter2idx: key-values from letters to indexs.
        
    output:
        enc_input_all: all data for encoder input.
        dec_input_all: all data for decoder input.
        dec_output_all: all data for decoder output.
    '''
    enc_input_all, dec_input_all, dec_output_all = [], [], []
    
    for seq in seq_data:
        for i in range(2):                                      # Padding for data.
            seq[i] = seq[i] + 'P' * (n_step - len(seq[i]))      # 'manPP', 'woman'

        enc_input = [letter2idx[n] for n in (seq[0] + 'E')]     # ['m', 'a', 'n', 'P', 'P', 'E'] -> [15, 3, 16, 2, 2, 1]
        dec_input = [letter2idx[n] for n in ('S' + seq[1])]     # ['S', 'w', 'o', 'm', 'a', 'n'] -> [0, 25, 17, 15, 3, 16]
        dec_output = [letter2idx[n] for n in (seq[1] + 'E')]    # ['w', 'o', 'm', 'a', 'n', 'E'] -> [25, 17, 15, 3, 16, 1]
        
        enc_input_all.append(np.eye(n_class)[enc_input])        # one-hot
        dec_input_all.append(np.eye(n_class)[dec_input])
        dec_output_all.append(dec_output)                       # no need to one-hot
        
    # return format:
    #   enc_input_all: [6 (Sample amount), (n_step+1), n_class]
    #   dec_input_all: [6 (Sample amount), (n_step+1), n_class]
    #   dec_output_all: [6 (Sample amount), (n_step+1)]
    # make tensor
    return torch.FloatTensor(enc_input_all), torch.FloatTensor(dec_input_all), torch.LongTensor(dec_output_all)

# make test batch
def make_testbatch(input_word):
    enc_input, dec_output = [], []

    input_w = input_word + 'P' * (n_step - len(input_word)) + 'E'
    input = [letter2idx[n] for n in input_w]
    output = [letter2idx[n] for n in 'S' + 'P' * n_step]

    enc_input = np.eye(n_class)[input]
    dec_output = np.eye(n_class)[output]

    return torch.FloatTensor(enc_input).unsqueeze(0), torch.FloatTensor(dec_output).unsqueeze(0)

# dataloader
class TranslateDataset(Data.Dataset):
    def __init__(self, enc_input_all, dec_input_all, dec_output_all):
        self.enc_input_all = enc_input_all
        self.dec_input_all = dec_input_all
        self.dec_output_all = dec_output_all
        
    def __len__(self):
        return len(self.enc_input_all)
    
    def __getitem__(self, idx):
        return self.enc_input_all[idx], self.dec_input_all[idx], self.dec_output_all[idx]

# model
class Seq2seq(nn.Module):
    def __init__(self):
        super(Seq2seq, self).__init__()
        self.encoder = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        self.decoder = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        self.fc = nn.Linear(n_hidden, n_class)
        
    def forward(self, enc_input, enc_hidden, dec_input): 
        enc_input = enc_input.transpose(0, 1)                       # enc_input: [n_step+1, batch_size, n_class]
        dec_input = dec_input.transpose(0, 1)                       # dec_input: [n_step+1, batch_size, n_class]
        
        _, enc_states = self.encoder(enc_input, enc_hidden)         # enc_states : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        output, _ = self.decoder(dec_input, enc_states)             # outputs : [n_step+1, batch_size, num_directions(=1) * n_hidden(=128)]
        
        model = self.fc(output)
        return model    

if __name__ == "__main__":
    letter = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz']
    letter2idx = {n: i for i, n in enumerate(letter)}

    seq_data = [['man', 'woman'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low']]

    # seq2seq parameter
    n_step = max([max(len(i), len(j)) for i, j in seq_data])        # n_step means how long for a time to read in.
    n_hidden = 128                                                  # dimension of RNN output vector.
    n_class = len(letter2idx)                                       # Regard as a classification problem, as class of n_class
    batch_size = len(seq_data)
    
    model = Seq2seq().to(device)
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    enc_input_all, dec_input_all, dec_output_all = make_batch(seq_data, letter2idx)
    
    loader = Data.DataLoader(TranslateDataset(enc_input_all, dec_input_all, dec_output_all), batch_size, True)
    
    for epoch in range(5000):
        for enc_input_batch, dec_input_batch, dec_output_batch in loader:
            # make hidden shape [num_layers * num_directions, batch_size, n_hidden]
            hidden = torch.zeros(1, batch_size, n_hidden).to(device)
            
            optimizer.zero_grad()
            
            #   enc_input_all: [6 (Sample amount), (n_step+1), n_class]
            #   dec_input_all: [6 (Sample amount), (n_step+1), n_class]
            #   dec_output_all: [6 (Sample amount), (n_step+1)]
            assert (enc_input_batch.shape == torch.Size([batch_size, n_step+1, n_class]))
            assert (dec_input_batch.shape == torch.Size([batch_size, n_step+1, n_class]))
            
            (enc_input_batch, dec_input_batch, dec_output_batch) = (enc_input_batch.to(device), dec_input_batch.to(device), dec_output_batch.to(device))
            
            output = model(enc_input_batch, hidden, dec_input_batch)        # output : [n_step+1, batch_size, n_class]
            
            output = output.transpose(0, 1)                             # output : [batch_size, n_step+1, n_class]
            loss = 0
            for i in range(0, len(dec_output_batch)):                     # output[i]: [n_step+1, n_class], dec_output_all[i] = [n_step+1]
                loss += criterion(output[i], dec_output_batch[i])
            if (epoch + 1) % 1000 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
            loss.backward()
            optimizer.step()
        
    # Test
    def translate(word):
        input_batch, output_batch = make_testbatch(word)
        
        (input_batch, output_batch) = (input_batch.to(device), output_batch.to(device))

        # make hidden shape [num_layers * num_directions, batch_size, n_hidden]
        hidden = torch.zeros(1, 1, n_hidden).to(device)
        output = model(input_batch, hidden, output_batch)
        # output : [max_len+1(=6), batch_size(=1), n_class]

        predict = output.data.max(2, keepdim=True)[1] # select n_class dimension
        decoded = [letter[i] for i in predict]
        end = decoded.index('E')
        translated = ''.join(decoded[:end])

        return translated.replace('P', '')
    
    print('test')
    print('man ->', translate('man'))
    print('mans ->', translate('mans'))
    print('king ->', translate('king'))
    print('black ->', translate('black'))
    print('upp ->', translate('upp'))
        