import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # 자동으로 device 설정 (cuda 사용 가능하면 cuda, 아니면 cpu)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.input_dim = configs.input_dim
        self.hidden_dim = configs.hidden_dim
        self.num_layers = configs.num_layers
        self.bidirectional = configs.bidirectional
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.lstm_encoder = LSTMEncoder(
            self.input_dim, self.hidden_dim, self.num_layers, 
            batch_first=True, bidirectional=self.bidirectional
        )
        self.lstm_decoder = LSTMDecoder(
            self.input_dim, self.hidden_dim, self.num_layers,
            batch_first=True, bidirectional=self.bidirectional
        )

    def init_hidden(self, x):
        # x 와 같은 device로 초기화
        # (기존에는 self.device 를 사용했지만, x가 이미 어느 device에 있는지 알 수 있으면 x.device 를 쓰는 편이 안전)
        h0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            x.size(0),
            self.hidden_dim,
            device=x.device
        )
        c0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            x.size(0),
            self.hidden_dim,
            device=x.device
        )        
        return h0, c0

    # def forward(self, x, teacher_forcing=None):
    #     self.encoder_hidden = self.init_hidden(x)  # 여기서 x.device 로 init
    #     self.out, self.encoder_hidden = self.lstm_encoder(x, self.encoder_hidden)

    #     outputs = torch.zeros(x.size(0), self.pred_len, x.size(2), device=x.device)
    #     decoder_input = x[:, -1, :].unsqueeze(1)

    #     if teacher_forcing is not None:
    #         # Train mode(teacher forcing)
    #         self.decoder_hidden = self.encoder_hidden
    #         for t in range(self.pred_len): 
    #             decoder_output, self.decoder_hidden = self.lstm_decoder(decoder_input, self.decoder_hidden)
    #             outputs[:, t, :] = decoder_output
    #             # teacher forcing으로 받기
    #             decoder_input = teacher_forcing[:, t, :].unsqueeze(1)
    #     else:
    #         # Test mode
    #         self.decoder_hidden = self.encoder_hidden
    #         for t in range(self.pred_len):
    #             decoder_output, self.decoder_hidden = self.lstm_decoder(decoder_input, self.decoder_hidden)
    #             outputs[:, t, :] = decoder_output
    #             # 지금까지 예측한 것을 다시 입력으로
    #             decoder_input = outputs[:, :t+1, :]

    #     return outputs
    def forward(self, x, pretrain_flag=None, teacher_forcing=None):
    # 1) 인코더
        self.encoder_hidden = self.init_hidden(x)
        lstm_out, self.encoder_hidden = self.lstm_encoder(x, self.encoder_hidden)

    # 2) 디코더
    #    - time step별 결과를 모을 리스트
        predictions = []
        decoder_input = x[:, -1, :].unsqueeze(1)
        self.decoder_hidden = self.encoder_hidden

    # (a) teacher forcing 모드
        if teacher_forcing is not None:
            for t in range(self.pred_len):
                decoder_output, self.decoder_hidden = self.lstm_decoder(decoder_input, self.decoder_hidden)
                predictions.append(decoder_output)  # 리스트에 저장
                # 다음 입력: ground truth
                decoder_input = teacher_forcing[:, t, :].unsqueeze(1)
    
    # (b) Autoregressive (test) 모드
        else:
            for t in range(self.pred_len):
                decoder_output, self.decoder_hidden = self.lstm_decoder(decoder_input, self.decoder_hidden)
                predictions.append(decoder_output)  # 리스트에 저장
                # 다음 입력: 방금 예측값
                decoder_input = decoder_output.unsqueeze(1)

    # 3) time-step별로 쌓았던 리스트를 최종 (batch, pred_len, input_dim) 텐서로 변환
    #    predictions == list of [batch, input_dim] -> stack -> [batch, pred_len, input_dim]
        outputs = torch.stack(predictions, dim=1)
        return outputs



class LSTMEncoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size, hidden_size, num_layers = 1, batch_first = True, bidirectional = False):
        
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define LSTM layer
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = batch_first, bidirectional = bidirectional)

    def forward(self, x_input, hidden):
        
        '''
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence 
        '''
        lstm_out, hidden = self.lstm(x_input, hidden)
        
        return lstm_out, hidden     


class LSTMDecoder(nn.Module):
    ''' Decodes hidden state output by encoder '''
    
    def __init__(self, input_size, hidden_size, num_layers = 1, batch_first = True, bidirectional = False):

        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        
        super(LSTMDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
      

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = batch_first, bidirectional = bidirectional)
        self.linear = nn.Linear(hidden_size*2, self.input_size)           

    def forward(self, x_input, encoder_hidden_states):
        
        '''        
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence 
 
        '''
        lstm_out, self.hidden = self.lstm(x_input, encoder_hidden_states)
        out_f = lstm_out[:, -1, :self.hidden_size]
        out_b = lstm_out[:, 0, self.hidden_size:]
        out = torch.cat((out_f, out_b), dim=1)


        # lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        output = self.linear(out)     
        
        return output, self.hidden
    
# class lstm_seq2seq(nn.Module):
#     ''' train LSTM encoder-decoder and make predictions '''
    
#     def __init__(self, input_size, hidden_size):

#         '''
#         : param input_size:     the number of expected features in the input X
#         : param hidden_size:    the number of features in the hidden state h
#         '''

#         super(lstm_seq2seq, self).__init__()

#         self.input_size = input_size
#         self.hidden_size = hidden_size

#         self.encoder = lstm_encoder(input_size = input_size, hidden_size = hidden_size)
#         self.decoder = lstm_decoder(input_size = input_size, hidden_size = hidden_size)


#     def train_model(self, input_tensor, target_tensor, n_epochs, target_len, batch_size, training_prediction = 'recursive', teacher_forcing_ratio = 0.5, learning_rate = 0.01, dynamic_tf = False):
        
#         '''
#         train lstm encoder-decoder
        
#         : param input_tensor:              input data with shape (seq_len, # in batch, number features); PyTorch tensor    
#         : param target_tensor:             target data with shape (seq_len, # in batch, number features); PyTorch tensor
#         : param n_epochs:                  number of epochs 
#         : param target_len:                number of values to predict 
#         : param batch_size:                number of samples per gradient update
#         : param training_prediction:       type of prediction to make during training ('recursive', 'teacher_forcing', or
#         :                                  'mixed_teacher_forcing'); default is 'recursive'
#         : param teacher_forcing_ratio:     float [0, 1) indicating how much teacher forcing to use when
#         :                                  training_prediction = 'teacher_forcing.' For each batch in training, we generate a random
#         :                                  number. If the random number is less than teacher_forcing_ratio, we use teacher forcing.
#         :                                  Otherwise, we predict recursively. If teacher_forcing_ratio = 1, we train only using
#         :                                  teacher forcing.
#         : param learning_rate:             float >= 0; learning rate
#         : param dynamic_tf:                use dynamic teacher forcing (True/False); dynamic teacher forcing
#         :                                  reduces the amount of teacher forcing for each epoch
#         : return losses:                   array of loss function for each epoch
#         '''
        
#         # initialize array of losses 
#         losses = np.full(n_epochs, np.nan)

#         optimizer = optim.Adam(self.parameters(), lr = learning_rate)
#         criterion = nn.MSELoss()

#         # calculate number of batch iterations
#         n_batches = int(input_tensor.shape[1] / batch_size)

#         with trange(n_epochs) as tr:
#             for it in tr:
                
#                 batch_loss = 0.
#                 batch_loss_tf = 0.
#                 batch_loss_no_tf = 0.
#                 num_tf = 0
#                 num_no_tf = 0

#                 for b in range(n_batches):
#                     # select data 
#                     input_batch = input_tensor[:, b: b + batch_size, :]
#                     target_batch = target_tensor[:, b: b + batch_size, :]

#                     # outputs tensor
#                     outputs = torch.zeros(target_len, batch_size, input_batch.shape[2])

#                     # initialize hidden state
#                     encoder_hidden = self.encoder.init_hidden(batch_size)

#                     # zero the gradient
#                     optimizer.zero_grad()

#                     # encoder outputs
#                     encoder_output, encoder_hidden = self.encoder(input_batch)

#                     # decoder with teacher forcing
#                     decoder_input = input_batch[-1, :, :]   # shape: (batch_size, input_size)
#                     decoder_hidden = encoder_hidden

#                     if training_prediction == 'recursive':
#                         # predict recursively
#                         for t in range(target_len): 
#                             decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
#                             outputs[t] = decoder_output
#                             decoder_input = decoder_output

#                     if training_prediction == 'teacher_forcing':
#                         # use teacher forcing
#                         if random.random() < teacher_forcing_ratio:
#                             for t in range(target_len): 
#                                 decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
#                                 outputs[t] = decoder_output
#                                 decoder_input = target_batch[t, :, :]

#                         # predict recursively 
#                         else:
#                             for t in range(target_len): 
#                                 decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
#                                 outputs[t] = decoder_output
#                                 decoder_input = decoder_output

#                     if training_prediction == 'mixed_teacher_forcing':
#                         # predict using mixed teacher forcing
#                         for t in range(target_len):
#                             decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
#                             outputs[t] = decoder_output
                            
#                             # predict with teacher forcing
#                             if random.random() < teacher_forcing_ratio:
#                                 decoder_input = target_batch[t, :, :]
                            
#                             # predict recursively 
#                             else:
#                                 decoder_input = decoder_output

#                     # compute the loss 
#                     loss = criterion(outputs, target_batch)
#                     batch_loss += loss.item()
                    
#                     # backpropagation
#                     loss.backward()
#                     optimizer.step()

#                 # loss for epoch 
#                 batch_loss /= n_batches 
#                 losses[it] = batch_loss

#                 # dynamic teacher forcing
#                 if dynamic_tf and teacher_forcing_ratio > 0:
#                     teacher_forcing_ratio = teacher_forcing_ratio - 0.02 

#                 # progress bar 
#                 tr.set_postfix(loss="{0:.3f}".format(batch_loss))
                    
#         return losses

#     def predict(self, input_tensor, target_len):
        
#         '''
#         : param input_tensor:      input data (seq_len, input_size); PyTorch tensor 
#         : param target_len:        number of target values to predict 
#         : return np_outputs:       np.array containing predicted values; prediction done recursively 
#         '''

#         # encode input_tensor
#         input_tensor = input_tensor.unsqueeze(1)     # add in batch size of 1
#         encoder_output, encoder_hidden = self.encoder(input_tensor)

#         # initialize tensor for predictions
#         outputs = torch.zeros(target_len, input_tensor.shape[2])

#         # decode input_tensor
#         decoder_input = input_tensor[-1, :, :]
#         decoder_hidden = encoder_hidden
        
#         for t in range(target_len):
#             decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
#             outputs[t] = decoder_output.squeeze(0)
#             decoder_input = decoder_output
            
#         np_outputs = outputs.detach().numpy()
        
#         return np_outputs
    
    # def forward(self, x):
    #     # x: [Batch, Input length, Channel]

    #     x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
    #     return x # [Batch, Output length, Channel]
    
    # class PlainLSTM(nn.Module):
    #     def __init__(self, config, hidden_dim, num_classes):
    #         super(PlainLSTM, self).__init__()
    #         self.config = config
    #         self.hidden_dim = hidden_dim
    #         self.num_layers = 2
    #         self.num_classes = num_classes
    #         self.bidirectional = config['bidirectional']

    #         self.input_dim = 128

    #         # architecture
    #         self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True, num_layers=self.num_layers, bidirectional=config['bidirectional'])
    #         self.fc = nn.Linear(self.hidden_dim * 2, 1)

    #     def init_hidden(self, x):
    #         h0 = torch.zeros((self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_dim)).cuda()
    #         c0 = torch.zeros((self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_dim)).cuda()
            
    #         return h0, c0

    #     def forward(self, x):
    #         hidden = self.init_hidden(x)
    #         out, hidden = self.lstm(x, hidden)

    #         out_f = out[:, -1, :self.hidden_dim]
    #         out_b = out[:, 0, self.hidden_dim:]
    #         out = torch.cat((out_f, out_b), dim=1)
    #         out = self.fc(out)

    #         return out
    
    # class MainModel(nn.Module):
    
    #     def __init__(self, config):

    #         super(MainModel, self).__init__()

    #         self.config = config
    #         self.feature = ResNetFeature(config)
    #         self.classifier = PlainLSTM(config, hidden_dim=config['hidden_dim'], num_classes=config['num_classes'])

    #     def forward(self, x):
    #         out = self.classifier(self.feature(x))

    #         return out