import math
import torch
import torch.nn as nn
import torch.nn.functional as F


###############################################
# Existing Models (Conv, ConvLSTM, MLP, LSTM, AR, TRMF, LSTNetSkip, DARNN)
###############################################

class ConvModel(nn.Module):
    """
    Implements the "conv" model:
      Input -> Conv1d (kernel_size=1, filters=32, ReLU) -> Flatten ->
      Dense(10) -> Dense(20) -> Dense(30) -> Dense(1)
    """

    def __init__(self, n_timesteps, n_features):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=32, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * n_timesteps, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 30)
        self.fc4 = nn.Linear(30, 1)

    def forward(self, x):
        # x: (batch, n_timesteps, n_features)
        x = x.permute(0, 2, 1)  # -> (batch, n_features, n_timesteps)
        x = self.relu(self.conv1(x))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


class ConvLSTMModel(nn.Module):
    """
    Implements the "conv-lstm" model:
      Input -> Conv1d (kernel_size=1, filters=32, ReLU) -> AveragePooling1d ->
      Transpose -> LSTM (hidden_size=50) -> Dropout(0.2) -> Dense(1)
    """

    def __init__(self, n_timesteps, n_features):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=32, kernel_size=1)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool1d(kernel_size=1)
        self.lstm = nn.LSTM(input_size=32, hidden_size=50, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        # x: (batch, n_timesteps, n_features)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.avgpool(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x


class MLPModel(nn.Module):
    """
    Implements the "mlp" model:
      Input (vector of size n_timesteps) -> Dense(10) -> Dense(20) -> Dense(30) -> Dense(1)
    """

    def __init__(self, n_timesteps):
        super().__init__()
        self.fc1 = nn.Linear(n_timesteps, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 30)
        self.fc4 = nn.Linear(30, 1)

    def forward(self, x):
        # x: (batch, n_timesteps)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


class LSTMModel(nn.Module):
    """
    Implements the "lstm" model:
      Input -> LSTM (hidden_size=100, with ReLU on output) -> Dense(1)
    """

    def __init__(self, n_timesteps, n_features):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=100, batch_first=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(100, 1)

    def forward(self, x):
        # x: (batch, n_timesteps, n_features)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.relu(x)
        x = self.fc(x)
        return x


class ARModel(nn.Module):
    """
    Implements the "ar" (autoregressive) linear model:
      Input -> Flatten -> Dense(1)
    """

    def __init__(self, n_timesteps, n_features):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(n_timesteps * n_features, 1)

    def forward(self, x):
        # x: (batch, n_timesteps, n_features)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class TRMFModel(nn.Module):
    """
    Implements the Temporal Regularized Matrix Factorization (TRMF) model.
    """

    def __init__(self, n_timesteps, n_features, rank=5):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.rank = rank
        self.F = nn.Parameter(torch.randn(n_timesteps, rank))
        self.G = nn.Parameter(torch.randn(rank, n_features))
        self.A = nn.Parameter(torch.eye(rank))
        self.bias = nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        # Compute the latent factor prediction as before
        f_last = self.F[-1]
        f_next = self.A @ f_last
        latent_pred = f_next @ self.G + self.bias  # (n_features,)

        # Compute an additional component from x, for example, a simple linear projection of the last timestep
        x_last = x[:, -1, :]  # (batch, n_features)
        x_component = nn.Linear(self.n_features, self.n_features).to(x.device)(x_last)

        # Combine both predictions
        combined_pred = latent_pred + x_component.mean(dim=0)

        batch_size = x.shape[0]
        combined_pred = combined_pred.unsqueeze(0).repeat(batch_size, 1)
        return combined_pred

    def reconstruction(self):
        return self.F @ self.G + self.bias


class LSTNetSkipModel(nn.Module):
    """
    Implements a simplified version of the LSTNet model with skip connections.
    """

    def __init__(self, n_timesteps, n_features,
                 kernel_size=3, conv_channels=50, rnn_hidden_size=100,
                 skip=2, skip_rnn_hidden_size=50, highway_window=3, dropout=0.2):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.kernel_size = kernel_size
        self.conv_channels = conv_channels
        self.rnn_hidden_size = rnn_hidden_size
        self.skip = skip
        self.skip_rnn_hidden_size = skip_rnn_hidden_size
        self.highway_window = highway_window

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=conv_channels, kernel_size=kernel_size)
        self.gru = nn.GRU(input_size=conv_channels, hidden_size=rnn_hidden_size, batch_first=True)
        self.skip_gru = nn.GRU(input_size=conv_channels, hidden_size=skip_rnn_hidden_size, batch_first=True)
        self.pt = (n_timesteps - kernel_size + 1) // skip
        self.fc = nn.Linear(rnn_hidden_size + self.pt * skip_rnn_hidden_size, n_features)
        if highway_window > 0:
            self.highway = nn.Linear(highway_window, 1)
        else:
            self.highway = None

    def forward(self, x):
        batch_size = x.size(0)
        c = x.permute(0, 2, 1)
        c = self.relu(self.conv(c))
        c = self.dropout(c)
        L = c.size(2)

        rnn_input = c.permute(0, 2, 1)
        rnn_output, _ = self.gru(rnn_input)
        h_main = rnn_output[:, -1, :]

        pt = self.pt
        if pt > 0 and L >= pt * self.skip:
            s = c[:, :, -pt * self.skip:]
            s = s.permute(0, 2, 1)
            s = s.view(batch_size, pt, self.skip, self.conv_channels)
            s = s.reshape(batch_size * pt, self.skip, self.conv_channels)
            skip_rnn_output, _ = self.skip_gru(s)
            s_last = skip_rnn_output[:, -1, :]
            s_last = s_last.view(batch_size, pt * self.skip_rnn_hidden_size)
        else:
            s_last = torch.zeros(batch_size, self.pt * self.skip_rnn_hidden_size, device=x.device)

        combined = torch.cat([h_main, s_last], dim=1)
        res = self.fc(combined)

        if self.highway is not None:
            if self.highway_window > x.size(1):
                hw = x[:, -x.size(1):, :]
            else:
                hw = x[:, -self.highway_window:, :]
            hw = hw.permute(0, 2, 1)
            hw = self.highway(hw)
            hw = hw.squeeze(-1)
            res = res + hw

        return res


class DARNNModel(nn.Module):
    """
    Implements a simplified version of the Dual-Stage Attention-Based RNN (DARNN).

    Architecture Overview:
      Encoder (Input-Attention Stage):
        - For each time step, an input-attention mechanism computes weights over the n_features,
          producing a weighted input vector that is fed to a GRU cell.
      Decoder (Temporal-Attention Stage):
        - A decoder GRUCell uses a learned initial input.
        - Temporal attention is applied over all encoder hidden states to produce a context vector.
      Final Prediction:
        - The decoder hidden state and context vector are concatenated and passed through a linear layer
          to produce the forecast.
    """

    def __init__(self, n_timesteps, n_features,
                 encoder_hidden_size=64, decoder_hidden_size=64, attention_dim=32):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.attention_dim = attention_dim

        # Encoder: GRUCell that takes weighted input (dim = n_features)
        self.encoder_gru = nn.GRUCell(input_size=n_features, hidden_size=encoder_hidden_size)

        # Input Attention parameters
        self.W_e = nn.Linear(encoder_hidden_size, attention_dim, bias=False)
        self.U_e = nn.Parameter(torch.Tensor(n_features, attention_dim))
        self.b_e = nn.Parameter(torch.Tensor(n_features, attention_dim))
        self.v_e = nn.Parameter(torch.Tensor(attention_dim))
        nn.init.xavier_uniform_(self.U_e)
        nn.init.zeros_(self.b_e)
        nn.init.xavier_uniform_(self.W_e.weight)
        nn.init.uniform_(self.v_e, -0.1, 0.1)

        # Decoder:
        self.fc_init = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        self.decoder_gru = nn.GRUCell(input_size=1, hidden_size=decoder_hidden_size)

        # Temporal Attention parameters
        self.W_d = nn.Linear(decoder_hidden_size, attention_dim, bias=False)
        self.U_d = nn.Linear(encoder_hidden_size, attention_dim, bias=False)
        self.v_d = nn.Parameter(torch.Tensor(attention_dim))
        nn.init.xavier_uniform_(self.W_d.weight)
        nn.init.xavier_uniform_(self.U_d.weight)
        nn.init.uniform_(self.v_d, -0.1, 0.1)

        # Final output layer
        self.fc_out = nn.Linear(decoder_hidden_size + encoder_hidden_size, 1)
        self.y0 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, T, n_features = x.size()
        device = x.device

        # Encoder with input attention
        h = torch.zeros(batch_size, self.encoder_hidden_size, device=device)
        encoder_hiddens = []
        for t in range(T):
            x_t = x[:, t, :]  # (batch, n_features)
            h_proj = self.W_e(h)  # (batch, attention_dim)
            h_proj_expanded = h_proj.unsqueeze(1).expand(-1, n_features, -1)
            x_t_expanded = x_t.unsqueeze(-1)
            x_proj = x_t_expanded * self.U_e.unsqueeze(0)
            attn_input = h_proj_expanded + x_proj + self.b_e.unsqueeze(0)
            attn_scores = torch.tanh(attn_input)
            scores = torch.sum(attn_scores * self.v_e, dim=2)
            alpha = F.softmax(scores, dim=1)
            x_t_weighted = alpha * x_t
            h = self.encoder_gru(x_t_weighted, h)
            encoder_hiddens.append(h)
        H = torch.stack(encoder_hiddens, dim=1)
        h_T = h

        # Decoder with temporal attention
        d = self.fc_init(h_T)
        y0 = self.y0.expand(batch_size, 1)
        d = self.decoder_gru(y0, d)
        d_proj = self.W_d(d).unsqueeze(1).expand(-1, T, -1)
        H_proj = self.U_d(H)
        attn_temp = torch.tanh(d_proj + H_proj)
        temp_scores = torch.sum(attn_temp * self.v_d, dim=2)
        beta = F.softmax(temp_scores, dim=1)
        beta_expanded = beta.unsqueeze(2)
        context = torch.sum(beta_expanded * H, dim=1)
        dec_concat = torch.cat([d, context], dim=1)
        output = self.fc_out(dec_concat)
        return output


###############################################
# New Model: Deep Global Local Forecaster (DeepGlo)
###############################################

class DeepGloModel(nn.Module):
    """
    Implements a simplified version of the Deep Global Local Forecaster (DeepGlo).

    Architecture Overview:
      Global Component:
        - An LSTM processes the entire input sequence to capture global temporal patterns.
        - The last hidden state is projected to produce a global forecast.
      Local Component:
        - The most recent 'local_window' time steps are flattened and processed via an MLP
          to capture local patterns.
      Fusion:
        - Global and local forecasts are concatenated and passed through a linear layer
          to produce the final forecast.

    Hyperparameters (defaults):
      - global_hidden_size: 64
      - local_hidden_size: 32
      - local_window: 3 (must be <= n_timesteps)
    """

    def __init__(self, n_timesteps, n_features,
                 global_hidden_size=64, local_hidden_size=32, local_window=3, dropout=0.2):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.global_hidden_size = global_hidden_size
        self.local_hidden_size = local_hidden_size
        self.local_window = local_window

        # Global Component: LSTM over the entire sequence
        self.global_lstm = nn.LSTM(input_size=n_features, hidden_size=global_hidden_size, batch_first=True)
        self.global_fc = nn.Linear(global_hidden_size, n_features)

        # Local Component: MLP on the most recent 'local_window' time steps
        if local_window > n_timesteps:
            raise ValueError("local_window must be less than or equal to n_timesteps")
        self.local_fc1 = nn.Linear(local_window * n_features, local_hidden_size)
        self.local_fc2 = nn.Linear(local_hidden_size, n_features)

        # Fusion Layer: combines global and local predictions
        self.fusion_fc = nn.Linear(n_features * 2, n_features)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, n_timesteps, n_features)
        batch_size = x.size(0)

        # Global Component
        global_out, (h_n, _) = self.global_lstm(x)  # h_n: (1, batch, global_hidden_size)
        h_global = h_n.squeeze(0)  # (batch, global_hidden_size)
        global_pred = self.global_fc(h_global)  # (batch, n_features)

        # Local Component: use the last 'local_window' time steps
        local_input = x[:, -self.local_window:, :]  # (batch, local_window, n_features)
        local_input_flat = local_input.reshape(batch_size, -1)  # (batch, local_window * n_features)
        local_hidden = self.relu(self.local_fc1(local_input_flat))
        local_pred = self.local_fc2(local_hidden)  # (batch, n_features)

        # Fusion of global and local predictions
        combined = torch.cat([global_pred, local_pred], dim=1)  # (batch, 2*n_features)
        fused = self.fusion_fc(combined)  # (batch, n_features)
        fused = self.dropout(fused)
        return fused


###############################################
# New Model: Simplified Temporal Fusion Transformer (TFT)
###############################################

class PositionalEncoding(nn.Module):
    """
    Implements the standard Positional Encoding as used in Transformers.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant 'pe' matrix with values dependent on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TFTModel(nn.Module):
    """
    A simplified version of the Temporal Fusion Transformer (TFT).

    This implementation projects the input features to a model dimension,
    adds positional encoding, passes the sequence through a Transformer encoder,
    and then uses the last time step's representation for forecasting.

    Note: A full TFT would include additional gating, variable selection, and
    interpretable attention mechanisms.
    """

    def __init__(self, n_timesteps, n_features, d_model=64, n_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.d_model = d_model

        # Project input features to the model dimension.
        self.input_projection = nn.Linear(n_features, d_model)

        # Positional encoding.
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=n_timesteps)

        # Transformer encoder layers.
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final prediction layer. Output dimension is set to n_features.
        self.fc = nn.Linear(d_model, n_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, n_timesteps, n_features)
        x = self.input_projection(x)  # (batch, n_timesteps, d_model)
        x = self.positional_encoding(x)  # (batch, n_timesteps, d_model)

        # Transformer expects input of shape (seq_len, batch, d_model)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)  # (n_timesteps, batch, d_model)

        # Use the representation from the last time step for forecasting.
        x = x[-1]  # (batch, d_model)
        x = self.dropout(x)
        x = self.fc(x)  # (batch, n_features)
        return x


###############################################
# New Model: Simplified DeepAR Model
###############################################

class DeepARModel(nn.Module):
    """
    A simplified implementation of the DeepAR model.

    This model uses an LSTM to encode the input time series and outputs
    parameters (mean and standard deviation) of a Gaussian distribution for forecasting.

    Note: In practice, DeepAR is a probabilistic forecasting model trained with likelihood loss.
    """

    def __init__(self, n_timesteps, n_features, hidden_size=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        # Output both mean and log(sigma) for each feature.
        self.fc = nn.Linear(hidden_size, n_features * 2)

    def forward(self, x):
        # x: (batch, n_timesteps, n_features)
        out, _ = self.lstm(x)
        # Use the last time step's hidden state.
        last_hidden = out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        params = self.fc(last_hidden)  # (batch, n_features*2)
        mean, log_sigma = params.chunk(2, dim=1)
        sigma = torch.exp(log_sigma)
        return mean, sigma


###############################################
# New Model: Simplified Deep State Space Model (DeepState)
###############################################

class DeepStateModel(nn.Module):
    """
    A simplified deep state space model.

    This model uses a GRU-based state transition (state encoder) and an emission
    network to produce forecasts from the latent state. In a full state space model,
    you might also incorporate stochasticity in the state transitions and
    perform Bayesian inference.
    """

    def __init__(self, n_timesteps, n_features, hidden_size=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # State encoder (transition function)
        self.state_rnn = nn.GRU(input_size=n_features, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True, dropout=dropout)
        # Emission network: maps latent state to observation space.
        self.emission = nn.Linear(hidden_size, n_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, n_timesteps, n_features)
        state, _ = self.state_rnn(x)
        # Use the final state's representation for forecasting.
        last_state = state[:, -1, :]
        last_state = self.dropout(last_state)
        y_pred = self.emission(last_state)
        return y_pred


###############################################
# Model Builder
###############################################

class ModelBuilder:
    """
    A builder class to create different PyTorch models based on a string identifier.

    Options include: 'conv', 'conv-lstm', 'mlp', 'lstm', 'ar', 'trmf', 'lstnet-skip', 'darnn',
    'deepglo', 'tft', 'deepar', and 'deepstate'.
    """

    def __init__(self, model_type="conv", n_timesteps=None, n_features=None, rank=5):
        self.model_type = model_type
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.rank = rank
        self.model = None

    def build_model(self):
        if self.model_type == "conv":
            if self.n_timesteps is None or self.n_features is None:
                raise ValueError("n_timesteps and n_features must be provided for the 'conv' model.")
            self.model = ConvModel(self.n_timesteps, self.n_features)
        elif self.model_type == "conv-lstm":
            if self.n_timesteps is None or self.n_features is None:
                raise ValueError("n_timesteps and n_features must be provided for the 'conv-lstm' model.")
            self.model = ConvLSTMModel(self.n_timesteps, self.n_features)
        elif self.model_type == "mlp":
            if self.n_timesteps is None:
                raise ValueError("n_timesteps must be provided for the 'mlp' model.")
            self.model = MLPModel(self.n_timesteps)
        elif self.model_type == "lstm":
            if self.n_timesteps is None or self.n_features is None:
                raise ValueError("n_timesteps and n_features must be provided for the 'lstm' model.")
            self.model = LSTMModel(self.n_timesteps, self.n_features)
        elif self.model_type == "ar":
            if self.n_timesteps is None or self.n_features is None:
                raise ValueError("n_timesteps and n_features must be provided for the 'ar' model.")
            self.model = ARModel(self.n_timesteps, self.n_features)
        elif self.model_type == "trmf":
            if self.n_timesteps is None or self.n_features is None:
                raise ValueError("n_timesteps and n_features must be provided for the 'trmf' model.")
            self.model = TRMFModel(self.n_timesteps, self.n_features, rank=self.rank)
        elif self.model_type == "lstnet-skip":
            if self.n_timesteps is None or self.n_features is None:
                raise ValueError("n_timesteps and n_features must be provided for the 'lstnet-skip' model.")
            self.model = LSTNetSkipModel(self.n_timesteps, self.n_features)
        elif self.model_type == "darnn":
            if self.n_timesteps is None or self.n_features is None:
                raise ValueError("n_timesteps and n_features must be provided for the 'darnn' model.")
            self.model = DARNNModel(self.n_timesteps, self.n_features)
        elif self.model_type == "deepglo":
            if self.n_timesteps is None or self.n_features is None:
                raise ValueError("n_timesteps and n_features must be provided for the 'deepglo' model.")
            self.model = DeepGloModel(self.n_timesteps, self.n_features)
        elif self.model_type == "tft":
            if self.n_timesteps is None or self.n_features is None:
                raise ValueError("n_timesteps and n_features must be provided for the 'tft' model.")
            self.model = TFTModel(self.n_timesteps, self.n_features)
        elif self.model_type == "deepar":
            if self.n_timesteps is None or self.n_features is None:
                raise ValueError("n_timesteps and n_features must be provided for the 'deepar' model.")
            self.model = DeepARModel(self.n_timesteps, self.n_features)
        elif self.model_type == "deepstate":
            if self.n_timesteps is None or self.n_features is None:
                raise ValueError("n_timesteps and n_features must be provided for the 'deepstate' model.")
            self.model = DeepStateModel(self.n_timesteps, self.n_features)
        else:
            raise ValueError(
                "Invalid model type. Choose from 'conv', 'conv-lstm', 'mlp', 'lstm', 'ar', 'trmf', 'lstnet-skip', 'darnn', 'deepglo', 'tft', 'deepar', or 'deepstate'.")
        return self.model

    @staticmethod
    def get_available_models():
        """
        Returns a list of available model names.
        """
        return [
            "conv", "conv-lstm", "mlp", "lstm", "ar", "trmf",
            "lstnet-skip", "darnn", "deepglo", "tft", "deepar", "deepstate"
        ]


###############################################
# Example Usage
###############################################

if __name__ == '__main__':
    n_timesteps = 10
    n_features = 1

    # Print available models
    print("Available models:", ModelBuilder.get_available_models())

    # Change model_type to one of:
    # "conv", "conv-lstm", "mlp", "lstm", "ar", "trmf", "lstnet-skip",
    # "darnn", "deepglo", "tft", "deepar", or "deepstate".
    builder = ModelBuilder(model_type="deepar", n_timesteps=n_timesteps, n_features=n_features)
    model = builder.build_model()
    print(model)

    dummy_input = torch.randn(8, n_timesteps, n_features)
    if builder.model_type == "deepar":
        mean, sigma = model(dummy_input)
        print("Mean shape:", mean.shape)
        print("Sigma shape:", sigma.shape)
    else:
        output = model(dummy_input)
        print("Output shape:", output.shape)
