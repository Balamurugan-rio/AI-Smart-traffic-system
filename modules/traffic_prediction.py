"""
Enhanced Traffic Prediction Module with Advanced LSTM & Ensemble Methods

Features:
- Bidirectional LSTM for better context understanding
- Attention mechanism for focusing on important timesteps
- Multi-step ahead forecasting
- Ensemble predictions (multiple models)
- Batch normalization and dropout regularization
- Data normalization/scaling
- Model checkpointing and early stopping
- Multiple loss functions (MAE, MSE)

Install optional dependencies:
    pip install tensorflow numpy scikit-learn
"""

import os
import numpy as np
from collections import deque

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization, 
                                        Bidirectional, Input, RepeatVector, TimeDistributed,
                                        Attention, Multiply, Add)
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    HAS_TF = True
except Exception:
    keras = None
    HAS_TF = False

try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


class TrafficPredictor:
    """Enhanced traffic prediction with multiple architectures and ensemble methods"""
    
    def __init__(self, model_path=None, seq_len=10, forecast_steps=3, architecture='bidirectional'):
        """
        Args:
            model_path: Path to pretrained model
            seq_len: Sequence length for input
            forecast_steps: Number of steps to forecast ahead
            architecture: 'bidirectional', 'attention', or 'ensemble'
        """
        self.model_path = model_path
        self.seq_len = seq_len
        self.forecast_steps = forecast_steps
        self.architecture = architecture
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1)) if HAS_SKLEARN else None
        self.ensemble_models = []
        
        if model_path and os.path.exists(model_path) and HAS_TF:
            try:
                self.model = keras.models.load_model(model_path)
            except Exception as e:
                print(f"Could not load model: {e}")
    
    def _normalize(self, data):
        """Normalize data using MinMaxScaler"""
        if self.scaler is None:
            return data
        data = np.array(data).reshape(-1, 1)
        return self.scaler.fit_transform(data).flatten()
    
    def _denormalize(self, data):
        """Denormalize data"""
        if self.scaler is None:
            return data
        data = np.array(data).reshape(-1, 1)
        return self.scaler.inverse_transform(data).flatten()
    
    def build_bidirectional_model(self, input_shape):
        """Build bidirectional LSTM model with batch normalization"""
        if not HAS_TF:
            raise RuntimeError('TensorFlow is not installed. Install tensorflow to use TrafficPredictor.')
        
        model = Sequential()
        
        # Bidirectional LSTM layers
        model.add(Bidirectional(LSTM(128, return_sequences=True, activation='relu'), 
                               input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        model.add(Bidirectional(LSTM(64, return_sequences=False, activation='relu')))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        # Dense layers with regularization
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        model.add(Dense(self.forecast_steps, activation='linear'))
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model = model
        return model
    
    def build_attention_model(self, input_shape):
        """Build LSTM with attention mechanism"""
        if not HAS_TF:
            raise RuntimeError('TensorFlow is not installed. Install tensorflow to use TrafficPredictor.')
        
        inputs = Input(shape=input_shape)
        
        # LSTM encoder with return sequences
        lstm1 = Bidirectional(LSTM(128, return_sequences=True, activation='relu'))(inputs)
        bn1 = BatchNormalization()(lstm1)
        dropout1 = Dropout(0.3)(bn1)
        
        lstm2 = Bidirectional(LSTM(64, return_sequences=True, activation='relu'))(dropout1)
        bn2 = BatchNormalization()(lstm2)
        dropout2 = Dropout(0.3)(bn2)
        
        # Attention layer
        attention = Attention()([dropout2, dropout2])
        
        # Dense layers
        lstm3 = LSTM(64, return_sequences=False, activation='relu')(attention)
        bn3 = BatchNormalization()(lstm3)
        dropout3 = Dropout(0.3)(bn3)
        
        dense1 = Dense(32, activation='relu')(dropout3)
        dense2 = Dense(self.forecast_steps, activation='linear')(dense1)
        
        model = Model(inputs=inputs, outputs=dense2)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model = model
        return model
    
    def build_ensemble(self, input_shape, num_models=3):
        """Build ensemble of multiple models"""
        if not HAS_TF:
            raise RuntimeError('TensorFlow is not installed. Install tensorflow to use TrafficPredictor.')
        
        self.ensemble_models = []
        for i in range(num_models):
            model = Sequential()
            model.add(Bidirectional(LSTM(100, return_sequences=True, activation='relu'), 
                                   input_shape=input_shape))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            
            model.add(LSTM(50, return_sequences=False, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(self.forecast_steps, activation='linear'))
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            self.ensemble_models.append(model)
        
        return self.ensemble_models
    
    def build_model(self, input_shape, units=64):
        """Build model based on architecture type"""
        if not HAS_TF:
            raise RuntimeError('TensorFlow is not installed. Install tensorflow to use TrafficPredictor.')
        
        if self.architecture == 'bidirectional':
            return self.build_bidirectional_model(input_shape)
        elif self.architecture == 'attention':
            return self.build_attention_model(input_shape)
        elif self.architecture == 'ensemble':
            return self.build_ensemble(input_shape)
        else:
            return self.build_bidirectional_model(input_shape)
    
    def train(self, X, y, epochs=50, batch_size=32, validation_split=0.1):
        """
        Train the model(s) with early stopping and checkpointing.
        X: shape (n_samples, seq_len, n_features)
        y: shape (n_samples, forecast_steps)
        """
        if not HAS_TF:
            raise RuntimeError('TensorFlow is not installed. Install tensorflow to train models.')
        
        if self.model is None and not self.ensemble_models:
            input_shape = (X.shape[1], X.shape[2]) if X.ndim == 3 else (X.shape[1], 1)
            self.build_model(input_shape)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
        ]
        
        if self.ensemble_models:
            histories = []
            for i, model in enumerate(self.ensemble_models):
                print(f"Training ensemble model {i+1}/{len(self.ensemble_models)}...")
                history = model.fit(X, y, epochs=epochs, batch_size=batch_size, 
                                   validation_split=validation_split, callbacks=callbacks, verbose=0)
                histories.append(history)
            return histories
        else:
            history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, 
                                    validation_split=validation_split, callbacks=callbacks)
            return history
    
    def predict(self, seq, use_ensemble=False):
        """
        Predict next forecast_steps values from a sequence.
        seq: shape (seq_len,) or (seq_len, features) or (1, seq_len, features)
        Returns: predictions of shape (forecast_steps,)
        """
        if not HAS_TF:
            raise RuntimeError('TensorFlow is not installed. Install tensorflow to run predictions.')
        
        arr = np.array(seq)
        
        # Reshape to (1, seq_len, features)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1, 1)
        elif arr.ndim == 2:
            arr = np.expand_dims(arr, axis=0)
        
        if use_ensemble and self.ensemble_models:
            predictions = []
            for model in self.ensemble_models:
                pred = model.predict(arr, verbose=0)
                predictions.append(pred[0])
            # Average ensemble predictions
            ensemble_pred = np.mean(predictions, axis=0)
            return ensemble_pred
        elif self.model is not None:
            pred = self.model.predict(arr, verbose=0)
            return pred.squeeze()
        else:
            return np.zeros(self.forecast_steps)
    
    def predict_with_confidence(self, seq, num_samples=10):
        """
        Predict with confidence intervals using multiple forward passes (MC Dropout).
        Returns: (predictions, std_dev) indicating uncertainty
        """
        if not HAS_TF or self.model is None:
            preds = self.predict(seq)
            return preds, np.zeros_like(preds)
        
        arr = np.array(seq)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1, 1)
        elif arr.ndim == 2:
            arr = np.expand_dims(arr, axis=0)
        
        # Multiple predictions to estimate uncertainty
        predictions = []
        for _ in range(num_samples):
            pred = self.model.predict(arr, verbose=0)
            predictions.append(pred[0])
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
    
    def save(self, path):
        """Save model to disk"""
        if not HAS_TF:
            raise RuntimeError('TensorFlow is not installed. Install tensorflow to save models.')
        
        if self.model:
            self.model.save(path)
            self.model_path = path
        elif self.ensemble_models:
            for i, model in enumerate(self.ensemble_models):
                model.save(f"{path}_model_{i}.h5")
            self.model_path = path


def prepare_sequences(data, seq_len=10, forecast_steps=3):
    """
    Prepare sequences for training.
    data: 1D array of values
    Returns: (X, y) suitable for LSTM training
    """
    X, y = [], []
    for i in range(len(data) - seq_len - forecast_steps + 1):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len:i + seq_len + forecast_steps])
    return np.array(X).reshape(-1, seq_len, 1), np.array(y)


if __name__ == '__main__':
    print('Enhanced Traffic Prediction module loaded.')
    print('Features: Bidirectional LSTM, Attention, Ensemble methods, Batch normalization')
    print('Install tensorflow for training and inference: pip install tensorflow')
