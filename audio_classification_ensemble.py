import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, Model
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Optional, Union
from dataclasses import dataclass
from functools import wraps
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    PCA = "pca"
    AUTOENCODER = "autoencoder"
    HYBRID = "hybrid"

@dataclass
class ModelConfig:
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 20
    patience: int = 7
    min_lr: float = 1e-6
    dropout_rate: float = 0.2

class FeatureProcessor(ABC):
    @abstractmethod
    def process(self, X: np.ndarray) -> np.ndarray:
        pass

class StatisticalFeatureProcessor(FeatureProcessor):
    def process(self, X: np.ndarray) -> np.ndarray:
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        max_val = np.max(X, axis=1, keepdims=True)
        min_val = np.min(X, axis=1, keepdims=True)
        return np.hstack([mean, std, max_val, min_val])

class InteractionFeatureProcessor(FeatureProcessor):
    def process(self, X: np.ndarray) -> np.ndarray:
        X_squared = X ** 2
        X_cubed = X ** 3
        return np.hstack([X_squared, X_cubed])

class FeatureEngineeringPipeline:
    def __init__(self):
        self.processors: List[FeatureProcessor] = [
            StatisticalFeatureProcessor(),
            InteractionFeatureProcessor()
        ]
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        enhanced_features = [X]
        for processor in self.processors:
            enhanced_features.append(processor.process(X))
        return np.hstack(enhanced_features)

class ModelFactory:
    @staticmethod
    def create_model(model_type: ModelType, input_shape: Tuple[int, ...], config: ModelConfig = ModelConfig()) -> BaseModel:
        if model_type == ModelType.PCA:
            return PCABasedModel(input_shape, config)
        elif model_type == ModelType.AUTOENCODER:
            return AutoencoderBasedModel(input_shape, config)
        else:
            return HybridModel(input_shape, config)

class BaseModel(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: Optional[Model] = None
        self.history: Optional[Dict[str, List[float]]] = None

    @abstractmethod
    def build(self, input_shape: Tuple[int, ...]) -> Model:
        pass

    def compile(self) -> None:
        if self.model is None:
            raise ValueError("Model not built yet")
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

class PCABasedModel(BaseModel):
    def __init__(self, input_shape: Tuple[int, ...], config: ModelConfig = ModelConfig()):
        super().__init__(config)
        self.input_shape = input_shape
        self.build(input_shape)  # Build model during initialization
    
    def build(self, input_shape: Tuple[int, ...]) -> Model:
        def conv_block(x: tf.Tensor, filters: int, kernel_size: int = 3, dilation_rate: int = 1) -> tf.Tensor:
            x = layers.Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(self.config.dropout_rate)(x)
            return x

        inputs = layers.Input(shape=input_shape)
        
        # Multi-scale feature extraction
        x1 = conv_block(inputs, 32, 1)
        x2 = conv_block(inputs, 32, 3)
        x3 = conv_block(inputs, 32, 3, dilation_rate=2)
        x4 = conv_block(inputs, 32, 3, dilation_rate=3)
        x = layers.Concatenate()([x1, x2, x3, x4])
        
        # Residual blocks
        for filters in [64, 128]:
            residual = conv_block(x, filters)
            x = conv_block(residual, filters)
            x = layers.Add()([residual, x])
        
        # Attention mechanism
        attention = layers.GlobalAveragePooling1D()(x)
        attention = layers.Dense(64, activation='tanh')(attention)
        attention = layers.Dense(x.shape[-1], activation='sigmoid')(attention)
        attention = layers.Reshape((1, x.shape[-1]))(attention)
        x = layers.Multiply()([x, attention])
        
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers with residual connections
        dense1 = layers.Dense(128, activation='relu')(x)
        dense1 = layers.BatchNormalization()(dense1)
        dense1 = layers.Dropout(0.3)(dense1)
        
        dense2 = layers.Dense(64, activation='relu')(dense1)
        dense2 = layers.BatchNormalization()(dense2)
        dense2 = layers.Dropout(0.3)(dense2)
        
        dense2 = layers.Add()([dense1, dense2])
        
        outputs = layers.Dense(1, activation='sigmoid')(dense2)
        
        self.model = Model(inputs, outputs, name='PCABasedModel')
        return self.model

class AudioClassifier:
    def __init__(self, data_path: str, model_type: ModelType = ModelType.PCA, n_components: int = 15):
        self.data_path = data_path
        self.model_type = model_type
        self.n_components = n_components
        self.config = ModelConfig()
        self.feature_pipeline = FeatureEngineeringPipeline()
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=n_components) if model_type == ModelType.PCA else None
        self.feature_selector = SelectKBest(f_classif, k=50)
        self.model = None
        
    def load_and_preprocess_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        logger.info("Loading dataset...")
        df = pd.read_excel(self.data_path)
        logger.info(f"Dataset loaded with shape: {df.shape}")
        
        X = df.drop(['filename', 'label'], axis=1).values
        y = df['label'].values
        
        logger.info("Applying feature engineering...")
        X = self.feature_pipeline.transform(X)
        
        logger.info("Selecting best features...")
        X = self.feature_selector.fit_transform(X, y)
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )
        
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        if self.model_type == ModelType.PCA:
            logger.info("Applying PCA...")
            X_train = self.pca.fit_transform(X_train)
            X_val = self.pca.transform(X_val)
            X_test = self.pca.transform(X_test)
            logger.info(f"PCA explained variance ratio: {sum(self.pca.explained_variance_ratio_):.3f}")
        
        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]
        X_test = X_test[..., np.newaxis]
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def train(self, train_data: Tuple[np.ndarray, np.ndarray], val_data: Tuple[np.ndarray, np.ndarray]) -> None:
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Create and compile model
        self.model = ModelFactory.create_model(self.model_type, X_train.shape[1:], self.config)
        self.model.compile()
        
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.patience,
                restore_best_weights=True
            ),
            callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss',
                save_best_only=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=self.config.min_lr
            )
        ]
        
        history = self.model.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.history = history.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        if self.model is None or self.model.model is None:
            raise ValueError("Model not trained yet")
            
        y_pred = (self.model.model.predict(X_test) > 0.5).astype(int).flatten()
        
        logger.info("\nModel Evaluation:")
        logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def plot_training_history(self) -> None:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['accuracy'], label='Train')
        plt.plot(self.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['loss'], label='Train')
        plt.plot(self.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

def main():
    # Initialize classifier
    classifier = AudioClassifier(
        'audio_mfcc_features.xlsx',
        model_type=ModelType.PCA,
        n_components=15
    )
    
    # Load and preprocess data
    train_data, val_data, test_data = classifier.load_and_preprocess_data()
    
    # Train model
    classifier.train(train_data, val_data)
    
    # Evaluate model
    classifier.evaluate(test_data[0], test_data[1])
    
    # Plot training history
    classifier.plot_training_history()

if __name__ == "__main__":
    main() 