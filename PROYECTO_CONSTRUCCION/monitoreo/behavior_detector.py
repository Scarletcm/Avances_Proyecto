"""
Sistema de detección de comportamiento con Machine Learning
Utiliza OpenCV + scikit-learn Random Forest para clasificar videos
"""

import cv2
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


class BehaviorDetector:
    """Clasificador de comportamientos en video usando Random Forest"""
    
    def __init__(self):
        """Inicializa el detector"""
        self.label_map = {
            'normal': 0,
            'robo': 1,
            'agresion': 2,
            'sospechoso': 3
        }
        self.reverse_map = {v: k for k, v in self.label_map.items()}
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def extract_features(self, video_path, max_frames=30):
        """Extrae características de un video para análisis"""
        try:
            cap = cv2.VideoCapture(video_path)
            features_list = []
            frame_count = 0
            prev_gray = None
            motion_scores = []
            edge_scores = []
            
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.resize(frame, (320, 240))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Histograma de intensidad
                hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                
                # Detección de movimiento
                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                    motion_score = np.sum(thresh) / (320 * 240 * 255)
                    motion_scores.append(motion_score)
                
                # Detección de bordes
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
                magnitude = np.sqrt(sobelx**2 + sobely**2)
                edge_score = np.mean(magnitude) / 255
                edge_scores.append(edge_score)
                
                features_list.append({
                    'hist': hist,
                    'mean': np.mean(gray) / 255,
                    'std': np.std(gray) / 255
                })
                
                prev_gray = gray.copy()
                frame_count += 1
            
            cap.release()
            
            if not features_list:
                return None
            
            # Compilar características agregadas
            avg_motion = np.mean(motion_scores) if motion_scores else 0
            avg_edges = np.mean(edge_scores) if edge_scores else 0
            max_motion = np.max(motion_scores) if motion_scores else 0
            max_edges = np.max(edge_scores) if edge_scores else 0
            avg_hist = np.mean([f['hist'] for f in features_list], axis=0)
            avg_mean = np.mean([f['mean'] for f in features_list])
            avg_std = np.mean([f['std'] for f in features_list])
            
            final_features = np.concatenate([
                avg_hist,  # 16
                [avg_motion, max_motion, avg_edges, max_edges, avg_mean, avg_std, len(features_list) / max_frames]
            ])
            
            # Padding a 80 características
            if len(final_features) < 80:
                final_features = np.pad(final_features, (0, 80 - len(final_features)), mode='constant')
            else:
                final_features = final_features[:80]
            
            return final_features
            
        except Exception as e:
            print(f"Error extrayendo características: {str(e)}")
            return None
    
    def train(self, data_dir, test_size=0.2):
        """Entrena el modelo con videos de data_dir"""
        X = []
        y = []
        
        for behavior, label in self.label_map.items():
            behavior_dir = os.path.join(data_dir, behavior)
            if not os.path.exists(behavior_dir):
                print(f"Directorio no encontrado: {behavior_dir}")
                continue
            
            video_files = [f for f in os.listdir(behavior_dir) 
                          if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))]
            
            print(f"Procesando {len(video_files)} videos de '{behavior}'...")
            
            for video_file in video_files:
                video_path = os.path.join(behavior_dir, video_file)
                try:
                    features = self.extract_features(video_path)
                    if features is not None:
                        X.append(features)
                        y.append(label)
                        print(f"  ✓ {video_file}")
                except Exception as e:
                    print(f"  ✗ Error en {video_file}")
                    continue
        
        if len(X) < 2:
            raise ValueError("Necesita al menos 2 videos de entrenamiento")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\nTotal de muestras: {len(X)}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Entrenando Random Forest...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        self.is_trained = True
        
        print(f"\n✅ Modelo entrenado")
        print(f"  Accuracy:  {accuracy*100:.2f}%")
        print(f"  Precision: {precision*100:.2f}%")
        print(f"  Recall:    {recall*100:.2f}%")
        print(f"  F1-Score:  {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'samples': len(X),
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
    
    def predict_frame(self, frame):
        """Predice el comportamiento en un frame individual"""
        if self.model is None or not self.is_trained:
            return None, 0.0
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            edge_score = np.mean(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)) / 255
            
            basic_features = np.array([
                *hist,
                0, 0, edge_score, edge_score,
                np.mean(gray) / 255,
                np.std(gray) / 255,
                1
            ])
            
            features = np.pad(basic_features, (0, max(0, 80 - len(basic_features))), mode='constant')
            features_scaled = self.scaler.transform([features])[0]
            
            prediction = self.model.predict([features_scaled])[0]
            probabilities = self.model.predict_proba([features_scaled])[0]
            confidence = float(max(probabilities))
            
            return prediction, confidence
        except Exception:
            return None, 0.0
    
    def save_model(self, filepath):
        """Guarda el modelo entrenado"""
        if self.model is None:
            raise ValueError("No hay modelo entrenado")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_map': self.label_map,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Carga un modelo entrenado"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_map = model_data['label_map']
        self.is_trained = model_data['is_trained']


# Instancia global
detector = BehaviorDetector()
