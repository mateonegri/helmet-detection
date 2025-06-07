import React, { useState, useRef, useEffect } from 'react';
import { Upload, Camera, AlertCircle, CheckCircle, Loader2, X, Video } from 'lucide-react';
import './App.css';
import LiveDetection from './LiveDetection';

interface PredictionResult {
  filename: string;
  prediction: string;
  message: string;
  confidence: number;
  is_wearing_helmet: boolean | null;
  details?: {
    with_helmet_count?: number;
    without_helmet_count?: number;
    total_riders?: number;
    max_with_helmet_confidence?: number;
    max_without_helmet_confidence?: number;
  };
  raw_detections: {
    with_helmet_count: number;
    without_helmet_count: number;
    total_detections: number;
    all_detections: Array<{
      class: string;
      confidence: number;
      bbox: number[];
    }>;
  };
}

const HelmetDetectionApp: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [activeTab, setActiveTab] = useState<'image' | 'live'>('image');

  const translatePrediction = (prediction: string): string => {
    const translations: { [key: string]: string } = {
      'Wearing Helmet': 'Usando Casco',
      'Not Wearing Helmet': 'Sin Usar Casco',
      'Mixed Detection': 'Detección Mixta',
      'No Detection': 'Sin Detección'
    };
    return translations[prediction] || prediction;
  };

  const translateClass = (className: string): string => {
    const translations: { [key: string]: string } = {
      'With_Helmet': 'Con Casco',
      'Without_Helmet': 'Sin Casco'
    };
    return translations[className] || className;
  };

  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Por favor seleccione un archivo de imagen válido');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setError('El tamaño del archivo debe ser menor a 10MB');
      return;
    }

    setSelectedFile(file);
    setPrediction(null);
    setError(null);

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files[0]) {
      handleFileSelect(files[0]);
    }
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    const files = event.dataTransfer.files;
    if (files && files[0]) {
      handleFileSelect(files[0]);
    }
  };

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
  };

  const clearSelection = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setPrediction(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const makePrediction = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await fetch('http://localhost:8080/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Prediction failed');
      }

      const result: PredictionResult = await response.json();
      setPrediction(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (prediction: string) => {
    switch (prediction) {
      case 'Wearing Helmet':
        return 'status-green';
      case 'Not Wearing Helmet':
        return 'status-red';
      case 'Mixed Detection':
        return 'status-orange';
      case 'No Detection':
        return 'status-gray';
      default:
        return 'status-blue';
    }
  };

  const getResultIcon = (prediction: string, isWearingHelmet: boolean | null) => {
    if (prediction === 'No Detection') {
      return <AlertCircle className="icon icon-gray" />;
    }
    if (prediction === 'Mixed Detection') {
      return <AlertCircle className="icon icon-orange" />;
    }
    return isWearingHelmet ? (
      <CheckCircle className="icon icon-green" />
    ) : (
      <AlertCircle className="icon icon-red" />
    );
  };

  useEffect(() => {
    // Clean up preview URL when component unmounts or file changes
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  return (
    <div className="app-container">
      <div className="content-container">
        <div className="header">
          <h1 className="title">Detección de Personas con Cascos</h1>
          <p className="subtitle">
            Sube una imagen o usa la cámara en vivo para detectar si se está usando casco.
          </p>
        </div>

        <div className="card">
          <div className="tabs">
            <button 
              className={`tab ${activeTab === 'image' ? 'active' : ''}`}
              onClick={() => setActiveTab('image')}
            >
              <Upload className="icon-tab" /> Analizar Imagen
            </button>
            <button 
              className={`tab ${activeTab === 'live' ? 'active' : ''}`}
              onClick={() => setActiveTab('live')}
            >
              <Video className="icon-tab" /> Detección en Vivo
            </button>
          </div>

          {activeTab === 'image' && (
            <>
              <div className="upload-section">
                <div
                  className={`drop-zone ${selectedFile ? 'drop-zone-active' : ''}`}
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                >
                  {!selectedFile ? (
                    <div className="upload-prompt">
                      <Upload className="icon-upload" />
                      <p className="upload-title">
                        Arrastra una imagen o haz clic para elegir una
                      </p>
                      <p className="upload-subtitle">
                        Soporte para JPG, PNG, GIF (máx. 10MB)
                      </p>
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        className="btn btn-primary"
                      >
                        Subir Imagen
                      </button>
                    </div>
                  ) : (
                    <div className="preview-container">
                      <div className="preview-header">
                        <Camera className="icon-camera" />
                        <span className="file-name">{selectedFile.name}</span>
                        <button
                          onClick={clearSelection}
                          className="btn-clear"
                        >
                          <X className="icon-x" />
                        </button>
                      </div>
                      {previewUrl && (
                        <div className="preview-image-container">
                          <img
                            src={previewUrl}
                            alt="Vista previa"
                            className="preview-image"
                          />
                        </div>
                      )}
                      <button
                        onClick={makePrediction}
                        disabled={loading}
                        className={`btn btn-detect ${loading ? 'btn-disabled' : ''}`}
                      >
                        {loading ? (
                          <>
                            <Loader2 className="icon-loading" />
                            Analizando...
                          </>
                        ) : (
                          'Detectar Casco'
                        )}
                      </button>
                    </div>
                  )}
                </div>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="hidden"
                />
              </div>

              {error && !loading && (
                <div className="error-message">
                  <div className="error-content">
                    <AlertCircle className="icon icon-red" />
                    <span className="error-text">{error}</span>
                  </div>
                </div>
              )}

              {loading && (
                <div className="loading-container">
                  <Loader2 className="icon-loading-main" />
                  <p className="loading-text">Analizando imagen...</p>
                </div>
              )}

              {prediction && !loading && (
                <div className="results-section">
                  <h2 className="results-title">Resultados del Análisis</h2>
                  <div className={`status-banner ${getStatusColor(prediction.prediction)}`}>
                    {getResultIcon(prediction.prediction, prediction.is_wearing_helmet)}
                    <span className="status-text">{translatePrediction(prediction.prediction)}</span>
                  </div>
                  <div className="results-grid">
                    <div className="result-item">
                      <p className="result-label">Confianza General</p>
                      <p className="result-value bold">{prediction.confidence.toFixed(1)}%</p>
                    </div>
                    <div className="result-item">
                      <p className="result-label">Usando Casco</p>
                      <p className={`result-value ${prediction.is_wearing_helmet ? 'text-green' : 'text-red'}`}>
                        {prediction.is_wearing_helmet === null ? 'N/A' : prediction.is_wearing_helmet ? 'Sí' : 'No'}
                      </p>
                    </div>
                    <div className="result-item">
                      <p className="result-label">Total Detectado</p>
                      <p className="result-value">{prediction.raw_detections.total_detections}</p>
                    </div>
                  </div>
                  <p className="message-text">{prediction.message}</p>
                  {prediction.raw_detections.all_detections.length > 0 && (
                    <div className="detections-list">
                      <h3 className="detections-title">Detecciones Detalladas</h3>
                      <ul>
                        {prediction.raw_detections.all_detections.map((det, index) => (
                          <li key={index} className="detection-item">
                            <span className={`detection-class ${det.class === 'With_Helmet' ? 'text-green' : 'text-red'}`}>
                              {translateClass(det.class)}
                            </span>
                            <span className="detection-confidence">
                              Confianza: {det.confidence.toFixed(1)}%
                            </span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </>
          )}

          {activeTab === 'live' && <LiveDetection />}
        </div>

        <div className="info-section">
          <h3 className="info-title">¿Cómo funciona?</h3>
          <p className="info-text">
            Esta IA analiza imágenes de personas conduciendo motocicletas y determina si están usando o no casco.
          </p>
        </div>
      </div>
    </div>
  );
};

export default HelmetDetectionApp;