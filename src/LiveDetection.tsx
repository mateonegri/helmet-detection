import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Camera, Zap, Loader2, AlertCircle } from 'lucide-react';

interface Detection {
  class: string;
  confidence: number;
  bbox: number[];
}

interface LivePredictionResult {
  prediction: string;
  is_wearing_helmet: boolean | null;
  raw_detections?: {
    all_detections: Detection[];
  };
}

const LiveDetection: React.FC = () => {
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const translateClass = (className: string): string => {
    const translations: { [key: string]: string } = {
      'With_Helmet': 'Con Casco',
      'Without_Helmet': 'Sin Casco'
    };
    return translations[className] || className;
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'environment'
        }
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsCameraOn(true);
        setError(null);
      }
    } catch (err) {
      console.error("Error accessing camera:", err);
      setError("No se pudo acceder a la cámara. Por favor, compruebe los permisos.");
    }
  };

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    setIsCameraOn(false);
    setLoading(false);
    setDetections([]);
    if (canvasRef.current) {
      const context = canvasRef.current.getContext('2d');
      context?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  }, []);

  const captureAndPredict = useCallback(async () => {
    if (!videoRef.current || videoRef.current.paused || videoRef.current.ended) {
      return;
    }
    
    setLoading(true);

    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const context = canvas.getContext('2d');
    context?.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
      if (!blob) {
        setLoading(false);
        return;
      }
      
      try {
        const formData = new FormData();
        formData.append('file', blob, 'live-frame.jpg');

        const response = await fetch('http://localhost:8080/predict', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error('Prediction failed');
        }

        const result: LivePredictionResult = await response.json();
        setDetections(result.raw_detections?.all_detections || []);
      } catch (err) {
        console.error("Prediction error:", err);
        // Don't show constant errors to user, just log them
      } finally {
        setLoading(false);
      }
    }, 'image/jpeg');
  }, []);

  useEffect(() => {
    if (isCameraOn) {
      intervalRef.current = setInterval(captureAndPredict, 1000); // Predict every second
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isCameraOn, captureAndPredict]);
  
  useEffect(() => {
    return () => {
      stopCamera(); // Cleanup on component unmount
    };
  }, [stopCamera]);

  useEffect(() => {
    if (videoRef.current && canvasRef.current && isCameraOn) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');

      const drawDetections = () => {
        if (!context) return;
        
        canvas.width = video.clientWidth;
        canvas.height = video.clientHeight;

        const scaleX = canvas.width / video.videoWidth;
        const scaleY = canvas.height / video.videoHeight;

        context.clearRect(0, 0, canvas.width, canvas.height);
        
        detections.forEach(det => {
          const [x1, y1, x2, y2] = det.bbox;
          
          const scaledX1 = x1 * scaleX;
          const scaledY1 = y1 * scaleY;
          const scaledWidth = (x2 - x1) * scaleX;
          const scaledHeight = (y2 - y1) * scaleY;

          const color = det.class === 'With_Helmet' ? '#28a745' : '#dc3545';
          context.strokeStyle = color;
          context.lineWidth = 3;
          context.strokeRect(scaledX1, scaledY1, scaledWidth, scaledHeight);
          
          context.fillStyle = color;
          const label = `${translateClass(det.class)}: ${Math.round(det.confidence)}%`;
          context.font = '16px Arial';
          const textWidth = context.measureText(label).width;
          context.fillRect(scaledX1, scaledY1 - 20, textWidth + 10, 20);
          
          context.fillStyle = '#ffffff';
          context.fillText(label, scaledX1 + 5, scaledY1 - 5);
        });
      };
      
      const animationFrameId = requestAnimationFrame(drawDetections);
      return () => cancelAnimationFrame(animationFrameId);
    }
  }, [detections, isCameraOn]);

  return (
    <div className="live-detection-container">
      <div className="video-wrapper">
        <video 
          ref={videoRef} 
          autoPlay 
          playsInline 
          className="live-video"
          onCanPlay={() => {
            if (videoRef.current) {
              videoRef.current.play().catch(e => console.error("Video play failed", e));
            }
          }}
        />
        <canvas ref={canvasRef} className="detection-canvas" />
        {!isCameraOn && (
          <div className="camera-prompt">
            <Camera className="icon-camera-big" />
            <p>Inicia la cámara para la detección en vivo</p>
          </div>
        )}
      </div>
      
      <div className="live-controls">
        {!isCameraOn ? (
          <button onClick={startCamera} className="btn btn-primary">
            <Camera className="icon-btn" /> Iniciar Cámara
          </button>
        ) : (
          <button onClick={stopCamera} className="btn btn-danger">
            <Zap className="icon-btn" /> Detener
          </button>
        )}
        {loading && <Loader2 className="icon-loading-live" />}
      </div>

      {error && (
        <div className="error-message">
          <div className="error-content">
            <AlertCircle className="icon icon-red" />
            <span className="error-text">{error}</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default LiveDetection; 