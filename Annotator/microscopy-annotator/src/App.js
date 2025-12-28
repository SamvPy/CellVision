import React, { useState, useEffect } from 'react';
import { Upload, AlertCircle, CheckCircle, XCircle, HelpCircle, Zap, AlertTriangle } from 'lucide-react';
import './App.css';
import AnnotationPage from './AnnotationPage';
import BatchMetadataPage from './BatchMetadataPage';
import DataManager from './DataManager';

function App() {
  // Icon mapping
  const iconMap = {
    CheckCircle, XCircle, HelpCircle, AlertCircle, AlertTriangle, Zap
  };

  // Customizable annotation categories
  const [annotationCategories, setAnnotationCategories] = useState([
    { id: 'good', label: 'Good Cell', color: 'green', iconName: 'CheckCircle', key: 'g' },
    { id: 'bad_artifact', label: 'Artifact', color: 'red', iconName: 'XCircle', key: 'b' },
    { id: 'bad_segmentation', label: 'Bad Segmentation', color: 'orange', iconName: 'AlertTriangle', key: 's' },
    { id: 'debris', label: 'Debris/Dead Cell', color: 'purple', iconName: 'Zap', key: 'd' },
    { id: 'unsure', label: 'Unsure', color: 'yellow', iconName: 'HelpCircle', key: 'u' },
  ]);

  const [dataset, setDataset] = useState([]);
  const [currentBatchInfo, setCurrentBatchInfo] = useState(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [annotations, setAnnotations] = useState({});
  const [metadata, setMetadata] = useState({
    cellType: '',
    fixationMethod: '',
    treated: '',
    dpi: '',
    infection: '',
    treatment: ''
  });
  const [stats, setStats] = useState({});
  const [activeTab, setActiveTab] = useState('cropped');
  const [currentPage, setCurrentPage] = useState('annotate');
  const [groupedData, setGroupedData] = useState({});
  const [expandedFolders, setExpandedFolders] = useState({});
  const [batchMetadata, setBatchMetadata] = useState({});

  // Load from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('microscopy_annotations');
    if (saved) {
      const data = JSON.parse(saved);
      setAnnotations(data.annotations || {});
      setCurrentIndex(data.currentIndex || 0);
      if (data.categories) {
        setAnnotationCategories(data.categories);
      }
      if (data.batchMetadata) {
        setBatchMetadata(data.batchMetadata);
      }
    }
  }, []);

  // Group dataset by folder -> run_folder -> filename
  useEffect(() => {
    if (dataset.length > 0) {
      const grouped = {};
      dataset.forEach((item, idx) => {
        const folder = item.folder || 'Unknown';
        const runFolder = item.run_folder || 'Unknown';
        const filename = item.filename || 'Unknown';

        if (!grouped[folder]) grouped[folder] = {};
        if (!grouped[folder][runFolder]) grouped[folder][runFolder] = {};
        if (!grouped[folder][runFolder][filename]) grouped[folder][runFolder][filename] = [];

        grouped[folder][runFolder][filename].push({ ...item, originalIndex: idx });
      });
      setGroupedData(grouped);
    }
  }, [dataset]);

  // Load metadata for current image (check batch metadata too)
  useEffect(() => {
    const currentItem = dataset[currentIndex];
    const currentAnnotation = annotations[currentIndex];

    if (currentAnnotation?.metadata) {
      setMetadata(currentAnnotation.metadata);
    } else if (currentItem) {
      const filenameKey = `${currentItem.folder}/${currentItem.run_folder}/${currentItem.filename}`;
      const runFolderKey = `${currentItem.folder}/${currentItem.run_folder}`;

      if (batchMetadata[filenameKey]) {
        setMetadata(batchMetadata[filenameKey]);
      } else if (batchMetadata[runFolderKey]) {
        setMetadata(batchMetadata[runFolderKey]);
      } else {
        setMetadata({
          cellType: '',
          fixationMethod: '',
          treated: '',
          dpi: '',
          infection: '',
          treatment: ''
        });
      }
    }
  }, [currentIndex, annotations, dataset, batchMetadata]);

  // Auto-save to localStorage
  useEffect(() => {
    if (Object.keys(annotations).length > 0 || annotationCategories.length > 0 || Object.keys(batchMetadata).length > 0) {
      localStorage.setItem('microscopy_annotations', JSON.stringify({
        annotations,
        currentIndex,
        categories: annotationCategories,
        batchMetadata,
        lastUpdated: new Date().toISOString()
      }));
    }
  }, [annotations, currentIndex, annotationCategories, batchMetadata]);

  // Calculate stats
  useEffect(() => {
    const annotated = Object.keys(annotations).length;
    const categoryCounts = {};
    annotationCategories.forEach(cat => {
      categoryCounts[cat.id] = Object.values(annotations).filter(a => a.quality === cat.id).length;
    });
    setStats({
      total: dataset.length,
      annotated,
      ...categoryCounts
    });
  }, [annotations, dataset, annotationCategories]);

  const handleAnnotation = (quality) => {
    const currentItem = dataset[currentIndex];
    const newAnnotations = {
      ...annotations,
      [currentIndex]: {
        index: currentItem?.index,
        filename: currentItem?.filename,
        run_folder: currentItem?.run_folder,
        quality,
        metadata: metadata,
        timestamp: new Date().toISOString()
      }
    };
    setAnnotations(newAnnotations);

    if (currentIndex < dataset.length - 1) {
      setCurrentIndex(currentIndex + 1);
    }
  };

  const handleKeyPress = React.useCallback((e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
      return;
    }

    if (e.key === 'ArrowLeft' && currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
      return;
    } else if (e.key === 'ArrowRight' && currentIndex < dataset.length - 1) {
      setCurrentIndex(currentIndex + 1);
      return;
    }

    const category = annotationCategories.find(cat =>
      cat.key.toLowerCase() === e.key.toLowerCase()
    );
    if (category) {
      handleAnnotation(category.id);
    }
  }, [currentIndex, metadata, dataset, annotationCategories]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [handleKeyPress]);

  const exportAnnotations = () => {
    const exportData = {
      annotations,
      categories: annotationCategories,
      batchMetadata,
      exportDate: new Date().toISOString(),
      stats
    };
    const dataStr = JSON.stringify(exportData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `annotations_${new Date().toISOString().split('T')[0]}.json`;
    link.click();
  };

  const applyBatchMetadata = (key, metadataToApply) => {
    setBatchMetadata(prev => ({
      ...prev,
      [key]: metadataToApply
    }));
  };

  const toggleFolder = (path) => {
    setExpandedFolders(prev => ({
      ...prev,
      [path]: !prev[path]
    }));
  };

  const handleLoadDataset = (data, batchInfo) => {
    setDataset(data);
    setCurrentBatchInfo(batchInfo);
    setCurrentIndex(0);
    // Reset to annotation page when new batch is loaded
    setCurrentPage('annotate');
  };

  const handleLoadAnnotations = (importedData) => {
    if (importedData.annotations) {
      setAnnotations(importedData.annotations);
    }
    if (importedData.batchMetadata) {
      setBatchMetadata(importedData.batchMetadata);
    }
    if (importedData.categories) {
      setAnnotationCategories(importedData.categories);
    }
  };

  const updateMetadata = (field, value) => {
    setMetadata({ ...metadata, [field]: value });
  };

  const getColorClasses = (color) => {
    const colors = {
      green: { bg: 'bg-green-600', border: 'border-green-500', text: 'text-green-500' },
      red: { bg: 'bg-red-600', border: 'border-red-500', text: 'text-red-500' },
      orange: { bg: 'bg-orange-600', border: 'border-orange-500', text: 'text-orange-500' },
      yellow: { bg: 'bg-yellow-600', border: 'border-yellow-500', text: 'text-yellow-500' },
      blue: { bg: 'bg-blue-600', border: 'border-blue-500', text: 'text-blue-500' },
      purple: { bg: 'bg-purple-600', border: 'border-purple-500', text: 'text-purple-500' },
      pink: { bg: 'bg-pink-600', border: 'border-pink-500', text: 'text-pink-500' },
      gray: { bg: 'bg-gray-600', border: 'border-gray-500', text: 'text-gray-500' }
    };
    return colors[color] || colors.gray;
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      <div className="max-w-7xl mx-auto p-6">

        {dataset.length === 0 ? (
          <DataManager
            onLoadDataset={handleLoadDataset}
            onLoadAnnotations={handleLoadAnnotations}
            currentBatchInfo={currentBatchInfo}
            annotations={annotations}
            batchMetadata={batchMetadata}
            stats={stats}
          />
        ) : (
          <>
            {/* Top Bar with Batch Info and Navigation */}
            <div className="mb-6 space-y-4">
              {/* Current Batch Info Bar */}
              <DataManager
                onLoadDataset={handleLoadDataset}
                onLoadAnnotations={handleLoadAnnotations}
                currentBatchInfo={currentBatchInfo}
                annotations={annotations}
                batchMetadata={batchMetadata}
                stats={stats}
              />

              {/* Page Navigation */}
              <div className="flex gap-3">
                <button
                  onClick={() => setCurrentPage('annotate')}
                  className={`flex-1 px-6 py-4 rounded-lg font-semibold transition-all shadow-lg ${
                    currentPage === 'annotate'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                  }`}
                >
                  📊 Image Annotation
                </button>
                <button
                  onClick={() => setCurrentPage('batch-metadata')}
                  className={`flex-1 px-6 py-4 rounded-lg font-semibold transition-all shadow-lg ${
                    currentPage === 'batch-metadata'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                  }`}
                >
                  📋 Batch Metadata
                </button>
              </div>
            </div>

            {currentPage === 'annotate' ? (
              <AnnotationPage
                dataset={dataset}
                currentIndex={currentIndex}
                setCurrentIndex={setCurrentIndex}
                annotations={annotations}
                metadata={metadata}
                updateMetadata={updateMetadata}
                handleAnnotation={handleAnnotation}
                exportAnnotations={exportAnnotations}
                stats={stats}
                annotationCategories={annotationCategories}
                iconMap={iconMap}
                getColorClasses={getColorClasses}
                activeTab={activeTab}
                setActiveTab={setActiveTab}
              />
            ) : (
              <BatchMetadataPage
                groupedData={groupedData}
                batchMetadata={batchMetadata}
                applyBatchMetadata={applyBatchMetadata}
                expandedFolders={expandedFolders}
                toggleFolder={toggleFolder}
              />
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default App;