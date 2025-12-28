import React, { useState } from 'react';
import { Upload, Download, FolderOpen, FileJson, AlertCircle, CheckCircle } from 'lucide-react';

const DataManager = ({ 
  onLoadDataset, 
  onLoadAnnotations,
  currentBatchInfo,
  annotations,
  batchMetadata,
  stats
}) => {
  const [batchFiles, setBatchFiles] = useState([]);
  const [selectedBatch, setSelectedBatch] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showBatchSelector, setShowBatchSelector] = useState(false);

  // Load multiple batch files from folder
  const handleFolderSelect = async (event) => {
    const files = Array.from(event.target.files);
    const jsonFiles = files.filter(f => f.name.endsWith('.json')).sort((a, b) => a.name.localeCompare(b.name));
    
    if (jsonFiles.length === 0) {
      alert('No JSON files found in selected folder');
      return;
    }

    setBatchFiles(jsonFiles);
    setShowBatchSelector(true);
    console.log(`Found ${jsonFiles.length} batch files`);
  };

  // Load a specific batch
  const loadBatch = async (file) => {
    setLoading(true);
    try {
      const text = await file.text();
      const data = JSON.parse(text);
      
      onLoadDataset(data, {
        filename: file.name,
        size: file.size,
        imageCount: data.length
      });
      
      setSelectedBatch(file.name);
      setShowBatchSelector(false);
      setLoading(false);
      
      console.log(`Loaded batch: ${file.name} with ${data.length} images`);
    } catch (error) {
      console.error('Error loading batch:', error);
      alert(`Error loading batch: ${error.message}`);
      setLoading(false);
    }
  };

  // Load single dataset file
  const handleSingleFileLoad = async (event) => {
    const file = event.target.files[0];
    if (file) {
      loadBatch(file);
    }
  };

  // Import annotations
  const handleImportAnnotations = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    try {
      const text = await file.text();
      const data = JSON.parse(text);
      
      if (!data.annotations) {
        alert('Invalid annotation file: missing annotations data');
        return;
      }

      onLoadAnnotations(data);
      alert(`Successfully imported annotations!\n\n` +
            `- ${Object.keys(data.annotations || {}).length} image annotations\n` +
            `- ${Object.keys(data.batchMetadata || {}).length} batch metadata entries`);
      
    } catch (error) {
      console.error('Error importing annotations:', error);
      alert(`Error importing annotations: ${error.message}`);
    }
  };

  // Export annotations
  const exportAnnotations = () => {
    const exportData = {
      annotations,
      batchMetadata,
      exportDate: new Date().toISOString(),
      batchInfo: currentBatchInfo,
      stats
    };
    
    const dataStr = JSON.stringify(exportData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    
    const batchName = currentBatchInfo?.filename?.replace('.json', '') || 'annotations';
    link.download = `${batchName}_annotations_${new Date().toISOString().split('T')[0]}.json`;
    link.click();
  };

  return (
    <div className="space-y-4">
      {/* Current Batch Info */}
      {currentBatchInfo && (
        <div className="bg-gray-800 rounded-lg p-4 shadow-lg border-l-4 border-blue-500">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-semibold text-white mb-1">Current Batch</h3>
              <p className="text-sm text-gray-400">📁 {currentBatchInfo.filename}</p>
              <p className="text-xs text-gray-500">
                {currentBatchInfo.imageCount} images • {(currentBatchInfo.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
            <button
              onClick={() => setShowBatchSelector(true)}
              className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg text-sm font-semibold transition-all"
            >
              Switch Batch
            </button>
          </div>
        </div>
      )}

      {/* Batch Selector Modal */}
      {showBatchSelector && batchFiles.length > 0 && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-6">
          <div className="bg-gray-800 rounded-lg p-6 max-w-2xl w-full max-h-[80vh] overflow-y-auto shadow-2xl">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-2xl font-bold text-white">Select Batch to Annotate</h2>
              <button
                onClick={() => setShowBatchSelector(false)}
                className="text-gray-400 hover:text-white text-2xl"
              >
                ×
              </button>
            </div>
            
            <p className="text-gray-400 text-sm mb-4">
              Found {batchFiles.length} batch files. Select one to start annotating.
            </p>

            <div className="space-y-2">
              {batchFiles.map((file, idx) => (
                <button
                  key={idx}
                  onClick={() => loadBatch(file)}
                  disabled={loading}
                  className={`w-full bg-gray-700 hover:bg-gray-600 disabled:opacity-50 p-4 rounded-lg text-left transition-all ${
                    selectedBatch === file.name ? 'ring-2 ring-blue-500' : ''
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <FileJson className="w-5 h-5 text-blue-400" />
                      <div>
                        <p className="font-semibold text-white">{file.name}</p>
                        <p className="text-xs text-gray-400">
                          {(file.size / 1024 / 1024).toFixed(2)} MB
                        </p>
                      </div>
                    </div>
                    {selectedBatch === file.name && (
                      <CheckCircle className="w-5 h-5 text-green-500" />
                    )}
                  </div>
                </button>
              ))}
            </div>

            {loading && (
              <div className="mt-4 text-center text-blue-400">
                <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-2"></div>
                Loading batch...
              </div>
            )}
          </div>
        </div>
      )}

      {/* Load Options */}
      {!currentBatchInfo && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          
          {/* Load Batch Folder */}
          <div className="bg-gray-800 rounded-lg p-6 shadow-lg">
            <div className="flex items-center gap-3 mb-3">
              <FolderOpen className="w-8 h-8 text-blue-400" />
              <div>
                <h3 className="font-semibold text-white">Load Batch Folder</h3>
                <p className="text-xs text-gray-400">Select folder with multiple batch files</p>
              </div>
            </div>
            <label className="block bg-blue-600 hover:bg-blue-700 px-4 py-3 rounded-lg cursor-pointer text-center font-semibold transition-all">
              <FolderOpen className="w-5 h-5 inline mr-2" />
              Select Folder
              <input
                type="file"
                webkitdirectory=""
                directory=""
                multiple
                accept=".json"
                onChange={handleFolderSelect}
                className="hidden"
              />
            </label>
            <p className="text-xs text-gray-500 mt-2 text-center">
              Choose a folder containing batch_001.json, batch_002.json, etc.
            </p>
          </div>

          {/* Load Single File */}
          <div className="bg-gray-800 rounded-lg p-6 shadow-lg">
            <div className="flex items-center gap-3 mb-3">
              <Upload className="w-8 h-8 text-green-400" />
              <div>
                <h3 className="font-semibold text-white">Load Single File</h3>
                <p className="text-xs text-gray-400">Load one JSON batch file</p>
              </div>
            </div>
            <label className="block bg-green-600 hover:bg-green-700 px-4 py-3 rounded-lg cursor-pointer text-center font-semibold transition-all">
              <Upload className="w-5 h-5 inline mr-2" />
              Select File
              <input
                type="file"
                accept=".json"
                onChange={handleSingleFileLoad}
                className="hidden"
              />
            </label>
            <p className="text-xs text-gray-500 mt-2 text-center">
              Load a single batch file to annotate
            </p>
          </div>

        </div>
      )}

      {/* Import/Export Annotations */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        
        {/* Import Annotations */}
        <div className="bg-gray-800 rounded-lg p-4 shadow-lg">
          <h3 className="font-semibold text-white mb-2 flex items-center gap-2">
            <Upload className="w-5 h-5 text-purple-400" />
            Import Previous Annotations
          </h3>
          <p className="text-xs text-gray-400 mb-3">
            Load annotations from a previous session
          </p>
          <label className="block bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded-lg cursor-pointer text-center font-semibold transition-all text-sm">
            <Upload className="w-4 h-4 inline mr-2" />
            Import Annotations
            <input
              type="file"
              accept=".json"
              onChange={handleImportAnnotations}
              className="hidden"
            />
          </label>
        </div>

        {/* Export Annotations */}
        <div className="bg-gray-800 rounded-lg p-4 shadow-lg">
          <h3 className="font-semibold text-white mb-2 flex items-center gap-2">
            <Download className="w-5 h-5 text-orange-400" />
            Export Current Annotations
          </h3>
          <p className="text-xs text-gray-400 mb-3">
            Save your progress ({Object.keys(annotations).length} annotations)
          </p>
          <button
            onClick={exportAnnotations}
            disabled={Object.keys(annotations).length === 0}
            className="w-full bg-orange-600 hover:bg-orange-700 disabled:bg-gray-700 disabled:opacity-50 px-4 py-2 rounded-lg font-semibold transition-all text-sm"
          >
            <Download className="w-4 h-4 inline mr-2" />
            Export Annotations
          </button>
        </div>

      </div>

      {/* Help Text */}
      <div className="bg-gray-800 rounded-lg p-4 shadow-lg border-l-4 border-yellow-500">
        <div className="flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
          <div>
            <h4 className="font-semibold text-white mb-1">Workflow Tips</h4>
            <ul className="text-xs text-gray-400 space-y-1">
              <li>• Load a batch folder to easily switch between batches</li>
              <li>• Export annotations frequently to save your progress</li>
              <li>• Import annotations to continue from where you left off</li>
              <li>• Annotations auto-save to browser, but exports create backup files</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DataManager;