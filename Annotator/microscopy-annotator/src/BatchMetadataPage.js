import React from 'react';

const BatchMetadataPage = ({
  groupedData,
  batchMetadata,
  applyBatchMetadata,
  expandedFolders,
  toggleFolder
}) => {
  return (
    <div className="space-y-6">
      <div className="bg-gray-800 rounded-lg p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-2 text-white">Batch Metadata Annotation</h2>
        <p className="text-gray-400 text-sm">
          Apply metadata to multiple images at once by folder, run, or filename.
          These settings will be used as defaults when annotating individual images.
        </p>
      </div>

      {/* Hierarchical View */}
      <div className="bg-gray-800 rounded-lg p-6 shadow-lg">
        {Object.keys(groupedData).map(folder => (
          <div key={folder} className="mb-6">
            {/* Folder Level */}
            <div
              className="flex items-center gap-3 mb-3 cursor-pointer hover:bg-gray-700 p-3 rounded-lg transition-all"
              onClick={() => toggleFolder(folder)}
            >
              <span className="text-2xl">{expandedFolders[folder] ? '📂' : '📁'}</span>
              <h3 className="text-lg font-bold text-white">{folder}</h3>
              <span className="text-gray-400 text-sm">
                ({Object.keys(groupedData[folder]).length} runs)
              </span>
            </div>

            {expandedFolders[folder] && (
              <div className="ml-8 space-y-4">
                {Object.keys(groupedData[folder]).map(runFolder => {
                  const runKey = `${folder}/${runFolder}`;
                  const runPath = `${folder}_${runFolder}`;
                  const fileCount = Object.keys(groupedData[folder][runFolder]).length;
                  const imageCount = Object.values(groupedData[folder][runFolder]).flat().length;
                  const currentMeta = batchMetadata[runKey] || {
                    cellType: '', fixationMethod: '', treated: '', dpi: '', infection: '', treatment: ''
                  };

                  return (
                    <div key={runFolder} className="bg-gray-700 rounded-lg p-4">
                      {/* Run Folder Level */}
                      <div
                        className="flex items-center justify-between mb-3 cursor-pointer"
                        onClick={() => toggleFolder(runPath)}
                      >
                        <div className="flex items-center gap-3">
                          <span className="text-xl">{expandedFolders[runPath] ? '▼' : '▶'}</span>
                          <div>
                            <h4 className="font-semibold text-white text-base">🏃 {runFolder}</h4>
                            <p className="text-xs text-gray-400">
                              {fileCount} files • {imageCount} images
                            </p>
                          </div>
                        </div>
                      </div>

                      {/* Run Folder Metadata Form */}
                      <div className="bg-gray-800 rounded-lg p-4 mb-3">
                        <h5 className="text-sm font-semibold text-gray-300 mb-3">
                          Apply metadata to all images in this run:
                        </h5>
                        <div className="grid grid-cols-3 gap-2 mb-3">
                          {[
                            { key: 'cellType', label: 'Cell Type' },
                            { key: 'fixationMethod', label: 'Fixation' },
                            { key: 'treated', label: 'Treated' },
                            { key: 'dpi', label: 'DPI' },
                            { key: 'infection', label: 'Infection' },
                            { key: 'treatment', label: 'Treatment' }
                          ].map(field => (
                            <input
                              key={field.key}
                              type="text"
                              value={currentMeta[field.key]}
                              onChange={(e) => {
                                const newMeta = { ...currentMeta, [field.key]: e.target.value };
                                applyBatchMetadata(runKey, newMeta);
                              }}
                              className="bg-gray-700 text-white rounded px-3 py-2 text-xs focus:outline-none focus:ring-2 focus:ring-blue-500"
                              placeholder={field.label}
                            />
                          ))}
                        </div>
                        <button
                          onClick={() => applyBatchMetadata(runKey, currentMeta)}
                          className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded text-sm font-semibold transition-all"
                        >
                          ✓ Save for Run
                        </button>
                      </div>

                      {/* Filename Level (Expandable) */}
                      {expandedFolders[runPath] && (
                        <div className="ml-6 space-y-3">
                          {Object.keys(groupedData[folder][runFolder]).map(filename => {
                            const filenameKey = `${folder}/${runFolder}/${filename}`;
                            const images = groupedData[folder][runFolder][filename];
                            const fileMeta = batchMetadata[filenameKey] || currentMeta;

                            return (
                              <div key={filename} className="bg-gray-900 rounded-lg p-3">
                                <div className="flex items-center gap-2 mb-2">
                                  <span className="text-sm">📄</span>
                                  <h5 className="text-sm font-medium text-white">{filename}</h5>
                                  <span className="text-xs text-gray-500">
                                    ({images.length} images)
                                  </span>
                                </div>

                                <div className="grid grid-cols-3 gap-2 mb-2">
                                  {[
                                    { key: 'cellType', label: 'Cell Type' },
                                    { key: 'fixationMethod', label: 'Fixation' },
                                    { key: 'treated', label: 'Treated' },
                                    { key: 'dpi', label: 'DPI' },
                                    { key: 'infection', label: 'Infection' },
                                    { key: 'treatment', label: 'Treatment' }
                                  ].map(field => (
                                    <input
                                      key={field.key}
                                      type="text"
                                      value={fileMeta[field.key]}
                                      onChange={(e) => {
                                        const newMeta = { ...fileMeta, [field.key]: e.target.value };
                                        applyBatchMetadata(filenameKey, newMeta);
                                      }}
                                      className="bg-gray-800 text-white rounded px-2 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-blue-500"
                                      placeholder={field.label}
                                    />
                                  ))}
                                </div>
                                <button
                                  onClick={() => applyBatchMetadata(filenameKey, fileMeta)}
                                  className="bg-green-600 hover:bg-green-700 px-3 py-1 rounded text-xs font-semibold transition-all"
                                >
                                  ✓ Save for File
                                </button>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default BatchMetadataPage;