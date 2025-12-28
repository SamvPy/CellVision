import React from 'react';
import { ChevronLeft, ChevronRight, Download, AlertCircle } from 'lucide-react';

const AnnotationPage = ({
  dataset,
  currentIndex,
  setCurrentIndex,
  annotations,
  metadata,
  updateMetadata,
  handleAnnotation,
  exportAnnotations,
  stats,
  annotationCategories,
  iconMap,
  getColorClasses,
  activeTab,
  setActiveTab
}) => {
  const currentItem = dataset[currentIndex];
  const currentAnnotation = annotations[currentIndex];
  const progress = dataset.length > 0 ? ((stats.annotated / dataset.length) * 100).toFixed(1) : 0;

  return (
    <>
      {/* Stats Header - Horizontal Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mb-6">
        {annotationCategories.map(cat => {
          const Icon = iconMap[cat.iconName] || AlertCircle;
          const colors = getColorClasses(cat.color);
          return (
            <div key={cat.id} className={`bg-gray-800 rounded-lg p-4 border-l-4 ${colors.border} hover:opacity-90 transition-all`}>
              <div className="flex items-center gap-2 mb-2">
                <Icon className={`w-5 h-5 ${colors.text}`} />
                <span className="text-xs text-gray-400 uppercase tracking-wider font-semibold">{cat.label}</span>
              </div>
              <div className="text-2xl font-bold text-white">{stats[cat.id] || 0}</div>
            </div>
          );
        })}
      </div>

      {/* Progress Bar */}
      <div className="bg-gray-800 rounded-lg p-4 mb-6 shadow-lg">
        <div className="flex justify-between text-sm mb-3">
          <span className="font-semibold text-white">Annotation Progress</span>
          <span className="text-gray-400 font-medium">{stats.annotated || 0} / {stats.total || 0} <span className="text-blue-400">({progress}%)</span></span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
          <div
            className="bg-blue-500 h-3 rounded-full transition-all"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 gap-6">
        <div className="space-y-4">
          
          {/* Image Caption */}
          <div className="bg-gray-800 rounded-lg p-6 shadow-lg">
            <h2 className="text-2xl font-bold mb-2 text-white">{currentItem?.filename}</h2>
            <div className="flex items-center gap-3 text-sm text-gray-400">
              <span className="bg-gray-700 px-3 py-1 rounded-full">Image {currentIndex + 1} / {dataset.length}</span>
              <span className="bg-blue-600 bg-opacity-20 text-blue-400 px-3 py-1 rounded-full font-medium">
                Run: {currentItem?.run_folder}
              </span>
            </div>
          </div>

          {/* Tabs */}
          <div className="bg-gray-800 rounded-lg overflow-hidden shadow-lg">
            <div className="flex border-b border-gray-700 bg-gray-900">
              <button
                onClick={() => setActiveTab('cropped')}
                className={`flex-1 px-6 py-4 font-semibold transition-all ${
                  activeTab === 'cropped'
                    ? 'bg-gray-800 text-white border-b-2 border-blue-500 shadow-lg'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
                }`}
              >
                🔬 Cropped View
              </button>
              <button
                onClick={() => setActiveTab('original')}
                className={`flex-1 px-6 py-4 font-semibold transition-all ${
                  activeTab === 'original'
                    ? 'bg-gray-800 text-white border-b-2 border-blue-500 shadow-lg'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
                }`}
              >
                🖼️ Original Context
              </button>
            </div>

            <div className="p-6 bg-gradient-to-br from-gray-800 to-gray-900">
              {activeTab === 'cropped' ? (
                <div className="grid grid-cols-2 gap-6">
                  <div>
                    <p className="text-xs text-gray-400 mb-3 font-semibold uppercase tracking-wider">Cropped Cell</p>
                    <div className="bg-gray-900 rounded-lg p-3 flex items-center justify-center shadow-xl border border-gray-700" style={{ minHeight: '400px', minWidth: '400px' }}>
                      {currentItem?.crop_img ? (
                        <img
                          src={currentItem.crop_img}
                          alt="Crop"
                          className="rounded"
                          style={{
                            imageRendering: 'pixelated',
                            width: '100%',
                            height: '100%',
                            objectFit: 'contain',
                            maxWidth: '400px',
                            maxHeight: '400px'
                          }}
                        />
                      ) : (
                        <span className="text-gray-500">No image</span>
                      )}
                    </div>
                  </div>
                  <div>
                    <p className="text-xs text-gray-400 mb-3 font-semibold uppercase tracking-wider">Crop Mask</p>
                    <div className="bg-gray-900 rounded-lg p-3 flex items-center justify-center shadow-xl border border-gray-700" style={{ minHeight: '400px', minWidth: '400px' }}>
                      {currentItem?.crop_mask_img ? (
                        <img
                          src={currentItem.crop_mask_img}
                          alt="Crop Mask"
                          className="rounded"
                          style={{
                            imageRendering: 'pixelated',
                            width: '100%',
                            height: '100%',
                            objectFit: 'contain',
                            maxWidth: '400px',
                            maxHeight: '400px'
                          }}
                        />
                      ) : (
                        <span className="text-gray-500">No image</span>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="grid grid-cols-2 gap-6">
                  <div>
                    <p className="text-xs text-gray-400 mb-3 font-semibold uppercase tracking-wider">Full Mask</p>
                    <div className="bg-gray-900 rounded-lg p-3 flex items-center justify-center shadow-xl border border-gray-700" style={{ minHeight: '400px' }}>
                      {currentItem?.masks_img ? (
                        <img
                          src={currentItem.masks_img}
                          alt="Mask"
                          className="rounded"
                          style={{
                            width: '100%',
                            height: '100%',
                            objectFit: 'contain',
                            maxHeight: '400px'
                          }}
                        />
                      ) : (
                        <span className="text-gray-500">No image</span>
                      )}
                    </div>
                  </div>
                  <div>
                    <p className="text-xs text-gray-400 mb-3 font-semibold uppercase tracking-wider">Full Image</p>
                    <div className="bg-gray-900 rounded-lg p-3 flex items-center justify-center shadow-xl border border-gray-700" style={{ minHeight: '400px', minWidth: '400px' }}>
                      {currentItem?.img ? (
                        <img
                          src={currentItem.img}
                          alt="Full Image"
                          className="rounded"
                          style={{
                            imageRendering: 'pixelated',
                            width: '100%',
                            height: '100%',
                            objectFit: 'contain',
                            maxHeight: '400px'
                          }}
                        />
                      ) : (
                        <span className="text-gray-500">No image</span>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Compact Annotation Section */}
          <div className="bg-gray-800 rounded-lg p-4 shadow-lg">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              
              {/* Metadata - Compact Horizontal */}
              <div>
                <h3 className="text-sm font-bold mb-3 text-white flex items-center gap-2">
                  📋 Metadata
                </h3>
                <div className="grid grid-cols-3 gap-2">
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
                      value={metadata[field.key]}
                      onChange={(e) => updateMetadata(field.key, e.target.value)}
                      className="bg-gray-700 text-white rounded px-3 py-2 text-xs focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all"
                      placeholder={field.label}
                      title={field.label}
                    />
                  ))}
                </div>
              </div>

              {/* Quality Assessment - Compact Horizontal */}
              <div>
                <h3 className="text-sm font-bold mb-3 text-white flex items-center gap-2">
                  ⭐ Quality Assessment
                </h3>
                <div className="grid grid-cols-5 gap-2">
                  {annotationCategories.map(cat => {
                    const Icon = iconMap[cat.iconName] || AlertCircle;
                    const colors = getColorClasses(cat.color);
                    const isSelected = currentAnnotation?.quality === cat.id;
                    return (
                      <button
                        key={cat.id}
                        onClick={() => handleAnnotation(cat.id)}
                        className={`${colors.bg} hover:opacity-90 px-2 py-3 rounded-lg flex flex-col items-center justify-center gap-1 font-semibold transition-all shadow-lg ${
                          isSelected ? 'ring-2 ring-white scale-105' : ''
                        }`}
                        title={cat.label}
                      >
                        <Icon className="w-5 h-5" />
                        <span className="text-xs">{cat.key.toUpperCase()}</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>

          {/* Navigation */}
          <div className="flex gap-4">
            <button
              onClick={() => setCurrentIndex(Math.max(0, currentIndex - 1))}
              disabled={currentIndex === 0}
              className="flex-1 bg-gray-800 hover:bg-gray-700 disabled:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed px-6 py-4 rounded-lg flex items-center justify-center gap-2 font-semibold transition-all shadow-lg text-lg"
            >
              <ChevronLeft className="w-6 h-6" />
              Previous
            </button>
            <button
              onClick={exportAnnotations}
              disabled={Object.keys(annotations).length === 0}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed px-6 py-4 rounded-lg flex items-center justify-center gap-2 font-bold transition-all shadow-xl"
            >
              <Download className="w-5 h-5" />
              Export
            </button>
            <button
              onClick={() => setCurrentIndex(Math.min(dataset.length - 1, currentIndex + 1))}
              disabled={currentIndex === dataset.length - 1}
              className="flex-1 bg-gray-800 hover:bg-gray-700 disabled:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed px-6 py-4 rounded-lg flex items-center justify-center gap-2 font-semibold transition-all shadow-lg text-lg"
            >
              Next
              <ChevronRight className="w-6 h-6" />
            </button>
          </div>

        </div>
      </div>
    </>
  );
};

export default AnnotationPage;