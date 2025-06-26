import React from 'react'

const ActionButtons = ({
  handleUploadToMongo,
  handleProcessFile,
  fetchProcessedOutput,
  selectedFile,
  processing,
}) => (
  <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', margin: '2rem 0' }}>
    <button
      className="var-process-btn"
      onClick={handleUploadToMongo}
      disabled={!selectedFile}
      style={{ background: 'rgba(0, 234, 255, 0.8)', color: '#181c24' }}

    >
      Save to Database
    </button>
    <button
      className="var-process-btn"
      onClick={handleProcessFile}
      disabled={!selectedFile || processing}
      style={{ background: 'rgba(0, 234, 255, 0.8)', color: '#181c24' }}

    >
      {processing ? 'Processing...' : 'Process'}
    </button>
    <button
      className="var-process-btn"
      onClick={fetchProcessedOutput}
      style={{ background: 'rgba(0, 234, 255, 0.8)', color: '#181c24' }}

    >
      Show Processed Output
    </button>
  </div>
)

export default ActionButtons
