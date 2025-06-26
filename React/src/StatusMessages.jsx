import React from 'react'

const StatusMessages = ({
  error,
  uploadStatus,
  uploadSuccess,
  processStatus,
  processSuccess,
}) => (
  <div style={{ marginBottom: '2rem', minHeight: '2.5em', textAlign: 'center' }}>
    {error && <div className="var-error">{error}</div>}
    {uploadStatus && (
      <div
        className="var-error"
        style={uploadSuccess ? { color: 'green', fontWeight: 'bold' } : {}}
      >
        {uploadStatus}
      </div>
    )}
    {processStatus && (
      <div
        className="var-error"
        style={processSuccess ? { color: 'green', fontWeight: 'bold' } : {}}
      >
        {processStatus}
      </div>
    )}
  </div>
)

export default StatusMessages
