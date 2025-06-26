import React from 'react'

const UploadSection = ({ handleFileChange }) => (
  <div style={{ margin: '2rem 0', textAlign: 'center' }}>
    
    <h2>Upload</h2>
    <label className="var-upload-label">
      <span>Upload Match Video or Image</span>
      <input type="file" accept="video/*,image/*" onChange={handleFileChange} />
    </label>
  </div>
)

export default UploadSection
