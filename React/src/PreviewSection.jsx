import React from 'react'

const PreviewSection = ({ previewUrl, isVideo, videoRef }) => (
  <section className="var-upload">
    <h2>Preview</h2>
    {previewUrl && (
      <div className="var-preview">
        <span>Original Preview:</span>
        {isVideo ? (
          <video ref={videoRef} src={previewUrl} controls width={400} />
        ) : (
          <img src={previewUrl} alt="preview" width={400} />
        )}
      </div>
    )}
  </section>
)

export default PreviewSection
