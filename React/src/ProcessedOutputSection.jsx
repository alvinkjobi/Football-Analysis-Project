import React from 'react'

const ProcessedOutputSection = ({ resultUrl, isResultVideo, isResultImage }) => (
  <section className="var-result">
    <h2>Processed Output</h2>
    {resultUrl && (
      <div className="var-preview">
        <span>Processed Output:</span>
        {isResultVideo ? (
          <video
            src={resultUrl}
            controls
            width={400}
            autoPlay
            muted
          />
        ) : isResultImage ? (
          <img src={resultUrl} alt="result" width={400} />
        ) : (
          <span>Unsupported output type</span>
        )}
      </div>
    )}
  </section>
)

export default ProcessedOutputSection
