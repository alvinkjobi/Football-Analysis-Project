import React, { useRef, useState, useCallback, useEffect } from 'react'
import Particles from './Particles'
import UploadSection from './UploadSection'
import ActionButtons from './ActionButtons'
import StatusMessages from './StatusMessages'
import PreviewSection from './PreviewSection'
import ProcessedOutputSection from './ProcessedOutputSection'

function App() {
  // State and refs
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [resultUrl, setResultUrl] = useState(null)
  const [resultMime, setResultMime] = useState(null)
  const [error, setError] = useState(null)
  const [processing, setProcessing] = useState(false)
  const [uploadStatus, setUploadStatus] = useState(null)
  const [uploadSuccess, setUploadSuccess] = useState(false)
  const [processStatus, setProcessStatus] = useState(null)
  const [processSuccess, setProcessSuccess] = useState(false)
  const videoRef = useRef(null)

  // Utility
  const revokeResultUrl = useCallback(() => {
    if (resultUrl) URL.revokeObjectURL(resultUrl)
  }, [resultUrl])

  // Handlers
  const handleFileChange = (e) => {
    const file = e.target.files[0]
    setSelectedFile(file)
    setPreviewUrl(file ? URL.createObjectURL(file) : null)
    setResultUrl(null)
    setResultMime(null)
    setUploadStatus(null)
    setUploadSuccess(false)
    setProcessStatus(null)
    setProcessSuccess(false)
    setError(null)
  }

  const handleUploadToMongo = async () => {
    setUploadStatus(null)
    setUploadSuccess(false)
    if (!selectedFile) {
      setUploadStatus('No file selected')
      return
    }
    try {
      const base64 = await new Promise((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = () => resolve(reader.result.split(',')[1])
        reader.onerror = reject
        reader.readAsDataURL(selectedFile)
      })
      const payload = {
        filename: selectedFile.name,
        mimetype: selectedFile.type,
        data: base64,
      }
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!response.ok) throw new Error('Failed to upload')
      const data = await response.json()
      setUploadStatus(`Saved to database with id: ${data.id}`)
      setUploadSuccess(true)
    } catch (err) {
      setUploadStatus('Failed to upload to database')
      setUploadSuccess(false)
    }
  }

  const handleProcessFile = async () => {
    setProcessStatus(null)
    setProcessSuccess(false)
    setError(null)
    if (!selectedFile) {
      setError('No file selected')
      return
    }
    setProcessing(true)
    try {
      const response = await fetch('http://localhost:5000/run-final', { method: 'POST' })
      if (!response.ok) throw new Error('Processing failed')
      const result = await response.json()
      if (result.error) throw new Error(result.error)
      setProcessStatus('Processing complete!')
      setProcessSuccess(true)
    } catch (err) {
      setProcessStatus('Processing failed: ' + (err.message || ''))
      setProcessSuccess(false)
    }
    setProcessing(false)
  }

  // Fetch processed output from MongoDB (base64) via backend endpoint
  const fetchProcessedOutput = async () => {
    setError(null)
    setResultUrl(null)
    setResultMime(null)
    try {
      const response = await fetch('http://localhost:5000/output-latest')
      if (!response.ok) throw new Error('No processed output found')
      const { mimetype, data } = await response.json()
      const byteString = atob(data)
      const byteArray = new Uint8Array(byteString.length)
      for (let i = 0; i < byteString.length; i++) byteArray[i] = byteString.charCodeAt(i)
      const isVideo = mimetype && mimetype.startsWith('video')
      const blob = new Blob([byteArray], { type: isVideo ? 'video/mp4' : mimetype })
      const url = URL.createObjectURL(blob)
      setResultUrl(url)
      setResultMime(isVideo ? 'video/mp4' : mimetype)
    } catch (err) {
      // Fallback: try to fetch static file directly
      try {
        console.log('Trying to fetch /output/output.mp4')
        const videoRes = await fetch('http://localhost:5000/output/output.mp4')
        if (videoRes.ok) {
          const blob = await videoRes.blob()
          setResultUrl(URL.createObjectURL(blob))
          setResultMime('video/mp4')
          return
        }
        console.log('Trying to fetch /output/output.jpg')
        const imgRes = await fetch('http://localhost:5000/output/output.jpg')
        if (imgRes.ok) {
          const blob = await imgRes.blob()
          setResultUrl(URL.createObjectURL(blob))
          setResultMime('image/jpeg')
          return
        }
        setError('Failed to fetch or display processed output')
      } catch {
        setError('Failed to fetch or display processed output')
      }
    }
  }

  useEffect(() => () => { revokeResultUrl() }, [revokeResultUrl])

  const isVideo = selectedFile && selectedFile.type.startsWith('video')
  const isImage = selectedFile && selectedFile.type.startsWith('image')
  const isResultVideo = resultMime && resultMime.startsWith('video')
  const isResultImage = resultMime && resultMime.startsWith('image')

  return (
    <div className="var-dashboard">
      {/* Particles background */}
      <div style={{ width: '100%', position: 'relative', zIndex: 0 }}>
        <Particles
          particleColors={['#ffffff', '#ffffff']}
          particleCount={200}
          particleSpread={10}
          speed={0.1}
          particleBaseSize={100}
          moveParticlesOnHover={true}
          alphaParticles={false}
          disableRotation={false}
        />
      </div>

      <header className="var-header" style={{ position: 'fixed', zIndex: 1 }}>
        <h1>Football Match Analyzer</h1>
      </header>

      {/* Move UploadSection directly under the header */}
      <div style={{ marginTop: '6rem' }}>
        <UploadSection handleFileChange={handleFileChange} />
      </div>

      <ActionButtons
        handleUploadToMongo={handleUploadToMongo}
        handleProcessFile={handleProcessFile}
        fetchProcessedOutput={fetchProcessedOutput}
        selectedFile={selectedFile}
        processing={processing}
      />

      <StatusMessages
        error={error}
        uploadStatus={uploadStatus}
        uploadSuccess={uploadSuccess}
        processStatus={processStatus}
        processSuccess={processSuccess}
      />

      <main className="var-main">
        <PreviewSection
          previewUrl={previewUrl}
          isVideo={isVideo}
          videoRef={videoRef}
        />
        <ProcessedOutputSection
          resultUrl={resultUrl}
          isResultVideo={isResultVideo}
          isResultImage={isResultImage}
        />
      </main>
      <footer className="var-footer">
        <span>Powered by YOLO & Computer Vision</span>
      </footer>
    </div>
  )
}



export default App
