import express from 'express'
import mongoose from 'mongoose'
import cors from 'cors'
import path from 'path'
import { exec } from 'child_process'
import fs from 'fs'

const app = express()
const PORT = 5000

app.use(cors())
app.use(express.json({ limit: '100mb' }))

// Connect to MongoDB analyser database
const MONGO_URI = 'mongodb://localhost:27017/analyser'

mongoose.connect(MONGO_URI)
  .then(() => console.log('MongoDB connected to analyser database'))
  .catch((err) => console.error('MongoDB connection error:', err))

// Define schema and model for 'input' collection
const inputSchema = new mongoose.Schema({
  filename: String,
  mimetype: String,
  data: String, // base64 string
  uploadedAt: { type: Date, default: Date.now }
})
const Input = mongoose.model('input', inputSchema, 'input')

// Upload endpoint: save file as base64 JSON in 'input' collection
app.post('/upload', async (req, res) => {
  try {
    const { filename, mimetype, data } = req.body
    if (!filename || !mimetype || !data) {
      return res.status(400).json({ error: 'Missing file data' })
    }
    const doc = new Input({ filename, mimetype, data })
    await doc.save()
    res.json({ message: 'File saved to MongoDB', id: doc._id })
  } catch (err) {
    res.status(500).json({ error: 'Failed to save file', details: err.message })
  }
})

// Endpoint to run final.py
app.post('/run-final', (req, res) => {
  const pythonScript = path.resolve('..', 'football', 'FINAL.py')
  exec(`python "${pythonScript}"`, (error, stdout, stderr) => {
    console.log('--- FINAL.py STDOUT ---\n', stdout)
    console.error('--- FINAL.py STDERR ---\n', stderr)
    if (error) {
      console.error('Error running final.py:', error)
      // Return both stderr and stdout for debugging
      return res.status(500).json({ error: stderr || error.message, stdout })
    }
    res.json({ message: 'final.py executed', output: stdout })
  })
})

// Endpoint to fetch the latest processed output from 'output' collection
app.get('/output-latest', async (req, res) => {
  try {
    // Use the correct collection and schema for output
    const Output = mongoose.model('output', inputSchema, 'output')
    const doc = await Output.findOne().sort({ uploadedAt: -1 })
    if (!doc) return res.status(404).json({ error: 'No output found' })
    res.json({
      filename: doc.filename,
      mimetype: doc.mimetype,
      data: doc.data,
    })
  } catch (err) {
    res.status(500).json({ error: 'Failed to fetch output', details: err.message })
  }
})

// Serve static files from the output folder at /output/*
app.use('/output', express.static('D:/FOOTBALL-ANALYSIS-PROJECT/Football-Analysis-Project/output'))

// Serve the latest processed video from output folder
app.get('/detected_mongo_input.mp4', (req, res) => {
  const outputDir = 'D:/FOOTBALL-ANALYSIS-PROJECT/Football-Analysis-Project/output'
  const files = fs.readdirSync(outputDir)
    .filter(f => f.endsWith('.mp4'))
    .map(f => ({ file: f, time: fs.statSync(path.join(outputDir, f)).mtime }))
    .sort((a, b) => b.time - a.time)
  if (files.length > 0) {
    const videoPath = path.join(outputDir, files[0].file)
    res.setHeader('Content-Type', 'video/mp4')
    fs.createReadStream(videoPath).pipe(res)
  } else {
    res.status(404).json({ error: 'Video not found' })
  }
})

// Serve the latest processed image from output folder
app.get('/detected_mongo_input.jpg', (req, res) => {
  const outputDir = 'D:/FOOTBALL-ANALYSIS-PROJECT/Football-Analysis-Project/output'
  const files = fs.readdirSync(outputDir)
    .filter(f => f.endsWith('.jpg') || f.endsWith('.jpeg') || f.endsWith('.png'))
    .map(f => ({ file: f, time: fs.statSync(path.join(outputDir, f)).mtime }))
    .sort((a, b) => b.time - a.time)
  if (files.length > 0) {
    const imagePath = path.join(outputDir, files[0].file)
    res.setHeader('Content-Type', 'image/jpeg')
    fs.createReadStream(imagePath).pipe(res)
  } else {
    res.status(404).json({ error: 'Image not found' })
  }
})

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`)
})
