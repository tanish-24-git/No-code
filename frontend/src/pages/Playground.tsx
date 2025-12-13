import { useState } from 'react'
import { Search, Download, Play, Upload, Settings } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { useSSE } from '@/hooks/useSSE'
import { formatBytes, formatTimestamp } from '@/utils/cn'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

interface Model {
  id: string
  name: string
  author: string
  downloads: number
  tags: string[]
  size_gb?: number
}

interface CachedModel {
  model_id: string
  s3_path: string
  size_gb: number
  files_count: number
}

export default function Playground() {
  const [activeTab, setActiveTab] = useState('model')
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<Model[]>([])
  const [cachedModels, setCachedModels] = useState<CachedModel[]>([])
  const [selectedModel, setSelectedModel] = useState<string | null>(null)
  const [isSearching, setIsSearching] = useState(false)
  const [runId, setRunId] = useState<string | null>(null)
  const [datasetFile, setDatasetFile] = useState<File | null>(null)
  
  // Training config
  const [config, setConfig] = useState({
    epochs: 3,
    batch_size: 4,
    learning_rate: 0.0002,
    lora_r: 16,
    lora_alpha: 32,
  })

  // SSE for logs
  const { logs, isConnected } = useSSE(
    runId ? `${API_URL}/api/v1/logs/${runId}/stream` : null
  )

  // Search HuggingFace models
  const handleSearch = async () => {
    if (!searchQuery) return
    
    setIsSearching(true)
    try {
      const response = await axios.get(`${API_URL}/api/v1/models/search`, {
        params: { q: searchQuery, limit: 20 }
      })
      setSearchResults(response.data.results)
    } catch (error) {
      console.error('Search failed:', error)
    } finally {
      setIsSearching(false)
    }
  }

  // Load cached models
  const loadCachedModels = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/v1/models/cached`)
      setCachedModels(response.data.models)
    } catch (error) {
      console.error('Failed to load cached models:', error)
    }
  }

  // Download model
  const handleDownload = async (modelId: string) => {
    try {
      const response = await axios.post(`${API_URL}/api/v1/models/download/${modelId}`)
      setRunId(response.data.run_id)
      setActiveTab('logs')
      
      // Reload cached models after download
      setTimeout(loadCachedModels, 2000)
    } catch (error) {
      console.error('Download failed:', error)
    }
  }

  // Upload dataset
  const handleDatasetUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setDatasetFile(file)
    }
  }

  // Start training
  const handleRun = async () => {
    if (!selectedModel || !datasetFile) {
      alert('Please select a model and upload a dataset')
      return
    }

    try {
      const formData = new FormData()
      formData.append('dataset', datasetFile)
      formData.append('model_id', selectedModel)
      formData.append('config', JSON.stringify(config))

      const response = await axios.post(`${API_URL}/api/v1/jobs`, formData)
      setRunId(response.data.run_id)
      setActiveTab('logs')
    } catch (error) {
      console.error('Training failed:', error)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#0a0f1e] to-[#1e293b] p-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">LLM Fine-Tuning Playground</h1>
          <p className="text-muted-foreground">Train and fine-tune language models with ease</p>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="model">
              <Search className="w-4 h-4 mr-2" />
              Model Selection
            </TabsTrigger>
            <TabsTrigger value="dataset">
              <Upload className="w-4 h-4 mr-2" />
              Dataset
            </TabsTrigger>
            <TabsTrigger value="config">
              <Settings className="w-4 h-4 mr-2" />
              Configuration
            </TabsTrigger>
            <TabsTrigger value="logs">
              <Play className="w-4 h-4 mr-2" />
              Logs
              {isConnected && <Badge variant="success" className="ml-2">Live</Badge>}
            </TabsTrigger>
            <TabsTrigger value="run" disabled={!selectedModel || !datasetFile}>
              <Play className="w-4 h-4 mr-2" />
              Run
            </TabsTrigger>
          </TabsList>

          {/* Model Selection Tab */}
          <TabsContent value="model" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Search HuggingFace Models</CardTitle>
                <CardDescription>Find and download models from HuggingFace Hub</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex gap-2 mb-4">
                  <Input
                    placeholder="Search models (e.g., Qwen/Qwen2-0.5B-Instruct)"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                  />
                  <Button onClick={handleSearch} disabled={isSearching}>
                    <Search className="w-4 h-4 mr-2" />
                    Search
                  </Button>
                </div>

                {searchResults.length > 0 && (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Model</TableHead>
                        <TableHead>Downloads</TableHead>
                        <TableHead>Tags</TableHead>
                        <TableHead>Action</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {searchResults.map((model) => (
                        <TableRow key={model.id}>
                          <TableCell>
                            <div>
                              <div className="font-medium">{model.name}</div>
                              <div className="text-sm text-muted-foreground">{model.author}</div>
                            </div>
                          </TableCell>
                          <TableCell>{model.downloads.toLocaleString()}</TableCell>
                          <TableCell>
                            <div className="flex gap-1 flex-wrap">
                              {model.tags.slice(0, 3).map((tag) => (
                                <Badge key={tag} variant="outline">{tag}</Badge>
                              ))}
                            </div>
                          </TableCell>
                          <TableCell>
                            <Button size="sm" onClick={() => handleDownload(model.id)}>
                              <Download className="w-4 h-4 mr-1" />
                              Download
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Cached Models</CardTitle>
                <CardDescription>Models available in local storage</CardDescription>
              </CardHeader>
              <CardContent>
                <Button onClick={loadCachedModels} className="mb-4">Refresh</Button>
                
                {cachedModels.length > 0 ? (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Model ID</TableHead>
                        <TableHead>Size</TableHead>
                        <TableHead>Files</TableHead>
                        <TableHead>Action</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {cachedModels.map((model) => (
                        <TableRow key={model.model_id}>
                          <TableCell className="font-medium">{model.model_id}</TableCell>
                          <TableCell>{model.size_gb} GB</TableCell>
                          <TableCell>{model.files_count} files</TableCell>
                          <TableCell>
                            <Button
                              size="sm"
                              variant={selectedModel === model.model_id ? 'default' : 'outline'}
                              onClick={() => setSelectedModel(model.model_id)}
                            >
                              {selectedModel === model.model_id ? 'Selected' : 'Select'}
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                ) : (
                  <p className="text-muted-foreground">No cached models. Download a model first.</p>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Dataset Tab */}
          <TabsContent value="dataset">
            <Card>
              <CardHeader>
                <CardTitle>Upload Training Dataset</CardTitle>
                <CardDescription>Upload a JSONL file with training examples</CardDescription>
              </CardHeader>
              <CardContent>
                <Input
                  type="file"
                  accept=".jsonl,.json"
                  onChange={handleDatasetUpload}
                />
                {datasetFile && (
                  <div className="mt-4 p-4 glass rounded-lg">
                    <p className="font-medium">{datasetFile.name}</p>
                    <p className="text-sm text-muted-foreground">{formatBytes(datasetFile.size)}</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Configuration Tab */}
          <TabsContent value="config">
            <Card>
              <CardHeader>
                <CardTitle>Training Configuration</CardTitle>
                <CardDescription>Configure LoRA parameters and hyperparameters</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Epochs</label>
                  <Input
                    type="number"
                    value={config.epochs}
                    onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) })}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Batch Size</label>
                  <Input
                    type="number"
                    value={config.batch_size}
                    onChange={(e) => setConfig({ ...config, batch_size: parseInt(e.target.value) })}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Learning Rate</label>
                  <Input
                    type="number"
                    step="0.0001"
                    value={config.learning_rate}
                    onChange={(e) => setConfig({ ...config, learning_rate: parseFloat(e.target.value) })}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">LoRA Rank (r)</label>
                  <Input
                    type="number"
                    value={config.lora_r}
                    onChange={(e) => setConfig({ ...config, lora_r: parseInt(e.target.value) })}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">LoRA Alpha</label>
                  <Input
                    type="number"
                    value={config.lora_alpha}
                    onChange={(e) => setConfig({ ...config, lora_alpha: parseInt(e.target.value) })}
                  />
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Logs Tab */}
          <TabsContent value="logs">
            <Card>
              <CardHeader>
                <CardTitle>Real-Time Logs</CardTitle>
                <CardDescription>
                  {isConnected ? (
                    <Badge variant="success">Connected</Badge>
                  ) : (
                    <Badge variant="outline">Disconnected</Badge>
                  )}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="bg-black/40 rounded-lg p-4 h-96 overflow-y-auto font-mono text-sm">
                  {logs.length === 0 ? (
                    <p className="text-muted-foreground">No logs yet. Start a job to see logs.</p>
                  ) : (
                    logs.map((log, i) => (
                      <div key={i} className="mb-2">
                        <span className="text-muted-foreground">{formatTimestamp(log.timestamp)}</span>
                        {' '}
                        <Badge variant={
                          log.level === 'ERROR' ? 'destructive' :
                          log.level === 'WARN' ? 'warning' :
                          log.level === 'METRIC' ? 'info' : 'outline'
                        }>
                          {log.level}
                        </Badge>
                        {' '}
                        <span className="text-blue-400">[{log.agent}]</span>
                        {' '}
                        <span className="text-foreground">{log.message}</span>
                      </div>
                    ))
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Run Tab */}
          <TabsContent value="run">
            <Card>
              <CardHeader>
                <CardTitle>Ready to Train</CardTitle>
                <CardDescription>Review your configuration and start training</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="glass p-4 rounded-lg">
                  <h3 className="font-medium mb-2">Selected Model</h3>
                  <p className="text-muted-foreground">{selectedModel || 'None'}</p>
                </div>
                <div className="glass p-4 rounded-lg">
                  <h3 className="font-medium mb-2">Dataset</h3>
                  <p className="text-muted-foreground">{datasetFile?.name || 'None'}</p>
                </div>
                <div className="glass p-4 rounded-lg">
                  <h3 className="font-medium mb-2">Configuration</h3>
                  <pre className="text-sm text-muted-foreground">{JSON.stringify(config, null, 2)}</pre>
                </div>
                <Button onClick={handleRun} size="lg" className="w-full">
                  <Play className="w-5 h-5 mr-2" />
                  Start Training
                </Button>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
