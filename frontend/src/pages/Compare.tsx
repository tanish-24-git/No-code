import { useState } from 'react'
import { Send, Zap } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

interface ModelResponse {
  text: string
  tokens: number
  latency_ms: number
}

export default function Compare() {
  const [baseModel, setBaseModel] = useState('Qwen/Qwen2-0.5B-Instruct')
  const [finetunedModel, setFinetunedModel] = useState('')
  const [prompt, setPrompt] = useState('')
  const [baseResponse, setBaseResponse] = useState<ModelResponse | null>(null)
  const [finetunedResponse, setFinetunedResponse] = useState<ModelResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const handleCompare = async () => {
    if (!prompt || !baseModel || !finetunedModel) {
      alert('Please fill in all fields')
      return
    }

    setIsLoading(true)
    try {
      // Call inference API for both models (parallel)
      const [baseResult, finetunedResult] = await Promise.all([
        axios.post(`${API_URL}/api/v1/inference`, {
          model_id: baseModel,
          prompt,
        }),
        axios.post(`${API_URL}/api/v1/inference`, {
          model_id: finetunedModel,
          prompt,
        }),
      ])

      setBaseResponse(baseResult.data)
      setFinetunedResponse(finetunedResult.data)
    } catch (error) {
      console.error('Inference failed:', error)
      alert('Inference failed. Make sure the inference API is implemented.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#0a0f1e] to-[#1e293b] p-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">Model Comparison</h1>
          <p className="text-muted-foreground">Compare base model vs fine-tuned model responses</p>
        </div>

        {/* Model Selection */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Select Models</CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">Base Model</label>
              <Input
                placeholder="e.g., Qwen/Qwen2-0.5B-Instruct"
                value={baseModel}
                onChange={(e) => setBaseModel(e.target.value)}
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Fine-tuned Model</label>
              <Input
                placeholder="e.g., run_abc123"
                value={finetunedModel}
                onChange={(e) => setFinetunedModel(e.target.value)}
              />
            </div>
          </CardContent>
        </Card>

        {/* Prompt Input */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Test Prompt</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex gap-2">
              <Input
                placeholder="Enter your prompt here..."
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleCompare()}
                className="flex-1"
              />
              <Button onClick={handleCompare} disabled={isLoading}>
                <Send className="w-4 h-4 mr-2" />
                {isLoading ? 'Generating...' : 'Compare'}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Side-by-Side Comparison */}
        <div className="grid grid-cols-2 gap-6">
          {/* Base Model Response */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Base Model</CardTitle>
                <Badge variant="outline">{baseModel}</Badge>
              </div>
            </CardHeader>
            <CardContent>
              {baseResponse ? (
                <div className="space-y-4">
                  <div className="bg-black/40 rounded-lg p-4 min-h-[200px]">
                    <p className="text-foreground whitespace-pre-wrap">{baseResponse.text}</p>
                  </div>
                  <div className="flex gap-4 text-sm text-muted-foreground">
                    <div className="flex items-center gap-1">
                      <Zap className="w-4 h-4" />
                      <span>{baseResponse.tokens} tokens</span>
                    </div>
                    <div>
                      <span>{baseResponse.latency_ms}ms</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="bg-black/40 rounded-lg p-4 min-h-[200px] flex items-center justify-center">
                  <p className="text-muted-foreground">No response yet</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Fine-tuned Model Response */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Fine-tuned Model</CardTitle>
                <Badge variant="success">{finetunedModel || 'Not selected'}</Badge>
              </div>
            </CardHeader>
            <CardContent>
              {finetunedResponse ? (
                <div className="space-y-4">
                  <div className="bg-black/40 rounded-lg p-4 min-h-[200px]">
                    <p className="text-foreground whitespace-pre-wrap">{finetunedResponse.text}</p>
                  </div>
                  <div className="flex gap-4 text-sm text-muted-foreground">
                    <div className="flex items-center gap-1">
                      <Zap className="w-4 h-4" />
                      <span>{finetunedResponse.tokens} tokens</span>
                    </div>
                    <div>
                      <span>{finetunedResponse.latency_ms}ms</span>
                    </div>
                  </div>
                  
                  {/* Performance Comparison */}
                  {baseResponse && (
                    <div className="glass p-4 rounded-lg">
                      <h4 className="font-medium mb-2">Performance Delta</h4>
                      <div className="space-y-1 text-sm">
                        <div>
                          Tokens: {finetunedResponse.tokens - baseResponse.tokens > 0 ? '+' : ''}
                          {finetunedResponse.tokens - baseResponse.tokens}
                        </div>
                        <div>
                          Latency: {finetunedResponse.latency_ms - baseResponse.latency_ms > 0 ? '+' : ''}
                          {finetunedResponse.latency_ms - baseResponse.latency_ms}ms
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="bg-black/40 rounded-lg p-4 min-h-[200px] flex items-center justify-center">
                  <p className="text-muted-foreground">No response yet</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
