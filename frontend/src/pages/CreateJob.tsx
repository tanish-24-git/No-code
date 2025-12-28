import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ArrowRight, ArrowLeft, Check, Sparkles, Upload, Target, 
  Settings, Play, BarChart3, Download, Loader2 
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import { projectsAPI, datasetsAPI, tasksAPI, aiAPI, jobsAPI } from '../utils/api';

type WizardStep = 'project' | 'dataset' | 'task' | 'training' | 'advanced' | 'review';

export default function CreateJob() {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState<WizardStep>('project');
  const [loading, setLoading] = useState(false);
  
  // Form state (22 fields total)
  const [formData, setFormData] = useState({
    // Project (3 fields)
    project_name: '',
    description: '',
    tags: [] as string[],
    
    // Dataset (5 fields)
    dataset_file: null as File | null,
    dataset_name: '',
    target_column: '',
    input_columns: [] as string[],
    split_ratio: 0.8,
    
    // Task (4 fields)
    task_type: 'chat' as any,
    output_type: 'text' as any,
    domain: 'general' as any,
    language: 'en',
    
    // Training (7 fields)
    training_mode: 'balanced' as any,
    base_model: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    epochs: 3,
    batch_size: 4,
    learning_rate: 0.0002,
    max_seq_len: 2048,
    lora_rank: 16,
    
    // Advanced (6 fields)
    gradient_accumulation: 1,
    precision: 'bf16' as any,
    early_stopping: true,
    class_balancing: false,
    data_augmentation: false,
    resume_checkpoint: '',
  });
  
  const [projectId, setProjectId] = useState<string>('');
  const [datasetId, setDatasetId] = useState<string>('');
  const [aiSuggestions, setAiSuggestions] = useState<any>(null);

  const steps: { key: WizardStep; title: string; icon: any }[] = [
    { key: 'project', title: 'Project', icon: Target },
    { key: 'dataset', title: 'Dataset', icon: Upload },
    { key: 'task', title: 'Task Type', icon: Sparkles },
    { key: 'training', title: 'Training', icon: Settings },
    { key: 'advanced', title: 'Advanced', icon: Settings },
    { key: 'review', title: 'Review', icon: Check },
  ];

  const currentStepIndex = steps.findIndex(s => s.key === currentStep);

  const handleNext = async () => {
    if (currentStep === 'project') {
      await createProject();
    } else if (currentStep === 'dataset') {
      await uploadDataset();
    } else if (currentStep === 'task') {
      await getSuggestions();
    } else if (currentStepIndex < steps.length - 1) {
      setCurrentStep(steps[currentStepIndex + 1].key);
    } else {
      await submitJob();
    }
  };

  const handleBack = () => {
    if (currentStepIndex > 0) {
      setCurrentStep(steps[currentStepIndex - 1].key);
    }
  };

  const createProject = async () => {
    if (!formData.project_name) {
      toast.error('Project name required');
      return;
    }
    
    setLoading(true);
    try {
      const response = await projectsAPI.create({
        project_name: formData.project_name,
        description: formData.description,
        tags: formData.tags,
      });
      setProjectId(response.data.project_id);
      toast.success('Project created!');
      setCurrentStep('dataset');
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to create project');
    } finally {
      setLoading(false);
    }
  };

  const uploadDataset = async () => {
    if (!formData.dataset_file) {
      toast.error('Please select a dataset file');
      return;
    }
    
    setLoading(true);
    try {
      const response = await datasetsAPI.upload(
        formData.dataset_file,
        formData.dataset_name || formData.dataset_file.name
      );
      setDatasetId(response.data.dataset_id);
      toast.success('Dataset uploaded!');
      setCurrentStep('task');
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to upload dataset');
    } finally {
      setLoading(false);
    }
  };

  const getSuggestions = async () => {
    setLoading(true);
    try {
      // Get AI suggestions for training config
      const response = await aiAPI.suggestConfig({
        dataset_stats: { rows: 10000 }, // TODO: Get from dataset upload response
        task_type: formData.task_type,
        gpu_count: 0,
      });
      
      setAiSuggestions(response.data);
      
      // Pre-fill training form
      setFormData(prev => ({
        ...prev,
        epochs: response.data.epochs,
        batch_size: response.data.batch_size,
        lora_rank: response.data.lora_rank,
        learning_rate: response.data.learning_rate,
      }));
      
      toast.success('AI configured optimal settings!');
      setCurrentStep('training');
    } catch (error: any) {
      toast.error('AI suggestion failed, using defaults');
      setCurrentStep('training');
    } finally {
      setLoading(false);
    }
  };

  const submitJob = async () => {
    setLoading(true);
    try {
      const jobConfig = {
        dataset_id: datasetId,
        task: {
          task_type: formData.task_type,
          output_type: formData.output_type,
          domain: formData.domain,
          language: formData.language,
        },
        training: {
          training_mode: formData.training_mode,
          base_model: formData.base_model,
          epochs: formData.epochs,
          batch_size: formData.batch_size,
          learning_rate: formData.learning_rate,
          max_seq_len: formData.max_seq_len,
          lora_rank: formData.lora_rank,
        },
        advanced: {
          gradient_accumulation: formData.gradient_accumulation,
          precision: formData.precision,
          early_stopping: formData.early_stopping,
          class_balancing: formData.class_balancing,
          data_augmentation: formData.data_augmentation,
          resume_checkpoint: formData.resume_checkpoint || undefined,
        },
      };

      const response = await jobsAPI.submit({
        pipeline_config: jobConfig
      });
      
      toast.success('Job submitted!');
      navigate(`/jobs/${response.data.job_id}`);
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Job submission failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-primary/10 p-6">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 text-center"
        >
          <h1 className="text-4xl font-bold gradient-text mb-2">Create Training Job</h1>
          <p className="text-muted-foreground">8-step wizard with AI assistance</p>
        </motion.div>

        {/* Step Progress */}
        <div className="flex items-center justify-between mb-12">
          {steps.map((step, idx) => {
            const Icon = step.icon;
            const isActive = idx === currentStepIndex;
            const isComplete = idx < currentStepIndex;
            
            return (
              <div key={step.key} className="flex items-center flex-1">
                <div className="flex flex-col items-center flex-1">
                  <div className={`
                    w-12 h-12 rounded-full flex items-center justify-center transition-all
                    ${isActive ? 'bg-primary text-primary-foreground glow-primary scale-110' : ''}
                    ${isComplete ? 'bg-accent text-accent-foreground' : ''}
                    ${!isActive && !isComplete ? 'bg-secondary text-muted-foreground' : ''}
                  `}>
                    {isComplete ? <Check className="w-6 h-6" /> : <Icon className="w-6 h-6" />}
                  </div>
                  <span className={`text-xs mt-2 ${isActive ? 'font-semibold' : ''}`}>
                    {step.title}
                  </span>
                </div>
                {idx < steps.length - 1 && (
                  <div className={`h-1 flex-1 mx-2 ${isComplete ? 'bg-accent' : 'bg-secondary'}`} />
                )}
              </div>
            );
          })}
        </div>

        {/* Form Content */}
        <motion.div
          key={currentStep}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          className="glass rounded-2xl p-8 mb-6"
        >
          <AnimatePresence mode="wait">
            {currentStep === 'project' && <ProjectForm formData={formData} setFormData={setFormData} />}
            {currentStep === 'dataset' && <DatasetForm formData={formData} setFormData={setFormData} />}
            {currentStep === 'task' && <TaskForm formData={formData} setFormData={setFormData} />}
            {currentStep === 'training' && <TrainingForm formData={formData} setFormData={setFormData} aiSuggestions={aiSuggestions} />}
            {currentStep === 'advanced' && <AdvancedForm formData={formData} setFormData={setFormData} />}
            {currentStep === 'review' && <ReviewForm formData={formData} />}
          </AnimatePresence>
        </motion.div>

        {/* Navigation */}
        <div className="flex gap-4">
          {currentStepIndex > 0 && (
            <button
              onClick={handleBack}
              className="flex-1 bg-secondary text-foreground px-6 py-3 rounded-xl hover:bg-secondary/80 transition-all flex items-center justify-center gap-2"
            >
              <ArrowLeft className="w-5 h-5" />
              Back
            </button>
          )}
          
          <button
            onClick={handleNext}
            disabled={loading}
            className="flex-1 bg-primary text-primary-foreground px-6 py-3 rounded-xl hover:opacity-90 disabled:opacity-50 transition-all glow-primary flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Processing...
              </>
            ) : currentStepIndex === steps.length - 1 ? (
              <>
                <Play className="w-5 h-5" />
                Start Training
              </>
            ) : (
              <>
                Next
                <ArrowRight className="w-5 h-5" />
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

// Individual form components (simplified - can expand)
function ProjectForm({ formData, setFormData }: any) {
  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Project Setup</h2>
      <div>
        <label className="block text-sm font-medium mb-2">Project Name *</label>
        <input
          type="text"
          value={formData.project_name}
          onChange={(e) => setFormData({ ...formData, project_name: e.target.value })}
          placeholder="CustomerSupportBot"
          className="w-full bg-secondary border border-border rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-primary"
        />
      </div>
      <div>
        <label className="block text-sm font-medium mb-2">Description</label>
        <textarea
          value={formData.description}
          onChange={(e) => setFormData({ ...formData, description: e.target.value })}
          placeholder="24/7 AI support agent..."
          rows={3}
          className="w-full bg-secondary border border-border rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-primary"
        />
      </div>
    </div>
  );
}

function DatasetForm({ formData, setFormData }: any) {
  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Dataset Upload</h2>
      <div>
        <label className="block text-sm font-medium mb-2">CSV/JSON/Parquet File *</label>
        <input
          type="file"
          accept=".csv,.json,.jsonl,.parquet"
          onChange={(e) => setFormData({ ...formData, dataset_file: e.target.files?.[0] || null })}
          className="w-full bg-secondary border border-border rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-primary"
        />
      </div>
    </div>
  );
}

function TaskForm({ formData, setFormData }: any) {
  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Task Type</h2>
      <div>
        <label className="block text-sm font-medium mb-2">Task Type *</label>
        <select
          value={formData.task_type}
          onChange={(e) => setFormData({ ...formData, task_type: e.target.value })}
          className="w-full bg-secondary border border-border rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-primary"
        >
          <option value="chat">Chat</option>
          <option value="classification">Classification</option>
          <option value="qa">Question Answering</option>
          <option value="summarization">Summarization</option>
        </select>
      </div>
    </div>
  );
}

function TrainingForm({ formData, setFormData, aiSuggestions }: any) {
  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Training Configuration</h2>
      {aiSuggestions && (
        <div className="bg-accent/10 border border-accent rounded-xl p-4">
          <p className="text-sm text-accent flex items-center gap-2">
            <Sparkles className="w-4 h-4" />
            AI Suggestion: {aiSuggestions.reasoning}
          </p>
        </div>
      )}
      
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium mb-2">Epochs</label>
          <input
            type="number"
            value={formData.epochs}
            onChange={(e) => setFormData({ ...formData, epochs: parseInt(e.target.value) })}
            min={1}
            max={10}
            className="w-full bg-secondary border border-border rounded-xl px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-2">Batch Size</label>
          <input
            type="number"
            value={formData.batch_size}
            onChange={(e) => setFormData({ ...formData, batch_size: parseInt(e.target.value) })}
            min={1}
            max={16}
            className="w-full bg-secondary border border-border rounded-xl px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary"
          />
        </div>
      </div>
    </div>
  );
}

function AdvancedForm({ formData, setFormData }: any) {
  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Advanced Settings</h2>
      <div className="space-y-4">
        <label className="flex items-center gap-3">
          <input
            type="checkbox"
            checked={formData.early_stopping}
            onChange={(e) => setFormData({ ...formData, early_stopping: e.target.checked })}
            className="w-5 h-5"
          />
          <span>Early Stopping</span>
        </label>
      </div>
    </div>
  );
}

function ReviewForm({ formData }: any) {
  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Review & Submit</h2>
      <div className="space-y-4">
        <div className="bg-secondary rounded-xl p-4">
          <h3 className="font-semibold mb-2">Project</h3>
          <p>{formData.project_name}</p>
        </div>
        <div className="bg-secondary rounded-xl p-4">
          <h3 className="font-semibold mb-2">Training</h3>
          <p>Epochs: {formData.epochs}, Batch Size: {formData.batch_size}</p>
        </div>
      </div>
    </div>
  );
}
