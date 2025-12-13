# LLM Platform Frontend

Modern, production-ready frontend for the LLM Fine-Tuning Platform.

## Features

- **Unified Playground**: Model search, dataset upload, training configuration, and real-time logs in one page
- **Model Comparison**: Side-by-side comparison of base vs fine-tuned models
- **Real-time Logs**: SSE-based log streaming with auto-reconnect
- **Modern UI**: Dark theme with glassmorphism effects using shadcn/ui
- **Responsive**: Mobile-friendly design

## Tech Stack

- React 18 + TypeScript
- Vite (build tool)
- Tailwind CSS + shadcn/ui
- TanStack Query (data fetching)
- React Router (routing)
- Axios (HTTP client)

## Development

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build
```

## Docker

```bash
# Build image
docker build -t llm-platform-frontend .

# Run container
docker run -p 3000:3000 llm-platform-frontend
```

## Environment Variables

- `VITE_API_URL`: Backend API URL (default: http://localhost:8000)
