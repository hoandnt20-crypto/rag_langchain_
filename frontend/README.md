# CLIP Image Retrieval UI

Modern, beautiful web interface for searching through video keyframes using CLIP model.

## Features

✨ **Dual Search Modes**
- 🔤 Text Search: Search using natural language descriptions
- 🖼️ Image Search: Upload an image to find similar keyframes

🎨 **Modern Design**
- Dark mode with glassmorphism effects
- Smooth animations and transitions
- Responsive layout for all devices
- Gradient accents and premium aesthetics

⚡ **User Experience**
- Drag-and-drop image upload
- Real-time search feedback
- Organized results by video
- Similarity scores for each keyframe
- Click to view full-size images

## Quick Start

### 1. Start the Backend

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 2. Open the Frontend

Simply open `frontend/index.html` in your web browser, or use a local server:

```bash
cd frontend
python -m http.server 3000
```

Then navigate to `http://localhost:3000`

## Usage

### Text Search
1. Click on the "📝 Text Search" tab
2. Enter a description of what you're looking for (e.g., "a person walking in the park")
3. Click "Search" or press Enter
4. View results organized by video with similarity scores

### Image Search
1. Click on the "🖼️ Image Search" tab
2. Click the upload area or drag and drop an image
3. Preview your uploaded image
4. Click "Search with Image"
5. View similar keyframes from the dataset

## API Endpoints

The frontend connects to these backend endpoints:

- `GET /health` - Check API status
- `POST /search/text` - Search by text query
- `POST /search/image` - Search by uploaded image
- `GET /static/*` - Serve keyframe images

## Configuration

To change the API URL, edit `frontend/app.js`:

```javascript
const API_BASE_URL = 'http://localhost:8000';
```

## Browser Compatibility

Works best on modern browsers:
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## Troubleshooting

**Images not loading?**
- Ensure the backend is running and serving static files
- Check that the `data/keyframes` directory exists
- Verify CORS is properly configured

**Search not working?**
- Check browser console for errors
- Verify the backend API is accessible
- Ensure the CLIP model is loaded

**Slow performance?**
- Large result sets may take time to render
- Consider reducing `TOP_K` in backend config
- Use a modern browser with hardware acceleration

## Tech Stack

- **Frontend**: Vanilla HTML, CSS, JavaScript
- **Styling**: Custom CSS with modern design patterns
- **Backend**: FastAPI with CLIP model
- **Fonts**: Google Fonts (Inter)
