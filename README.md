# Picture Book Generator

A Python script that generates complete picture books for children using AI.

## Features

- **Story Generation**: Uses Claude 4 for creating age-appropriate stories with consistent characters
- **Image Generation**: Supports two backends:
  - **Imagen 3**: Google's Imagen 3 model for high-quality illustrations
  - **Gemini Flash**: Google's Gemini 2.0 Flash with image generation capabilities
- **Image Validation**: Uses Gemini 2.5 Pro to validate generated images and determine optimal text placement
- **Text Overlay**: Automatically adds story text and dialog to images with smart positioning
- **PDF Generation**: Compiles everything into a professional picture book PDF

## Installation

1. Install required dependencies:
```bash
pip install anthropic google-genai opencv-python pillow reportlab python-dotenv
```

2. Set up environment variables in a `.env` file:
```
ANTHROPIC_API_KEY=your_claude_api_key
GEMINI_API_KEY=your_gemini_api_key
```

## Usage

### Basic Usage

Generate a picture book for a 5-year-old:
```bash
python create.py 5
```

### Image Generation Backends

Choose between Imagen 3 (default) and Gemini Flash:

```bash
# Use Imagen 3 (default)
python create.py 5 --image-backend imagen

# Use Gemini Flash
python create.py 5 --image-backend gemini
```

### Advanced Options

```bash
python create.py 5 \
  --image-backend gemini \
  --max-workers 2 \
  --generation-retries 5 \
  --validation-retries 3 \
  --reference-file "my_story.txt"
```

### Command Line Arguments

- `age`: Target age for the child (1-12)
- `--image-backend, -i`: Choose image generation backend (`imagen` or `gemini`)
- `--reference, -r`: Reference story text for adaptation
- `--reference-file, -f`: Path to file containing reference story
- `--story-structure, -s`: Path to pre-generated story structure JSON
- `--list-stories, -l`: List available story structure files
- `--max-workers, -m`: Number of parallel image generation threads (default: 4)
- `--generation-retries, -g`: Max retries for image generation (default: 3)
- `--validation-retries, -v`: Max retries for validation failures (default: 2)

### Working with Story Structures

You can save and reuse story structures:

1. Generate a story structure (automatically saved):
```bash
python create.py 5
```

2. List available story structures:
```bash
python create.py --list-stories
```

3. Use a pre-generated story structure:
```bash
python create.py 5 --story-structure output/story_structure_adventure_20241201_143022.json
```

## Architecture

### Core Components

1. **`create.py`**: Main script with `PictureBookGenerator` class
2. **`image_generator.py`**: Modular image generation with `ImageGenerator` class

### Image Generation Backends

#### Imagen 3
- High-quality, specialized image generation model
- Optimized for artistic illustrations
- 4:3 aspect ratio suitable for picture books
- Safety filters for child-appropriate content

#### Gemini Flash
- Fast, multimodal model with image generation
- Text and image generation in single API call
- Good for rapid prototyping and testing
- Newer model with evolving capabilities

### Workflow

1. **Story Generation**: Claude 4 creates structured story with character descriptions
2. **Image Generation**: Selected backend generates illustrations with character consistency
3. **Image Validation**: Gemini 2.5 Pro validates images and determines text placement
4. **Text Overlay**: OpenCV adds story text and dialog with optimal positioning
5. **PDF Compilation**: ReportLab creates final picture book PDF

## Output

The script generates:
- Individual page images in `output/images/`
- Story structure JSON in `output/`
- Final PDF picture book in `output/`

## Error Handling

- Automatic retry logic for failed image generation
- Placeholder images for persistent failures
- Graceful fallbacks for validation errors
- Detailed logging throughout the process

## Tips

- **Imagen 3**: Better for final production-quality books
- **Gemini Flash**: Faster for testing and iteration
- Use `--max-workers 2` for better rate limit management
- Save story structures to reuse with different image backends
- Increase retry counts for better success rates with challenging prompts
