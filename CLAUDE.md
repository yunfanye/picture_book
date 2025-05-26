# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based picture book generator that creates complete children's picture books using AI. The system combines Claude 4 for story generation, Imagen 3 or Gemini Flash for image generation, and automated text overlay to produce PDFs.

## Core Architecture

### Main Components

- **`create.py`**: Main script containing `PictureBookGenerator` class - handles story generation, image processing pipeline, and PDF compilation
- **`image_generator.py`**: Modular `ImageGenerator` class with support for multiple backends (Imagen 3 and Gemini Flash)

### Workflow Pipeline

1. **Story Generation**: Claude 4 creates structured JSON with character descriptions and page-by-page content
2. **Parallel Image Generation**: ThreadPoolExecutor generates images with character consistency validation
3. **Image Scoring & Validation**: Gemini 2.5 Pro scores images (0-100) based on description alignment and character accuracy, with heavy penalties for malformed text
4. **Score-Based Selection**: System only replaces images if new score is higher than previous best
5. **Smart Text Placement**: Gemini identifies optimal background areas (sky, grass, walls) for text placement to avoid covering important visual elements
6. **Text Overlay**: OpenCV adds story text and dialog with semi-transparent backgrounds for better readability
7. **PDF Compilation**: ReportLab creates final picture book

### Image Generation Backends

- **Imagen 3** (`imagen`): High-quality artistic illustrations, optimized for production books
- **Gemini Flash** (`gemini`): Fast multimodal generation, good for testing and iteration

### Picture Styles

- **`minimalist`**: Simple lines, limited color palettes (2-4 colors), clean uncluttered compositions with emphasis on negative space
- **`watercolor`**: Hand-drawn style with soft flowing effects, natural bleeding, gentle brush strokes, whimsical dreamy atmosphere
- **`digital`**: Modern vibrant illustrations with saturated colors, smooth gradients, contemporary children's book aesthetic (default)
- **`collage`**: Mixed media style combining textures, papers, materials with layered compositions and handcrafted appearance
- **`comic`**: Bold graphic novel style with confident line work, dynamic compositions, punchy colors, expressive character poses

## Common Commands

### Basic Usage
```bash
# Generate picture book for 5-year-old with Imagen 3 (default)
python create.py 5

# Use Gemini Flash backend
python create.py 5 --image-backend gemini

# Use reference story for adaptation
python create.py 5 --reference-file "story.txt"

# Use different picture styles
python create.py 5 --picture-style watercolor
python create.py 5 --picture-style minimalist
```

### Development Commands
```bash
# Install dependencies
pip install -r requirements.txt

# List available story structures
python create.py --list-stories

# Use pre-generated story structure
python create.py 5 --story-structure output/story_structure_*.json

# Adjust performance and reliability
python create.py 5 --max-workers 2 --generation-retries 5 --validation-retries 3

# Advanced combination example
python create.py 5 --image-backend gemini --picture-style watercolor --max-workers 2
```

### Environment Setup
Required `.env` file:
```
ANTHROPIC_API_KEY=your_claude_api_key
GEMINI_API_KEY=your_gemini_api_key
```

## Key Design Patterns

### Character Consistency System
- Characters defined once in story structure JSON with detailed physical descriptions
- Each page references established character appearances
- Image generation prompts include character consistency requirements
- Validation ensures character appearance matches across pages

### Score-Based Image Selection
- Gemini 2.5 Pro scores images 0-100 based on alignment with description and character accuracy
- Heavy penalties (-50 points) only for BIG CHUNKS of malformed text (multiple words/sentences), not minor text artifacts
- System only replaces images if new score exceeds previous best score
- Scoring criteria: character accuracy (30pts), scene alignment (30pts), child-appropriateness (15pts), visual quality (15pts), style consistency (10pts)

### Error Handling & Retry Logic
- Multi-layered retry system: generation failures, validation failures, score-based improvements
- Graceful fallbacks with placeholder images
- Score-based selection ensures best available image is always used

### Parallel Processing
- ThreadPoolExecutor for concurrent image generation
- Configurable worker threads (default: 4, recommended: 2 for rate limits)
- Results collected and ordered by page number

## File Structure & Output

- **`output/`**: Generated content directory
  - **`images/`**: Individual page images (page_XX.png, final_page_XX.png)
  - **`story_structure_*.json`**: Reusable story structures with timestamps
  - **`*_picture_book.pdf`**: Final compiled picture books

## Configuration Guidelines

### Performance Tuning
- Use `--max-workers 2` for better rate limit management with Google APIs
- Increase retry counts for challenging prompts or unreliable network conditions
- Imagen 3 for final production, Gemini Flash for rapid iteration

### Story Structure Workflow
- Generate story structures once, reuse with different backends
- Save successful structures for consistent character regeneration
- Pre-generated structures enable A/B testing different image backends

## API Integration Notes

- **Claude 4**: Story generation with structured JSON output, character consistency prompts
- **Imagen 3**: Specialized for artistic illustrations, safety-filtered for children
- **Gemini 2.5 Pro**: Image validation and text placement analysis
- **Gemini Flash**: Fast image generation with text+image capabilities

## Memories
- Added support for memorizing project-specific context and learning with `memorize` command