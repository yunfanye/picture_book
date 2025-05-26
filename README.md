# Picture Book Generator for Children

This script generates complete picture books for children using AI. It combines Claude 4 for story generation, Imagen 3 for image creation, OpenCV for text overlay, and creates a final PDF book.

## Features

1. **AI Story Generation**: Uses Claude 4 to create age-appropriate stories with structured output
2. **Reference Story Adaptation**: Can adapt existing stories to be age-appropriate while maintaining core themes
3. **Character Consistency**: Maintains consistent character appearances across all illustrations
4. **Image Generation**: Uses Google's Imagen 3 to create beautiful illustrations
5. **Text Overlay**: Uses OpenCV to add story text and dialog to images
6. **PDF Creation**: Compiles everything into a professional PDF book

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Make sure your `.env` file contains the following API keys:

```env
ANTHROPIC_API_KEY=your_claude_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional, for future features
```

**Note**: The Gemini API key is used for Imagen 3 image generation. You need to have access to the paid tier to use Imagen 3.

### 3. Run the Script

```bash
# Generate an original story
python create.py <age>

# Generate a story based on a reference story
python create.py <age> --reference "Your reference story text here"

# Generate a story from a reference story file
python create.py <age> --reference-file path/to/story.txt
```

Where `<age>` is the target age for the child (1-12 years old).

## Usage Examples

```bash
# Generate a picture book for a 5-year-old
python create.py 5

# Generate a picture book for a 3-year-old based on a classic fairy tale
python create.py 3 --reference "Once upon a time, there was a little girl named Goldilocks..."

# Generate a picture book for an 8-year-old from a story file
python create.py 8 --reference-file stories/adventure.txt

# Generate using short reference flag
python create.py 6 -r "A brave little mouse goes on an adventure"
python create.py 4 -f fairy_tales/cinderella.txt
```

## Character Consistency

The script now ensures character consistency across all illustrations by:

1. **Character Definition**: Claude 4 creates detailed physical descriptions for each character
2. **Consistent Descriptions**: Each character's appearance (hair color, eye color, clothing, build, etc.) is defined once and maintained throughout
3. **Image Generation**: Imagen 3 receives character descriptions with each image prompt to ensure consistency
4. **Character Tracking**: Each page specifies which characters appear, ensuring proper consistency prompts

## Output

The script will create:
- `output/images/` - Individual page images
- `output/<title>_picture_book.pdf` - Final PDF book

## How It Works

1. **Story Generation**: Claude 4 creates a complete story structure with 20-30 pages, including:
   - Age-appropriate title and content
   - Detailed character descriptions for consistency
   - Story text for each page
   - Character dialog
   - Detailed image descriptions
   - Character tracking per page

2. **Image Creation**: For each page, the script generates images using Google's Imagen 3 model with:
   - Scene descriptions
   - Character consistency prompts
   - Established character appearances

3. **Text Overlay**: OpenCV adds the story text and dialog to each image with proper formatting

4. **PDF Compilation**: All pages are compiled into a professional PDF book

## Reference Story Adaptation

When using a reference story, the script will:
- Adapt the story to be age-appropriate for the target age
- Maintain core narrative structure and themes
- Create consistent character descriptions
- Generate new illustrations that match the adapted story
- Ensure proper pacing and vocabulary for the target age

## Customization

You can modify the script to:
- Change the number of pages (currently 20-30)
- Adjust text formatting and positioning
- Modify image generation prompts
- Change PDF layout and styling
- Adjust image aspect ratios and safety settings
- Customize character description requirements

## Image Generation

The script uses Google's Imagen 3 model for high-quality image generation. Features include:
- 4:3 aspect ratio optimized for picture books
- Child-friendly safety filters
- Bright, vibrant colors suitable for children
- Cartoon/animated style illustrations
- Character consistency across all pages
- Detailed character appearance prompts

If image generation fails (e.g., due to API limits or safety filters), the script automatically creates placeholder images with the story descriptions and character information.

## Requirements

- Python 3.8+
- Valid Anthropic API key (for Claude)
- Valid Google AI API key (for Imagen 3)
- Access to Google AI's paid tier (required for Imagen 3)

## Troubleshooting

- **API Key Issues**: Make sure your `.env` file is properly configured
- **Missing Dependencies**: Run `pip install -r requirements.txt`
- **Age Range**: Use ages between 1-12 years old
- **Output Directory**: The script automatically creates `output/` and `output/images/` directories
- **Image Generation Errors**: If Imagen 3 fails, the script will create placeholder images and continue
- **Paid Tier Required**: Imagen 3 requires access to Google AI's paid tier
- **Reference Story Format**: Reference stories can be plain text, the script will adapt them appropriately
- **Character Consistency**: If characters appear inconsistent, check that the character descriptions are detailed enough

## Example Output Structure

```
output/
├── images/
│   ├── page_01.png
│   ├── page_02.png
│   ├── ...
│   ├── final_page_01.png
│   ├── final_page_02.png
│   └── ...
└── the_magical_garden_picture_book.pdf
```

## API Costs

- **Claude API**: Approximately $0.01-0.05 per book (depending on length)
- **Imagen 3**: Approximately $0.04 per image (20-30 images per book = $0.80-1.20)
- **Total estimated cost**: $0.85-1.25 per picture book

## Future Enhancements

- Support for different art styles
- Interactive elements
- Audio narration
- Custom character creation
- Multi-language support
- Batch generation for multiple ages
- Custom story themes and topics
- Character relationship mapping
- Advanced character consistency validation # picture_book
