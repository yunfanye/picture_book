#!/usr/bin/env python3
"""
Picture Book Generator for Children

This script generates a complete picture book with:
1. Claude 4 for story generation and structured output
2. Imagen 3 for image generation
3. OpenCV for adding text to images
4. PDF compilation of the final book
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Spacer
from reportlab.lib.units import inch
import anthropic
from google import genai
from google.genai import types
from io import BytesIO
from dotenv import load_dotenv
import base64

# Load environment variables
load_dotenv()

class PictureBookGenerator:
    def __init__(self, max_workers: int = 4):
        # Initialize API clients
        self.claude_client = anthropic.Anthropic(
            api_key=os.getenv('ANTHROPIC_API_KEY')
        )
        
        # Initialize Google GenAI client for Imagen
        self.genai_client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
        
        # Configure thread pool
        self.max_workers = max_workers
        
        # Create output directories
        self.output_dir = Path("output")
        self.images_dir = self.output_dir / "images"
        self.output_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
    
    def generate_story_structure(self, age: int, reference_story: str = None, save_to_file: bool = True) -> Dict[str, Any]:
        """Generate story structure using Claude 4"""
        
        reference_prompt = ""
        if reference_story:
            reference_prompt = f"""
            
            REFERENCE STORY TO ADAPT:
            {reference_story}
            
            Please adapt this reference story to be age-appropriate for a {age}-year-old while maintaining the core narrative structure and themes.
            """
        
        prompt = f"""
        Create a children's picture book story suitable for a {age}-year-old child. 
        The book should have 20-25 pages with engaging content appropriate for this age group.
        {reference_prompt}
        
        IMPORTANT: First, define all the main characters in the story with detailed physical descriptions that will be used consistently throughout all illustrations.
        
        Please provide a structured response in JSON format with:
        1. A compelling title
        2. A brief summary
        3. A "characters" section with detailed physical descriptions for each main character
        4. For each page (20-25 pages), provide:
           - Page number
           - Story text (1-3 sentences appropriate for age {age})
           - Dialog (if any characters are speaking)
           - Detailed image description for AI image generation (MUST reference character descriptions when characters appear)
           - List of characters appearing on this page
        
        CHARACTER CONSISTENCY REQUIREMENTS:
        - Define each character's appearance once in the "characters" section
        - Include specific details: hair color, eye color, clothing style, size/build, distinctive features
        - When characters appear in page descriptions, reference their established appearance
        - Maintain these descriptions consistently across all pages
        
        Make sure the story has:
        - Age-appropriate vocabulary and concepts
        - Engaging characters with consistent appearances
        - A clear beginning, middle, and end
        - Positive messages and learning opportunities
        - Vivid, descriptive scenes for illustration
        
        Format as valid JSON with this structure:
        {{
            "title": "Book Title",
            "summary": "Brief story summary",
            "target_age": {age},
            "characters": {{
                "character_name": {{
                    "description": "Detailed physical description including hair, eyes, clothing, build, distinctive features",
                    "personality": "Brief personality traits"
                }}
            }},
            "pages": [
                {{
                    "page_number": 1,
                    "story_text": "Story text for this page",
                    "dialog": "Character dialog if any",
                    "characters_on_page": ["character1", "character2"],
                    "image_description": "Detailed description for AI image generation that references character descriptions when characters appear"
                }}
            ]
        }}
        """
        
        try:
            response = self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=20000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract JSON from response
            content = response.content[0].text
            # Find JSON in the response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            json_str = content[start_idx:end_idx]
            
            story_data = json.loads(json_str)
            
            # Save to file if requested
            if save_to_file:
                timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_title = "".join(c for c in story_data.get('title', 'story') if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_title = safe_title.replace(' ', '_').lower()
                filename = f"story_structure_{safe_title}_{timestamp}.json"
                story_file_path = self.output_dir / filename
                
                # Save the complete story structure
                with open(story_file_path, 'w', encoding='utf-8') as f:
                    json.dump(story_data, f, indent=2, ensure_ascii=False)
                
                print(f"📁 Story structure saved to: {story_file_path}")
            
            print(f"Generated story: '{story_data['title']}'")
            print(f"Number of pages: {len(story_data['pages'])}")
            print(f"Characters defined: {list(story_data.get('characters', {}).keys())}")
            
            return story_data
            
        except Exception as e:
            print(f"Error generating story: {e}")
            raise
    
    def list_story_files(self) -> List[str]:
        """List available story structure files in the output directory"""
        story_files = []
        if self.output_dir.exists():
            for file_path in self.output_dir.glob("story_structure_*.json"):
                story_files.append(str(file_path))
        return sorted(story_files)
    
    def load_story_structure(self, file_path: str) -> Dict[str, Any]:
        """Load story structure from a JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                story_data = json.load(f)
            
            print(f"📁 Loaded story structure from: {file_path}")
            print(f"Story title: '{story_data['title']}'")
            print(f"Number of pages: {len(story_data['pages'])}")
            print(f"Characters defined: {list(story_data.get('characters', {}).keys())}")
            
            return story_data
            
        except Exception as e:
            print(f"Error loading story structure: {e}")
            raise
    
    def generate_image(self, description: str, page_number: int, characters_on_page: List[str] = None, character_descriptions: Dict[str, Dict] = None, max_retries: int = 3, custom_prompt: str = None) -> str:
        """Generate image using Imagen 3 with character consistency and immediate retry on failure"""
        
        # Use custom prompt if provided, otherwise build enhanced prompt
        if custom_prompt:
            print(f"🎯 Using custom rewritten prompt for page {page_number}")
            enhanced_prompt = custom_prompt
        else:
            # Build character consistency prompt
            character_prompt = ""
            if characters_on_page and character_descriptions:
                character_prompt = "\n\nCHARACTER DESCRIPTIONS (maintain consistency):\n"
                for char_name in characters_on_page:
                    if char_name in character_descriptions:
                        char_info = character_descriptions[char_name]
                        character_prompt += f"- {char_name}: {char_info['description']}\n"
            
            # Enhanced prompt for better children's book illustrations
            enhanced_prompt = f"""
            Create a beautiful, colorful children's book illustration in a warm, friendly art style.
            The image should be suitable for a picture book with bright colors, clear details, and engaging characters.
            
            Scene description: {description}
            {character_prompt}
            
            Style requirements:
            - Bright, vibrant colors
            - Child-friendly and non-scary
            - Clear, simple composition
            - Storybook illustration style
            - High quality and detailed
            - Cartoon or animated style
            - Safe for children
            - MAINTAIN CHARACTER CONSISTENCY: If characters appear, they must match their established descriptions exactly
            """
        
        # Retry logic for image generation
        for attempt in range(max_retries):
            try:
                print(f"Generating image for page {page_number} (attempt {attempt + 1}/{max_retries})...")
                
                # Generate image using Imagen 3
                response = self.genai_client.models.generate_images(
                    model='imagen-3.0-generate-002',
                    prompt=enhanced_prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio="4:3",  # Good for picture books
                        safety_filter_level="BLOCK_ONLY_HIGH",
                        person_generation="ALLOW_ADULT"
                    )
                )
                
                # Check if images were generated
                if response.generated_images and len(response.generated_images) > 0:
                    # Save the generated image
                    image_path = self.images_dir / f"page_{page_number:02d}.png"
                    
                    # Get the first (and only) generated image
                    generated_image = response.generated_images[0]
                    
                    # Convert bytes to PIL Image and save
                    image = Image.open(BytesIO(generated_image.image.image_bytes))
                    image.save(image_path)
                    
                    print(f"✅ Successfully generated image for page {page_number}")
                    return str(image_path)
                else:
                    print(f"❌ No images generated for page {page_number} on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        print(f"🔄 Retrying immediately...")
                        continue
                    else:
                        print(f"⚠️  Failed to generate image after {max_retries} attempts, creating placeholder")
                        break
                        
            except Exception as e:
                print(f"❌ Error generating image for page {page_number} on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    print(f"🔄 Retrying immediately...")
                    continue
                else:
                    print(f"⚠️  Failed to generate image after {max_retries} attempts due to errors, creating placeholder")
                    break
        
        # Create placeholder image if all attempts failed
        print("Creating placeholder image...")
        image_path = self.images_dir / f"page_{page_number:02d}.png"
        img = Image.new('RGB', (1280, 896), color='lightblue')  # 4:3 aspect ratio
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Add character info to placeholder if available
        placeholder_text = f"PLACEHOLDER - Page {page_number}\n\n{description}"
        if characters_on_page and character_descriptions:
            placeholder_text += f"\n\nCharacters: {', '.join(characters_on_page)}"
        
        # Wrap text
        words = placeholder_text.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            line_text = ' '.join(current_line)
            bbox = draw.textbbox((0, 0), line_text, font=font)
            if bbox[2] > 1200:  # If line is too wide
                if len(current_line) > 1:
                    current_line.pop()
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)
                    current_line = []
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Draw text
        y_offset = 50
        for line in lines[:20]:  # Limit to 20 lines
            draw.text((50, y_offset), line, fill='black', font=font)
            y_offset += 35
        
        img.save(image_path)
        return str(image_path)
    
    def validate_image_and_get_dialog_placement(self, image_path: str, character_descriptions: Dict[str, Dict], 
                                               image_description: str, story_text: str, dialog: str, 
                                               characters_on_page: List[str]) -> Tuple[bool, Dict[str, Any]]:
        """
        Use Gemini 2.5 Pro to:
        1. Validate if the generated image aligns with requirements
        2. Determine optimal dialog placement coordinates and colors
        3. Provide rewritten prompts for regeneration when image is invalid
        
        Returns:
            Tuple of (is_valid, placement_info)
            placement_info contains: {
                'dialog_positions': [{'text': str, 'x': int, 'y': int, 'color': tuple}],
                'story_text_position': {'x': int, 'y': int, 'color': tuple},
                'validation_feedback': str,
                'regeneration_suggestions': str,
                'rewritten_prompt': str (when image is invalid)
            }
        """
        try:
            # Read and encode the image
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
            
            # Prepare character context
            character_context = ""
            if characters_on_page and character_descriptions:
                character_context = "\n\nExpected characters in this image:\n"
                for char_name in characters_on_page:
                    if char_name in character_descriptions:
                        char_info = character_descriptions[char_name]
                        character_context += f"- {char_name}: {char_info['description']}\n"
            
            # Create validation prompt
            validation_prompt = f"""
            Please analyze this children's book illustration and provide feedback on three tasks:

            TASK 1 - IMAGE VALIDATION:
            Check if the generated image aligns with the requirements:
            
            Expected scene: {image_description}
            {character_context}
            Story context: {story_text}
            
            Evaluate:
            - Do the characters match their described appearances?
            - Does the scene match the description?
            - Is it appropriate for children?
            - Are the colors bright and engaging?
            - Is the composition clear and not cluttered?
            
            TASK 2 - DIALOG PLACEMENT:
            Determine optimal text placement for the following content:
            
            Story text: "{story_text}"
            Dialog (if any): "{dialog}"
            
            For text placement, consider:
            - Avoid covering important visual elements
            - Ensure good readability with contrasting colors
            - Place dialog near speaking characters when possible
            - Use bottom area for story text if no better option
            - Provide RGB color values that contrast well with the background
            - Prefer white text for story text and black text for dialog
            
            TASK 3 - PROMPT REWRITING (only if image is invalid):
            If the image doesn't meet requirements, provide a rewritten prompt that addresses the specific issues found.
            The rewritten prompt should:
            - Fix character appearance inconsistencies
            - Improve scene composition
            - Enhance visual elements that were missing or incorrect
            - Maintain the core scene description while adding specific improvements
            - Include all character consistency requirements from the original prompt
            
            Please respond in JSON format:
            {{
                "is_valid": true/false,
                "validation_feedback": "Detailed feedback on what matches/doesn't match requirements",
                "story_text_position": {{
                    "x": pixel_x_coordinate,
                    "y": pixel_y_coordinate,
                    "color": [r, g, b],
                    "background_needed": true/false,
                    "background_color": [r, g, b]
                }},
                "dialog_positions": [
                    {{
                        "text": "dialog text chunk",
                        "x": pixel_x_coordinate,
                        "y": pixel_y_coordinate,
                        "color": [r, g, b],
                        "background_needed": true/false,
                        "background_color": [r, g, b]
                    }}
                ],
                "regeneration_suggestions": "If not valid, specific suggestions for improvement",
                "rewritten_prompt": "If not valid, provide a complete rewritten prompt for image generation that addresses the identified issues"
            }}
            
            Image dimensions are approximately 1280x896 pixels (4:3 aspect ratio).
            Coordinate (0,0) is top-left corner.
            """
            
            # Call Gemini 2.5 Pro with the image
            response = self.genai_client.models.generate_content(
                model='gemini-2.5-pro-preview-05-06',
                contents=[
                    types.Part.from_bytes(
                        data=image_data,
                        mime_type='image/png'
                    ),
                    validation_prompt
                ]
            )
            
            # Parse the response
            response_text = response.text
            print(f"Gemini validation response: {response_text[:200]}...")
            
            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx == -1 or end_idx == 0:
                print("Warning: Could not find JSON in Gemini response, using defaults")
                return True, self._get_default_placement_info(story_text, dialog)
            
            json_str = response_text[start_idx:end_idx]
            result = json.loads(json_str)
            
            is_valid = result.get('is_valid', True)
            
            # Ensure we have valid placement info with proper defaults
            story_text_pos = result.get('story_text_position', {})
            if story_text_pos is None:
                story_text_pos = {}
            
            dialog_positions = result.get('dialog_positions', [])
            if dialog_positions is None:
                dialog_positions = []
            
            # Ensure story_text_position has all required fields with safe defaults
            if story_text_pos:
                story_text_pos.setdefault('x', 40)
                story_text_pos.setdefault('y', 800)
                story_text_pos.setdefault('color', [255, 255, 255])
                story_text_pos.setdefault('background_color', [0, 0, 0])
                story_text_pos.setdefault('background_needed', True)
                
                # Ensure color values are not None
                if story_text_pos['color'] is None:
                    story_text_pos['color'] = [255, 255, 255]
                if story_text_pos['background_color'] is None:
                    story_text_pos['background_color'] = [0, 0, 0]
            
            # Ensure dialog_positions have all required fields with safe defaults
            for dialog_info in dialog_positions:
                if dialog_info is not None:
                    dialog_info.setdefault('x', 40)
                    dialog_info.setdefault('y', 750)
                    dialog_info.setdefault('color', [255, 255, 200])
                    dialog_info.setdefault('background_color', [0, 0, 0])
                    dialog_info.setdefault('background_needed', True)
                    dialog_info.setdefault('text', '')
                    
                    # Ensure color values are not None
                    if dialog_info['color'] is None:
                        dialog_info['color'] = [255, 255, 200]
                    if dialog_info['background_color'] is None:
                        dialog_info['background_color'] = [0, 0, 0]
            
            placement_info = {
                'story_text_position': story_text_pos,
                'dialog_positions': dialog_positions,
                'validation_feedback': result.get('validation_feedback', ''),
                'regeneration_suggestions': result.get('regeneration_suggestions', ''),
                'rewritten_prompt': result.get('rewritten_prompt', '')
            }
            
            print(f"Image validation result: {'VALID' if is_valid else 'INVALID'}")
            if not is_valid:
                print(f"Validation feedback: {placement_info['validation_feedback']}")
                print(f"Suggestions: {placement_info['regeneration_suggestions']}")
            
            return is_valid, placement_info
            
        except Exception as e:
            print(f"Error in image validation: {e}")
            # Return default placement if validation fails
            return True, self._get_default_placement_info(story_text, dialog)
    
    def _get_default_placement_info(self, story_text: str, dialog: str) -> Dict[str, Any]:
        """Provide default text placement when Gemini validation fails"""
        placement_info = {
            'story_text_position': {
                'x': 40,
                'y': 800,  # Near bottom
                'color': [255, 255, 255],  # White
                'background_needed': True,
                'background_color': [0, 0, 0]  # Black background
            },
            'dialog_positions': [],
            'validation_feedback': 'Using default placement due to validation error',
            'regeneration_suggestions': '',
            'rewritten_prompt': ''
        }
        
        if dialog:
            # Split dialog into chunks if too long
            dialog_chunks = [dialog] if len(dialog) < 50 else [dialog[i:i+50] for i in range(0, len(dialog), 50)]
            for i, chunk in enumerate(dialog_chunks):
                placement_info['dialog_positions'].append({
                    'text': chunk,
                    'x': 40,
                    'y': 750 - (i * 30),  # Stack dialog above story text
                    'color': [255, 255, 200],  # Light yellow
                    'background_needed': True,
                    'background_color': [0, 0, 0]
                })
        
        return placement_info
    
    def generate_and_validate_image(self, page_data: Dict[str, Any], character_descriptions: Dict[str, Dict], 
                                   max_validation_retries: int = 2, max_generation_retries: int = 3) -> Tuple[str, Dict[str, Any]]:
        """
        Generate an image and validate it with Gemini 2.5 Pro, regenerating if necessary.
        
        Args:
            page_data: Page information including description, text, dialog, etc.
            character_descriptions: Character appearance descriptions
            max_validation_retries: Maximum retries for validation failures (default: 2)
            max_generation_retries: Maximum retries for image generation failures (default: 3)
        
        Returns:
            Tuple of (image_path, placement_info)
        """
        page_number = page_data['page_number']
        image_description = page_data['image_description']
        story_text = page_data['story_text']
        dialog = page_data.get('dialog', '')
        characters_on_page = page_data.get('characters_on_page', [])
        
        # Store the original description for potential reuse
        original_description = image_description
        custom_prompt_to_use = None
        
        for attempt in range(max_validation_retries + 1):
            print(f"🎨 Generating and validating image for page {page_number} (validation attempt {attempt + 1}/{max_validation_retries + 1})")
            
            # Generate the image (with its own retry logic)
            image_path = self.generate_image(
                image_description, 
                page_number, 
                characters_on_page, 
                character_descriptions,
                max_retries=max_generation_retries,
                custom_prompt=custom_prompt_to_use
            )
            
            # Validate the image and get placement info
            is_valid, placement_info = self.validate_image_and_get_dialog_placement(
                image_path, character_descriptions, image_description, 
                story_text, dialog, characters_on_page
            )
            
            if is_valid:
                print(f"✅ Image for page {page_number} validated successfully")
                return image_path, placement_info
            else:
                print(f"❌ Image for page {page_number} failed validation")
                if attempt < max_validation_retries:
                    print(f"🔄 Regenerating image for page {page_number} with improvements...")
                    
                    # Use Gemini's rewritten prompt if available, otherwise enhance the original
                    rewritten_prompt = placement_info.get('rewritten_prompt', '').strip()
                    if rewritten_prompt:
                        print(f"📝 Using Gemini's rewritten prompt for regeneration")
                        custom_prompt_to_use = rewritten_prompt
                        # Keep original description for validation context
                        image_description = original_description
                    elif placement_info.get('regeneration_suggestions'):
                        print(f"📝 Enhancing original prompt with suggestions")
                        enhanced_description = f"{original_description}\n\nIMPROVEMENT NEEDED: {placement_info['regeneration_suggestions']}"
                        image_description = enhanced_description
                        custom_prompt_to_use = None
                    else:
                        print(f"📝 No specific improvements provided, using original prompt")
                        custom_prompt_to_use = None
                else:
                    print(f"⚠️  Using image for page {page_number} despite validation failure after {max_validation_retries} retries")
                    return image_path, placement_info
        
        # This should never be reached, but just in case
        return image_path, placement_info
    
    def add_text_to_image(self, image_path: str, story_text: str, dialog: str, page_number: int, 
                         placement_info: Dict[str, Any] = None) -> str:
        """Add story text and dialog directly onto the image using OpenCV with Gemini-determined placement"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")

            height, width = img.shape[:2]
            
            # Configure text properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            story_font_scale = 0.8
            dialog_font_scale = 0.7
            thickness = 2
            
            def draw_text_with_background(img, text, position, font, font_scale, text_color, bg_color, thickness, bg_needed=True):
                """Draw text with optional background for better readability"""
                x, y = position
                
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                if bg_needed:
                    # Draw background rectangle
                    padding = 5
                    cv2.rectangle(img, 
                                (x - padding, y - text_height - padding), 
                                (x + text_width + padding, y + baseline + padding), 
                                bg_color, -1)
                
                # Draw text
                cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)
            
            def wrap_text_for_position(text, max_width, font, font_scale, thickness):
                """Wrap text to fit within specified width"""
                words = text.split()
                lines = []
                current_line = []
                
                for word in words:
                    test_line = ' '.join(current_line + [word])
                    (text_width, text_height), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
                    
                    if text_width < max_width:
                        current_line.append(word)
                    else:
                        if current_line:
                            lines.append(' '.join(current_line))
                            current_line = [word]
                        else:
                            lines.append(word)
                
                if current_line:
                    lines.append(' '.join(current_line))
                
                return lines
            
            # Use placement info if provided, otherwise use defaults
            if placement_info and isinstance(placement_info, dict):
                # Draw story text using Gemini-determined position
                if story_text and placement_info.get('story_text_position'):
                    pos_info = placement_info['story_text_position']
                    if pos_info and isinstance(pos_info, dict):
                        x = pos_info.get('x', 40)
                        y = pos_info.get('y', height - 100)
                        
                        # Safe color handling with null checks
                        color_val = pos_info.get('color', [255, 255, 255])
                        if color_val is None:
                            color_val = [255, 255, 255]
                        color = tuple(color_val)
                        
                        bg_color_val = pos_info.get('background_color', [0, 0, 0])
                        if bg_color_val is None:
                            bg_color_val = [0, 0, 0]
                        bg_color = tuple(bg_color_val)
                        
                        bg_needed = pos_info.get('background_needed', True)
                        
                        # Wrap text to fit within reasonable width
                        max_width = width - x - 40  # Leave margin on right
                        story_lines = wrap_text_for_position(story_text, max_width, font, story_font_scale, thickness)
                        
                        # Draw each line
                        line_height = 35
                        for i, line in enumerate(story_lines):
                            line_y = y + (i * line_height)
                            if line_y < height - 20:  # Don't draw below image
                                draw_text_with_background(img, line, (x, line_y), font, story_font_scale, 
                                                        color, bg_color, thickness, bg_needed)
                
                # Draw dialog using Gemini-determined positions
                dialog_positions = placement_info.get('dialog_positions', [])
                if dialog_positions is None:
                    dialog_positions = []
                    
                for dialog_info in dialog_positions:
                    if dialog_info is None or not isinstance(dialog_info, dict):
                        continue
                        
                    dialog_text = dialog_info.get('text', '')
                    if dialog_text:
                        x = dialog_info.get('x', 40)
                        y = dialog_info.get('y', height - 200)
                        
                        # Safe color handling with null checks
                        color_val = dialog_info.get('color', [255, 255, 200])
                        if color_val is None:
                            color_val = [255, 255, 200]
                        color = tuple(color_val)
                        
                        bg_color_val = dialog_info.get('background_color', [0, 0, 0])
                        if bg_color_val is None:
                            bg_color_val = [0, 0, 0]
                        bg_color = tuple(bg_color_val)
                        
                        bg_needed = dialog_info.get('background_needed', True)
                        
                        # Add quotes if not present
                        if not dialog_text.startswith('"'):
                            dialog_text = f'"{dialog_text}"'
                        
                        # Wrap dialog text
                        max_width = width - x - 40
                        dialog_lines = wrap_text_for_position(dialog_text, max_width, font, dialog_font_scale, thickness)
                        
                        # Draw each line
                        line_height = 30
                        for i, line in enumerate(dialog_lines):
                            line_y = y + (i * line_height)
                            if line_y < height - 20:  # Don't draw below image
                                draw_text_with_background(img, line, (x, line_y), font, dialog_font_scale, 
                                                        color, bg_color, thickness, bg_needed)
            
            else:
                # Fallback to original overlay method if no placement info
                # [Keep original overlay code as fallback]
                text_margin = 40
                max_text_width = width - (text_margin * 2)
                
                # Default colors
                text_color = (255, 255, 255)  # White text
                outline_color = (0, 0, 0)     # Black outline
                dialog_color = (255, 255, 200)  # Light yellow for dialog
                
                def draw_text_with_outline(img, text, position, font, font_scale, text_color, outline_color, thickness):
                    """Draw text with black outline for better readability"""
                    x, y = position
                    # Draw outline (thicker black text)
                    cv2.putText(img, text, (x, y), font, font_scale, outline_color, thickness + 2)
                    # Draw main text (white)
                    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)
                
                def wrap_text(text, max_width, font, font_scale, thickness):
                    """Wrap text to fit within specified width"""
                    words = text.split()
                    lines = []
                    current_line = []
                    
                    for word in words:
                        test_line = ' '.join(current_line + [word])
                        (text_width, text_height), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
                        
                        if text_width < max_width:
                            current_line.append(word)
                        else:
                            if current_line:
                                lines.append(' '.join(current_line))
                                current_line = [word]
                            else:
                                lines.append(word)
                    
                    if current_line:
                        lines.append(' '.join(current_line))
                    
                    return lines
                
                # Process story text
                story_lines = []
                if story_text:
                    story_lines = wrap_text(story_text, max_text_width, font, story_font_scale, thickness)
                
                # Process dialog text
                dialog_lines = []
                if dialog:
                    dialog_text = f'"{dialog}"'
                    dialog_lines = wrap_text(dialog_text, max_text_width, font, dialog_font_scale, thickness)
                
                # Calculate total text height needed
                line_height = 35
                total_lines = len(story_lines) + len(dialog_lines) + (1 if dialog_lines else 0)
                total_text_height = total_lines * line_height + 40
                
                # Create semi-transparent overlay for text area
                if story_lines or dialog_lines:
                    overlay_y_start = height - total_text_height - 20
                    overlay_y_end = height - 10
                    
                    # Create overlay
                    overlay = img.copy()
                    cv2.rectangle(overlay, (10, overlay_y_start), (width - 10, overlay_y_end), (0, 0, 0), -1)
                    
                    # Blend overlay with original image (semi-transparent)
                    alpha = 0.7
                    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                    
                    # Add text to the overlay area
                    y_offset = overlay_y_start + 30
                    
                    # Draw story text
                    for line in story_lines:
                        draw_text_with_outline(img, line, (text_margin, y_offset), font, story_font_scale, text_color, outline_color, thickness)
                        y_offset += line_height
                    
                    # Add space between story and dialog
                    if dialog_lines:
                        y_offset += 10
                    
                    # Draw dialog text
                    for line in dialog_lines:
                        draw_text_with_outline(img, line, (text_margin, y_offset), font, dialog_font_scale, dialog_color, outline_color, thickness)
                        y_offset += line_height
            
            # Add page number in top-right corner
            page_text = f"Page {page_number}"
            (page_text_width, page_text_height), _ = cv2.getTextSize(page_text, font, 0.6, 1)
            page_x = width - page_text_width - 20
            page_y = 30
            
            # Semi-transparent background for page number
            cv2.rectangle(img, (page_x - 10, page_y - page_text_height - 5), (page_x + page_text_width + 10, page_y + 5), (0, 0, 0), -1)
            cv2.rectangle(img, (page_x - 10, page_y - page_text_height - 5), (page_x + page_text_width + 10, page_y + 5), (255, 255, 255), 1)
            cv2.putText(img, page_text, (page_x, page_y), font, 0.6, (255, 255, 255), 1)
            
            # Save the final image
            final_path = self.images_dir / f"final_page_{page_number:02d}.png"
            cv2.imwrite(str(final_path), img)
            
            print(f"Added text to page {page_number}")
            return str(final_path)
            
        except Exception as e:
            print(f"Error adding text to page {page_number}: {e}")
            return image_path  # Return original if text overlay fails
    
    def create_pdf(self, image_paths: List[str], title: str) -> str:
        """Create PDF from images"""
        try:
            pdf_path = self.output_dir / f"{title.replace(' ', '_').lower()}_picture_book.pdf"
            
            # Create PDF document
            doc = SimpleDocTemplate(
                str(pdf_path),
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Build PDF content
            story = []
            
            # Add title page
            story.append(Spacer(1, 2*inch))
            
            for image_path in sorted(image_paths):
                if os.path.exists(image_path):
                    # Add image to PDF
                    img = RLImage(image_path, width=6*inch, height=4.5*inch)
                    story.append(img)
                    story.append(Spacer(1, 0.5*inch))
            
            # Build PDF
            doc.build(story)
            print(f"Created PDF: {pdf_path}")
            
            return str(pdf_path)
            
        except Exception as e:
            print(f"Error creating PDF: {e}")
            raise
    
    def generate_book(self, age: int, reference_story: str = None, story_structure_file: str = None, 
                     max_validation_retries: int = 2, max_generation_retries: int = 3) -> str:
        """Main function to generate the complete picture book"""
        print(f"Generating picture book for {age}-year-old...")
        if reference_story:
            print("Using reference story for adaptation...")
        
        # Step 1: Generate or load story structure
        if story_structure_file:
            print("Step 1: Loading story structure from file...")
            story_data = self.load_story_structure(story_structure_file)
        else:
            print("Step 1: Generating story structure...")
            story_data = self.generate_story_structure(age, reference_story)
        
        # Step 2: Generate images in parallel using ThreadPoolExecutor
        print(f"Step 2: Generating images using {self.max_workers} threads...")
        
        def generate_single_image(page_data):
            """Helper function to generate a single image"""
            return self.generate_and_validate_image(
                page_data, 
                story_data.get('characters', {}), 
                max_validation_retries=max_validation_retries,
                max_generation_retries=max_generation_retries
            )
        
        # Use ThreadPoolExecutor for parallel image generation
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all image generation tasks
            future_to_page = {
                executor.submit(generate_single_image, page): page 
                for page in story_data['pages']
            }
            
            # Collect results as they complete
            image_results = [None] * len(story_data['pages'])
            for future in as_completed(future_to_page):
                page = future_to_page[future]
                try:
                    image_path, placement_info = future.result()
                    # Store in correct order based on page number
                    page_index = page['page_number'] - 1
                    image_results[page_index] = (image_path, placement_info)
                except Exception as e:
                    print(f"Error generating image for page {page['page_number']}: {e}")
                    # Create placeholder result
                    page_index = page['page_number'] - 1
                    placeholder_path = str(self.images_dir / f"page_{page['page_number']:02d}.png")
                    default_placement = self._get_default_placement_info(page['story_text'], page.get('dialog', ''))
                    image_results[page_index] = (placeholder_path, default_placement)
        
        # Step 3: Add text to images
        print("Step 3: Adding text to images...")
        final_image_paths = []
        
        for i, page in enumerate(story_data['pages']):
            if i < len(image_results) and image_results[i]:
                image_path, placement_info = image_results[i]
                final_path = self.add_text_to_image(
                    image_path,
                    page['story_text'],
                    page.get('dialog', ''),
                    page['page_number'],
                    placement_info
                )
                final_image_paths.append(final_path)
        
        # Step 4: Create PDF
        print("Step 4: Creating PDF...")
        pdf_path = self.create_pdf(final_image_paths, story_data['title'])
        
        print(f"Picture book '{story_data['title']}' generated successfully!")
        print(f"PDF saved to: {pdf_path}")
        
        return pdf_path

def main():
    parser = argparse.ArgumentParser(description='Generate a picture book for children')
    parser.add_argument('age', type=int, nargs='?', help='Target age for the child (e.g., 5)')
    parser.add_argument('--reference', '-r', type=str, help='Reference story text or path to text file')
    parser.add_argument('--reference-file', '-f', type=str, help='Path to file containing reference story')
    parser.add_argument('--story-structure', '-s', type=str, help='Path to JSON file containing pre-generated story structure')
    parser.add_argument('--list-stories', '-l', action='store_true', help='List available story structure files')
    parser.add_argument('--max-workers', '-m', type=int, default=4, help='Maximum number of worker threads for image generation (default: 4)')
    parser.add_argument('--generation-retries', '-g', type=int, default=3, help='Maximum retries for image generation failures (default: 3)')
    parser.add_argument('--validation-retries', '-v', type=int, default=2, help='Maximum retries for validation failures (default: 2)')
    
    args = parser.parse_args()
    
    # Handle list stories option
    if args.list_stories:
        generator = PictureBookGenerator()
        story_files = generator.list_story_files()
        if story_files:
            print("📚 Available story structure files:")
            for i, file_path in enumerate(story_files, 1):
                file_name = os.path.basename(file_path)
                # Try to extract title and timestamp from filename
                parts = file_name.replace('story_structure_', '').replace('.json', '').split('_')
                if len(parts) >= 2:
                    title_part = '_'.join(parts[:-1]).replace('_', ' ').title()
                    timestamp_part = parts[-1]
                    print(f"  {i}. {title_part} ({timestamp_part})")
                    print(f"     File: {file_path}")
                else:
                    print(f"  {i}. {file_name}")
                    print(f"     File: {file_path}")
                print()
        else:
            print("📚 No story structure files found in output directory.")
        return
    
    if args.age is None:
        print("Error: age is required unless using --list-stories")
        parser.print_help()
        sys.exit(1)
    
    if args.age < 1 or args.age > 12:
        print("Age should be between 1 and 12 years old")
        sys.exit(1)
    
    # Handle reference story input
    reference_story = None
    if args.reference_file:
        try:
            with open(args.reference_file, 'r', encoding='utf-8') as f:
                reference_story = f.read().strip()
            print(f"Loaded reference story from: {args.reference_file}")
        except FileNotFoundError:
            print(f"Error: Reference file '{args.reference_file}' not found")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading reference file: {e}")
            sys.exit(1)
    elif args.reference:
        reference_story = args.reference
        print("Using provided reference story text")
    
    # Validate story structure file if provided
    if args.story_structure:
        if not os.path.exists(args.story_structure):
            print(f"Error: Story structure file '{args.story_structure}' not found")
            sys.exit(1)
        print(f"Using pre-generated story structure from: {args.story_structure}")
    
    # Create generator and run
    generator = PictureBookGenerator(max_workers=args.max_workers)
    print(f"Using {args.max_workers} worker threads for image generation")
    print(f"Image generation retries: {args.generation_retries}")
    print(f"Validation retries: {args.validation_retries}")
    
    try:
        # Run the function
        pdf_path = generator.generate_book(
            args.age, 
            reference_story, 
            story_structure_file=args.story_structure,
            max_validation_retries=args.validation_retries,
            max_generation_retries=args.generation_retries
        )
        print(f"\n✅ Success! Picture book generated: {pdf_path}")
        
    except Exception as e:
        print(f"\n❌ Error generating picture book: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
