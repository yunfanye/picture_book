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
import asyncio
import argparse
from typing import List, Dict, Any
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

# Load environment variables
load_dotenv()

class PictureBookGenerator:
    def __init__(self):
        # Initialize API clients
        self.claude_client = anthropic.Anthropic(
            api_key=os.getenv('ANTHROPIC_API_KEY')
        )
        
        # Initialize Google GenAI client for Imagen
        self.genai_client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
        
        # Create output directories
        self.output_dir = Path("output")
        self.images_dir = self.output_dir / "images"
        self.output_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
    
    def generate_story_structure(self, age: int, reference_story: str = None) -> Dict[str, Any]:
        """Generate story structure using Claude 4"""
        
        reference_prompt = ""
        page_guidance = "The book should have 20-30 pages with engaging content appropriate for this age group."
        
        if reference_story:
            reference_prompt = f"""
            
            REFERENCE STORY TO ADAPT:
            {reference_story}
            
            Please adapt this reference story to be age-appropriate for a {age}-year-old while maintaining the core narrative structure and themes.
            
            IMPORTANT ADAPTATION GUIDELINES:
            - Maintain the story's pacing and key plot points
            - Each page should represent a meaningful story moment or scene
            - Don't rush through the story - give each important event its own page(s)
            - For repetitive elements (like the hen asking for help), create separate pages for each instance
            - Include setup, character introduction, problem development, climax, and resolution
            - Ensure smooth narrative flow across all pages
            """
        
        prompt = f"""
        Create a children's picture book story suitable for a {age}-year-old child. 
        {page_guidance}
        {reference_prompt}
        
        IMPORTANT: First, define all the main characters in the story with detailed physical descriptions that will be used consistently throughout all illustrations.
        
        Please provide a structured response in JSON format with:
        1. A compelling title
        2. A brief summary
        3. A "characters" section with detailed physical descriptions for each main character
        4. For each page, provide:
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
        
        PACING AND STRUCTURE REQUIREMENTS:
        - Don't rush through the story - each important moment deserves its own page
        - Include character introduction pages
        - Show the problem/conflict developing gradually
        - Include emotional moments and character reactions
        - Build to a satisfying climax and resolution
        - For repetitive story elements, create separate pages for each instance
        
        Make sure the story has:
        - Age-appropriate vocabulary and concepts
        - Engaging characters with consistent appearances
        - A clear beginning, middle, and end
        - Positive messages and learning opportunities
        - Vivid, descriptive scenes for illustration
        - Proper story pacing with adequate page count
        
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
                max_tokens=16000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract JSON from response
            content = response.content[0].text
            # Find JSON in the response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            json_str = content[start_idx:end_idx]
            
            story_data = json.loads(json_str)
            print(f"Generated story: '{story_data['title']}'")
            print(f"Number of pages: {len(story_data['pages'])}")
            print(f"Characters defined: {list(story_data.get('characters', {}).keys())}")
            
            return story_data
            
        except Exception as e:
            print(f"Error generating story: {e}")
            raise
    
    async def generate_image(self, description: str, page_number: int, characters_on_page: List[str] = None, character_descriptions: Dict[str, Dict] = None) -> str:
        """Generate image using Imagen 3 with character consistency"""
        try:
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
            
            print(f"Generating image for page {page_number}...")
            
            # Generate image using Imagen 3
            response = self.genai_client.models.generate_images(
                model='imagen-3.0-generate-002',
                prompt=enhanced_prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio="4:3",  # Good for picture books
                    safety_filter_level="BLOCK_LOW_AND_ABOVE",
                    person_generation="ALLOW_ADULT"
                )
            )
            
            # Save the generated image
            image_path = self.images_dir / f"page_{page_number:02d}.png"
            
            if response.generated_images:
                # Get the first (and only) generated image
                generated_image = response.generated_images[0]
                
                # Convert bytes to PIL Image and save
                image = Image.open(BytesIO(generated_image.image.image_bytes))
                image.save(image_path)
                
                print(f"Generated image for page {page_number}")
                return str(image_path)
            else:
                raise Exception("No images generated")
            
        except Exception as e:
            print(f"Error generating image for page {page_number}: {e}")
            print("Creating placeholder image...")
            
            # Create a placeholder image if generation fails
            image_path = self.images_dir / f"page_{page_number:02d}.png"
            img = Image.new('RGB', (1280, 896), color='lightblue')  # 4:3 aspect ratio
            draw = ImageDraw.Draw(img)
            
            # Try to load a font, fallback to default if not available
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            # Add character info to placeholder if available
            placeholder_text = description
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
    
    def add_text_to_image(self, image_path: str, story_text: str, dialog: str, page_number: int) -> str:
        """Add story text and dialog directly onto the image using OpenCV"""
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
            
            # Colors
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
            
            # Calculate text area dimensions
            text_margin = 40
            max_text_width = width - (text_margin * 2)
            
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
            total_lines = len(story_lines) + len(dialog_lines) + (1 if dialog_lines else 0)  # Extra space between story and dialog
            total_text_height = total_lines * line_height + 40  # Extra padding
            
            # Create semi-transparent overlay for text area
            if story_lines or dialog_lines:
                # Position text area at bottom of image
                overlay_y_start = height - total_text_height - 20
                overlay_y_end = height - 10
                
                # Create overlay
                overlay = img.copy()
                cv2.rectangle(overlay, (10, overlay_y_start), (width - 10, overlay_y_end), (0, 0, 0), -1)
                
                # Blend overlay with original image (semi-transparent)
                alpha = 0.7  # Transparency level
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
    
    async def generate_book(self, age: int, reference_story: str = None) -> str:
        """Main function to generate the complete picture book"""
        print(f"Generating picture book for {age}-year-old...")
        if reference_story:
            print("Using reference story for adaptation...")
        
        # Step 1: Generate story structure
        print("Step 1: Generating story structure...")
        story_data = self.generate_story_structure(age, reference_story)
        
        # Step 2: Generate images in parallel
        print("Step 2: Generating images...")
        image_tasks = []
        
        async def generate_all_images():
            tasks = []
            for page in story_data['pages']:
                task = self.generate_image(
                    page['image_description'], 
                    page['page_number'], 
                    page.get('characters_on_page', []), 
                    story_data.get('characters', {})
                )
                tasks.append(task)
            
            return await asyncio.gather(*tasks)
        
        image_paths = await generate_all_images()
        
        # Step 3: Add text to images
        print("Step 3: Adding text to images...")
        final_image_paths = []
        
        for i, page in enumerate(story_data['pages']):
            if i < len(image_paths):
                final_path = self.add_text_to_image(
                    image_paths[i],
                    page['story_text'],
                    page.get('dialog', ''),
                    page['page_number']
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
    parser.add_argument('age', type=int, help='Target age for the child (e.g., 5)')
    parser.add_argument('--reference', '-r', type=str, help='Reference story text or path to text file')
    parser.add_argument('--reference-file', '-f', type=str, help='Path to file containing reference story')
    
    args = parser.parse_args()
    
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
    
    # Create generator and run
    generator = PictureBookGenerator()
    
    try:
        # Run the async function
        pdf_path = asyncio.run(generator.generate_book(args.age, reference_story))
        print(f"\n✅ Success! Picture book generated: {pdf_path}")
        
    except Exception as e:
        print(f"\n❌ Error generating picture book: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
