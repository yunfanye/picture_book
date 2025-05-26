#!/usr/bin/env python3
"""
Image Generation Module for Picture Book Generator

Supports multiple image generation backends:
1. Imagen 3 (Google GenAI)
2. Gemini Flash (Google GenAI)
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from google import genai
from google.genai import types
from io import BytesIO


class ImageGenerator:
    def __init__(self, backend: str = "imagen", api_key: str = None):
        """
        Initialize image generator with specified backend
        
        Args:
            backend: "imagen" or "gemini" for image generation backend
            api_key: Google GenAI API key
        """
        self.backend = backend.lower()
        if self.backend not in ["imagen", "gemini"]:
            raise ValueError("Backend must be 'imagen' or 'gemini'")
        
        # Initialize Google GenAI client
        self.genai_client = genai.Client(api_key=api_key or os.getenv('GEMINI_API_KEY'))
        
        print(f"🎨 Initialized image generator with {self.backend.upper()} backend")
    
    def generate_image(self, description: str, page_number: int, characters_on_page: List[str] = None, 
                      character_descriptions: Dict[str, Dict] = None, max_retries: int = 3, 
                      custom_prompt: str = None, images_dir: Path = None) -> str:
        """
        Generate image using the configured backend
        
        Args:
            description: Image description
            page_number: Page number for naming
            characters_on_page: List of character names appearing on this page
            character_descriptions: Character appearance descriptions
            max_retries: Maximum retry attempts
            custom_prompt: Custom prompt to use instead of building one
            images_dir: Directory to save images
            
        Returns:
            Path to generated image file
        """
        if images_dir is None:
            images_dir = Path("output/images")
            images_dir.mkdir(parents=True, exist_ok=True)
        
        # Use custom prompt if provided, otherwise build enhanced prompt
        if custom_prompt:
            print(f"🎯 Using custom rewritten prompt for page {page_number}")
            enhanced_prompt = custom_prompt
        else:
            enhanced_prompt = self._build_enhanced_prompt(description, characters_on_page, character_descriptions)
        
        # Retry logic for image generation
        for attempt in range(max_retries):
            try:
                print(f"Generating image for page {page_number} using {self.backend.upper()} (attempt {attempt + 1}/{max_retries})...")
                
                if self.backend == "imagen":
                    image_path = self._generate_with_imagen(enhanced_prompt, page_number, images_dir)
                else:  # gemini
                    image_path = self._generate_with_gemini(enhanced_prompt, page_number, images_dir)
                
                if image_path:
                    print(f"✅ Successfully generated image for page {page_number}")
                    return image_path
                else:
                    print(f"❌ No image generated for page {page_number} on attempt {attempt + 1}")
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
        return self._create_placeholder_image(description, page_number, characters_on_page, character_descriptions, images_dir)
    
    def _build_enhanced_prompt(self, description: str, characters_on_page: List[str] = None, 
                              character_descriptions: Dict[str, Dict] = None) -> str:
        """Build enhanced prompt for image generation"""
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
        - Aspect ratio: 4:3
        """
        
        return enhanced_prompt
    
    def _generate_with_imagen(self, prompt: str, page_number: int, images_dir: Path) -> Optional[str]:
        """Generate image using Imagen 3"""
        try:
            response = self.genai_client.models.generate_images(
                model='imagen-3.0-generate-002',
                prompt=prompt,
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
                image_path = images_dir / f"page_{page_number:02d}.png"
                
                # Get the first (and only) generated image
                generated_image = response.generated_images[0]
                
                # Convert bytes to PIL Image and save
                image = Image.open(BytesIO(generated_image.image.image_bytes))
                image.save(image_path)
                
                return str(image_path)
            
            return None
            
        except Exception as e:
            print(f"Imagen generation error: {e}")
            return None
    
    def _generate_with_gemini(self, prompt: str, page_number: int, images_dir: Path) -> Optional[str]:
        """Generate image using Gemini Flash"""
        try:
            response = self.genai_client.models.generate_content(
                model="gemini-2.5-flash-preview-image-generation",
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )
            
            # Look for image in response
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    # Save the generated image
                    image_path = images_dir / f"page_{page_number:02d}.png"
                    
                    # Convert inline data to PIL Image and save
                    image = Image.open(BytesIO(part.inline_data.data))
                    image.save(image_path)
                    
                    return str(image_path)
                elif part.text is not None:
                    print(f"Gemini response text: {part.text}")
            
            return None
            
        except Exception as e:
            print(f"Gemini generation error: {e}")
            return None
    
    def _create_placeholder_image(self, description: str, page_number: int, characters_on_page: List[str] = None, 
                                 character_descriptions: Dict[str, Dict] = None, images_dir: Path = None) -> str:
        """Create placeholder image when generation fails"""
        print("Creating placeholder image...")
        image_path = images_dir / f"page_{page_number:02d}.png"
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